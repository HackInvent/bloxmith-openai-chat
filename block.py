# -----------------------------------------------------------------------------
# Role: Implements the OpenAI Chat block runtime and UI contract.
# File Name: block.py
# Author: Alexandre EL
# Email: alex@hackinvent.com
# Created Date: 2026-05-19
# -----------------------------------------------------------------------------

from __future__ import annotations

from html import escape
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
import json
import os
import re
import time

from bloxsmith_app.block_api import (
    append_chat_exchange,
    APPLICATION_JSON,
    BlockDefinition,
    BlockRuntimeContext,
    BlockRuntimeOutput,
    BlockRuntimeResult,
    load_chat_history,
    normalize_history_turns,
    render_inspector_template,
    render_node_card_template,
    TEXT_PLAIN,
)


OPENAI_CHAT_MODELS = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-chat-latest",
    "chat-latest",
)
OPENAI_REASONING_EFFORTS = ("default", "none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_OPENAI_CHAT_MODEL = "gpt-5.5"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_MAX_PROMPT_CHARS = 250_000
MAX_TIMEOUT_SEC = 3600
MAX_PROMPT_CHARS = 1_000_000
MAX_OUTPUT_TOKENS_LIMIT = 128_000


# Functional behavior:
# FB1 - Build one OpenAI chat prompt from current inputs and the response output instruction.
# FB2 - Call the OpenAI-compatible Responses API at /v1/responses with a GPT-5.x chat model.
# FB3 - Emit normalized assistant text on text outputs and the full response JSON on raw_json outputs.
# FB4 - Refuse missing API keys and oversized prompts before calling the API.
# FB5 - Capture model, endpoint, usage, request id, and response metadata without leaking api_key.
# FB6 - Render model/API configuration through block-owned inspector, modal, and node card UI.
# FB7 - Run through the generic runtime path used by both centralized and zeromq_active execution modes.
# FB8 - Reuse and update the last configured user/assistant exchanges through the run-local chat history store.
class OpenAiChatBlockError(ValueError):
    """Raised when the OpenAI Chat block cannot generate a response."""


class OpenAiChatBlock(BlockDefinition):
    """Autonomous block implementation for `OpenAiChatBlock`."""
    kind = "openai_chat"

    def render_node_card(self, *, node: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the OpenAI Chat canvas card from the block-owned template."""

        config = self._ui_config(node)
        return render_node_card_template(
            block=self,
            node=node,
            node_classes=["openai-chat-node"],
            replacements={
                "title": node.get("title") or self.default_title(),
                "instruction": self._truncate(config["instruction"] or "Instruction vide", 62),
                "model": config["model"],
                "effort": config["reasoning_effort"],
            },
        )

    def render_modal(self, *, node: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the OpenAI Chat modal with prompt, attributes, and last response tabs."""

        payload = payload or {}
        title = str(node.get("title") or self.default_title())
        config = self._ui_config(node)
        template = (self.directory / "block_modal.html").read_text(encoding="utf-8")
        replacements = {
            "node_id": escape(str(node.get("id") or ""), quote=True),
            "node_title": escape(title),
            "node_kind": escape(self.kind, quote=True),
            "node_kind_title": escape(str(self.model.get("title") or self.default_title())),
            "modal_tabs_html": self._render_modal_tabs(node, title, config, payload),
        }
        html = template
        for key, value in replacements.items():
            html = html.replace(f"{{{{ {key} }}}}", str(value))
        return {"html": html, "context": {"node_id": str(node.get("id") or ""), "node_kind": self.kind}}

    def render_inspector_panel(self, *, node: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the OpenAI Chat inspector with generic field bindings."""

        config = self._ui_config(node)
        template = (self.directory / "inspector_panel.html").read_text(encoding="utf-8")
        html = render_inspector_template(
            template=(
                template
                .replace("{{ instruction }}", escape(config["instruction"]))
                .replace("{{ instruction_output_id }}", str(config["instruction_output_id"]))
                .replace("{{ model_options }}", self._select_options(OPENAI_CHAT_MODELS, config["model"]))
                .replace(
                    "{{ reasoning_effort_options }}",
                    self._select_options(OPENAI_REASONING_EFFORTS, config["reasoning_effort"]),
                )
                .replace("{{ api_key_placeholder }}", "Cle configuree" if config["api_key"] else "sk-...")
                .replace("{{ api_base_url }}", escape(config["api_base_url"], quote=True))
                .replace("{{ system_instruction }}", escape(config["system_instruction"]))
                .replace("{{ max_output_tokens }}", str(config["max_output_tokens"]))
                .replace("{{ history_turns }}", str(config["history_turns"]))
                .replace("{{ timeout_sec }}", str(config["timeout_sec"]))
                .replace("{{ max_prompt_chars }}", str(config["max_prompt_chars"]))
            ),
            node={**node, "type": self.kind, "kind": self.kind},
            payload=payload,
        )
        return {
            "html": html,
            "context": {
                "node_id": str(node.get("id") or ""),
                "api_key_configured": bool(config["api_key"]),
                "full_panel": True,
            },
        }

    def execute_runtime(self, context: BlockRuntimeContext) -> BlockRuntimeResult:
        """Call OpenAI Responses API and publish text/raw JSON outputs."""

        logs: list[str] = []
        started = time.perf_counter()
        try:
            config = self.normalize_config(context.config)
            instruction = self._response_instruction(context)
            prompt = self._build_prompt(context, instruction=instruction)
            history = load_chat_history(context, turns=int(config["history_turns"]))
            messages = [*history, {"role": "user", "content": prompt}]
            request_chars = self._messages_char_count(messages)
            if not config["api_key"]:
                raise OpenAiChatBlockError("api_key OpenAI manquante.")
            if not prompt.strip():
                raise OpenAiChatBlockError("Prompt OpenAI Chat vide.")
            if request_chars > int(config["max_prompt_chars"]):
                raise OpenAiChatBlockError(
                    "Prompt OpenAI Chat trop long: "
                    f"{request_chars} caracteres > limite {config['max_prompt_chars']}."
                )

            endpoint = self._responses_endpoint(config["api_base_url"])
            logs.append(
                f"[openai-chat] {context.node_id}: model={config['model']} "
                f"endpoint={endpoint} prompt_chars={request_chars} "
                f"history_turns={len(history) // 2} timeout={config['timeout_sec']}s."
            )
            response = self._post_response(messages=messages, config=config, endpoint=endpoint)
            answer = self._extract_output_text(response)
            updated_history = append_chat_exchange(
                context,
                user_message=prompt,
                assistant_message=answer,
                turns=int(config["history_turns"]),
            )
            raw_json = json.dumps(response, ensure_ascii=False, indent=2)
            outputs = self._runtime_outputs(context, answer=answer, raw_json=raw_json)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            request_id = str(response.get("id") or "")
            logs.append(
                f"[done] OpenAI Chat {context.node_id}: {len(answer)} caractere(s) "
                f"usage_total={usage.get('total_tokens', '?')}."
            )
            return BlockRuntimeResult(
                status="success",
                outputs=outputs,
                logs=logs,
                last_message=answer,
                content_type=TEXT_PLAIN,
                worker_received=answer or "-",
                metadata={
                    "openai_chat": {
                        "model": config["model"],
                        "endpoint": endpoint,
                        "request_id": request_id,
                        "usage": usage,
                        "duration": round(time.perf_counter() - started, 3),
                        "history_turns": len(updated_history) // 2,
                    },
                    "last_openai_chat_request": self._last_request_preview(config=config, messages=messages),
                },
            )
        except OpenAiChatBlockError as exc:
            message = self._mask_secret(str(exc), context.config.get("api_key") if isinstance(context.config, dict) else "")
            logs.append(f"[openai-chat-error] {context.node_id}: {message}")
            return BlockRuntimeResult(
                status="failed",
                outputs=[],
                logs=logs,
                error=message,
                exit_code=1,
                last_message=message,
                content_type=TEXT_PLAIN,
                worker_received="-",
            )

    def normalize_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        """Return safe OpenAI Chat runtime configuration from raw node config."""

        raw = config if isinstance(config, dict) else {}
        return {
            "model": self._normalize_model(raw.get("model")),
            "api_key": str(raw.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip(),
            "api_base_url": self._normalize_base_url(raw.get("api_base_url")),
            "system_instruction": str(raw.get("system_instruction") or "").strip(),
            "reasoning_effort": self._normalize_reasoning_effort(raw.get("reasoning_effort")),
            "max_output_tokens": self._normalize_int(
                raw.get("max_output_tokens"),
                default=0,
                minimum=0,
                maximum=MAX_OUTPUT_TOKENS_LIMIT,
            ),
            "history_turns": normalize_history_turns(raw.get("history_turns"), default=5),
            "timeout_sec": self._normalize_int(
                raw.get("timeout_sec"),
                default=DEFAULT_TIMEOUT_SEC,
                minimum=1,
                maximum=MAX_TIMEOUT_SEC,
            ),
            "max_prompt_chars": self._normalize_int(
                raw.get("max_prompt_chars"),
                default=DEFAULT_MAX_PROMPT_CHARS,
                minimum=1,
                maximum=MAX_PROMPT_CHARS,
            ),
        }

    def _post_response(self, *, messages: list[dict[str, str]], config: dict[str, Any], endpoint: str) -> dict[str, Any]:
        """Send one JSON request to the OpenAI Responses API."""

        payload: dict[str, Any] = {
            "model": config["model"],
            "input": messages,
        }
        if config["system_instruction"]:
            payload["instructions"] = config["system_instruction"]
        if config["reasoning_effort"] != "default":
            payload["reasoning"] = {"effort": config["reasoning_effort"]}
        if int(config["max_output_tokens"]) > 0:
            payload["max_output_tokens"] = int(config["max_output_tokens"])

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urlrequest.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlrequest.urlopen(request, timeout=int(config["timeout_sec"])) as response:
                response_body = response.read().decode("utf-8", errors="replace")
        except urlerror.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise OpenAiChatBlockError(
                f"OpenAI Chat HTTP {exc.code}: {self._mask_secret(response_body, config['api_key'])[:800]}"
            ) from exc
        except urlerror.URLError as exc:
            raise OpenAiChatBlockError(f"OpenAI Chat API inaccessible: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OpenAiChatBlockError(f"Timeout OpenAI Chat apres {config['timeout_sec']}s.") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise OpenAiChatBlockError(f"Reponse OpenAI Chat non JSON: {response_body[:600]}") from exc
        if not isinstance(parsed, dict):
            raise OpenAiChatBlockError("Reponse OpenAI Chat invalide.")
        return parsed

    def _build_prompt(self, context: BlockRuntimeContext, *, instruction: str) -> str:
        """Compose a prompt from named inputs and the response instruction."""

        input_sections: list[str] = []
        seen: set[str] = set()
        for input_port in context.input_ports:
            port_id = str(getattr(input_port, "id", "") or "")
            port_name = str(getattr(input_port, "name", "") or "").strip() or port_id
            if port_name in seen:
                continue
            seen.add(port_name)
            value = str(context.input_value(port_name, port_id) or "")
            if not value:
                continue
            input_sections.append(
                "\n".join(
                    [
                        f"[BEGIN INPUT {port_name}]",
                        value,
                        f"[END INPUT {port_name}]",
                    ]
                )
            )

        parts: list[str] = []
        if input_sections:
            parts.append("Voici les donnees recues.\n\n" + "\n\n".join(input_sections))
        if instruction.strip():
            parts.append("Instruction:\n\n" + instruction.strip())
        return "\n\n".join(parts).strip()

    def _response_instruction(self, context: BlockRuntimeContext) -> str:
        """Return the primary text-output instruction used for the API request."""

        for output_port in context.output_ports:
            name = str(getattr(output_port, "name", "") or "").strip().lower()
            if name == "raw_json":
                continue
            instruction = str(getattr(output_port, "instruction", "") or "").strip()
            if instruction:
                return instruction
        return ""

    def _runtime_outputs(self, context: BlockRuntimeContext, *, answer: str, raw_json: str) -> list[BlockRuntimeOutput]:
        """Map the OpenAI response text and raw JSON to declared output ports."""

        outputs: list[BlockRuntimeOutput] = []
        for output_port in context.output_ports:
            port_id = int(getattr(output_port, "id", 0) or 0)
            port_name = str(getattr(output_port, "name", "") or "")
            emits = getattr(output_port, "emits", getattr(output_port, "accepts", ()))
            emits_values = tuple(str(item) for item in emits) if isinstance(emits, (list, tuple)) else ()
            is_raw_json = port_name == "raw_json" or APPLICATION_JSON in emits_values
            outputs.append(
                BlockRuntimeOutput(
                    port_id=port_id,
                    port_name=port_name,
                    value=raw_json if is_raw_json else answer,
                    content_type=APPLICATION_JSON if is_raw_json else TEXT_PLAIN,
                )
            )
        return outputs

    def _extract_output_text(self, response: dict[str, Any]) -> str:
        """Extract assistant text from the current Responses API shape."""

        direct_text = response.get("output_text")
        if isinstance(direct_text, str):
            return direct_text

        collected: list[str] = []
        raw_output = response.get("output")
        if isinstance(raw_output, list):
            for output_item in raw_output:
                if not isinstance(output_item, dict):
                    continue
                content = output_item.get("content")
                if isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, dict):
                            text = content_item.get("text")
                            if isinstance(text, str):
                                collected.append(text)
                            elif isinstance(content_item.get("content"), str):
                                collected.append(str(content_item["content"]))
                        elif isinstance(content_item, str):
                            collected.append(content_item)
                elif isinstance(content, str):
                    collected.append(content)
        if collected:
            return "\n".join(part for part in collected if part)

        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return str(message["content"])
        return ""

    def _ui_config(self, node: dict[str, Any]) -> dict[str, Any]:
        """Return normalized values used by the inspector, modal, and node card."""

        raw_config = node.get("config") if isinstance(node.get("config"), dict) else {}
        config = self.normalize_config(raw_config)
        outputs = node.get("outputs") if isinstance(node.get("outputs"), list) else []
        first_text_output = next(
            (
                port
                for port in outputs
                if isinstance(port, dict) and str(port.get("name") or "").strip().lower() != "raw_json"
            ),
            {},
        )
        return {
            **config,
            "instruction": str(first_text_output.get("instruction") or ""),
            "instruction_output_id": int(first_text_output.get("id") or 1),
        }

    def _render_modal_tabs(
        self,
        node: dict[str, Any],
        title: str,
        config: dict[str, Any],
        payload: dict[str, Any],
    ) -> str:
        """Render the tabbed OpenAI Chat modal content."""

        node_dom_id = self._safe_dom_id(str(node.get("id") or "openai-chat"))
        response_output = self._first_response_output(node)
        response_port_id = self._dict_port_id(response_output, 1)
        tabs = [
            self._render_tab(
                tab_id="prompt",
                tab_dom_id=f"openai-chat-{node_dom_id}-tab-prompt",
                panel_dom_id=f"openai-chat-{node_dom_id}-panel-prompt",
                title="Prompt",
                subtitle=f"#{response_port_id} - response",
                selected=True,
            ),
            self._render_tab(
                tab_id="attributes",
                tab_dom_id=f"openai-chat-{node_dom_id}-tab-attributes",
                panel_dom_id=f"openai-chat-{node_dom_id}-panel-attributes",
                title="Attributs",
                subtitle="Modele et API",
                selected=False,
            ),
            self._render_tab(
                tab_id="last-response",
                tab_dom_id=f"openai-chat-{node_dom_id}-tab-last-response",
                panel_dom_id=f"openai-chat-{node_dom_id}-panel-last-response",
                title="Last response",
                subtitle="Reponse brute",
                selected=False,
            ),
        ]
        panels = [
            self._render_prompt_panel(
                node_dom_id=node_dom_id,
                selected=True,
                config=config,
                node=node,
                response_output=response_output,
            ),
            self._render_attributes_panel(
                node_dom_id=node_dom_id,
                selected=False,
                title=title,
                config=config,
                node=node,
                payload=payload,
            ),
            self._render_last_response_panel(
                node_dom_id=node_dom_id,
                selected=False,
                response=self._ui_last_response(payload),
            ),
        ]
        return (
            '<div class="openai-chat-modal-body" data-openai-chat-modal-tabs>'
            '<nav class="openai-chat-modal-tablist" role="tablist" aria-label="Configuration OpenAI Chat">'
            + "".join(tabs)
            + '</nav>'
            + '<div class="openai-chat-modal-panels">'
            + "".join(panels)
            + '</div>'
            + '</div>'
        )

    def _render_prompt_panel(
        self,
        *,
        node_dom_id: str,
        selected: bool,
        config: dict[str, Any],
        node: dict[str, Any],
        response_output: dict[str, Any],
    ) -> str:
        """Render the primary prompt editor panel."""

        panel_id = f"openai-chat-{node_dom_id}-panel-prompt"
        tab_id = f"openai-chat-{node_dom_id}-tab-prompt"
        port_id = self._dict_port_id(response_output, 1)
        instruction = str(response_output.get("instruction") or "")
        return (
            '<section class="openai-chat-modal-panel openai-chat-prompt-panel" data-openai-chat-modal-panel '
            f'data-openai-chat-tab-id="prompt" id="{escape(panel_id, quote=True)}" role="tabpanel" '
            f'aria-labelledby="{escape(tab_id, quote=True)}"{"" if selected else " hidden"}>'
            '<div class="openai-chat-prompt-layout">'
            '<div class="openai-chat-prompt-editor">'
            '<div class="openai-chat-prompt-header">'
            '<div>'
            '<span class="group-label">OpenAI Responses API</span>'
            '<h3>Prompt</h3>'
            f'<p>Modele <code>{escape(config["model"])}</code> via <code>/v1/responses</code>.</p>'
            '</div>'
            '</div>'
            '<div class="field-group openai-chat-system-field">'
            '<label>System instruction</label>'
            '<textarea data-block-config-field="system_instruction" rows="5" spellcheck="false" '
            'placeholder="Role, ton, contraintes globales du modele.">'
            f'{escape(config["system_instruction"])}'
            '</textarea>'
            '</div>'
            '<div class="field-group openai-chat-instruction-field">'
            '<label>Instruction de sortie</label>'
            '<textarea data-openai-chat-instruction data-block-output-field="instruction" '
            f'data-block-output-port-id="{escape(str(port_id), quote=True)}" rows="18" spellcheck="false" '
            'placeholder="Decris ce que le modele doit produire avec les inputs recus.">'
            f'{escape(instruction)}'
            '</textarea>'
            '</div>'
            '</div>'
            '<aside class="openai-chat-reference-panel">'
            '<div class="ports-editor-header"><span class="group-label">Inputs disponibles</span></div>'
            '<p class="field-hint">Les inputs sont inclus automatiquement dans le prompt final.</p>'
            f'{self._render_input_references(node)}'
            '</aside>'
            '</div>'
            '</section>'
        )

    def _render_attributes_panel(
        self,
        *,
        node_dom_id: str,
        selected: bool,
        title: str,
        config: dict[str, Any],
        node: dict[str, Any],
        payload: dict[str, Any],
    ) -> str:
        """Render identity, API config, ports, and latest runtime state."""

        panel_id = f"openai-chat-{node_dom_id}-panel-attributes"
        tab_id = f"openai-chat-{node_dom_id}-tab-attributes"
        return (
            '<section class="openai-chat-modal-panel openai-chat-attributes-panel" data-openai-chat-modal-panel '
            f'data-openai-chat-tab-id="attributes" id="{escape(panel_id, quote=True)}" role="tabpanel" '
            f'aria-labelledby="{escape(tab_id, quote=True)}"{"" if selected else " hidden"}>'
            '<div class="openai-chat-attributes-grid">'
            '<section class="openai-chat-modal-section">'
            '<div class="ports-editor-header"><span class="group-label">Identite</span></div>'
            f'{self._render_title_field(title)}'
            '</section>'
            '<section class="openai-chat-modal-section">'
            '<div class="ports-editor-header"><span class="group-label">Configuration API</span></div>'
            f'{self._render_config_fields(config)}'
            '</section>'
            '<section class="openai-chat-modal-section">'
            '<div class="ports-editor-header"><span class="group-label">Ports</span></div>'
            f'{self._render_generic_modal_ports(node)}'
            '</section>'
            '<section class="openai-chat-modal-section">'
            '<div class="ports-editor-header"><span class="group-label">Dernier etat</span></div>'
            f'{self._render_generic_modal_runtime(payload)}'
            '</section>'
            '</div>'
            '</section>'
        )

    def _render_last_response_panel(self, *, node_dom_id: str, selected: bool, response: str) -> str:
        """Render the latest raw OpenAI response panel."""

        panel_id = f"openai-chat-{node_dom_id}-panel-last-response"
        tab_id = f"openai-chat-{node_dom_id}-tab-last-response"
        source_id = f"{panel_id}-source"
        empty_class = " hidden" if response else ""
        response_class = "" if response else " hidden"
        return (
            '<section class="openai-chat-modal-panel openai-chat-last-response-panel" data-openai-chat-modal-panel '
            f'data-openai-chat-tab-id="last-response" id="{escape(panel_id, quote=True)}" role="tabpanel" '
            f'aria-labelledby="{escape(tab_id, quote=True)}"{"" if selected else " hidden"}>'
            '<div class="openai-chat-last-response-layout">'
            '<div class="openai-chat-last-response-header">'
            '<div>'
            '<span class="group-label">Derniere reponse</span>'
            '<h3>Last response</h3>'
            '<p>Reponse JSON brute conservee dans le dernier etat runtime.</p>'
            '</div>'
            f'<button class="ghost-btn openai-chat-last-response-copy{response_class}" data-block-modal-copy="#{escape(source_id, quote=True)}" type="button">Copier</button>'
            '</div>'
            f'<p class="openai-chat-last-response-empty{empty_class}">Aucune reponse OpenAI Chat enregistree pour ce bloc.</p>'
            f'<pre class="openai-chat-last-response-output{response_class}" id="{escape(source_id, quote=True)}" data-block-modal-copy-source>{escape(response)}</pre>'
            '</div>'
            '</section>'
        )

    def _render_config_fields(self, config: dict[str, Any]) -> str:
        """Render editable OpenAI model and API settings."""

        return (
            '<div class="openai-chat-config-grid">'
            '<div class="field-group">'
            '<label>Modele</label>'
            f'<select data-block-config-field="model">{self._select_options(OPENAI_CHAT_MODELS, config["model"])}</select>'
            '</div>'
            '<div class="field-group">'
            '<label>Reasoning effort</label>'
            f'<select data-block-config-field="reasoning_effort">{self._select_options(OPENAI_REASONING_EFFORTS, config["reasoning_effort"])}</select>'
            '</div>'
            '<div class="field-group">'
            '<label>API base URL</label>'
            f'<input data-block-config-field="api_base_url" type="text" autocomplete="off" spellcheck="false" value="{escape(config["api_base_url"], quote=True)}" />'
            '</div>'
            '<div class="field-group">'
            '<label>API key</label>'
            f'<input data-block-config-field="api_key" data-block-skip-empty="true" type="password" autocomplete="off" spellcheck="false" placeholder="{escape("Cle configuree" if config["api_key"] else "sk-...", quote=True)}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Max output tokens</label>'
            f'<input data-block-config-field="max_output_tokens" data-block-value-type="integer" type="number" min="0" max="{MAX_OUTPUT_TOKENS_LIMIT}" step="1" value="{config["max_output_tokens"]}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Historique échanges</label>'
            f'<input data-block-config-field="history_turns" data-block-value-type="integer" type="number" min="0" max="50" step="1" value="{config["history_turns"]}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Timeout secondes</label>'
            f'<input data-block-config-field="timeout_sec" data-block-value-type="integer" type="number" min="1" max="{MAX_TIMEOUT_SEC}" step="1" value="{config["timeout_sec"]}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Limite prompt</label>'
            f'<input data-block-config-field="max_prompt_chars" data-block-value-type="integer" type="number" min="1" max="{MAX_PROMPT_CHARS}" step="1000" value="{config["max_prompt_chars"]}" />'
            '</div>'
            '</div>'
            '<p class="field-hint">Le bloc appelle <code>/v1/responses</code>. Laisse reasoning effort sur <code>default</code> pour omettre le parametre.</p>'
        )

    def _render_title_field(self, title: str) -> str:
        """Render the editable title field without duplicating the modal Apply button."""

        return (
            '<div class="field-group">'
            '<label>Nom du bloc</label>'
            f'<input data-block-title-field type="text" autocomplete="off" value="{escape(title, quote=True)}" />'
            '</div>'
        )

    def _render_input_references(self, node: dict[str, Any]) -> str:
        """Render input names that will be included in the OpenAI prompt."""

        inputs = node.get("inputs") if isinstance(node.get("inputs"), list) else []
        if not inputs:
            return '<div class="ports-editor-empty">Aucune entree disponible.</div>'
        rows: list[str] = []
        for index, port in enumerate(inputs):
            if not isinstance(port, dict):
                continue
            port_id = self._dict_port_id(port, index + 1)
            raw_name = str(port.get("name") or "").strip()
            title = str(port.get("title") or raw_name or f"Input {port_id}").strip()
            rows.append(
                '<div class="openai-chat-reference-row">'
                f'<code>@{escape(raw_name or str(port_id))}</code>'
                '<div>'
                f'<strong>{escape(title)}</strong>'
                f'<small>#{escape(str(port_id))} - {escape(raw_name or str(port_id))}</small>'
                '</div>'
                '</div>'
            )
        return '<div class="openai-chat-reference-list">' + "".join(rows) + '</div>'

    def _render_tab(
        self,
        *,
        tab_id: str,
        tab_dom_id: str,
        panel_dom_id: str,
        title: str,
        subtitle: str,
        selected: bool,
    ) -> str:
        """Render one modal tab button."""

        return (
            '<button class="openai-chat-modal-tab" data-openai-chat-modal-tab '
            f'data-openai-chat-tab-id="{escape(tab_id, quote=True)}" id="{escape(tab_dom_id, quote=True)}" '
            f'type="button" role="tab" aria-selected="{str(selected).lower()}" '
            f'aria-controls="{escape(panel_dom_id, quote=True)}" tabindex="{0 if selected else -1}">'
            f'<span>{escape(title)}</span>'
            f'<small>{escape(subtitle)}</small>'
            '</button>'
        )

    def _ui_last_response(self, payload: dict[str, Any]) -> str:
        """Return the latest raw JSON response from runtime payload outputs."""

        runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
        result = runtime.get("result") if isinstance(runtime.get("result"), dict) else {}
        outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
        for output in outputs.values():
            if not isinstance(output, dict):
                continue
            if str(output.get("port_name") or "").strip() == "raw_json":
                value = output.get("value") or output.get("last_message")
                return str(value or "")
        return ""

    def _last_request_preview(self, *, config: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return a metadata-safe preview of the API request."""

        return {
            "endpoint": self._responses_endpoint(config["api_base_url"]),
            "model": config["model"],
            "prompt_chars": self._messages_char_count(messages),
            "messages": len(messages),
            "history_turns": config["history_turns"],
            "reasoning_effort": config["reasoning_effort"],
            "max_output_tokens": config["max_output_tokens"],
        }

    def _first_response_output(self, node: dict[str, Any]) -> dict[str, Any]:
        """Return the first non-raw output dictionary from the node."""

        outputs = node.get("outputs") if isinstance(node.get("outputs"), list) else []
        return next(
            (
                port
                for port in outputs
                if isinstance(port, dict) and str(port.get("name") or "").strip().lower() != "raw_json"
            ),
            {},
        )

    def _select_options(self, values: tuple[str, ...], selected: str) -> str:
        """Render select options for a normalized selected value."""

        return "\n".join(
            f'<option value="{escape(value)}"{" selected" if value == selected else ""}>{escape(value)}</option>'
            for value in values
        )

    def _normalize_model(self, value: Any) -> str:
        """Return a supported GPT-5.x/chat model id."""

        model = str(value or DEFAULT_OPENAI_CHAT_MODEL).strip()
        if model in OPENAI_CHAT_MODELS:
            return model
        return DEFAULT_OPENAI_CHAT_MODEL

    def _normalize_reasoning_effort(self, value: Any) -> str:
        """Return a supported reasoning effort selector value."""

        effort = str(value or "default").strip().lower() or "default"
        return effort if effort in OPENAI_REASONING_EFFORTS else "default"

    def _normalize_base_url(self, value: Any) -> str:
        """Return a normalized OpenAI-compatible API base URL."""

        base_url = str(value or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
        return base_url or DEFAULT_OPENAI_BASE_URL

    def _responses_endpoint(self, base_url: str) -> str:
        """Return the Responses API endpoint for the configured base URL."""

        return f"{self._normalize_base_url(base_url)}/v1/responses"

    def _normalize_int(self, raw_value: Any, *, default: int, minimum: int, maximum: int) -> int:
        """Return a bounded integer config value."""

        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    def _dict_port_id(self, port: dict[str, Any], fallback: int) -> int | str:
        """Return a stable port id for modal field bindings."""

        raw_id = port.get("id")
        try:
            parsed = int(raw_id)
        except (TypeError, ValueError):
            return str(raw_id or fallback)
        return parsed if parsed > 0 else fallback

    def _safe_dom_id(self, value: str) -> str:
        """Normalize a value so it can be embedded in modal DOM ids."""

        normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "").strip())
        return normalized.strip("-") or "openai-chat"

    def _truncate(self, value: str, max_length: int) -> str:
        """Return a compact one-line preview for the node card."""

        text = str(value or "").replace("\n", " ").strip()
        return text if len(text) <= max_length else f"{text[: max_length - 1]}..."

    def _mask_secret(self, text: str, secret: str) -> str:
        """Mask API key occurrences in runtime errors and logs."""

        if not secret:
            return text
        return str(text).replace(str(secret), "***")

    def _messages_char_count(self, messages: list[dict[str, str]]) -> int:
        """Return the total text size sent to the chat API."""

        return sum(len(str(message.get("content") or "")) for message in messages)
