#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Role: Verifies OpenAI Chat block behavior.
# File Name: F5.18_openai_chat_block.py
# Author: Alexandre EL
# Email: alex@hackinvent.com
# Created Date: 2026-05-19
# -----------------------------------------------------------------------------

"""F5.18 - OpenAI Chat block.

The test runs against a local OpenAI-compatible Responses API endpoint and
verifies prompt construction, auth, response/raw JSON outputs, UI rendering,
API-key masking, and both centralized and zeromq_active runtimes.
"""

# Test cases:
# - FB1/FB2/FB3/FB7 - Run text -> OpenAI Chat -> display in centralized and active runtime, verify /v1/responses request and output propagation.
# - FB4/FB5 - Refuse missing keys and oversized prompts before network calls and ensure api_key never leaks in logs/errors.
# - FB6 - Render modal, inspector, node-card, and assets with model/API fields and masked api_key.
# - FB8 - Execute the block twice with a run_data store and verify the second request includes the previous exchange.

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Any
import json
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blocs.openai_chat.block import OpenAiChatBlock
from bloxsmith_app.block_runtime import BlockRuntimeContext
from bloxsmith_app.block_ui import render_block_inspector_panel, render_block_modal, render_block_node_card
from bloxsmith_app.run_data_store import RunDataStore
from ui_smoke_common import (
    create_run_api,
    data_edge,
    display_node,
    expect,
    graph_payload,
    isolated_server,
    text_node,
    wait_for_run_terminal,
)
from urllib.parse import quote
from block_test_packages import install_test_package, release_key, surface_payload


SECRET = "sk-test-openai-chat-secret"
ANSWER = "bonjour depuis le faux openai chat"


class FakeOpenAiChatHttpServer(ThreadingHTTPServer):
    requests_log: list[dict[str, Any]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeOpenAiChatHandler)
        self.requests_log = []

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


class FakeOpenAiChatHandler(BaseHTTPRequestHandler):
    server: FakeOpenAiChatHttpServer

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}
        self.server.requests_log.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization") or "",
                "content_type": self.headers.get("Content-Type") or "",
                "payload": payload,
            }
        )
        response = json.dumps(
            {
                "id": "resp_test_openai_chat",
                "model": payload.get("model"),
                "output_text": ANSWER,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": ANSWER}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        return


class FakeChatServer:
    def __enter__(self) -> FakeOpenAiChatHttpServer:
        self.server = FakeOpenAiChatHttpServer()
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.server

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def openai_chat_node(api_base_url: str, *, max_prompt_chars: int = 250000) -> dict[str, Any]:
    return {
        "id": "openai-chat-1",
        "kind": "openai_chat",
        "title": "OpenAI Chat test",
        "position": {"x": 360, "y": 120},
        "inputs": [
            {"id": 1, "name": "in", "title": "In", "accepts": ["message/*", "text/plain"], "multiplicity": "many"}
        ],
        "outputs": [
            {
                "id": 1,
                "name": "response",
                "title": "Response",
                "emits": ["message/*", "text/plain"],
                "multiplicity": "many",
                "instruction": "Reponds en une phrase a @in.",
            },
            {
                "id": 2,
                "name": "raw_json",
                "title": "Raw JSON",
                "emits": ["application/json", "message/*"],
                "multiplicity": "many",
            },
        ],
        "config": {
            "model": "gpt-5.5",
            "api_key": SECRET,
            "api_base_url": api_base_url,
            "system_instruction": "Tu es concis.",
            "reasoning_effort": "default",
            "max_output_tokens": 0,
            "history_turns": 5,
            "timeout_sec": 10,
            "max_prompt_chars": max_prompt_chars,
        },
    }


def run_openai_chat_case(runtime_mode: str, fake_server: FakeOpenAiChatHttpServer) -> dict[str, Any]:
    with isolated_server() as server:
        # Surfaces are release assets: a bundled kind serves none of them.
        model = install_test_package(server, "openai_chat")
        key = quote(release_key(model), safe="")
        served = lambda payload, suffix: next(
            asset["path"] for asset in payload["assets"] if asset["path"].endswith(suffix))
        document = graph_payload(
            f"F5 OpenAI Chat {runtime_mode}",
            [
                text_node("text-1", "Texte Chat", "hello openai chat", 80, 120),
                openai_chat_node(fake_server.base_url),
                display_node("display-1", "Affichage", 700, 120),
            ],
            [
                data_edge("edge-text-chat", "text-1", 1, "openai-chat-1", 1),
                data_edge("edge-chat-display", "openai-chat-1", 1, "display-1", 1),
            ],
        )
        created = create_run_api(server, document, runtime_mode=runtime_mode)
        run = wait_for_run_terminal(server, str(created.get("run_id") or ""), timeout_sec=25)

    logs = "\n".join(run.get("logs", []))
    node_logs = "\n".join(run.get("node_logs", {}).get("openai-chat-1", []))
    expect(run.get("status") == "success", f"The OpenAI Chat {runtime_mode} run must succeed.")
    expect(
        run.get("output_values", {}).get("openai-chat-1:1", {}).get("value") == ANSWER,
        "The text response must leave on the response port.",
    )
    raw_json = run.get("output_values", {}).get("openai-chat-1:2", {}).get("value") or ""
    expect("resp_test_openai_chat" in raw_json and ANSWER in raw_json, "La reponse JSON brute doit sortir sur raw_json.")
    expect(SECRET not in logs and SECRET not in node_logs, "La cle API ne doit pas apparaitre dans les logs.")
    expect("fallback centralized" not in logs, "Le run ne doit pas fallback centralise.")
    if runtime_mode == "zeromq_active":
        expect(
            run.get("results", {}).get("openai-chat-1", {}).get("transport") == "zeromq_active",
            "openai_chat doit etre execute via zeromq_active.",
        )
    return run


def test_http_requests(fake_server: FakeOpenAiChatHttpServer) -> None:
    """TC1 - Verify OpenAI-compatible Responses API requests."""

    expect(len(fake_server.requests_log) >= 2, "Le faux endpoint Chat doit recevoir une requete par run.")
    for request in fake_server.requests_log:
        payload = request["payload"]
        expect(request["path"] == "/v1/responses", "The block must call /v1/responses.")
        expect(request["authorization"] == f"Bearer {SECRET}", "The block must send the API key as a bearer token.")
        expect(request["content_type"] == "application/json", "The block must send JSON.")
        expect(payload.get("model") == "gpt-5.5", "The GPT-5.x model must be sent.")
        input_payload = payload.get("input")
        expect(isinstance(input_payload, list), "The block must send the Responses API input as a message list.")
        expect("hello openai chat" in str(input_payload or ""), "The prompt must contain the input.")
        expect("Reponds en une phrase" in str(input_payload or ""), "The prompt must contain the instruction.")
        expect(payload.get("instructions") == "Tu es concis.", "The system instruction must be sent.")


def test_prompt_guards(fake_server: FakeOpenAiChatHttpServer) -> None:
    """TC2 - Verify missing keys and prompt limits fail before network calls."""

    before = len(fake_server.requests_log)
    block = OpenAiChatBlock()
    common_context = {
        "run_id": "unit-run",
        "node_id": "openai-chat-guard",
        "kind": "openai_chat",
        "title": "OpenAI Chat guard",
        "inputs": {"in": "0123456789"},
        "input_content_types": {"in": "text/plain"},
        "input_message": "0123456789",
        "input_ports": (SimpleNamespace(id=1, name="in"),),
        "output_ports": (
            SimpleNamespace(id=1, name="response", instruction="too long", emits=("message/*", "text/plain")),
            SimpleNamespace(id=2, name="raw_json", emits=("application/json", "message/*")),
        ),
        "root_dir": ROOT,
        "run_dir": ROOT,
    }
    missing_key = block.execute_runtime(
        BlockRuntimeContext(
            **common_context,
            config={"api_key": "", "api_base_url": fake_server.base_url, "max_prompt_chars": 100},
        )
    )
    expect(missing_key.status == "failed", "A missing API key must fail.")
    expect("api_key" in missing_key.error, "The error must mention the missing API key.")

    oversized = block.execute_runtime(
        BlockRuntimeContext(
            **common_context,
            config={"api_key": SECRET, "api_base_url": fake_server.base_url, "max_prompt_chars": 8},
        )
    )
    expect(oversized.status == "failed", "A prompt that is too long must fail.")
    expect("trop long" in oversized.error, "The error must explain the prompt limit.")
    expect(len(fake_server.requests_log) == before, "Local errors must not call the fake endpoint.")


def test_conversation_history(fake_server: FakeOpenAiChatHttpServer) -> None:
    """TC3 - Verify run-local conversation memory keeps the previous exchange."""

    before = len(fake_server.requests_log)
    block = OpenAiChatBlock()
    store = RunDataStore()
    common_context = {
        "run_id": "history-run",
        "node_id": "openai-chat-history",
        "kind": "openai_chat",
        "title": "OpenAI Chat history",
        "input_content_types": {"in": "text/plain"},
        "input_ports": (SimpleNamespace(id=1, name="in"),),
        "output_ports": (
            SimpleNamespace(id=1, name="response", instruction="", emits=("message/*", "text/plain")),
            SimpleNamespace(id=2, name="raw_json", emits=("application/json", "message/*")),
        ),
        "root_dir": ROOT,
        "run_dir": ROOT,
        "store": store,
        "config": {
            "api_key": SECRET,
            "api_base_url": fake_server.base_url,
            "model": "gpt-5.5",
            "history_turns": 1,
            "max_prompt_chars": 10000,
        },
    }
    first = block.execute_runtime(
        BlockRuntimeContext(
            **common_context,
            inputs={"in": "premiere question"},
            input_message="premiere question",
        )
    )
    expect(first.status == "success", "The first call with memory must succeed.")
    second = block.execute_runtime(
        BlockRuntimeContext(
            **common_context,
            inputs={"in": "seconde question"},
            input_message="seconde question",
        )
    )
    expect(second.status == "success", "The second call with memory must succeed.")
    requests = fake_server.requests_log[before:]
    expect(len(requests) == 2, "The memory test must produce two API calls.")
    second_messages = requests[1]["payload"].get("input")
    expect(isinstance(second_messages, list) and len(second_messages) == 3, "The second call must include the previous exchange and the new question.")
    expect(second_messages[0]["role"] == "user" and "premiere question" in second_messages[0]["content"], "The first question must be restored.")
    expect(second_messages[1]["role"] == "assistant" and ANSWER in second_messages[1]["content"], "The first response must be restored.")
    expect(second_messages[2]["role"] == "user" and "seconde question" in second_messages[2]["content"], "The new question must close the payload.")


def test_openai_chat_ui_contract(fake_server: FakeOpenAiChatHttpServer) -> None:
    """TC4 - Render OpenAI Chat block-owned modal, inspector, node-card, and assets."""

    node = openai_chat_node(fake_server.base_url)
    rendered = render_block_modal("openai_chat", {"node": node, "runtime": {}})
    html = str(rendered.get("html") or "")
    assets = rendered.get("assets") or []
    css = (ROOT / "blocs/openai_chat/assets/css/block_modal.css").read_text(encoding="utf-8")
    js = (ROOT / "blocs/openai_chat/assets/js/block_modal.js").read_text(encoding="utf-8")

    expect("cw-openai-chat-modal" in html, "Le modal OpenAI Chat doit venir du bloc.")
    expect('data-block-runtime-refresh="autonomous"' in html, "Le modal OpenAI Chat doit gerer son refresh runtime.")
    expect('data-openai-chat-tab-id="prompt"' in html, "The modal must expose the Prompt tab.")
    expect('data-openai-chat-tab-id="attributes"' in html, "The modal must expose the Attributs tab.")
    expect('data-openai-chat-tab-id="last-response"' in html, "The modal must expose the Last response tab.")
    expect('data-block-output-field="instruction"' in html, "L'instruction doit rester liee a output.instruction.")
    expect('data-block-config-field="model"' in html, "Le modele must be editable.")
    expect('data-block-config-field="api_key"' in html, "La cle API must be editable.")
    expect('data-block-config-field="history_turns"' in html, "Le nombre d'echanges memorises must be editable.")
    expect(SECRET not in html, "The API key must not be rendered in clear text in the modal.")
    expect("gpt-5.5" in html and "gpt-5-chat-latest" in html, "Les modeles GPT-5.x/chat doivent etre proposes.")
    expect(".openai-chat-modal-panel[hidden]" in css, "Le CSS doit cacher les panels inactifs.")
    expect("export function mount" in js, "Le JS doit monter le modal via le registre block UI.")

    inspector = render_block_inspector_panel("openai_chat", {"node": node})
    inspector_html = str(inspector.get("html") or "")
    expect("cw-openai-chat-inspector" in inspector_html, "The OpenAI Chat inspector must come from the block.")
    expect('data-block-config-field="system_instruction"' in inspector_html, "The inspector must edit the system instruction.")
    expect('data-block-config-field="history_turns"' in inspector_html, "The inspector must edit the conversation memory.")
    expect(SECRET not in inspector_html, "The API key must not be rendered in clear text in the inspector.")

    card = render_block_node_card("openai_chat", {"node": node})
    card_html = str(card.get("html") or "")
    expect("data-openai-chat-node-card" in card_html, "La node-card OpenAI Chat doit venir du bloc.")


def main() -> None:
    with FakeChatServer() as fake_server:
        test_openai_chat_ui_contract(fake_server)
        run_openai_chat_case("centralized", fake_server)
        run_openai_chat_case("zeromq_active", fake_server)
        test_http_requests(fake_server)
        test_prompt_guards(fake_server)
        test_conversation_history(fake_server)
    print("[ok] F5.18_openai_chat_block")


if __name__ == "__main__":
    main()
