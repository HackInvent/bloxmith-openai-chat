# OpenAI Chat Block

<!-- block-metadata:start -->
[![Block version: unversioned](https://img.shields.io/badge/block-unversioned-lightgrey)](model.json)
[![BloxSmith compatibility: 1.0.9](https://img.shields.io/badge/BloxSmith-1.0.9-brightgreen)](compatibility.json)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Verified BloxSmith versions: **1.0.9** (bundled-block tests; see [test evidence](compatibility.json)).
<!-- block-metadata:end -->


## Role

`openai_chat` calls the OpenAI-compatible Responses API and emits an assistant response.

## Files

- `block.py`: prompt assembly, OpenAI API request, response parsing, API-key masking, runtime outputs, and UI rendering.
- `model.json`: default input, response/raw JSON outputs, API config, and runtime capabilities.
- `inspector_panel.html`: block-owned inspector UI for prompt and API settings.
- `block_modal.html`: prompt-first modal UI with attributes and last-response tabs.
- `assets/css/block_modal.css`: modal layout and prompt editor styles.
- `assets/js/block_modal.js`: modal tab keyboard/click behavior.
- `node_card.html`: block-owned canvas card body.

## Ports

- Inputs:
  - `in` (`id: 1`): optional input; accepts generic messages, text, and JSON.
- Outputs:
  - `response` (`id: 1`): emits assistant text as `text/plain`.
  - `raw_json` (`id: 2`): emits the full OpenAI JSON response as `application/json`.

## Configuration

- `model`: GPT-5.x/chat model id. Default: `gpt-5.5`.
- `api_key`: OpenAI API key. If empty at runtime, `OPENAI_API_KEY` is used.
- `api_base_url`: OpenAI-compatible API root. Default: `https://api.openai.com`.
- `system_instruction`: optional instruction sent as the Responses API `instructions` field.
- `reasoning_effort`: optional Responses API reasoning effort. `default` omits the parameter.
- `max_output_tokens`: optional maximum output token count. `0` omits the parameter.
- `history_turns`: number of previous user/assistant exchanges stored in the current run and resent on the next call. `0` disables memory.
- `timeout_sec`: HTTP timeout.
- `max_prompt_chars`: local prompt guard before the HTTP request.

## Runtime Behavior

`execute_runtime()` builds one prompt from all non-empty named inputs and the response output instruction, then posts JSON to:

```text
POST <api_base_url>/v1/responses
```

The request body contains at least `model` and `input`. `input` is sent as a Messages-style list containing the previous configured exchanges plus the current user prompt. Optional `instructions`, `reasoning`, and `max_output_tokens` are included only when configured.

The block extracts assistant text from `output_text` first, then from the structured `output[].content[]` shape. The raw JSON response is also emitted for diagnostics or downstream parsing.

Conversation history is stored in the current run data under an internal `chat_history.openai_chat.<node_id>` key. A new full run starts empty; run resume or active runtime waves can reuse the restored history.

The same implementation runs in One Shot Simulation (`centralized`) and Active Runtime (`zeromq_active`) through the generic block executor.

## UI Behavior

The inspector and modal expose prompt, system instruction, GPT-5.x model selection, API base URL, API key, reasoning effort, output token limit, history turn count, timeout, and prompt guard.

API keys are not rendered back in clear text. Leaving the API key field empty keeps the existing value when the generic frontend binding skips empty sensitive fields. Editable prompt and API settings stay local until the user clicks **Apply**.

The modal declares `data-block-runtime-refresh="autonomous"`; block-owned JS keeps tabs, command/response diagnostics, focus, and draft fields stable during runtime polling.

## Maintenance Notes

OpenAI Chat behavior belongs in this block. Do not add OpenAI Chat-specific branches to the orchestrator or active runtime worker; use the generic block executor contract instead.

## Compatibility policy

[compatibility.json](compatibility.json) records HackInvent's verified BloxSmith versions and test evidence. Only the versions listed above have been verified, using the block-owned suites in a **bundled-block test installation**. This is not a certification of managed-package installation, every browser/OS, or live provider availability. Other framework versions are unverified, not necessarily incompatible.

The block-version badge follows `model.json`, not a published Git tag. `unversioned` means that no block release version is declared; no number is inferred from the framework version. The framework still uses `model.json` for its runtime/install contract; the tester-owned JSON does not replace it. Official integration tests run in the private `bloxmith-blocs` workspace. Test helpers and the proprietary framework are not bundled in this public block repository.
