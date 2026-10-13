# Coding Agent 1.2.0

A single-file CLI coding agent for local LLMs with an OpenAI-compatible API. It creates and edits code, investigates failures, and runs approved checks. The application uses Python 3.8+ and the standard library.

[Русский README](README-ru.md) · [User guide](UserGuide.md) · [Changelog](CHANGELOG.md)

New tasks and each /continue default to 500 LLM requests. Configure with `--max-steps N`, `--continue-steps N`, or session commands `/max-steps [N]`, `/continue-steps [N]`. Russian-layout commands are recognized automatically (for example, `.й` as `/q`); Russian-layout arguments are corrected too.

User messages and `/edit` submissions continue the current conversation, so you can ask to save the previous answer to a file. Use `/new` for a separate task and `/resume` to recover the conversation after restarting.

LLM profiles are backend-scoped: /profile shows actual settings, /profile fast disables reasoning and /profile deep applies analysis settings. Strata fast, balanced and deep preserve the selected model and set max_tokens=8192; balanced uses medium effort with a 2048-token reasoning budget, while deep uses high effort without a separate budget. Manual overrides are marked modified.

Strata profiles follow the [Qwen3.8-Flash-Next recommendations](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF#best-practices): fast uses temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0; deep uses temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0. Switching profiles replaces all these parameters. Accepted values: temperature 0..2, top_p >0 and ≤1, integer top_k 1..64, min_p 0..1, presence_penalty 0..2, and finite repetition_penalty >0. Tune profiles separately for other models. These settings do not guarantee loop prevention; presence_penalty may cause language mixing and reduced quality. balanced shares deep sampling settings. /profile balanced sets medium effort and a 2048-token reasoning budget; /profile deep sets high effort and reasoning_budget_tokens=0 (no separate budget). The total max_tokens=8192 includes reasoning and the answer. Compare whole-task time, corrections and cumulative tokens: shorter responses may increase retries. The model may exhaust the total limit before producing a tool call. The agent does not change server reasoning_loop_recovery or YaRN settings.

### Ollama

Run `python coding_agent.py --workspace /path/to/project --backend ollama --model YOUR_MODEL`. The default address is http://127.0.0.1:11434; change BACKENDS for a remote server. Use an installed model supporting tool calls. /model lists available IDs.

Fast is the startup profile and disables thinking with reasoning_effort=none. /profile deep preserves the model and sets low, temperature=0.6, top_p=0.95 and max_tokens=8192. Thinking support depends on model and Ollama version. /thinking on without an effort uses medium; /thinking auto retains profile parameters. Separate reasoning budgets and top_k are not sent through the OpenAI-compatible adapter; profile fields are rejected.

Set --context-window N for active-context statistics: the adapter does not assume training capacity is Ollama runtime context. This option does not configure the server. API reference: [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility).

## Features

- llama.cpp, vLLM, Strata and Ollama adapters; model and LLM profile switching without restart.
- Targeted file listing, UTF-8 reading, substring/regex search, file creation and exact block replacement.
- Isolated working copy, diffs, initial-baseline restore and user-confirmed publication to the original.
- Disposable check snapshots in Linux bubblewrap; portable mode runs without a systemd user service, while strict mode verifies aggregate cgroup limits.
- Python compilation, unittest, pytest, Bash syntax, ShellCheck, Ansible syntax and offline Ansible lint.
- AUTO/REVIEW editing, bounded iterations, repetition detection and automatic recovery of unapplied changes after restart.
- Readline history, cursor movement, TAB completion, multiline clipboard paste, detailed tool progress and token/context statistics.
- `/edit` opens a terminal task editor: Alt+Enter submits, Esc twice cancels, Ctrl+P/Ctrl+N browse saved drafts.

## Requirements

Use a system Python installed under `/usr`, an unprivileged Linux account, trusted root-owned `bwrap` and usable unprivileged namespaces. The default `--sandbox portable` requires no user systemd service or cgroup delegation. `--sandbox strict` additionally requires trusted `systemd-run`/`systemctl`, a reachable user manager and cgroups v2 memory, PID and CPU controllers. `/security` probes the selected mode and shows its actual availability. Checks refuse to run when isolation is unavailable.

The LLM server must support structured tool calls. Optional checkers must already be installed in trusted system locations: `bash`, `shellcheck`, `ansible-playbook`, `ansible-lint`; pytest must be available to the system Python visible inside the sandbox. The agent never installs dependencies automatically. Interactive editing requires the standard `readline` module and a terminal.

For multiline task editing, use `/edit` in an interactive terminal with Python's standard `curses` module. The editor occupies the bottom half of the screen, with its centered title and keyboard hints on the line border. Enter adds a newline; paste text using your terminal's clipboard shortcut. Alt+Enter closes the editor, prints the full task and starts it. Press Esc twice to close without execution; nonempty drafts are saved even when cancelled. Ctrl+P/Ctrl+N navigate the separate editor history, preserving the current draft while browsing.

## Quick start

Configure `BACKENDS` and `LLM_PROFILES` near the top of [coding_agent.py](coding_agent.py), using your server URL and actual model IDs. The shipped URLs are examples for a private LAN.

```bash
python3.8 coding_agent.py --version
python3.8 coding_agent.py --workspace /path/to/project --backend home-llama --model qwen36-fast
python3.8 coding_agent.py --workspace /path/to/project --sandbox strict  # optional aggregate resource limits
```

The banner includes `Coding Agent 1.2.0`, the original and agent workspace paths, session status and security status. Application state must be in a separate directory outside the original project; use `--state-dir /path/to/private-state` if needed.

Restarting with the same `--workspace` and `--state-dir` automatically reconnects the most recently saved session when it has unapplied changes. `/apply` then publishes those files after a per-file summary and confirmation. Use `--new-session` to start a fresh copy intentionally; `/resume` remains available for manual task recovery.

```text
/security
/model
/mode review
Add a --timeout option to backup.py and update the relevant tests.
/task
/diff
/check python_compile backup.py
/check unittest tests
/apply
```

After a task ends, including blocked validation, the application lists each changed file in the agent copy with added and removed line counts (`+N -M`). It does not print the full diff automatically. Use `/diff` for the current task, `/diff path/to/file.py` for one task file, or `/diff --session` for all changes since the session began. You can also paste the displayed absolute path. Quote paths containing spaces. `/apply` shows only file names and line counts before asking for confirmation. A sandbox failure leaves the agent-copy files available.

`/apply` displays pending changes and asks for `y`; otherwise it leaves the original unchanged. `AUTO` permits edits to the agent copy only. `/revert` immediately restores the entire copy to its initial session baseline. It does not undo changes already applied to the original.

## Skills

Project skills live in `.agents/skills/<name>/SKILL.md` inside the target workspace. Each file needs a YAML header with `name` matching the directory and a `description`, followed by Markdown instructions. `/skills` lists available skills; `/skills name` previews one. Include `$name` in a task to load it explicitly, or let the model choose relevant skills from the brief catalog using `read_skill`. Referenced text resources can be read within the skill directory. Skills follow existing workspace and sandbox restrictions. See the [user guide](UserGuide.md#project-skills) for the format and examples.

## Validation

| Check | Purpose |
| --- | --- |
| `python_compile` | Compile selected Python files. |
| `unittest` | Discover tests in one directory; zero discovered tests is a failure. |
| `pytest` | Run pytest through the system Python. |
| `bash_syntax` | Run `bash -n` separately for each file. |
| `shellcheck` | Static shell analysis without rc discovery or external source following. |
| `ansible_syntax` | Syntax-check playbooks or generated role wrappers using private inventory/config. |
| `ansible_lint` | Offline static lint using controlled Ansible and lint configuration. |

Results distinguish `PASS`, `FAIL`, `SKIPPED`, `TIMEOUT`, `SANDBOX_ERROR` and `INVALID_TARGET`. Missing external static tools are `SKIPPED`. Syntax and lint do not establish functional correctness, deployment success or Ansible idempotence. Ansible lint uses [offline mode](https://docs.ansible.com/projects/lint/configuring/) to prevent requirements installation and schema refresh.

## Configuration and server setup

```python
BACKENDS = {
    "home-llama": {"type": "llama.cpp", "url": "http://127.0.0.1:12345",
                   "api_key": None, "model": None},
    "vllm": {"type": "vllm", "url": "http://127.0.0.1:8000",
             "api_key_env": "VLLM_API_KEY", "model": None},
}
LLM_PROFILES = {
    "strata": {
        "fast": {"model": None, "thinking": "off", "max_tokens": 8192,
                 "temperature": 0.7, "top_p": 0.80, "top_k": 20,
                 "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
        "balanced": {"model": None, "thinking": "on", "reasoning_effort": "medium",
                     "reasoning_budget_tokens": 2048, "max_tokens": 8192,
                     "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                     "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0},
        "deep": {"model": None, "thinking": "on", "reasoning_effort": "high",
                 "reasoning_budget_tokens": 0, "max_tokens": 8192,
                 "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                 "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0},
    },
}
```

For llama.cpp, enable function calling with `--jinja` and a suitable model/chat template; see the [official llama.cpp guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md). For vLLM, configure automatic tool choice and a parser compatible with the model; see the [official vLLM guide](https://docs.vllm.ai/en/latest/features/tool_calling/). All agent tool execution remains in this application.

Use `--context-window TOKENS` when backend context metadata is unavailable. File, output, iteration and sandbox limits are editable constants in `coding_agent.py`; their defaults and effects are documented in the user guide.

Each LLM request shows a brief task summary. SSE streaming is enabled by default, with elapsed time and data counters every 5 seconds without response contents. Responses are limited to 8192 tokens (`--max-tokens N`, `/max-tokens N`). Truncated responses never execute tools or retry automatically; use `/continue`. `--llm-timeout` sets socket waiting (600 seconds); `--no-stream` enables compatibility with non-streaming servers. Restore the task with `/resume` after restarting. Malformed tool argument JSON is also rejected as a whole; /continue instructs the model to reduce the edit and correct its JSON.

LLM diagnostics separately count received reasoning, response text and tool-argument characters (not tokens). During streaming and at completion, interruption or cancellation, diagnostics show started tool names, including incomplete names, without their contents. Call identifiers are excluded from these counts. Zero reasoning characters means no separate reasoning/reasoning_content fields were received: reasoning mixed into content by the server is counted as text. For ordinary JSON, these details are available only after receiving the response. A started call does not imply an executed tool.


Portable mode enforces per-process address-space, CPU-time, open-file and file-size limits plus a wall timeout. It does not enforce an aggregate memory, CPU or process-count limit. The private `/tmp` has no separate hard quota with the supported bubblewrap 0.6.3 syntax. Strict mode adds verified aggregate cgroup limits. Neither mode starts a checker outside bubblewrap.

## Verification

```bash
python3.8 -B coding_agent.py --self-test
```

Self-tests need no LLM server. Integration tests are skipped when the required checker or verified sandbox is unavailable; a skip is not evidence that a checker ran. On the current machine, live portable sandbox integrations pass without user-service `cpu.max` delegation; the strict resource-limit test and absent `ansible-lint` are skipped.

## Architecture and documentation

The [diagramms](diagramms) directory contains one PlantUML diagram per `.puml` file, with matching `-ru.puml` translations:

- [Components](diagramms/components.puml).
- [Core classes](diagramms/classes.puml).
- [Workspace and publication lifecycle](diagramms/workspace-lifecycle.puml).
- [Sandbox check sequence](diagramms/sandbox-sequence.puml).
- [Model/tool iteration](diagramms/agent-loop.puml).
- [Static validation selection](diagramms/static-validation.puml).

The [user guide](UserGuide.md) covers configuration, all commands, practical workflows, storage, troubleshooting and security boundaries. Source docstrings document purpose, typed inputs, return values and error conditions in English and Russian.

### Context and AGENTS.md

Loading the root `AGENTS.md` is reported when starting a task; `/context` displays saved instruction information and statistics. `/compact` summarizes the current task conversation through the LLM, retaining project and skill instructions, the goal, file state and check results. The complete conversation is archived. Continue with `/continue`. See [UserGuide.md](UserGuide.md).

### Strata

Select `--backend strata`; its configuration must use `"type": "strata"`. The adapter uses Chat Completions, SSE and local tool execution; context capacity comes from `meta.n_ctx`. `/thinking off` is supported. Compatibility was checked against [Strata documentation](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md) and [API source](https://github.com/Niko1221/Strata/blob/main/serve/server.py), with response contract tests. Live checks on October 7, 2026 passed with qwen3.8-flash-next-iq3_xxs and a 131072-token context: JSON/SSE, a tool call and reasoning controls. The earlier connection refusal was observed before the server became available.

## License

[MIT](LICENSE).

