# Coding Agent 1.2.0 — User Guide

This guide describes the behavior implemented in `coding_agent.py`. [Русская версия](UserGuide-ru.md) has the same section structure and examples. For a short introduction, see [README.md](README.md).

Strata profiles follow the [Qwen3.8-Flash-Next recommendations](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF#best-practices): fast uses temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0; deep uses temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0. Switching profiles replaces all these parameters. Accepted values: temperature 0..2, top_p >0 and ≤1, integer top_k 1..64, min_p 0..1, presence_penalty 0..2, and finite repetition_penalty >0. Tune profiles separately for other models. These settings do not guarantee loop prevention; presence_penalty may cause language mixing and reduced quality. balanced shares deep sampling settings. /profile balanced sets medium effort and a 2048-token reasoning budget; /profile deep sets high effort and reasoning_budget_tokens=0 (no separate budget). The total max_tokens=8192 includes reasoning and the answer. Compare whole-task time, corrections and cumulative tokens: shorter responses may increase retries. The model may exhaust the total limit before producing a tool call. The agent does not change server reasoning_loop_recovery or YaRN settings.

### Ollama

Run `python coding_agent.py --workspace /path/to/project --backend ollama --model YOUR_MODEL`. The default address is http://127.0.0.1:11434; change BACKENDS for a remote server. Use an installed model supporting tool calls. /model lists available IDs.

Fast is the startup profile and disables thinking with reasoning_effort=none. /profile deep preserves the model and sets low, temperature=0.6, top_p=0.95 and max_tokens=8192. Thinking support depends on model and Ollama version. /thinking on without an effort uses medium; /thinking auto retains profile parameters. Separate reasoning budgets and top_k are not sent through the OpenAI-compatible adapter; profile fields are rejected.

Set --context-window N for active-context statistics: the adapter does not assume training capacity is Ollama runtime context. This option does not configure the server. API reference: [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility).

## 1. Purpose and operating model

Coding Agent is a terminal application for code creation, focused changes, traceback investigation, refactoring and test updates. You enter a task in natural language; a local LLM requests a small set of validated tools. The application applies edits and returns actual tool results to the model.

Three locations have different roles:

| Location | Role |
| --- | --- |
| Original project | Source selected by `--workspace`; model tools cannot write to it. |
| Agent workspace | Independent session copy where edits accumulate. |
| Check snapshot | Fresh disposable copy used by a sandboxed checker; changes are never copied back. |

A completed task means the agent has finished its bounded workflow. It does not mean that edits have been published: `/apply` is a separate user action. Static check success proves only the corresponding syntax/lint result.

## 2. Requirements and preflight

The application needs Python 3.8+ and no third-party Python library. Secure project checks require Linux, an unprivileged OS account, trusted root-owned `bwrap` and usable unprivileged namespaces. Portable mode is the default and needs no user systemd manager or delegated cgroup controllers. Strict mode additionally requires trusted `systemd-run`/`systemctl`, a reachable user manager, cgroups v2 and usable memory/PID/CPU controllers. The executing Python must resolve to a trusted system installation under `/usr`; a runtime in a user HOME is rejected for secure checks.

The backend must expose model discovery and OpenAI-compatible chat completion with structured tool calls. Configure the server separately. The agent does not install packages or configure system delegation.

Optional tools must already be available in trusted system locations. This includes Bash, ShellCheck and Ansible tools. A project virtual environment is excluded from snapshots by default and is not mounted into checks. pytest must be importable by the system Python used inside the sandbox.

After startup, run `/security`. `ENABLED` requires a successful real probe; `DEGRADED` means secure validation is unavailable. File editing can still operate in the isolated session, but a `SANDBOX_ERROR` prevents the agent from claiming validated completion. The startup probe does not require an LLM request.

## 3. Launching and command-line options

Run the script from a shell:

```bash
python3.8 coding_agent.py --version
python3.8 coding_agent.py --help
python3.8 coding_agent.py --workspace /home/user/projects/demo --backend home-llama --model qwen36-fast
python3.8 coding_agent.py --workspace /home/user/projects/demo --sandbox strict  # optional cgroup mode
```

The normal banner prints version `1.2.0`, original/copy paths, `NEW` or `RESUMED` session status, backend, model, editing and sandbox modes, step limit, state path and security status. Startup reconnects the most recently saved compatible session if it has unapplied changes; otherwise it creates a new copy. A resumed session restores the task and saved model automatically, so `/apply` can publish its files immediately after confirmation. Use `/resume` to reconnect a task manually when it has no pending file changes.

| Option | Meaning and default |
| --- | --- |
| `--workspace PATH` | Original project directory; default is the current directory. |
| `--backend NAME` | A key in BACKENDS: home-llama, vllm, strata, ollama. DEFAULT_BACKEND sets the local default. |
| `--model NAME` | Explicit server model ID; overrides the backend's configured model. |
| `--context-window TOKENS` | Positive token capacity override for statistics. |
| `--llm-timeout SECONDS` | Positive chat HTTP socket timeout; default 600 seconds. |
| `--max-tokens N` | Per-response token limit, default 8192; sent as max_tokens. |
| `--no-stream` | Non-streaming mode for incompatible servers. |
| `--review` | Start in REVIEW instead of AUTO. |
| `--new-session` | Force a fresh isolated copy even if an earlier session has unapplied changes. |
| `--sandbox portable\|strict` | Checker resource policy; default `portable`, `strict` requires a user systemd service and verified cgroup limits. |
| `--state-dir PATH` | Private state path; default described in section 16. |
| `--self-test` | Run built-in tests without an LLM server and exit. |
| `--version` | Print the script name and version, then exit. |
| `--help` | Print argparse help and exit. |

State and original project paths must be disjoint: neither may contain the other. For example, use `/home/user/projects/demo` with `/home/user/.local/state/coding-agent`, not `demo/.agent-state`. The initial copy rejects unsafe links/special files and has configured size/entry limits.

## 4. Backend configuration

Edit the configuration constants near the top of `coding_agent.py`; there is no separate application config file. Replace shipped LAN addresses with reachable URLs. URLs may include `/v1` or the server root; the chat adapter supplies the missing prefix. Embedded URL credentials, query strings, fragments and redirects are rejected.

```python
BACKENDS = {
    "home-llama": {
        "type": "llama.cpp",
        "url": "http://127.0.0.1:12345",
        "api_key": None,
        "model": None,
        "context_windows": {"your-fast-model-id": 32768},
    },
    "vllm": {
        "type": "vllm",
        "url": "http://127.0.0.1:8000/v1",
        "api_key_env": "VLLM_API_KEY",
        "model": None,
    },
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

A configured `api_key` takes precedence over the named environment variable. Prefer `api_key_env` to embedding credentials in the source. The vLLM example reads `VLLM_API_KEY` from the agent environment when constructing its adapter; this API credential is not needed by project checks.

For llama.cpp, use `llama-server` with `--jinja` and a suitable function-calling chat template. Model aliases and router model IDs must match `/model` output. Consult the [official function-calling documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md).

For vLLM, configure `--enable-auto-tool-choice` and a model-compatible `--tool-call-parser`; parser selection depends on the model and server version. Consult [vLLM tool-calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling/). The application performs its own file/check operations; configuring the model API does not grant server-side access to the project.

### Project skills

Store each skill in `.agents/skills/<name>/SKILL.md` inside the **target project**. Skills are copied into the isolated workspace with the project; edits to the original directory are not synchronized into an already open session. `/skills` lists the current copy's skills and any invalid entries; `/skills name` previews its instructions without launching a task. TAB completes names after `/skills`.

Example `.agents/skills/docstrings/SKILL.md`:

```markdown
---
name: docstrings
description: Add concise function docstrings matching the project style.
---
Read the function implementation before documenting it.
Describe its purpose, parameter types, return value and actual error conditions.
Preserve existing behavior.
```

The directory and `name` must match: 1–64 lowercase letters, digits, hyphens or underscores, starting with a letter or digit. `description` is required. The supported YAML subset accepts plain strings, single/double quoted strings, and `>`/`|` multiline descriptions. Other frontmatter fields are ignored; this is not a general YAML parser. A catalog contains up to 50 directory entries with descriptions shortened to 256 characters. Each skill or resource is limited to 64 KiB and 24000 characters; oversized or malformed instructions are reported rather than silently truncated.

At task start, only skill names, descriptions and paths enter the context. The model can use `list_skills` to refresh this catalog and `read_skill(name)` to load a relevant skill. To explicitly request a known skill, include `$docstrings` in your task, including in `/edit`: `Use $docstrings to document sort_words.py`. Recognized mentions load instructions before the first LLM request (up to eight skills). Unknown dollar-prefixed words remain ordinary task text. Loaded instructions are persisted in the conversation and restored by `/resume`; automatic selection depends on the model.

Referenced text files are read using `read_skill(name, path)`, with `path` relative to that skill directory, for example `references/style.md`. Absolute paths, parent traversal, symlinks and hard-linked files are rejected. Bundled scripts are not executed by skill loading. Skills remain subject to the user's task, existing tool capabilities and sandbox rules. The source directory is project-local; user-wide skill directories are not searched automatically.

### Strata server

Use type `strata` to discover the selected model and alias context capacity from `/v1/models` → `meta.n_ctx`. Type `vllm` supported chat but left the limit unknown because vLLM uses `max_model_len`. Plain responses, SSE, `reasoning_content`, `tools`, fragmented `tool_calls` and final-event usage are covered by tests. The agent executes tools locally; requests do not enable `strata_mcp`.

Example configuration:

```python
"strata": {"type": "strata", "url": "http://127.0.0.1:8080",
           "api_key_env": "STRATA_API_KEY", "model": "strata"},
```

Use the actual model ID from `/model`; the URL may end in `/v1`. `/thinking off` sends `chat_template_kwargs.enable_thinking=false`. Otherwise the server setting applies. `LLM_PROFILES` supplies reasoning_effort and reasoning_budget_tokens per backend profile; verify support in your installed server version. Strata rejects requests when prompt plus output budget exceed context: run `/compact` beforehand or reduce `/max-tokens`.

Sources: [DETAILS.md](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md), [serve/server.py](https://github.com/Niko1221/Strata/blob/main/serve/server.py), [serve/frontend.py](https://github.com/Niko1221/Strata/blob/main/serve/frontend.py). Checked main on October 7, 2026; local contract tests do not replace validation against a specific deployment version. A subsequent live probe passed on October 7, 2026: qwen3.8-flash-next-iq3_xxs, n_ctx=131072; streaming answers, a valid tool call without execution and reasoning controls were verified. This adapter does not use Responses API, Anthropic Messages or server-hosted MCP tools.

### Reasoning controls in Strata

For Strata, use /profile fast without reasoning, /profile balanced for medium effort with a 2048-token budget, and /profile deep for high effort without a separate budget. All three bound the complete response at 8192 tokens. /thinking off temporarily disables reasoning; /thinking on enables it with current profile parameters; /thinking auto removes the agent switch, retaining profile parameters and server settings. Manual changes appear as modified in /profile. A positive reasoning_budget_tokens bounds reasoning; 0 means no separate budget. Configure levels and budgets in LLM_PROFILES; dedicated commands for them are not implemented.

Strata defaults to fast. Keep it for routine edits: earlier long reasoning exhausted output budgets. Deep is a starting point for complex analysis, not a measured optimum. Reasoning and the server wrap-up share max_tokens. Without overrides, the server uses high. Live probes confirmed off, low and enforcement of the separate budget.

## 5. Models, profiles and thinking

```text
/model
/model your-fast-model-id
/profile fast
/profile deep
/thinking auto
```

`/model` prints the current model, requests available IDs and refreshes model completion suggestions. `/model NAME` selects an ID immediately; availability is determined by the next server request. If no model is configured and exactly one is discoverable, a new task selects it automatically; otherwise choose explicitly. The standalone `DEFAULT_MODEL` constant is not the model-selection fallback: use the backend's `model` field or `--model`.

`/profile` shows current-backend profiles, the selected model, actual parameters and modified status. /profile fast and /profile deep replace the complete generation settings, removing prior parameters; LLM_PARAMETERS applies only before a profile is selected. Strata activates fast at startup. /model, /thinking and /max-tokens override settings and mark the profile modified until reapplied. model=None preserves the model; if unset, a single available model is selected or /model NAME is required. Explicit IDs are checked before settings change. Unsupported fields and parameters are rejected. By default effort is enabled for Strata and Ollama, and a separate reasoning budget only for Strata; other backends need verified support declared through BACKENDS profile_capabilities. Conversation and file changes are retained. Profiles are session settings; /resume restores conversation but not the previous launch profile.

`/thinking auto` keeps server defaults. `on` or `off` sends `chat_template_kwargs.enable_thinking`. A server/model template must support that parameter; the command does not implement reasoning itself.

## 6. Terminal input, history and completion

The first message starts a task. Subsequent messages, including submissions from `/edit`, continue the same conversation with previous answers, instructions and tool results. For example, after an analysis, “save the previous analysis to report.md” includes that previous answer in the model context. Use `/new` to clear the conversation for a separate task. `/continue` proceeds without a new user instruction. A line beginning with `/` is a command. Follow-up messages use the `/continue-steps` budget; the initial task uses `/max-steps`. In an interactive terminal that supports bracketed paste, paste a multiline task and press Enter once after the paste. The pasted newlines remain part of one task; Enter typed outside the paste submits it.

| Key | Behavior in an interactive readline terminal |
| --- | --- |
| Up / Down | Previous / next history entry. |
| Left / Right | Move the cursor within the current line. |
| Home / End | Beginning / end of the line. |
| TAB | Complete the current command or supported argument; show choices when ambiguous. |
| Ctrl+C during input | Cancel the line and return to the prompt. |
| Ctrl+C during a task | Cancel the active iteration; inspect changes before continuing. |
| EOF, usually Ctrl+D | End the input loop. |

History includes entered commands and tasks, including multiline pasted tasks, keeps up to `HISTORY_LENGTH = 500` entries, and is saved atomically per original project in private state. Consecutive identical entries are not added again. Treat history and saved task messages as local records of your input.

Completion supports slash commands, help topics, modes, profiles, thinking values, known/configured model IDs, directories for `/files` and check names for `/check`. `/model` refreshes server IDs. File-target completion for `/check` is not implemented; enter targets explicitly. Quoted targets with spaces work in `/check`, for example `/check bash_syntax "scripts/backup copy"`. `/files` receives its whole trailing argument directly: use `/files directory with spaces` without shell-style quoting.

When stdin/stdout are not terminals or `readline` is unavailable, plain input remains available, but readline editing, completion and persistent interactive history are not active. Multiline clipboard paste relies on GNU Readline and a terminal that supports bracketed paste.

### Multiline task editor

Run `/edit` to open a terminal modal in the bottom half of the screen. Its line border contains a centered title at the top and keyboard hints at the bottom. This requires an interactive terminal and Python's standard `curses` module. The text is editable before submission; slash-prefixed text inside the editor is task text, not a REPL command. After Alt+Enter, the entire task is printed in the terminal before the LLM starts processing it.

| Key | Editor behavior |
| --- | --- |
| Enter | Insert a newline; keep editing. |
| Left / Right / Up / Down | Move through text and across lines. |
| Home / End | Move to the beginning / end of the current line. |
| Page Up / Page Down | Move ten lines up / down; scroll the viewport. |
| Backspace / Delete | Delete before / after the cursor; join lines at a boundary. |
| TAB | Insert a tab, displayed at four-column tab stops. |
| Terminal paste shortcut (e.g. Ctrl+Shift+V) | Insert clipboard text at the cursor; multiline bracketed paste is inserted literally. |
| Ctrl+P / Ctrl+N | Previous / next editor history entry. Return to the newest position to restore the fresh draft. |
| Alt+Enter | Close and start the entire task; empty text does not run. |
| Esc, then Esc again | Close without running the task. |

Nonempty text is saved on both submission and cancellation. Editor history is separate from command history, keeps up to `HISTORY_LENGTH = 500` entries per original project, and survives restarts. Consecutive identical texts are not repeated. While browsing history, edits and cursor positions are preserved for the open editor. Cancelled drafts can be recalled through `/edit`, then Ctrl+P. Input cancellation with Ctrl+C also preserves nonempty text.

## 7. Command reference

Exact command names take priority. Other names may use any unique prefix; an ambiguous prefix returns candidate names. For example, `/sec` resolves to `/security`, while both `/mo` and `/mod` are ambiguous between `/model` and `/mode`.

| Command | Behavior |
| --- | --- |
| `/help [tools]` | Print all commands, or model tools; always include allowed checks. |
| `/skills [NAME]` | List project skills or preview one `SKILL.md`; use `$name` in a task to request it. |
| `/edit` | Open the multiline task editor with persistent submitted/cancelled draft history. |
| `/tools` | Shortcut to tool/check help. |
| `/new` | Clear task context while retaining session file changes. |
| `/task` | Show task ID/status, counters, changed paths and recorded check statuses. |
| `/model [NAME]` | List models or select an ID. |
| `/profile [fast\|deep]` | Show profile mapping or select a profile. |
| `/mode [auto\|review]` | Show or set per-edit approval behavior. |
| `/files [PATH]` | List one safe directory in the agent copy. |
| `/changes` | List session changes against the initial baseline. |
| `/diff [--session] [PATH]` | Show current-task changes by default, or all session changes with `--session`; accepts a relative path or an absolute path inside the agent copy. |
| `/revert` | Immediately restore the entire copy to its initial session baseline. |
| `/context` | Show conversation counters and token/context statistics. |
| `/compact` | Summarize the current dialogue, retaining instructions and an archive. |
| `/continue` | Give the current task another bounded iteration budget. |
| `/thinking [auto\|on\|off]` | Show or set the optional template parameter. |
| `/max-tokens [N]` | Show or change the response limit without restarting. |
| `/resume` | Manually load the most recently persisted compatible task/session; do not run it yet. |
| `/apply` | Summarize pending files and line counts, confirm, back up and publish. |
| `/discard` | Confirm if unapplied changes exist, delete this session copy and exit. |
| `/security` | Refresh isolation probe and display limits, blockers and checker availability. |
| `/check [NAME TARGET ...]` | Run approved semantic validation with optional relative targets. |
| `/quit` | Exit; keep the session copy and persisted state. |

`/task`, `/context`, `/compact` and `/continue` require a current task. `/check` creates a manual-validation task when necessary and defaults to `python_compile`. `/apply` and `/discard` operate on the isolated session independently of a model task. The model cannot invoke these publication commands as tools.

A task run defaults to 500 LLM requests; /continue grants another 500. This step budget is separate from token and context limits. Limit stops and cancellation are highlighted in bold yellow; task errors use bold red. Redirected output, TERM=dumb or the presence of NO_COLOR disable ANSI colors while retaining textual labels.

Set budgets with `--max-steps N` and `--continue-steps N`. Session commands `/max-steps [N]` and `/continue-steps [N]` display or update them. Values must be positive integers. Settings last until exit; the first applies to new tasks and the second to the next follow-up or /continue.

Russian keyboard-layout commands are recognized: `.й` → `/q`, `.рудз` → `/help`, `/вшаа кщдуы.тпштч` → `/diff roles/nginx`. Russian-layout arguments are corrected too: /thinking щаа becomes /thinking off. Already-English arguments remain unchanged, and path letter case is preserved. A dot in place of / is corrected only for existing command names or prefixes; ambiguous prefixes still require clarification.

The conversation remains available after task completion. `/diff` and statistics cover the current conversation until `/new`; follow-ups retain original file versions and check results. Use `/resume` after restarting to recover the conversation. If an older version already replaced the task, `/resume` restores the latest saved task rather than automatically combining earlier conversations.

## 8. Editing, review, publication and rollback

A typical workflow is:

```text
/security
/mode review
Add a --timeout option to backup.py; default 30 seconds; update tests.
/task
/diff
/check python_compile backup.py
/check unittest tests
/apply
```

In AUTO, authorized edits are written to the agent copy immediately. In REVIEW, every proposed creation/edit shows a diff and asks `Apply change? [y/N]`; only `y` accepts it. Rejected edits are returned to the model as an actual rejection. Both modes still require explicit confirmation for `/apply` to the original.

`/apply` compares pending files with the last-applied source hashes. It rejects unsupported deletion, unsafe paths, existing destinations for new files and external source changes. It shows each pending file with added and removed line counts against the **current original**, asks `Confirm action? [y/N]`, checks conflicts again, creates original backups and an intent journal, then publishes each file atomically. It does not print a full diff; use `/diff` explicitly when needed.

Publication is atomic per file, not transactional across the whole project. If publication fails midway, the error lists already-applied paths and the backup directory. Inspect that directory's `apply.json` and backups before choosing recovery; automatic rollback of the original is not provided.

After each task, the terminal lists task file paths and added/removed line counts (`+N -M`) against each file's state before that task. A replacement counts as both removed and added lines. It does not print a full diff automatically. `/diff` compares files with their state before the current task, including files created by an earlier task; `/diff path/to/file.py` filters to one task file. `/diff --session [PATH]` compares with the initial session baseline. The displayed absolute agent-copy path also works as an argument; paths with spaces must be quoted. `/changes` lists session changes, so already-applied edits can still appear there and in `/diff --session`. Repeating `/apply` without new pending changes prints `No unapplied changes.`

`/revert` has no confirmation prompt. It restores the **entire session copy**, including edits from earlier tasks in the same session, to its initial baseline. Already-applied original files remain applied. Consequently, a later `/apply` may propose restoring the original to baseline content; compare the original and agent copy before confirming. `/new` only clears task context and is not rollback.

`/discard` removes the session's project, baseline and manifest after any required pending-change confirmation, then exits. `/quit` leaves the copy available. A discarded session cannot be restored through `/resume`.

## 9. Tools available to the model

| Tool | Inputs and result |
| --- | --- |
| `list_files` | Relative `path`, optional `recursive`; entries and omitted count. |
| `read_file` | Relative file, inclusive 1-based `start_line`/`end_line`; numbered UTF-8 text. |
| `search_text` | Pattern, relative path, filename glob, regex/case flags, result limit; matching paths, lines and text. |
| `create_file` | New relative path and text; refuses an existing file. |
| `edit_file` | Relative path, exact `old_text`, replacement `new_text`; requires exactly one match. |
| `run_check` | Approved name, relative targets, bounded timeout; structured real check result. |
| `get_diff` | No arguments; current-task diff. |
| `list_skills` | No arguments; bounded catalog with names, descriptions, paths and discovery errors. |
| `read_skill` | Required `name`, optional `path` (default `SKILL.md`); read one skill or its local text resource. |

Unknown tools and arguments are rejected. No arbitrary command, executable, shell string, package installation or network tool is exposed to the model. Project `AGENTS.md` is read when starting a task, but it cannot expand these capabilities. The prompt asks the model to inspect relevant files before editing and to keep replacements focused.

Paths must be relative. Absolute/UNC/drive paths, `..`, `.git`, alternate streams, symlinks and reparse points are refused; text reads also reject hard links, binary NUL bytes, invalid UTF-8 and oversized files. New files use UTF-8/LF; edits preserve an existing UTF-8 BOM and uniformly CRLF file when present.

Substring search runs in the application. Regex matching uses a sandboxed worker with a three-second timeout; unavailable sandbox protection makes regex search unavailable too. Large file reads require line ranges. Tool progress shows filenames, selected ranges/checks, targets, timeout and actual check duration/status.

## 10. Python validation

```text
/check python_compile backup.py src/config.py
/check unittest tests
/check pytest tests/test_backup.py
```

`python_compile` runs the system Python with `-I -m py_compile` on `.py` file targets. It does not prove runtime correctness. The model's default targets are changed Python files. For a manual `/check python_compile` with no changed Python files, the CLI enumerates Python files in the agent workspace.

`unittest` runs discovery in at most one directory. Without a target it uses `tests` when present, otherwise the workspace default discovery location. A report with zero discovered tests is `FAIL`, even if the process exits zero.

`pytest` runs the system Python's pytest module. If that module is not installed in the visible runtime, preserve its real failure output; there is no automatic installation and no project-venv fallback.

The completion gate requires compilation of changed Python files and test validation when test files/directories are present. Actual failures must be investigated and rerun within the task budget. All imports and test execution happen in disposable snapshots.

## 11. Bash validation

```text
/check bash_syntax scripts/backup.sh bin/deploy
/check shellcheck scripts/backup.sh bin/deploy
```

`bash_syntax` runs trusted `bash -n` independently per file, preserving per-file status/exit code/output. Explicit targets may have any extension. Auto-selection includes `.sh` and Bash shebangs such as `#!/bin/bash`, `#!/usr/bin/bash` and `#!/usr/bin/env bash`. A `#!/bin/sh` file is recommended for ShellCheck without pretending Bash syntax is POSIX validation.

Use ShellCheck after successful Bash syntax validation. Its fixed invocation disables rc discovery and does not enable `-x`; diagnostics preserve line numbers and rule IDs such as `SC2086`. External-source control follows [ShellCheck's documented restriction](https://www.shellcheck.net/wiki/SC1144). Sourced files are not automatically traversed under this policy.

Missing trusted Bash/ShellCheck tools return `SKIPPED`; checker diagnostics return `FAIL`; sandbox initialization failure returns `SANDBOX_ERROR`. Results establish static syntax/analysis only. The application does not run shell scripts functionally and does not provide Bats or `bash_run`.

## 12. Ansible validation

```text
/check ansible_syntax site.yml playbooks/upgrade.yml
/check ansible_syntax roles/nginx
/check ansible_lint roles/nginx
/check ansible_lint .
```

`ansible_syntax` invokes `ansible-playbook --syntax-check` separately for each recognized playbook/role target. Roles are recognized by directories such as `tasks`, `handlers`, `defaults`, `vars`, `templates`, `files` or `meta`. A file under `roles/nginx/tasks/main.yml` is mapped to `roles/nginx`; it is not treated as a standalone playbook. Arbitrary YAML, such as OpenAPI or CI configuration, is not automatically selected unless the playbook heuristic matches its path/content.

Trusted preparation creates a read-only `/sandbox` metadata mount outside the snapshot's project tree. The generated inventory is:

```ini
[agent_test]
localhost ansible_connection=local
```

`ANSIBLE_CONFIG` points to `/sandbox/ansible.cfg`, and syntax commands explicitly select `/sandbox/inventory`. Configuration fixes forks to one, host key checking to true, retry files to false, and project-local roles/collections paths. A role target gets a generated playbook with `hosts: agent_test`, `gather_facts: false` and a safely quoted role name. The wrapper never appears in the agent workspace.

Ansible checks do not use project production inventory, project `ansible.cfg`, host HOME configuration, SSH-agent sockets or host vault/Ansible secrets. Project-local plugins and collections can still execute untrusted code during validation; they remain subject to the same sandbox. No secret is requested to unlock encrypted dependencies; missing dependencies/keys remain actual diagnostics.

`ansible_lint` accepts recognized playbooks and validated role/roles/project directories. It uses a generated lint config and fixed `--offline` options; [offline mode](https://docs.ansible.com/projects/lint/configuring/) disables requirements installation and schema refresh. Network remains unavailable. Missing installed lint tooling gives `SKIPPED`; missing project dependencies after a checker starts generally give `FAIL` with the original output and rule IDs.

After an Ansible edit, recommendation order is syntax then lint. The application does not implement normal playbook execution, `--check`, real converge, idempotence, Molecule, containers or SSH targets. A syntax/lint PASS must not be described as a successful deployment.

## 13. Check results and task statuses

| Check status | Interpretation |
| --- | --- |
| `PASS` | The isolated checker ran successfully. |
| `FAIL` | Started validation returned failure or hit a detected resource/growth condition. |
| `SKIPPED` | An external static checker was not found in a trusted system location. |
| `TIMEOUT` | The check exceeded its duration and descendant termination was verified. |
| `SANDBOX_ERROR` | Protection/startup failed, or termination could not be verified; no validated result. |
| `INVALID_TARGET` | A static check target was unsafe, missing or unrecognized; checker not started. |

Ordinary Python target errors can be returned as `TOOL_ERROR` rather than the static check's `INVALID_TARGET`. Do not treat either as PASS. A resource failure may add `resource_limit` and cgroup event counters. If termination is unverified, `SANDBOX_ERROR` reports quarantine instead of claiming cleanup.

Recorded results include name, status, exit code, duration, stdout/stderr, targets, task revision, timeout/resource details and sandbox metadata. Static aggregates include `files`, `sandboxed` and a static-only message. Multi-target syntax checks use separate snapshots; infrastructure failure or timeout stops the remaining group. Large diagnostics preserve the beginning and end. `/task` shows recorded statuses; `/check` prints the structured result.

Task statuses are `ACTIVE`, `COMPLETED`, `LIMIT`, `BLOCKED`, `CANCELLED`, `ERROR` and `REVERTED`. `BLOCKED` means the current revision has a sandbox infrastructure failure. `LIMIT` can be continued with `/continue`; `REVERTED` requires a new task. A skipped optional tool must be reported explicitly, not folded into “all checks passed.”

## 14. Tokens and context occupancy

Before each chat request, the agent prints its response number, model, HTTP timeout and a short summary: the task goal for the first request, recent tool names/paths/check targets for later requests, or a continuation/validation notice. At most three recent tools are listed. This is a description of the request metadata, not the model's reasoning. System prompts, earlier conversation and tool result bodies are not printed in these lines. After a response, elapsed time is shown. The actual backend request still includes the conversation required to perform the task.

Chat uses SSE streaming by default. Every 5 seconds, progress reports elapsed waiting time, event count and received characters, including reasoning; reasoning and response text are not printed. These are not token counts: measured usage comes from the final server usage object when supplied. `--llm-timeout` (600 seconds) limits socket waiting, not the total duration of an active stream. Use `--no-stream` for servers without SSE or stream_options support; there is no automatic request retry. An ordinary JSON response returned instead of SSE is also accepted. Stream errors or interruption never execute partially received tools. Use `/continue` to proceed, or `/resume` then `/continue` after restarting.

After each model response, the agent prints input/output/total tokens reported by the backend and cumulative task totals. Missing usage is `n/a`; partial totals show how many responses provided that counter. These counts include repeated prompt tokens across model requests and do not represent a monetary invoice.

`/context` also displays message/step/tool/file counts and a rough next-request estimate from UTF-8 JSON bytes divided by four, including tool definitions. This estimate is not a tokenizer measurement and can differ substantially with language and chat template.

Context limit precedence is: `--context-window`, per-model `context_windows`, backend `context_window`, then cached server metadata. llama.cpp reads `/props` and compatible model metadata may supply `max_model_len`. With a known limit, output includes percentage and remaining tokens; otherwise it says the limit is unknown.

Statistics do not automatically truncate or summarize the conversation. When context is exhausted, finish or save the work, inspect the diff, and use `/new` for a fresh task. File changes remain in the session copy.

LLM diagnostics separately count received reasoning, response text and tool-argument characters (not tokens). During streaming and at completion, interruption or cancellation, diagnostics show started tool names, including incomplete names, without their contents. Call identifiers are excluded from these counts. Zero reasoning characters means no separate reasoning/reasoning_content fields were received: reasoning mixed into content by the server is counted as text. For ordinary JSON, these details are available only after receiving the response. A started call does not imply an executed tool.

### Long and truncated responses

The response limit defaults to 8192 tokens. Set it with `--max-tokens N` or `/max-tokens N`; it takes precedence over `LLM_PARAMETERS["max_tokens"]`. The agent asks the model to create one or two small files per step. The server enforces the limit; some models also spend the budget on reasoning.

For `finish_reason=length`, another unsupported finish reason or an empty response, the agent enters `LIMIT`: the partial response is excluded from history, no tools execute and no automatic retry occurs. Argument JSON for the entire batch is checked before any tool executes. If necessary, set `/max-tokens 8192`, disable reasoning with `/thinking off` (requires server template support), then `/continue`. Large tasks should proceed across multiple steps. These settings also apply to `/compact`, which rejects truncated summaries. A skill requiring unavailable `molecule test` execution cannot expand agent capabilities: the agent creates scenarios but reports functional validation as not run.

Malformed argument JSON rejects the entire response tool batch and stops without automatic retry. Diagnostics show the call number and name, argument size and error position without file contents. Token usage is displayed for rejected responses too; the response timing line includes finish_reason. On /continue, the model receives a saved instruction to send one small call with valid JSON and a minimal old_text/new_text block. Earlier file changes are retained. An unterminated string alone does not prove that the token limit was reached: check finish_reason.

### Context compaction and project instructions

When starting a task, the agent prints `AGENTS.md загружен: <path>` if the file exists at the isolated project root. `/context` reports whether instructions were loaded, their path, the number of skill catalog/instruction messages, the compaction count and the latest archive path. These describe the current task's saved instructions: editing the source file does not automatically reload them.

Run `/compact` to ask the selected LLM to summarize the current conversation. The request preserves the goal, changed files, failing command, current hypothesis, test results and next exact command, following any `Compact Instructions` in `AGENTS.md`. Summary quality depends on the model.

The system prompt, project instructions, skill catalog, explicitly loaded skills and resources successfully read through `read_skill` are retained in full. The original goal and actual task state accompany the summary. File changes, backups and check results are not reset. The full pre-compaction conversation is archived as `backups/<task-id>/conversation-<uuid>.json` inside the state directory; its path is printed. This archive is not sent with subsequent requests.

The command executes no tools and does not automatically continue the task. Use `/context` to inspect the estimated size and `/continue` to proceed; `/resume` restores the saved summary after restart. The compaction request contributes to token statistics. Before/after sizes are JSON UTF-8 bytes, not tokens; the latest measured input count describes the compaction request, while the next-request estimate describes the new context.

A backend error, empty response or truncated response leaves the conversation unchanged. A summary that does not reduce size also leaves the original context intact. Compaction itself requires a request containing the current conversation, so run it before exhausting the context window. An ordinary instruction such as “Compress context” is sent to the model within the current conversation; use `/compact` for actual compaction.

## 15. Security boundaries and configurable limits

Every project checker, including static analysis and regex workers, goes through `SandboxRunner`. It verifies the selected mode, creates a disposable snapshot, sets `no_new_privs` and enters bubblewrap namespaces. Portable mode launches the trusted Python gate directly with per-process `RLIMIT_AS`, `RLIMIT_CPU`, open-file, file-size and core limits. Strict mode starts a sanitized systemd user service and verifies actual kernel cgroup memory, swap, task and CPU limits before bubblewrap. Only the disposable project, safe system mounts, private HOME/tmp and generated metadata are visible. Checker network is disabled; backend HTTP requests are a separate application operation.

In portable mode the checker is PID 1 of a private PID namespace, so its exit kills remaining namespace descendants; timeout kills the bubblewrap process group, then the launcher and output pipes must close before snapshot removal. Strict mode stops and verifies the whole service cgroup. If termination cannot be verified, the run directory is quarantined and reported. Portable per-process limits do not cap aggregate memory, CPU use or process count. The private `/tmp` has no separate hard quota on bubblewrap 0.6.3. Do not remove quarantined files while processes may still use them.

| Constant | Shipped value / effect |
| --- | --- |
| `HTTP_TIMEOUT` | 120 s socket timeout for model discovery and other ordinary HTTP requests. |
| `LLM_TIMEOUT` | 600 s socket timeout for chat requests; overridden by `--llm-timeout`. |
| `MAX_AGENT_STEPS` / `CONTINUE_STEPS` | 500 / 500; --max-steps, --continue-steps and session commands. |
| `MAX_TOOL_CALLS` / `REPEAT_THRESHOLD` | 100 calls per run budget / 5 repeated identical calls. |
| `CHECK_TIMEOUT` / `SANDBOX_WALL_TIMEOUT` | 120 s; supplied check timeout is clamped. |
| `STATIC_CHECK_TIMEOUT` | 60 s maximum per static process. |
| `MAX_FILE_BYTES` / `MAX_READ_BYTES` | 10 MiB file / 1 MiB selected read. |
| `MAX_LIST_RESULTS` / `MAX_SEARCH_RESULTS` | 1000 entries / 200 matches. |
| `MAX_TOOL_OUTPUT_CHARS` / `STATIC_CHECK_OUTPUT_MAX_CHARS` | 30000; preserve bounded beginning/end. |
| `MAX_WORKSPACE_BYTES` / `MAX_WORKSPACE_ENTRIES` | 512 MiB copied content / 100000 entries. |
| `SANDBOX_MEMORY_MAX_MB` / `SANDBOX_TASKS_MAX` | 2048 MiB per-process address space in portable mode; 2048 MiB aggregate memory / 64 tasks and no swap in strict mode. |
| `SANDBOX_CPU_QUOTA_PERCENT` | 200 percent aggregate CPU quota in strict mode only; portable mode limits each process's CPU time to the check timeout plus two seconds. |
| `SANDBOX_OPEN_FILES_MAX` / `SANDBOX_FILE_SIZE_MAX_MB` | 256 open files / 256 MiB per-file resource limit. |
| `SANDBOX_SNAPSHOT_MAX_MB` | 1024 MiB monitored project snapshot threshold; private `/tmp` has no separate hard quota. |
| `SANDBOX_CORE_DUMPS` | False; launcher sets core limit to zero. |
| `HISTORY_LENGTH` / `REGEX_TIMEOUT` | 500 history lines / 3 s isolated regex matching. |

Limits are source constants, not arbitrary model-supplied options. Static checks use the selected sandbox mode. `SANDBOX_ENV_ALLOWLIST` is empty by default; changing it can expose selected host values to non-Ansible checks. Ansible still filters to the controlled environment. `ALLOW_UNSANDBOXED_CHECKS = False` remains a fail-closed safeguard.

`IGNORE_PATTERNS` excludes `.git`, Python caches, common virtual environments, `node_modules` and editor directories from copies/listings. This is not a general secret-file filter: readable project text may be sent to the configured LLM. Choose the project and backend accordingly.

The filesystem jail assumes no hostile same-UID host process racing its operations. Snapshot growth monitoring is not a hard disk quota. The tool set has no model-visible delete, arbitrary shell, installation or Git mutation operation. Publication is restricted to the user command.

## 16. State, persistence and recovery

On Linux the default state is `${XDG_STATE_HOME:-~/.local/state}/coding-agent`; on Windows path selection uses `LOCALAPPDATA` or `~/AppData/Local`. Secure checks themselves require Linux. New state/run/session directories are created with private permissions. State is plain JSON/files, not encrypted storage.

```text
state/
  coding-agent.log
  current-task.json
  history-<original-project-hash>.json
  editor-history-<original-project-hash>.json
  backups/<task-id>/
    task.json
    <path-hash>.original
  workspaces/<session-id>/
    session.json
    baseline/
    project/
  apply-backups/<apply-id>/
    apply.json
    <path-hash>
  runs/check-<id>/
    workspace/
    sandbox/           # Ansible checks only
    started
    cgroup.json        # when the resource gate records it
```

Ordinary run directories are deleted; only active/quarantined directories should remain. Task backups and apply backups are separate. Log rotation uses a 2 MiB threshold and two backup logs. Known backend API keys are redacted, but saved conversation/history contains task text and can contain project content. `DEBUG` enables additional response previews; protect state access and storage.

`current-task.json` identifies the most recently saved task in this state directory. Automatic recovery uses this pointer only when the original project matches and the session contains unapplied files. Sharing one state directory among unrelated projects can replace that pointer. `/resume` also validates the original project and saved isolated path; it rejects a foreign or legacy direct-workspace task. Use separate state directories if independent recent-task pointers are needed.

Recovery after a clean exit:

```bash
python3.8 coding_agent.py --workspace /home/user/projects/demo --state-dir /home/user/.local/state/coding-agent --model your-model-id
```

```text
/task
/diff
/apply
```

Check the startup `Session: RESUMED (unapplied changes)` line before `/apply`. If the earlier agent process is still running, or a fresh copy was explicitly requested, run `/resume` first. To continue the task before publication, inspect `/task` and `/security`, then use `/continue`. `/resume` restores task messages and checks; it does not itself contact the model.

## 17. Practical workflows

### 17.1. A Python feature

Ask for a precise change, behavior and expected test coverage. Review actual modified paths, run compilation and relevant tests, then publish explicitly:

```text
Add --timeout with default 30 to backup.py. Cover the default and explicit value in tests/test_backup.py.
/diff
/check python_compile backup.py tests/test_backup.py
/check unittest tests
/apply
```

### 17.2. A Bash correction

```text
Fix quoting in scripts/backup.sh. Preserve the script's current arguments and exit-code behavior.
/check bash_syntax scripts/backup.sh
/check shellcheck scripts/backup.sh
/diff
/apply
```

If ShellCheck is missing, retain the `SKIPPED` report. Bash syntax success is not a functional backup test.

### 17.3. An Ansible role

```text
Correct the nginx role tasks and add names and fully qualified module names where needed.
/check ansible_syntax roles/nginx
/check ansible_lint roles/nginx
/diff
/apply
```

Use the role directory for syntax. Both checks use the synthetic inventory, not production hosts. Do not infer converge/idempotence from these results.

### 17.4. Reject or abandon changes

Reject an individual REVIEW proposal with Enter or `n`. To abandon all session edits in the copy, use `/revert`; to remove the whole session and exit, use `/discard`. `/quit` preserves the copy; pending edits reconnect automatically on the next matching startup.

## 18. Verification and architecture sources

```bash
python3.8 -B coding_agent.py --self-test
```

Tests use temporary fixtures inside the current directory, so it must be writable. They require no LLM server. Unit tests exercise path validation, edits/backups, schemas, iteration limits, API normalization, statistics, workspace publication, static command construction and metadata/environment cleanup. Real checker/namespace/resource tests run only when the installed tools and verified sandbox are available. Review the skip reasons instead of interpreting a successful test suite as complete integration coverage.

The release environment lacks user-service `cpu.max` and `ansible-lint`; live integration checks are skipped there. This guide does not claim those checks passed.

Six architecture sources live in [diagramms](diagramms): [components](diagramms/components.puml), [classes](diagramms/classes.puml), [workspace lifecycle](diagramms/workspace-lifecycle.puml), [sandbox sequence](diagramms/sandbox-sequence.puml), [agent loop](diagramms/agent-loop.puml) and [static validation](diagramms/static-validation.puml). Each file contains one PlantUML diagram and has a corresponding `-ru.puml` translation. Use an installed PlantUML renderer/editor to view them; rendering is not part of agent runtime.

Source docstrings use `EN:` and `RU:` purpose sentences with shared `Args / Аргументы`, `Returns / Результат` and `Errors / Ошибки` sections. They describe argument types/meaning, results and expected error handling. Trust-boundary comments explain atomic publication, descriptor-relative copying, environment cleanup, descendant termination and approximate token accounting in both languages.

## 19. Troubleshooting

| Symptom | Action |
| --- | --- |
| Backend connection/HTTP error | Verify `BACKENDS` URL, network access, credentials and server process; restart after configuration edits. REPL remains available. |
| `BACKEND_TIMEOUT` while the model is generating | Check the reported limit and elapsed time. Restart with a larger `--llm-timeout`, then `/resume` and `/continue`; the timeout concerns HTTP waiting, not project validation. |
| Model not found or model selection requested | Run `/model` and select an actual ID; update `LLM_PROFILES` aliases. |
| Answers describe actions but no tools run | Verify the model's function-calling template/parser on the server; inspect `/tools`. |
| `bubblewrap ... no trusted ... executable` | Ensure the selected executable is already installed, root-owned and under trusted system directories; no automatic installation is attempted. |
| `cpu.max` or another controller unavailable in strict mode | The user-service cgroup lacks a required controller. Use default `--sandbox portable`, or arrange delegation with the administrator if aggregate limits are needed. |
| User systemd manager unavailable in strict mode | Use `--sandbox portable`, or check the user manager/bus and runtime environment if strict mode is required. |
| Trusted system Python required | Launch with a system interpreter under `/usr`, rather than a HOME/venv runtime. |
| `SKIPPED` | The external static checker is absent or untrusted. It was not run and is not PASS. |
| `INVALID_TARGET` / path violation | Use an existing safe relative path; choose a recognized playbook or role for Ansible syntax. |
| pytest import failure | pytest is absent from the sandbox-visible system Python; project `.venv` is not a fallback. |
| `TIMEOUT` / resource limit | Inspect diagnostics and limits. Check the project before increasing bounded source settings. |
| `EDIT_FAILED` | Read the current file; use an exact block that matches once. |
| `APPLY CONFLICT` | Original changed after the source fingerprint. Compare copy/original and start a fresh session or reconcile manually; no force-apply is provided. |
| Apply failed after some files | Inspect listed applied paths, backup directory and `apply.json` before recovery. |
| Resume rejected | Use the same original and state directory; a discarded/foreign session cannot be attached. |
| Context limit unknown | Set `--context-window` or per-model `context_windows`; the estimate is not exact tokenization. |
| No arrows/TAB/history | Use an interactive terminal and a Python with readline; piped input uses the plain input path. |
| Quarantine reported | Inspect the reported service/run location and verify termination before cleanup; never assume the snapshot was removed. |

## 20. Scope and release information

Version `1.2.0` is defined by `APP_VERSION`, shown in the normal startup banner and available through `--version`. Release records are [CHANGELOG.md](CHANGELOG.md) and [CHANGELOG-ru.md](CHANGELOG-ru.md); licensing is [MIT](LICENSE).

The application does not implement automatic dependency installation, deployment, arbitrary shell execution, functional Bash runs, full Ansible converge/idempotence, container/VM orchestration, Git mutation or automatic conversation compaction. Use the approved tool results to judge what was actually validated, and `/apply` to decide which copy changes reach the original.
