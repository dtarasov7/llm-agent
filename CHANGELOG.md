# Changelog

## [Unreleased]

- Strata balanced uses medium effort with a 2048-token reasoning budget; deep uses high effort with no separate budget. Both retain max_tokens=8192.
- Updated Strata fast/deep sampling to Qwen3.8-Flash-Next recommendations; added validated min_p, presence_penalty and repetition_penalty profile fields. The complete-response limit remains 8192.

Changes are recorded by released version. The application version is defined by `APP_VERSION` in `coding_agent.py`.

## [1.2.0] - 2026-10-07

### Added

- Ollama backend and fast/deep profiles with an 8192-token limit; thinking maps to reasoning_effort and unsupported extensions are rejected.

- Profile sampling settings: Strata deep uses temperature=0.6, top_p=0.95 and top_k=20; values are validated and switching to fast removes them.

## [1.1.0] - 2026-10-07

### Added

- LLM_PROFILES replaces global model aliases with backend-scoped model/thinking/reasoning settings and max_tokens=8192. Strata defaults to fast; deep uses low with budget 512. Applying profiles replaces parameters, marks manual overrides and rejects unsupported settings.

- Verified live Strata compatibility (qwen3.8-flash-next-iq3_xxs, context 131072); documented reasoning switches, effort levels, separate budgets and the off override.

- Configurable 500/500 step budgets via --max-steps, --continue-steps and matching session commands; Russian keyboard-layout command correction including argument correction.

- Terminal highlighting for important stops and errors; the step-limit notice explains the LLM request budget and continuation through /continue.

- Separate reasoning, text and tool-argument character counts, with started tool names during generation and on exit, without exposing contents.

- SSE streaming with progress, a 8192-token response budget, `--max-tokens`, `/max-tokens` and `--no-stream`. Truncated responses stop without tool execution or automatic retry; large tasks proceed incrementally.

- Report loaded `AGENTS.md`, display instruction information in `/context`, and add `/compact` to summarize the current task while retaining instructions, skills, state and a conversation archive.

- Project-local skills in `.agents/skills/<name>/SKILL.md`: `/skills [NAME]`, TAB completion, brief task catalogs, explicit `$name` loading and model tools `list_skills`/`read_skill`; loaded instructions survive task resume.
- Brief LLM request summaries, response elapsed times and `--llm-timeout` for configurable chat waiting.
- `/edit` terminal task editor with multiline clipboard paste, cursor editing, Alt+Enter submission, double-Esc cancellation and persistent draft history navigated by Ctrl+P/Ctrl+N; cancelled text is retained.
- Default `--sandbox portable` runs bubblewrap checks without a user systemd manager or cgroup delegation; `--sandbox strict` retains verified aggregate cgroup limits.
- `/security` reports selected mode, active limits and the reduced resource guarantees of portable mode.

### Fixed

- Added a dedicated Strata backend type: context capacity uses meta.n_ctx rather than vLLM max_model_len. Verified JSON/SSE, tool fragments, final usage and disabling reasoning.

- Subsequent messages and /edit continue the current conversation with prior answers and tool results; /new explicitly clears context. Follow-ups retain file state and support explicit skill loading.

- Malformed tool argument JSON now reports the tool, size and error position without contents, rejects the whole batch and persists repair instructions for /continue. Rejected responses display token usage and finish_reason.

- Give non-streaming chat responses a default 600-second socket timeout instead of the shared 120 seconds; report timeout duration and recovery steps explicitly.
- Recognize alternate Home/End terminal sequences in `/edit`; place its line-bordered modal in the bottom half of the screen with title/hints on the borders, and print the submitted task before LLM execution.
- Make `/diff` show only current-task edits, so files created earlier in the session are not shown as wholly new; use `/diff --session [PATH]` for the original session-wide view.
- Keep pasted multiline tasks together with GNU Readline bracketed paste and preserve them as one history entry.
- Translate `/help`, `/help tools` and `/tools` output into Russian while keeping command and tool identifiers unchanged.
- Show only pending file names and added/removed line counts before `/apply` confirmation; keep full diffs available through explicit `/diff`.
- Reconnect the latest compatible session with unapplied changes on restart, so `/apply` no longer sees a new empty copy; add `--new-session` to request a fresh copy.
- Replace the automatic full task diff with per-file added/removed line counts; `/diff [PATH]` now shows one selected file, including paths copied from the final output.
- Remove unsupported bubblewrap `--size` arguments, allowing live checks on bubblewrap 0.6.3; private `/tmp` has no separate hard quota.
- Show absolute agent-copy file paths and `/diff` → `/apply` guidance after completion, blocking, errors, cancellation and iteration limits.
- Clarify staging versus original publication at startup and in model instructions; discourage unchanged retries after sandbox infrastructure failures.

## [1.0.0] - 2026-10-05

### Added

- Single-file Python 3.8+ CLI agent using only the standard library.
- OpenAI-compatible llama.cpp/vLLM adapters, model discovery, fast/deep profiles and optional thinking parameters.
- UTF-8 file tools, bounded listing/search, exact unique edits, atomic publication and original-content backups.
- Isolated session copies, read-only originals, conflict-checked `/apply`, initial-baseline `/revert`, `/discard` and explicit `/resume`.
- Mandatory disposable bubblewrap checks with network isolation, fake HOME, private temporary storage, clean environment, no_new_privs and kernel-verified cgroup/resource limits.
- Python compilation, unittest and pytest checks; Bash syntax/ShellCheck and controlled offline Ansible syntax/lint checks with temporary role wrappers.
- Explicit check statuses, per-file diagnostics, resource/timeout reporting and quarantine when descendant termination cannot be verified.
- AUTO/REVIEW editing, bounded model/tool loops, repetition protection and task persistence.
- Readline editing, per-project history, TAB completion and detailed tool/check progress.
- Per-response and cumulative token statistics, context metadata discovery and explicit rough occupancy estimates.
- Built-in offline self-tests and guarded integration tests.
- English/Russian source documentation, README, user guides and six translated architecture diagrams.
- Version 1.0.0 in the startup banner and the `--version` command.

### Validation limitations

- Live sandbox integrations were skipped in the verification environment because the user service lacks the required `cpu.max` controller; `ansible-lint` is also absent.
- Static syntax/lint checks do not establish script runtime correctness, Ansible converge or idempotence.
- Snapshot growth is monitored rather than enforced by a hard filesystem quota; same-UID hostile host races are outside the file-jail threat model.
