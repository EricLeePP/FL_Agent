# Real Vantage6 CLI Demonstration

Synthetic sandbox only; not a production authorization system.

## Data Snapshot

The default data snapshot was verified through authenticated node metadata and
expired at `2026-09-18T01:01:43.497792+00:00`. Reverify the nodes and publish a new
snapshot before new computations. An open SSH tunnel alone does not refresh a data
inventory.

## Start on This Windows Computer

Keep the existing SSH tunnel terminal open. From a NEW PowerShell:

```powershell
cd "C:\Users\p70077044\Desktop\Postdoc_project\Agent_fed"
.\agent.cmd
```

The short launcher uses the existing project environment and forwards all CLI
options; no environment activation or PowerShell policy change is needed. It
does not start the SSH tunnel. Read
[the English operations and registry guide](../../AGENT_OPERATIONS_AND_REGISTRY.md)
for one-time setup, current/production connections, JSON structures and the
proposed metadata synchronization. The longer Python entry point remains valid.

The default is DeepSeek planning and REAL Vantage6 execution. The launcher reads
`~/Desktop/deepseek-api.txt` at runtime when present. This must be a UTF-8 text
file containing only the bare API key (one line, no quotes or labels). The key is
not copied into the project or command line. On this Windows computer the file
is `C:\Users\p70077044\Desktop\deepseek-api.txt`.

Credential precedence for the launcher: explicit `--deepseek-api-key-file PATH`,
then `DEEPSEEK_API_KEY_FILE`, then the desktop file, then `DEEPSEEK_API_KEY`, then
hidden terminal input. Thus the new desktop file overrides an older key in the
environment. An explicitly configured but unreadable/malformed file stops startup;
it does not silently fall back. The ordinary CLI and benchmark accept the same
file option/environment variable but do not auto-discover the desktop file.
Local/demo providers do not read the DeepSeek key.

Do not type keys, passwords or patient records
at `Researcher>` or `Clarification>`; these research turns are saved locally and
sent to the selected planner. Vantage6 credentials are requested only after
task review and fresh checks. No credentials are written to artifacts or LLM
prompts. Results are parsed and plotted locally, not sent to DeepSeek.

At the research prompt, enter:

```text
Describe age at diagnosis and survival status in hospital 1 and hospital 2 using synthetic_clean.
```

The agent selects the registered algorithm, maps classes, checks site/database
bindings and data metadata, and shows an exact-task review. In a normal terminal,
use Up/Down to select `Submit and wait for results`, then Enter. The default is
`Keep files without running`. Confirm only if the displayed task matches your
intent. In plain/piped mode, type `confirm` instead. Enter `sandbox_researcher` and its
Vantage6 password, not the Ubuntu SSH password. From there the agent logs in,
checks the sandbox version and encryption mode, submits the task, polls status,
fetches the aggregate result, validates requested fields and produces a chart
and HTML report. No manual SDK script execution is needed.

Expected values for this specific synthetic example: age count 300, mean
23.7783666667, adjusted std 3.3533924612, alive 215 and deceased 85. Numerical
count is non-missing observations, not inherently a unique patient count.

## Streaming Terminal Interface

The CLI now uses a compact, inline conversation layout, with a model/backend
header, trusted status, live progress, streamed explanation and task review.
It does not replace the terminal with a fullscreen dashboard or write secrets to
shell history. Terminal history for research input stays in memory; existing
per-turn research artifacts are still written to the session directory.

- DeepSeek planning uses real SSE content deltas. The progress line updates as
  content arrives; incomplete JSON cannot become an executable plan.
- After validation, a separate streaming model call explains allowlisted public
  preparation facts in English by default, switching to Chinese for a Chinese
  researcher message. This is additional API usage, not a typewriter
  animation. Its prose never updates a plan, chooses a tool or authorizes execution.
- The checked status and exact-task review remain deterministic and authoritative.
  Warnings remain in the explanation facts; allow-all policy is not a claim of
  live node availability or server authentication.
- Results, raw backend errors, passwords, API keys, server connection objects and
  patient records are excluded from explanation input. Completed-result reporting
  stays local. Backend failures also use local diagnostics, not remote narration.
- Up/Down moves through choices. Space toggles hospitals in a multi-select; Enter
  accepts the selection. Escape cancels. `Type a reply instead` retains free text.
  `/menu` offers common actions; Tab completes commands; `/details` shows the full
  last checked response. `/algorithms` still rereads the file without any LLM call.
- `--plain` disables cursor controls. Redirection or `TERM=dumb` also selects
  plain mode automatically. `--no-stream` disables SSE and the extra explanation
  call, retaining deterministic feedback and ordinary JSON planning. A local
  OpenAI-compatible server must implement SSE to use streaming mode.

If the explanation stream breaks, the partial text is marked incomplete and the
checked response remains available. Ctrl+C exits locally and never silently retries
a remote task. ANSI/control characters in external text are not executed. Live
text is reflowed as a whole per delta to avoid split-CJK wrapping on Windows.

Policy stays `AllowAllPolicyChecker`; no additional policy-approval dialog is
introduced. Task review is study-intent confirmation, not a permissions gate.
Actual Vantage6 login and server/node access controls are not bypassed.

## Greetings and Language

Pure greetings and thanks (`hi`, `hello`, `good morning`, `thanks`, and common
Chinese equivalents) get an immediate conversational reply without a model call,
catalog query, algorithm error or task preparation. Matching covers the entire
message: `Hi, describe age ...` remains a research request. This is a conservative
small-talk route, not an unrestricted general-purpose chat assistant.

Greetings during clarification do not clear the pending study. Normal typed
messages select the explanation language: English by default, Chinese when the
researcher writes Chinese, switching back for an English message. Commands,
numeric-only replies and cursor-selected catalog values keep the previous
language. English catalog names cannot switch a Chinese conversation by accident.
Static terminal controls, algorithm IDs and canonical field names stay English.
`request.json` and `stream.json` record the selected response language.

Local model variant:

```powershell
.\agent.cmd --provider local --llm-endpoint http://127.0.0.1:8000/v1/chat/completions --model YOUR_DEPLOYED_MODEL
```

Only use that address/model if your local model server is actually configured
there. To test the real backend without an LLM, append `--provider demo` instead.
The demo planner recognizes a narrow set of English phrases and does NOT prove
natural-language understanding or LLM performance. It also does not provide
free-form multi-turn clarification; use a complete revised request in demo mode.

On another installation, install this project with `pip install -e '.[live]'`
under Python 3.10 and supply the appropriate public deployment/data configuration.
No change to an existing system Python is necessary.

## Conversation and Failure Demonstration

Use DeepSeek or a configured local model for these examples:

| Researcher input | Expected behavior |
| --- | --- |
| `Describe age at diagnosis.` | Ask for hospitals and a dataset; retain supplied fields across clarification |
| `Describe age at diagnosis and survival status in hospital 1 and hospital 2.` | Ask which dataset; do not generate a task/script or use the deployment default |
| `Describe age at diagnosis in hospital 99.` | Explain unknown hospital and list registered site candidates; no substitution |
| `Describe age at diagnosis in hospital 1 and hospital 2 using synthetic_nonexistent.` | Ask for a registered database; do not fall back silently |
| `Describe gender in hospital 1 and hospital 2 using synthetic_clean.` | Explain that the schema variable is absent from this synthetic fixture inventory |
| `Fit a Cox model in hospital 1 and hospital 2.` | Explain unsupported algorithm and show registered alternatives; do not switch analyses automatically |
| `Describe age at diagnosis in hospital 1 and hospital 2, only for patients younger than 25.` | Refuse to drop the unsupported filter |
| `Describe age at diagnosis in hospital 1 using synthetic_clean.` | Reject the pinned algorithm's known single-site numeric defect; do not add a hospital |
| `Describe survival status in hospital 1.` | Categorical-only single-site preparation is allowed; this is not the failing numeric path |
| `/algorithms` | Reload the JSON catalog and show current capabilities and limitations without an LLM call |
| `/new` or `/cancel` | Clear a pending clarification; no submission |
| `/exit` | End the CLI session |

After a refusal, enter a complete revised request. Alternatives never replace
the research question automatically. After a clarification, a model-assisted
reply is combined with the previous structured plan; the full task must still
be reviewed and confirmed.

HTTP 401 and 403 from Vantage6 have distinct authentication/access diagnostics.
They do not disclose restricted-resource details. HTTP 404 does not prove a
hidden resource exists or does not exist. A crashed task is reported as failed,
with its ID, not as a successful empty analysis. Its exact hospital-side cause
may still require a protected administrator log inspection.

## Artifacts and Reproducibility

Each launch creates a new directory under `integration/agent-demo/session-*/`.
Every turn has its own `turn-NNN/` directory:

- `request.json`, `response.json`, `researcher_response.md`: conversation and diagnostics.
- `assistant_reply.txt`, `stream.json`: optional streamed explanation and its
  completion/chunk/timing record. It is explicitly non-authoritative.
- `catalog_snapshot.json`: exact algorithm knowledge used for that turn.
- `prepared_task.json`, `task_spec.json`, `run_vantage6_task.py`: reviewable plan,
  task and script. The exported script deliberately refuses unmanaged execution
  when data/policy rechecks are required; the CLI executes through the trusted SDK worker.
- `backend/submission.json`, `backend/state.json`, `backend/result.raw.json`:
  real task ID, execution state and aggregate return. Passwords are excluded.
- `execution.json`, `parsed_result.json`, `result.png`, `result.html`: completed
  execution record and local reporting. Report validation is structural/coverage
  validation, not proof of clinical validity or exact pooled quantiles.

The 600-second client deadline terminates only the local SDK worker, NOT the
remote task. A timed-out submission may have succeeded without a recorded ID.
Never blindly retry. A second execution of the same prepared artifact is blocked
when its `backend/` attempt folder exists. Creating another reviewed task is an
explicit new attempt, not server-side deduplication or automatic recovery.

## Source of Constraints

- `src/federated_agent/data/catalogs/algorithms.json`: versioned algorithm input schema, deployment scope,
  capabilities and machine-enforced `constraints.min_sites_for_numerical=2`.
  The S01 restriction comes from Ubuntu task 5's reproduced upstream defect.
- `src/federated_agent/data/config/ubuntu_sandbox.json`: returned collaboration/site IDs and tunnel binding.
- `src/federated_agent/data/config/ubuntu_data_availability.json`: operator snapshot derived from the
  returned fixture manifest and Ubuntu node-mapping evidence. The two columns
  are a complete inventory ONLY for these synthetic CSVs. Absence from a normal
  incomplete metadata snapshot still means unknown, not unavailable.
- The snapshot expires on 2026-09-14. Refresh it only after checking actual
  mappings; do not extend its timestamp simply to bypass a failed check.

Catalog and deployment are reloaded for planning/review; catalog hash, task
confirmation, deployment and data/policy checks are checked again at submission.
File metadata does not prove live node health, nonempty cohorts or authorization.
Policy is still explicitly allow-all; authenticated Vantage6 server/node controls
remain active. Server-side policy enforcement, tenant-filtered catalog visibility
and production result-privacy controls remain future work.

## Validation Status

The pre-existing Windows smoke S02 task 40 passed. It did NOT use an LLM.
New CLI unit/integration tests use fixed plans, HTTP mocks, and a reduced fixture
derived from that real wrapper. Those tests do not count as new real task runs.
The full suite now passes 152 tests, including actual SDK authentication,
serialization and result decoding against mocked HTTP responses. Dependency
and compilation checks also passed. The real CLI entry point was exercised
through catalog listing, clarification, task review and cancellation.
The new real CLI run and end-to-end LLM run must be recorded separately.

The earlier environment-key probe returned HTTP 402. The user-supplied desktop
file key subsequently succeeded on 2026-09-07, generating a real DeepSeek plan,
task configuration and Python script. A Chinese two-turn launcher session also
completed missing-hospital clarification and task review; execution was cancelled.
This is real LLM evidence, but NOT a new Vantage6 execution.

`examples/cli_requests.json` contains 30 planning acceptance cases. Run them with:

```powershell
.\integration\.venv-vantage6\Scripts\python.exe .\agent_mvp\scripts\evaluate_requests.py --provider deepseek --deepseek-api-key-file "$HOME\Desktop\deepseek-api.txt"
```

This evaluator never submits Vantage6 tasks. It checks statuses and, for applicable
cases, exact sites, variables and database labels. It stops at provider failures
and reports untested cases accurately. These cases are a baseline, not a complete
clinical benchmark or proof that an LLM never drops a requirement.

Implementation references: [DeepSeek SSE API](https://api-docs.deepseek.com/api/create-chat-completion/),
[prompt-toolkit applications](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/full_screen_apps.html),
[Rich live layout](https://rich.readthedocs.io/en/stable/live.html).
