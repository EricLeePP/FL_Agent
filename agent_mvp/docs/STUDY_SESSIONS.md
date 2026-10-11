# Stateful Studies and Read-Only Recovery

Implemented on 2026-09-17. This is a synthetic/public-data MVP, not production
authorization, an autonomous coding environment, or a guarantee of scientific
validity.

## Start and resume

From PowerShell in the project root:

```powershell
.\agent.cmd
```

Within the CLI:

```text
/sessions
/resume session-<the-id-shown-in-the-list>
/plan
/history
/report turn-001
```

Alternatively, reopen a known session directly:

```powershell
.\agent.cmd --resume session-<the-id-shown-in-the-list>
```

Session IDs are directory names under the configured `--output` directory. Use
the same output directory and backend that created the session. No filesystem
paths are accepted as session/turn IDs. Only one CLI process may own a session
at a time; the OS releases its lock when the process exits. Existing pre-upgrade
turn folders can be imported locally on first resume without executing anything.

Resuming does **not** submit, retrieve results, or replay old confirmation. The
default provider still initializes its normal local credential configuration;
`--provider demo` can inspect/recover sessions without an LLM API key.

If remote computation completed but local report generation failed, `/history`
shows the report status. `/report <turn-id>` rebuilds from the saved result under
current result-access policy, without calling the model or submitting a task.

## Continue a study

For example, after a prepared or completed request:

```text
Keep the same hospitals and add survival status. Leave everything else unchanged.
```

The LLM now proposes a partial `{"changes": {...}}` update. Python preserves
omitted fields, validates the merged ResearchPlan and rechecks the current
catalogs. Replacement lists must retain the existing members when the request
is to add a member. Explicitly requested unsupported conditions stay in the
study until withdrawn; they continue to block execution. Invalid data requests
remain in the active proposal so a later hospital change cannot silently forget
the missing variable. Malformed model output does not replace the current plan.

Explicit dataset selections and selected algorithm IDs are pinned in the
saved plan. Deployment defaults are no longer researcher choices. Missing
selection requires clarification; an omitted follow-up field retains an existing
selection visibly, while `changes.database_label=null` clears it and requires
reselection. Updated defaults must not silently change an ongoing study. The plan
is still a proposal, not proof of data availability or permission. Study context
retains a follow-up when the model leaves its old purpose unchanged.

Choosing a candidate from a clarification menu does not call the model. The
selection is one of the validators' own candidates, so it is applied to the plan
directly: a dataset or algorithm choice sets that field, and a hospital or
variable choice replaces exactly the unresolved name it answers. The revised plan
then passes the same validators. Typed replies, and selections for any other
field, still go to the model. A selection leaves the study context unchanged.

A study's dataset is always an explicit choice; see
[DATASET_SELECTION.md](DATASET_SELECTION.md).

Before each new execution, the CLI displays the changed fields, current task,
organization IDs, and confirmation identifier. A new explicit confirmation is
required. `/review` revalidates an unexecuted plan without calling the LLM; a
cancelled credential prompt or known pre-submission rejection can be reviewed
again, but an already submitted or uncertain task cannot be duplicated. `/new` or
`/cancel` clears the active plan, retains history, and **does not cancel any
remote task**. Starting a separate study can intentionally create a new task;
this is not server-wide deduplication.
An unresolved earlier execution in the same session also blocks submission of a
revised plan, even though planning can continue. Recover/reconcile it first;
`/new` is not a bypass for an uncertain submission in that session.

## Recover an existing remote task

Restore the SSH/VPN connection, resume the original session, then:

```text
/history
/recover turn-003
```

`/recover` without a turn ID uses the active turn. Recovery:

1. Validates the saved task, confirmation hash, recorded task ID and current
   connection against the original deployment.
2. Acquires execution/worker locks and obtains Vantage6 credentials locally.
3. Applies the existing policy extension point, authenticates through the SDK,
   and reads the existing task status and results.
4. Validates result structure/variable coverage and writes the local report.

The recovery worker has a separate branch that never calls `task.create()`.
Its only write to the server is authentication; task operations are reads.
Expired data metadata blocks **new computations**, not read-only retrieval of an
already submitted task. Server authorization still applies; the development
allow-all hook is not an access grant. The current policy interface is task-based,
not yet a production action-specific result-access policy.

If the submission response was lost before its task ID was durably recorded,
recovery stops with `submission_unresolved`. The administrator must reconcile
the recorded timestamp, task parameters and server history. There is no unsafe
automatic retry, automatic task matching or server-side exactly-once guarantee.
Do not manually invent/edit a task ID. An ongoing original worker must stop before
another worker can attach; terminating local monitoring does not cancel Vantage6.

## Local files

| File | Purpose |
| --- | --- |
| `session.json` | Active plan, pending clarification, language, revision, current turn and result-free last status |
| `turn-*/request.json` | Researcher-authored input and response language |
| `turn-*/plan_revision.json` | Previous plan, proposed/effective plan and field-level changes |
| `turn-*/planning_response.json` | Preparation/validation response retained separately from execution |
| `turn-*/prepared_task.json` | Validated task and original contract references |
| `turn-*/researcher_confirmation.json` | Exact confirmation ID and confirmation time; never auto-replayed |
| `turn-*/run_manifest.json` | Model name, prompt hash, template identity/source hashes, metadata provenance, data checks and task ID/status |
| `turn-*/backend/submission_intent.json` | Durable intent immediately before the create call; prevents replay of an uncertain attempt |
| `turn-*/backend/submission.json` | Returned task ID and task confirmation binding |
| `turn-*/backend/recovery-*.json` | Timestamped read-only recovery attempts |

The fixed template is identified as `joshi_vantage6_userclient`, version `1.0.0`.
This is an audit identity, **not** a dynamically selectable template/plugin
registry. Old runs retain their original provenance; importing them does not
retroactively create missing model or template evidence.

State writes use atomic replacement and flushed file contents. Results are not
included in LLM follow-up context, and configured API keys/passwords are not
serialized into session or manifest files. Researcher-authored text is stored
locally: do not enter credentials or patient records in the research prompt.
Filesystem access/retention, encrypted storage and production identity controls
are still deployment responsibilities.

## Current evidence and limitations

- Real DeepSeek checks used public requests with mock-backend preparation:
  add a variable, restart, add an unsupported age filter, restart, change sites
  while retaining that filter, then explicitly remove it. SSE narration was
  also exercised. No new real Vantage6 computation was submitted.
- Automated tests exercise restart, contextual updates, defaults, locks,
  uncertain submission, denial, policy hooks and recovery. A real installed
  Vantage6 SDK worker reconnects to a loopback HTTP **test server** without a
  task-creation POST; that is not an Ubuntu integration run.
- Ubuntu recovery still needs a connected-sandbox acceptance run.
- The bundled operator snapshot has expired. It was not extended.
  An operator must reverify site metadata and publish a new snapshot, then
  supply it through the existing data-availability configuration. Do not change
  timestamps merely to enable execution.
- Automatic Ubuntu publishing/synchronization, a unified immutable metadata
  release, pluggable template packages, independent model benchmarks and OPA
  enforcement remain follow-on work. No new algorithms were registered here.

Run the full local regression suite from the project root:

```powershell
.\integration\.venv-vantage6\Scripts\python.exe -m unittest discover -s agent_mvp/tests -v
```
