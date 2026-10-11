# Fed Agent Architecture

This describes the layers of Fed Agent V2, the main flows through them, and the
invariants that the tests keep.

## Layers

```text
Transports      cli.py               mcp_server.py           terminal_ui.py
                one request or a     read-only tools for     rendering only
                saved study          other agents
                      \                    |                    /
Composition     bootstrap.py: flags > environment > config file > defaults -> AppConfig
                app.py: build_application(), build_agent(), build_planner(), describe()
                      |
Application     services/study.py      saved-study rules behind a UI port (StudyChat)
services        services/execution.py  executor setup, consent, persisting an outcome
                services/reporting.py  result artifacts, reports, report regeneration
                services/responses.py  every failure mapped to a researcher response
                service.py             FederatedResearchAgent: respond, check_plan,
                                       prepare, execute_response, recover_response
                      |
Domain          metadata.py (MetadataSnapshot)  task_compiler.py  confirmation.py
                policy.py  data_availability.py  requirements.py  clarification.py
                lookups.py  prompts.py  narration.py  capabilities.py
                      |
Adapters        adapters/vantage6.py   the only module importing the Vantage6 SDK
                sdk_runtime.py         the only module importing the Agents SDK
                planner.py             JSON-mode HTTP providers and the planners
                result_adapters.py     result checks and reports, with plugins
                      |
Execution core  runs/store.py       RunStore: the run state on disk, one owner
                runs/worker.py      the trusted worker process
                runs/submission.py  the single task.create call
                runs/executor.py    the parent side of the worker
```

Dependencies point downwards. Architecture tests stop services from importing a
transport and stop the command line from reaching into run state, policy
enforcement, storage or manifests.

## The main flows

**A research request.** `respond()` answers greetings and local questions without
metadata or a model. Otherwise it captures one immutable `MetadataSnapshot` and
uses only it. The planner proposes a `ResearchPlan` from the shared prompts in
`prompts.py`; with `llm.tool_lookups` the SDK model may first call read-only
lookups bound to the same snapshot. Requirement protection, name resolution,
compilation, the result-adapter check, the policy's `prepare` decision and the
data checks follow. Any failure becomes a researcher response in
`services/responses.py`.

**A saved study.** `StudyChat` runs commands and research turns. A menu selection
is applied to the plan without a model call. A ready task is written with its
review copy and manifest. An unresolved earlier attempt blocks a new submission.
Consent is saved before execution, and recovery or report regeneration never
submits.

**Execution.** `execute_response()` checks the confirmation, captures fresh
metadata, prepares the task again and requires an identical confirmation ID. It
then applies the runtime policy and hands the task to the executor. The mock
executor runs in process. The sandbox executor starts the trusted worker, gives it
credentials once over stdin and follows it only through `RunStore`. The named
result adapter checks the returned result, and the report comes from the same
adapter.

**Recovery.** `recover_response()` never compiles or submits. The worker in resume
mode reads the recorded task; it accepts a task prepared by another worker
version and logs that it used the installed adapter.

## Invariants and the tests that keep them

| Invariant | Test |
| --- | --- |
| Services branch on declared capabilities, never on a collaborator's class | `test_architecture` (hard zero) |
| Services never import a transport; the CLI holds no run, policy or storage rules | `test_architecture` |
| One atomic JSON writer; bundled files resolve through `resources.py`, not the source tree | `test_architecture` |
| Vantage6, Agents SDK, OpenAI and MCP imports each stay in the one module that owns them | `test_architecture` |
| Every model instruction lives in `prompts.py` | `test_architecture` |
| The agent, services and CLI never name a result type | `test_architecture` |
| Every error code has a researcher message | `test_error_codes` |
| The exported script contains the worker's own submission function | `test_fixed_template_workflow` |
| Confirmations hold no file paths; a task from another worker version is recoverable but never submitted | `test_execution_core` |
| Researcher-visible outcomes of the 30-case corpus stay fixed | `test_behavior_snapshots` |

## Extension points

| To add | Do this |
| --- | --- |
| An executor, policy provider, data checker, planner or model provider | Implement the protocol and declare its `capabilities` (`capabilities.py`) |
| A result type for a new algorithm | Publish a result adapter through the `federated_agent.result_adapters` entry points ([registry](ALGORITHM_REGISTRY.md)) |
| A metadata source | Return a `MetadataSnapshot`, as `capture_local` and `release_snapshot` do |
| A transport | Drive `StudyChat` through its UI port, or call the agent's public methods |
| A model-facing tool | Add a read-only method to `lookups.py`; the MCP server and SDK loop expose it |

A new way of submitting tasks is deliberately not a plugin. It needs a reviewed
runtime adapter in the trusted worker, whose sources form the code identity that
confirmations bind.

## Confirmation fingerprints

A prepared task carries fingerprint version 2: catalog and metadata are identified
by content hashes without file paths, and code by the newline-normalized hash of
the worker sources. Any other version is refused, so the task must be prepared again.

## Not yet verified

- The Windows CI job has not run on this repository yet.
- Planning quality with tool lookups has not been measured with a real model.
  Local planner models are evaluated in [LLM_EVALUATION.md](LLM_EVALUATION.md).
- No real Vantage6 task has run on this version yet. The bundled sandbox inventory
  expired on 2026-09-18 and must be reverified first.
- The MCP server offers stdio only; a network transport needs authentication.
