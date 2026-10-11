# Fixed Colleague Template and LLM-Generated Task Parameters

Requirement clarified by the researcher on 2026-09-16.

## Core Requirement

Preserve the fixed researcher-side workflow from Joshi's Vantage6 example.
Use the LLM to interpret the researcher's request and propose task-dependent
parameters. Resolve and validate those proposals against the configured algorithm,
semantic, deployment and data-availability documents before filling the template.
Do not make the LLM rewrite the workflow or execute unconstrained generated code.

The intended pipeline is:

```text
Research request + current algorithm contract/public context
    -> LLM parameter proposal (ResearchPlan)
    -> catalog-backed resolution, validation and clarification
    -> validated task parameters
    -> researcher confirmation and fresh checks
    -> fixed connect/authenticate/submit/wait/retrieve/save workflow
```

This is the required product behavior, not a claim that the model always extracts
every condition correctly. Existing evaluation limitations remain relevant.

## What Stays Fixed

Joshi's example creates a client, authenticates, configures encryption, calls
`client.task.create`, waits, retrieves a result and saves it. The current SDK
worker retains this lifecycle. The submission call exists exactly once, in
`runs/submission.py`, and exposes the same parameter positions as the
colleague's example:

```python
def submit_task(client, payload):
    """Create the Vantage6 task; only validated task values vary."""
    return client.task.create(
        collaboration=payload["collaboration"],
        organizations=payload["organizations"],
        name=payload["name"],
        image=payload["image"],
        description=payload["description"],
        input_=payload["input_"],
        databases=payload["databases"],
    )
```

`payload` comes from a validated task, not a dictionary accepted directly from
the LLM. The runtime uses bounded polling, durable task IDs and explicit failure
handling around the original lifecycle. It is not a byte-for-byte copy of the
Gist and does not retain its evaluation-only `SCENARIOS` loop as a default.
The synthetic worker retains its existing loopback/version/unencrypted checks.

The reviewable Python export embeds the source of that same function verbatim,
so the script a researcher reads shows the call the worker actually makes. A
test fails if the two ever differ. The export also recomputes its confirmation
ID and refuses to run if the embedded task content changed. It never reads
credentials and never submits: managed submissions need fresh data-availability
and policy checks, so the export ends by printing the agent command that
performs them:

```text
python -m federated_agent --execute-prepared prepared_task.json --confirm <ID> \
    --backend vantage6 --confirm-synthetic-unencrypted-sandbox
```

The CLI always executes the installed trusted worker rather than an editable
exported script. These boundaries do not require the LLM to understand or modify
execution code.

## Confirmation Fingerprints

The confirmation ID binds everything that affects a submission: connection,
client payload, result adapter, catalog identity, safety flags, registry
release, execution template and local metadata content.

Catalog and metadata are identified by canonical JSON content hashes, without
paths, and code by the newline-normalized hash of the worker sources only
(fingerprint version 2). The same content therefore gives the same ID in another
directory, on another machine or with Windows line endings. Editing the CLI,
planner or reporting code does not invalidate prepared tasks; only the files in
`execution_templates.WORKER_SOURCES` form the code identity.

A task that was already submitted can be recovered after the worker changed,
because recovery only reads the recorded remote task. A task that was never
submitted cannot be submitted by a different worker version; prepare it again.

## What Varies

| Item | LLM responsibility | Deterministic responsibility |
| --- | --- | --- |
| Analysis/algorithm | Interpret the requested analysis; propose an existing algorithm when unambiguous | Confirm the enabled contract and deployment scope; clarify ambiguity |
| Research variables | Extract requested concepts, preserving unsupported or unknown terms | Resolve semantic classes and supported datatypes; check declared data availability |
| Hospitals and dataset | Extract the requested names and labels, without inventing IDs | Resolve deployment-scoped IDs and dataset bindings |
| Algorithm options | Extract explicitly requested options described in the parameter schema | Check required fields, allowed names/types/ranges and documented defaults |
| Image, method and argument names | Do not invent or override these | Read them from the selected versioned contract |
| Credentials and connection | No access to credentials; do not generate destinations | Use the reviewed connection profile and local credential provider |
| Execution order and error handling | Do not change them | Run the fixed workflow after confirmation and preflight |

LLM-generated values are proposals, not trusted executable parameters. Missing
values are clarified unless an applicable default is explicitly declared. If a
requested filter or algorithm is unsupported, the system must explain that fact
and offer catalog-backed alternatives without changing the study automatically.

The policy extension currently returns allow; this does not bypass task validity,
data-evidence checks or the Vantage6 server/node's own authentication and controls.

## Algorithm-Specific Parameters Are Not Universal

Joshi's original example targeted a synthetic-data generator and passed
`variables_of_interest`, `training_config` and `column_config`. The currently
registered STRONG AYA descriptive-statistics contract instead accepts
`variables_to_describe`, `variables_to_stratify` and `organisation_ids`.
The execution lifecycle can stay fixed while these input contracts differ.

Do not copy the training kwargs into descriptive statistics, or claim that the
original synthetic-data training algorithm is already supported. Supporting that
algorithm requires its exact versioned input/output contract, deployment/data
bindings, suitable adapters and an actual sandbox test. Merely supplying a script
example is not evidence that its image and data are available now.

## Current Evidence and Files

- `src/federated_agent/planner.py`: LLM proposal contract and current catalog context.
- `src/federated_agent/prompts.py`: every instruction given to a model, shared by both runtimes.
- `src/federated_agent/clarification.py`: menu selections applied to the plan without a model.
- `src/federated_agent/task_compiler.py`: parameter bindings and schema checks.
- `src/federated_agent/runs/submission.py`: the single submission call.
- `src/federated_agent/runs/worker.py`: fixed SDK execution lifecycle in the trusted worker.
- `src/federated_agent/runs/store.py`: the on-disk run state that the worker and the CLI share.
- `src/federated_agent/script_renderer.py`: fixed reviewable script export.
- `tests/test_fixed_template_workflow.py`: fixed-workflow/variable-parameter tests.
- `tests/test_execution_core.py`: fingerprint portability and worker upgrades.

The new tests use a fake JSON provider and a mocked Vantage6 client, not remote
DeepSeek or an actual submission. They check that different valid requests change
only the generated parameter slots, invalid proposals cannot produce a prepared
task, undeclared executable fields are rejected, and runtime/export submission
fields remain aligned. They do not measure natural-language model accuracy.

Validation on 2026-09-16: all 156 baseline tests passed before edits; all 161 tests
passed after adding the five fixed-template tests. No remote model call or real
Vantage6 submission was performed for this change.

The actual Ubuntu data snapshot expired on 2026-09-14 and was not extended in
this change. Tests explicitly use the historical evidence time; real execution
still requires genuinely refreshed evidence. Automatic Ubuntu-to-Windows metadata
synchronization remains a proposal, not an implemented feature.

Original input: [Joshi's researcher-side example](https://gist.github.com/JHogenboom/2dc551b58595d1900710e74c5e3c7c6a).
