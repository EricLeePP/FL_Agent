# Researcher Diagnostics and Catalog Suggestions

## What Changed

Every response now has structured `issues` and, when appropriate, `suggested_algorithms`.
The CLI writes a readable English `researcher_response.md` alongside `response.json`.
The researcher can see the problem, affected field, clarification candidates,
registered parameter constraints, next steps, and whether anything was submitted.

The explanation is generated deterministically from validator/checker outcomes.
It does not ask an LLM to invent a refusal or a replacement algorithm. The LLM
still extracts the research plan; these checks do not prove the plan faithfully
captured every natural-language requirement. The planned request benchmark remains
necessary. English is the current presentation language, not automatic localization.

## Behavior

| Situation | Status / issue | Action |
|---|---|---|
| Unknown or approximate variable | needs_clarification / unknown_variable | Ask for confirmation, including a single fuzzy candidate; no automatic semantic substitution |
| Unknown or missing hospital | needs_clarification | Show configured public candidates; do not include all hospitals implicitly |
| Unknown/disabled algorithm | unsupported / unsupported_analysis | Show compatible catalog options, explicitly not equivalent replacements |
| Unsupported filter or condition | unsupported / unmet_requirements | Preserve the condition and request clarification or an explicit change |
| Invalid parameter | invalid_request / invalid_parameters | Show field and registered type/range/allowed-name constraint |
| Algorithm not declared for selected sites | invalid_request / algorithm_site_unavailable | Identify missing declared coverage and request administrator verification |
| Data explicitly unavailable | data_unavailable | Identify the checked site/variable and evidence; no task preparation |
| Data omitted from metadata | ready with data_not_listed warning | Unknown, not unavailable; do not claim absence |
| Expired data snapshot | ready with data_metadata_expired warning | Draft only; managed execution requires fresh known availability |
| Authorization returns deny | permission_denied | Stop before data checks; suppress resource suggestions and private policy reasons |
| Preflight service/file fails | preflight_error | Report verification failure, not missing data or a permission denial |
| Backend returns HTTP 401/403/404 | execution_failed | Map to sanitized authentication/access/resource diagnostics; do not reveal backend bodies |
| Submission times out | execution_failed, submission_state=unknown | Submission may have succeeded; check server history before retrying |
| Result retrieval times out | execution_failed, submission_state=submitted | Retain task ID; do not declare the remote task failed or canceled |
| Backend reports failure | execution_failed, submission_state=failed | Do not generate a successful scientific report |

`ready` still means a task draft exists, not that execution is authorized or that
the dataset/image has been tested. The mock success message explicitly states
that no real Vantage6 call was made.

## Which Algorithms Are Suggested

Selection uses the SAME freshly loaded catalog snapshot as planning and
compilation. Candidates must be enabled, use a supported adapter, match the
deployment's Vantage6 major/minor version, cover the selected database label,
and include both the selected organizations and the central execution organization.

If no sites were specified, options are marked `select_sites_first`. If a site
cannot be resolved, the agent asks about the site rather than suggesting an
alternative that silently changes it. Disabled entries and unsupported protocols
are not recommended. A metadata update affects the next response immediately.

Suggestions carry catalog version/hash, capabilities, limitations and declared
or sandbox-verified evidence. `authorization=not_verified` and
`replaces_original_request=false` are explicit. Even a sandbox-verified algorithm
does not prove current per-user permission, data presence, node health or image
availability. No suggestion is executed automatically.

Only public, administrator-maintained metadata belong in these registries.
The current catalog/deployment registries are NOT filtered per researcher.
Before adding confidential resources, implement authenticated, per-user discovery
before metadata are sent to the LLM or shown as clarification candidates.
Suppressing suggestions after a denial is not a substitute for that earlier boundary.

## Data Availability Extension

`DataAvailabilityChecker.check(resolved_plan, task_spec)` is a trusted Python
integration point. The default `UnverifiedDataChecker` returns unknown for every
selected site/variable. A future adapter may query a hospital-side metadata API,
but should not export patient records, unrestricted SQL or credentials to the LLM.

An optional `FileDataAvailabilityChecker` is implemented now:

```powershell
$env:PYTHONPATH = "src"
python -m federated_agent --provider demo `
  --request "Describe age at diagnosis and survival status in hospital 1 and hospital 2" `
  --data-availability-config src/federated_agent/data/config/data_availability.example.json `
  --output output-data-check
```

`src/federated_agent/data/config/data_availability.example.json` is an illustrative declaration, NOT hospital
evidence. Its dates may expire; replace them only after observing actual metadata,
not simply to make a failing check pass. For repeatable simulated examples use the
generator below, which produces fresh, explicitly labeled test inputs.

The file is a closed JSON document with `schema_version`, `version`, `deployment`,
timezone-aware `observed_at` and `expires_at`, and one entry per organization/database.
Each `variables` dictionary maps semantic class IDs to available/unavailable/unknown.
Absent entries mean unknown. No row-level data or access grants belong in this file.

It is reread at preparation and immediately before execution. Wrong deployment,
corrupt/missing files, duplicate declarations and invalid timestamps stop the
preflight. Expired evidence becomes unknown. Provenance includes file version and
SHA-256. The source is always labeled `metadata_snapshot`, never `live_check`.
Availability only covers declared variables; it does not prove cohort size,
statistical validity, row-level consent, data quality or successful mapping.

## Authorization and Execution Boundaries

`AllowAllPolicyChecker` remains the default, as requested for the MVP. Its
`verification=not_checked` is visible in the response. **No real identity provider,
OPA service or permission discovery was installed in this change.**

The existing policy interface now handles deny/unavailable outcomes at preparation
and is invoked again before execution. A real adapter must obtain the researcher
identity from an authenticated application/session, not the natural-language plan
or an arbitrary user-provided ID. It must make an actual authorization decision;
setting the `verification` field is not an implementation of access control.

When a non-default checker is configured, generated tasks are marked
`requires_managed_execution`. The flag is part of the confirmation material.
Standalone exported scripts stop before credentials/network access and direct the
operator back to the configured agent for fresh checks. Unknown data checks block
managed execution, even though a draft can be shown for discussion. This avoids
pretending a frozen script can call an in-process policy or data service.

The default allow-all/unverified standalone export remains available for the
synthetic sandbox, with an explicit warning. Its existing SDK wait has no hard
deadline; this change handles timeout exceptions, it does not add a universal
network timeout or cancel remote tasks. The Ubuntu smoke helper has a separate
bounded-worker implementation. Production still requires backend-enforced access
control, bounded calls, current per-user visibility and result privacy checks.

`execute_response()` is the user-facing execution boundary. Raw SDK exception
bodies and private policy reasons are not included in it. Low-level `execute()`
continues to raise typed errors for callers that need them. Failure-code mapping
does not infer algorithm/data existence from an ambiguous backend 404. A backend
that only raises a generic exception receives a generic execution diagnostic,
not a guessed permission explanation.

## Try the Feedback

From `agent_mvp`:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_feedback_examples.py --output output-feedback-examples
```

This creates seven readable examples: unsupported algorithm, unknown variable,
unknown hospital, unsupported filter, declared-unavailable data, simulated access
denial and a ready-but-unverified draft. It uses fixed plans, synthetic metadata
and a fake denial policy, with no DeepSeek or Vantage6 calls. Use a new output
directory on subsequent runs; existing example folders are not overwritten.

For an actual model-planning test, use `--provider deepseek` with runtime credentials
already configured, then inspect `researcher_response.md`, `response.json` and the
plan. The CLI defaults to mock but now supports `--backend vantage6` for the
explicit synthetic sandbox, with a separate bounded worker, no automatic write
retries and distinct HTTP diagnostics. Read `LIVE_CLI_DEMO.md` for current limits.
Thirty authored planning cases and a no-submission evaluator are now available;
an authored case is not a completed real-LLM benchmark result.
