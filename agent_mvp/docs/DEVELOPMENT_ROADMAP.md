# Mandatory Development Milestones

The three milestones below are user requirements, not optional suggestions.
Adding a live algorithm registry does not replace any of them.

## Foundation: live algorithm contracts

Status: implemented and tested offline.

One JSON registry now supplies every default request with current algorithm
knowledge and executable contracts. Tests cover hot updates, source snapshots,
capability queries, default/parameter validation, availability declarations and
stale task rejection. Policy remains allow-all. See `ALGORITHM_REGISTRY.md`.

## 1. Research-request evaluation corpus

Status: a 30-case development corpus exists in `examples/cli_requests.json`, and
the planner evaluation suite (`evaluation/`, see `LLM_EVALUATION.md`) adds 371
core and 123 held-out synthetic cases with validated labels, deterministic scoring
and adversarial safeguard tests. Gold review with Len/Joshi and evaluation on real
researcher requests remain pending. Automated unit/integration tests are NOT
substitutes for this corpus.

Build 40 initial research requests (within the requested 30-50 range), with gold
expectations reviewed with Len/Joshi. Proposed distribution: 10 valid requests,
8 missing/ambiguous inputs, 8 synonyms/multilingual formulations, 6 unsupported
analyses and 8 filter/time/stratification constraints. Include catalog-update and
multiple-algorithm variants without leaking evaluation examples into the prompt.

Each case should store an ID, request, expected status, required concepts and
constraints, expected semantic classes, allowed algorithm IDs, deployment fixture,
catalog version/hash and expected clarification fields. Pin fixture versions for
reproducibility. Changes to a catalog require explicit expectation review.

Compare demo extraction, the existing LLM compiler, and the live-registry version.
Record provider/model/prompt version, latency, calls, task validity, exact contract
matching, omitted requirements and incorrect unsupported-request acceptance.
Repeat nondeterministic runs and report variability. Preventing silent loss of
filters is more important than maximizing the number of generated scripts.

Acceptance: every gold case has a reviewed expected outcome; executable outputs
use the specified snapshot and preserve all supported constraints. Unsupported
constraints cannot produce a silently simplified task. Publish measured results,
not a claim that a test count demonstrates natural-language accuracy.

## 2. Stateful multi-turn clarification

Status: persistent study state, cross-process resume, partial plan updates,
revision diffs, pinned defaults and read-only task recovery are implemented.
Real DeepSeek follow-ups across restarts retained hospitals/variables and an
unsupported filter until explicitly withdrawn. See `STUDY_SESSIONS.md`.
The broader independently reviewed adversarial dialogue evaluation remains pending.

Persist session ID, turn history, latest plan, unresolved fields, deployment
reference and catalog hash. Accept a follow-up as a constrained update to the
previous plan, preserving fields not changed by the researcher. Revalidate every
turn against a fresh catalog snapshot; do not reuse a now-disabled algorithm or
silently change the previously selected one. Record old/new values and sources.

Acceptance: "hospital missing -> follow-up question -> researcher adds hospital ->
updated plan -> generated script -> confirmation" completes in one session.
Also test changed variables, conflicting answers, cancellation, algorithm removal
between turns, and confirmation invalidation after any executable change. Test
both a deterministic fake provider and the selected real LLM backend.

## 3. Real Vantage6 sandbox validation

Status: real tasks have run against the synthetic sandbox. The recovery path has
unit tests and an installed-SDK subprocess test against a loopback HTTP test
server, NOT yet a fresh Ubuntu recovery run. The bundled data metadata has
expired: restore connectivity and reverify metadata for new computations, and
separately validate recovery and expanded scientific/failure acceptance cases.

Confirm with Len/Joshi: server/client version, endpoint/API prefix, collaboration,
central organization, participating organization IDs, database labels, synthetic
datasets and their RDF/SPARQL representation, image digest/version/signature,
algorithm-store/node configuration, and access credentials. Keep credentials out
of prompts, registries and logs. `src/federated_agent/data/config/ubuntu_sandbox.json` points to the returned
synthetic sandbox via a local SSH tunnel; the separate demo `.invalid` URL is not
a real backend.

Deploy the pinned algorithm using synthetic data in at least two nodes. Submit
the generated research-side script, retain task IDs/status/errors and parse the
actual result envelopes. Compare with direct local computations using the SAME
definitions, missing-value handling and aggregation semantics. An average of site
means or medians is not automatically a pooled mean or median; check the upstream
implementation before choosing numerical acceptance criteria.

Acceptance: a reproducible run produces the expected payload, real node execution,
traceable results and a valid report; count/summary comparisons meet explicitly
chosen exactness/tolerance criteria. Add timeout, unreachable-node and algorithm
failure cases. Report which comparisons pass and which are unsupported; mock
fixtures must never be presented as real federated results.

## Intended value

The measurable contribution is reliable translation of research intent into
versioned, deployment-aware, reproducible federated workflows. The registry
separates model knowledge from verified execution contracts and allows updates
without rewriting the planner. Evaluation and real execution must establish
whether this reduces researcher errors, manual corrections and time-to-valid-task.
The registry alone does not establish scientific novelty, clinical validity or
privacy protection; those claims need the respective studies and enforcement.
