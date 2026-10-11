# Live Algorithm Registry

Implemented: 2026-09-06. Policy remains `AllowAllPolicyChecker`.

## Decision and scope

`src/federated_agent/data/catalogs/algorithms.json` is the canonical, administrator-maintained source for
algorithm knowledge AND executable input contracts. Markdown explains this
format; it is not a second source of algorithm truth. Updating the JSON requires
neither retraining the model nor rebuilding an embedding index or restarting an
existing agent object.

This is an agent-facing registry, not a replacement for a Vantage6 algorithm
store, node configuration, container registry, or hospital authorization system.
Adding a record does not install an algorithm, make a node reachable, make its
data compatible, or prove permission. Site administrators still have to deploy
and configure their infrastructure. The file records those facts/declarations.

The current entry is STRONG AYA descriptive statistics v2.1.0. Its local demo
availability remains `declared`. A separate `ubuntu-synthetic-sandbox` scope now
records real returned evidence, including Windows smoke task 40. This is not a
production availability claim. Numeric requests with fewer than two sites are
blocked by `constraints.min_sites_for_numerical`, based on the reproduced S01
defect. These machine-readable constraints are also included in planner knowledge.

## Data ownership

| File | Responsibility |
| --- | --- |
| `src/federated_agent/data/catalogs/algorithms.json` | Algorithm identity, capabilities, limitations, lifecycle, version, image, method, schemas, bindings, result adapter, deployment scope, evidence |
| `src/federated_agent/data/config/demo_deployment.json` | Endpoint/version, collaboration, central organization, site aliases/IDs and database labels |
| `src/federated_agent/data/catalogs/AYA_cancer_schema.jsonld` | Semantic variable classes and data types |
| `src/federated_agent/data/catalogs/algorithm_store.json` | Upstream metadata snapshot used as supporting evidence, not the live execution contract |

A deployment profile does not list algorithms; images and signatures live only
in `algorithms.json`. Updating an algorithm for
existing sites changes only `algorithms.json`; introducing a new site also
requires its deployment/semantic metadata to exist.

## Record structure

The complete editable example is `src/federated_agent/data/catalogs/algorithms.json`. Strict models are in
`models.py`; `AlgorithmRegistryDocument` in `algorithm_catalog.py` validates the
file and cross-field consistency.

| Field | Meaning |
| --- | --- |
| `schema_version` | File format, currently `1.0`; unsupported formats fail loading |
| `catalog_version`, `updated_at` | Human-maintained revision and change date |
| `algorithms[].algorithm_id` | Stable unique ID; one selected version per ID in this registry |
| `algorithm_version`, `image`, `vantage6_version`, `function` | Pinned implementation and its Vantage6 entry point; no implicit or `latest` image tags |
| `status` | `enabled`, `draft`, or `disabled`; only enabled records are selectable |
| `analysis_types`, `request_keywords` | Intent categories and matching hints; not proof of scientific suitability |
| `description`, `capabilities`, `limitations` | Public planning context; unsupported scientific capabilities must be explicit |
| `accepted_arguments` | Exact keyword names of the pinned implementation, not a guess from a different release |
| `source`, `metadata_source`, `contract_warnings` | Code/metadata versions and known discrepancies |
| `availability[]` | Deployment name, allowed participant/central organization IDs, database labels, verification state and evidence |
| `execution.adapter` | Compiler adapter name; an unknown name is visible but cannot compile |
| `execution.parameters_schema` | Closed JSON Schema for researcher-specified options and documented defaults |
| `execution.input_bindings` | Declarative mapping from validated research concepts/options to algorithm kwargs |
| `execution.kwargs_schema` | Closed JSON Schema for the final argument payload |
| `execution.result_adapter` | Name of an installed result adapter (built in: `descriptive_statistics_v1`, `raw_json`); a task cannot be prepared while its adapter is missing, and new result types never inherit descriptive-statistics reporting |

`enabled` is an operational lifecycle setting, not a substitute for future OPA
approval. `sandbox_verified` is still an administrator's recorded assertion;
the agent does not automatically verify that evidence today.

### Input bindings

The generic `vantage6_kwargs_v1` adapter supports four sources:

| Binding source | Value supplied by the compiler |
| --- | --- |
| `variables_by_class` | `{class_id: {datatype: ...}}`, resolved from the semantic catalog |
| `organisation_ids` | IDs resolved from the deployment registry |
| `parameter` | A named researcher option validated against `parameters_schema`, or its explicit schema default |
| `constant` | An explicit catalog value, including `null` |

For the actual v2.1.0 descriptive-statistics entry:

```json
{
  "variables_to_describe": {"source": "variables_by_class"},
  "variables_to_stratify": {"source": "constant", "value": null},
  "organisation_ids": {"source": "organisation_ids"}
}
```

For a future compatible algorithm that truly accepts an `iterations` argument,
a maintainer could add this binding and declare the corresponding option in both
schemas. This is an illustration, not a capability of the descriptive algorithm:

```json
{
  "iterations": {"source": "parameter", "parameter": "iterations"}
}
```

Its option schema could be `{"type":"integer","minimum":1,"maximum":10,
"default":3}`. The compiler records whether 3 came from the registry or another
value came from the research plan. A required option with no default prompts for
clarification. Unknown options, wrong types and out-of-range values fail
validation. There is no `eval`, Python import, shell command or script template
execution inside the binding system.

This adapter currently requires variables and one site selector. It does not
implement stratification, cross-task iterative orchestration, arbitrary code,
automatic model training, or all possible Vantage6 algorithm signatures.
Compatible additions can be file-only; a fundamentally different input or result
protocol requires a tested adapter. Do not label such algorithms executable merely
because their names are present in the registry.

Input validation uses the established
[jsonschema Draft202012Validator](https://python-jsonschema.readthedocs.io/en/stable/validate/).
Catalog schema version 1.0 requires inline schemas without `$ref`/`$dynamicRef` and does not retrieve
network schemas. Both outer input schemas must reject additional properties.

### Result adapters

A result adapter is the trusted code that checks a completed result against the
researcher's resolved plan and writes the local report. The catalog only names
it; `result_adapters.py` holds the registry. Two adapters are built in:

| Adapter | Check | Report |
| --- | --- | --- |
| `descriptive_statistics_v1` | Column-oriented tables that cover every requested variable | Parsed tables, optional chart and `result.html` |
| `raw_json` | None; any completed result is kept | None; the raw result stays in `execution.json` |

An installed package adds a result type through the `federated_agent.result_adapters`
entry-point group:

```toml
[project.entry-points."federated_agent.result_adapters"]
row_count_v1 = "my_package.results:RowCountResult"
```

The object (or class) provides `name`, `reports`, `check(raw, plan,
semantic_catalog)` and `write_report(raw, output, *, semantic_catalog, context,
chart, emit)`. `check` raises `ValueError`, `TypeError` or `KeyError` for a result
that does not answer the plan; the researcher then sees `result_invalid` and the
task stays recorded as submitted. A plugin can never replace a built-in adapter,
and a plugin that fails to load is skipped with a warning. Preparing a task whose
adapter is not installed returns `result_adapter_missing`, so nothing is submitted
that the agent could not check afterwards. `--describe` lists each algorithm's
adapter and whether it is installed. A new execution protocol (how a task is
submitted) is different: it needs a reviewed runtime adapter in the trusted
worker, not a plugin.

## Every request uses the file

The supported request boundary is `FederatedResearchAgent.respond(request,
planner)`. It performs these steps:

1. Read the entire configured file afresh and validate it. Missing, malformed,
   duplicate-key, inconsistent or unsupported-version files return `catalog_error`.
   There is no previous-copy fallback and no mtime-based cache.
2. Compute SHA-256 of the exact bytes and record path, format and catalog version.
3. Supply the public catalog knowledge to the selected LLM, including during
   retries. There is no longer a hard-coded "only descriptive statistics" prompt.
4. Use that SAME snapshot for ID selection, input construction, validation and
   task provenance. A file edit during an LLM call takes effect on the next turn,
   not halfway through the current task.
5. Return a source-linked prepared task, clarification, unsupported response or
   validation failure. Catalog questions return structured catalog records rather
   than model-authored claims of capability. `--list-algorithms` does this without
   an LLM call.
6. Save `response.json` and a byte-exact `catalog_snapshot.json`. The source hash
   can be checked against the archived snapshot. Task confirmation includes the
   catalog reference as well as the executable payload.
7. Before agent execution, reload the catalog, reject stale preparation, and
   recompile against the current contracts. Generated standalone scripts also
   check the live file hash before credentials and again immediately before task
   submission. A changed file requires regeneration and confirmation.

These checks ensure catalog freshness and contract consistency. They do NOT prove
that an LLM captured every sentence correctly or that an administrator's capability
description is true. Research-request evaluation, clarification and actual sandbox
tests remain necessary. File hashes are change detectors, not digital signatures
or a security gateway; protect catalog write access outside the LLM. A narrow
check-to-submit race still requires a future server-side enforcement mechanism if
immediate revocation must be guaranteed.

Application callers should use the file-backed agent, not a hand-constructed
in-memory `AlgorithmCatalog`, to obtain the freshness guarantees above.

## Updating an algorithm

1. Obtain the exact deployed image tag/digest, function signature, parameter
   meanings, result contract, data requirements and deployment scope from the
   maintainer. Check the source/image version together.
2. Add or update one record in `algorithms.json`, then update its catalog version
   and change date. Use `draft` while the contract is incomplete or unverified for
   the intended workflow. Keep `verification: declared` until sandbox evidence exists.
3. Validate a candidate file using `--catalog <candidate> --validate-catalog`, run
   relevant tests, then publish it with an atomic file replacement. An incomplete
   in-place edit will deliberately stop new requests until corrected.
4. The next request sees the change. Disabling/removing an entry stops new
   selection. Multiple enabled algorithms for one analysis trigger clarification
   instead of choosing the first record by ordering.

Run from `agent_mvp`:

```powershell
python -m pip install -e .
python -m federated_agent --validate-catalog --output output-catalog-validation
python -m federated_agent --list-algorithms --output output-catalog-list
python -m federated_agent --provider demo --request "Describe age at diagnosis in hospital 1" --output output-catalog-demo
python -m unittest discover -s tests -v
```

DeepSeek and local providers use the same `respond` boundary. Prompts contain the
research request AND public algorithm metadata. Never place patient data or
secrets in either. A remote model receives these public metadata as well as the
request; it does not receive endpoint credentials or raw hospital records.

Use separate output directories for separate runs. Old generated files are not
deleted when a newer request fails; `response.json` identifies the current
outcome. Scripts generated before this feature do not gain freshness checks
retroactively and should be regenerated. When moving a newly generated script to
another machine, `FEDERATED_AGENT_CATALOG` may point to the live registry copy;
its bytes must still match the recorded hash. Do not point it to a historical
snapshot and mistake that for live freshness.

## Verification and limitations

`tests/test_live_algorithm_catalog.py` covers file-only additions with a different
method/kwargs contract, fresh prompts, same-turn snapshots, retries, removal,
disablement, malformed files, provenance, parameter/default validation, site scope,
ambiguity, stale confirmation, script freshness and the offline CLI.

The added algorithm in that test file is explicitly a synthetic contract fixture,
not a deployed hospital algorithm. Fake providers verify orchestration, not LLM
understanding. The existing mock returns fixed descriptive-statistics values and
is not scientific evidence for any new algorithm. No new remote LLM or Vantage6
execution was performed for this change.

See `DEVELOPMENT_ROADMAP.md` for the three mandatory next deliverables.
