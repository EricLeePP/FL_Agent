# Policy, Registry and Execution-Template Architecture

Updated: 2026-09-17. Paths below are relative to the repository root.

## 1. Requirements in Software-Engineering Terms

| User requirement | Architectural requirement |
| --- | --- |
| Allow requests now; introduce policy constraints later | Pluggable Policy Decision Point (PDP), non-optional Policy Enforcement Points (PEPs), an explicit development allow-all provider, and fail-closed handling of policy errors |
| Hospitals update algorithms/data using simple files; researchers see updates | An authoritative, versioned metadata registry; schema-validated publication; pull-based synchronization; atomic snapshot activation; freshness and provenance guarantees |
| Execution workflows can evolve without rewriting the agent | Separation of task parameters from execution code; versioned execution-template descriptors; a trusted adapter registry; compatibility checks and confirmation invalidation |

These are extension contracts, not instructions added to an LLM prompt. The LLM
proposes a research plan; deterministic code owns validation, policy enforcement,
version selection and execution. Registry metadata is a control plane, not a way
to move patient records between hospitals and researchers.

## 2. Audit Findings and Changes

| Before this change | Implemented change |
| --- | --- |
| A replaceable policy checker existed, but had no action-specific contract | Decisions now distinguish `prepare`, `submit`, `read_result` and `release_result` |
| Adding a CLI check alone could leave the SDK worker outside the policy boundary | The service binds trusted policy configuration into the worker; the worker checks before submission, result retrieval and result persistence |
| Local files were refreshed separately; there was no remote publication protocol | Five documents are validated and activated as one immutable, content-addressed release |
| Semantic definitions could remain loaded from startup | Registry mode refreshes the semantic map and the LLM's public planning context with the other documents |
| Execution code and the exported script were embedded separately without a template registry | Export assets, installed runtime adapters and versioned descriptors now have explicit bindings and hashes |
| Confirmation bound the task/catalog, but not a complete metadata release or template | Confirmation and run manifests also bind the registry release and installed execution-template reference |

The default startup still uses existing local metadata and allow-all policy.
Remote synchronization is opt-in through `--registry-source`. No registry daemon
has been installed on Ubuntu by this change.

## 3. Architecture and Ownership

```text
Hospital / federation operators
    edit approved JSON metadata
        -> validate + publish a versioned release
        -> authenticated HTTPS current.json
                  |
Researcher agent  | fetch at each research/catalog request and before submission
    RegistryClient -> validate -> immutable cache -> atomic current pointer
                  |
Researcher -> LLM plan -> deterministic compile -> prepare policy -> exact review
                  |
             confirmed execution
                  |
    trusted SDK worker -> submit policy -> Vantage6 task.create
                       -> read_result policy -> retrieve
                       -> release_result policy -> save / report
                  |
           one configured PDP
           allow_all now / OPA later
```

The worker does not ask the LLM whether a policy check is necessary. Every normal
execution path crosses these checks. Parsing a sentence or rendering a menu does
not need a separate privacy decision. Hospital/server controls remain essential:
a researcher who owns their computer could modify this Python client. An agent-side
PEP is not an unbypassable hospital security perimeter. Production must enforce
the applicable policy at institution-controlled server/node boundaries as well.

## 4. Policy Contract

Operator-owned `agent_mvp/src/federated_agent/data/config/policy.json` currently contains:

```json
{
  "schema_version": "1.0",
  "mode": "allow_all",
  "version": "development-allow-all-v1"
}
```

This permits policy decisions only. Invalid algorithms, unavailable data,
unsupported requirements, expired evidence and mismatching confirmations remain
blocked. Vantage6's own authentication and permissions are never bypassed.

`policy.opa.example.json` shows the alternative OPA provider. It sends a JSON
`input` containing `schema_version`, `action`, `task_id`, `subject` and `task`.
The expected response contract is:

```json
{
  "result": {
    "allowed": false,
    "reasons": ["example_reason_code"],
    "obligations": [],
    "verification": "verified",
    "policy_version": "hospital-policy-1",
    "decision_id": "example-decision-id"
  }
}
```

Missing/invalid responses, connection errors and timeouts deny progress; there is
no fallback from failed OPA to allow-all. Strings such as `"true"` are not valid
booleans. Any non-empty `obligations` list is rejected until an explicit handler
exists. Listing `apply_dp` is not equivalent to implementing differential privacy.

The service's `prepare` input is a pre-authentication feasibility check with
`subject.authenticated=false`. Authoritative runtime checks for a file-configured
policy happen inside the SDK worker after authentication, using the available
Vantage6 subject ID and organization ID. These are never taken from LLM text.
Production still needs a reviewed identity/role/purpose mapping and actual Rego
policies; a JSON decision contract does not supply those policies automatically.

A custom policy provider states what it can do by declaring `PolicyCapabilities`
(`capabilities.py`); the agent never branches on the provider's class.
`worker_config` names the policy file from which the trusted worker repeats the
same decisions. `worker_decides_runtime` states that runtime decisions need the
authenticated identity and therefore happen only inside the worker. A provider
without a worker configuration can still prepare tasks and run with the mock
backend. The agent refuses to start the trusted worker with such a provider.

Policy configuration is reread at each PEP, including recovery. Recovery only
checks read/release, never submits a new task. `backend/policy_decisions.json`
records decisions, safe error flags, task IDs and timestamps, preserving earlier
entries on recovery. It does not store credentials or raw result content.

The policy URL/configuration is operator-owned, separate from remotely published
metadata. `token_env` refers to a local environment variable; its value is not
embedded in the release or task. Redirects are disabled. OPA sees the structured
task specification, not patient records or raw results. Future content-sensitive
release checks need an additional trusted result-inspection interface.

## 5. Registry Data Model

Publish one JSON envelope. Operators can maintain the five source files separately;
`registry_admin` assembles them into a single coherent release:

```json
{
  "release": {
    "schema_version": "1.0",
    "release_id": "hospital-release-20260917-1",
    "sequence": 1,
    "deployment": "ubuntu-synthetic-sandbox",
    "created_at": "2026-09-17T08:00:00Z",
    "expires_at": "2026-09-18T08:00:00Z",
    "documents": {
      "algorithms": {},
      "deployment": {},
      "availability": {},
      "semantics": {},
      "templates": {}
    }
  },
  "signature": "BASE64_ED25519_SIGNATURE"
}
```

The empty objects above are structural placeholders, not a valid release.

| Key / role | Purpose |
| --- | --- |
| `schema_version` | Publication protocol compatibility, distinct from algorithm versions |
| `release_id` | Human-readable release identity for audit and support |
| `sequence` | Strictly increasing publication order; clients reject rollbacks and same-sequence mutations |
| `deployment` | Prevents mixing metadata from different federations |
| `created_at`, `expires_at` | Reject future-dated or expired releases; release validity cannot outlive its inventory |
| `algorithms` | Algorithm IDs, images/versions, functions, JSON Schemas, input bindings, capabilities, limitations and site scopes |
| `deployment` document | Hospital names/aliases, organization IDs and database labels |
| `availability` | Per-organization, per-dataset, per-semantic-class availability, completeness and observation/expiry timestamps |
| `semantics` | STRONG AYA class identifiers, data types and variable names |
| `templates` | Supported template versions and installed adapter/asset identifiers |
| `signature` | Publisher authenticity and integrity, verified against a locally pinned Ed25519 public key |

Stable IDs are used for joins. Human labels can change without becoming identity
keys. Dataset existence and variable semantics are separate facts: being defined
in the semantic map does not prove that a hospital has that variable.

The publisher validates cross-file references. For example, an availability row
cannot reference an unknown hospital, database label or semantic class. A declared
algorithm must reference supported installed templates. Publishing a contract
does not install its Docker image, register a database on a node, grant access,
or prove a scientifically correct result.

An institution/federation publisher is the authoritative writer. Nodes may supply
observations, but this implementation does not automatically infer an algorithm's
argument contract from a newly installed Docker image. An operator must register
the validated contract and scope. The agent never edits the authoritative source.

## 6. Publication and Synchronization Workflow

For local development, from the repository root with the project environment:

```powershell
.\integration\.venv-vantage6\Scripts\python.exe -m federated_agent.registry_admin --config agent_mvp/src/federated_agent/data/config/registry_publish.json --output integration/registry-release.json --release-id sandbox-1 --sequence 1 --unsigned-local
.\agent.cmd --registry-source agent_mvp/src/federated_agent/data/config/registry_source.local.example.json
```

The inventory must still be fresh. Refresh it from real node evidence when it
expires; do not merely extend dates. Every subsequent publication must increase
`--sequence`; the publisher refuses to overwrite an equal/newer release.

For Ubuntu or a federation-operated metadata service:

1. Install the same reviewed agent package/checkout, including the `registry`
   extra (`python -m pip install -e './agent_mvp[registry]'` from its parent).
2. Maintain the source JSON files and a publisher mapping modeled on
   `agent_mvp/src/federated_agent/data/config/registry_publish.json`. Paths are relative to that mapping.
3. Publish using `python -m federated_agent.registry_admin --config ... --output
   ... --release-id ... --sequence ... --signing-key /private/publisher.pem`.
4. Serve only the published `current.json` through an institution-managed HTTPS
   endpoint with appropriate access controls. Never serve the project root,
   credential files, private key, or patient data. The hosting service itself is
   infrastructure to deploy separately, not included as a production web server.
5. On Windows, configure `registry_source.production.example.json` with that URL,
   the pinned base64 raw Ed25519 public key, deployment and local connection route.
   Keep the publisher's private key only on the publishing side.
6. Start `agent.cmd --registry-source <operator-config.json>`. No manual file copy
   or agent restart is required for subsequent compatible metadata releases.

Unsigned JSON is accepted only with the explicit development flag and a local
file/loopback source. Non-local HTTP is rejected; non-local HTTPS releases require
a pinned signing key. The current real executor STILL only accepts the synthetic
loopback 4.15 sandbox. The production registry example does not enable a production
Vantage6 executor, TLS deployment, SSO, VPN or output privacy controls.

At each research/catalog request, the agent fetches the release, checks its size,
JSON/schema, signature, deployment, timestamps, sequence and cross-file references,
then stages the entire snapshot. Only after full validation does it atomically
replace `current.json` in the cache. A local lock serializes activation. The five
files are never updated individually in the active snapshot.

There is another refresh before executing a confirmed task. Any changed release,
task contract or selected template invalidates the old confirmation. The researcher
must review the newly prepared task. Failed refreshes do not silently use stale
metadata for new work. An already verified cached snapshot can support read-only
recovery of an existing recorded task, even offline from the registry or after
expiry; the execution backend and current policy still must be reachable.

This is pull-on-use synchronization, not background polling or push notification.
Greetings do not query the registry because they assert no federation capability.
Switching a registry URL/key/local route requires a separate cache, so its authority
history cannot be confused with a different source. Historical signatures are not
a full key-revocation system; production key rotation and revocation need an
institution-managed procedure.

The researcher-side connection route is locally pinned. The publisher cannot
redirect Vantage6 credentials to a different host by changing the deployment
document. Registry credentials, execution credentials and signing keys are separate.

## 7. Execution-Template Updates

`agent_mvp/src/federated_agent/data/catalogs/execution_templates.json` describes the installed workflow:

```json
{
  "schema_version": "1.0",
  "defaults": {"joshi_vantage6_userclient": "1.0.0"},
  "templates": [{
    "id": "joshi_vantage6_userclient",
    "version": "1.0.0",
    "runtime_adapter": "vantage6_userclient_v1",
    "export_asset": "joshi_vantage6_userclient.py.tpl",
    "vantage6_version": "4.15.0",
    "poll_interval_seconds": 3
  }]
}
```

Algorithms can select `execution.template_id` and optionally pin
`execution.template_version`. Omitting the version follows the registry's default;
the prepared task always resolves and records an exact version and code/asset hashes.

| Change | Required work |
| --- | --- |
| New algorithm with an already supported input/result contract | Update `algorithms.json`, its verified site scope and any relevant inventory; publish a release |
| New dataset / hospital / variable | Update the appropriate deployment, inventory and semantic documents; publish together |
| Select another already installed template or adjust supported polling configuration | Add a new descriptor version, update its default or algorithm binding, publish |
| Change exported Python formatting/workflow | Update the installed `.py.tpl` asset, bump the descriptor version and run regression tests |
| Change SDK execution behavior or support a new Vantage6 version | Implement/test a trusted runtime adapter and necessary worker changes; distribute a reviewed software release before publishing metadata that selects it |

`runtime_adapters.py` provides configured hooks for encryption setup, validated
payload construction and result retrieval. The trusted worker (`runs/worker.py`)
retains non-optional policy boundaries, bounded execution, task recording and
recovery; `runs/submission.py` holds its single submission call. A larger
workflow/API change may require extending this interface; merely renaming a
template in JSON cannot create an implementation.

The renderer uses installed assets and literal substitution of validated task
data, never `eval`, remote imports or LLM-generated Python. Both the export asset
and installed worker/adapter code hashes are bound into confirmation. Unknown
adapters, path traversal, missing assets and changed code fail closed.

Generated scripts remain review artifacts showing Joshi's fixed submission shape.
New bound tasks must execute through the configured agent so a standalone export
cannot bypass registry/data/policy checks. Already-submitted tasks retain their
recorded bindings for recovery. Recovery only reads the recorded remote task, so
it keeps working after a worker upgrade; the worker logs that it used the
installed adapter. A prepared task that was never submitted must be prepared
again once the worker code it was bound to has changed. Deleting a runtime adapter
that recorded tasks name is still a migration decision, not routine cleanup.

## 8. Verification and Remaining Work

`tests/test_architecture_extensions.py` covers coherent refresh, live HTTP updates,
signature tampering, rollback, expired/offline caches, concurrent activation,
cross-file integrity, secret-safe transport, runtime policy rejection, unsupported
obligations, result release, template updates and recovery audit preservation.

Still not implemented as production infrastructure: an Ubuntu-hosted registry
service and institutional publisher ownership, automatic node-to-contract
discovery, production identity/OPA rules, server/node-side policy enforcement,
content-sensitive disclosure control, cryptographic code distribution, and a
non-sandbox production executor. These are explicit extension/deployment tasks,
not capabilities implied by the current allow-all setting.
