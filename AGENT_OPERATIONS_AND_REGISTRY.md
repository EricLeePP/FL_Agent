# Federated Agent: Startup, Connectivity, Metadata and Synchronization

Original proposal: 2026-09-07. Audience: researchers, Len/Joshi, and deployment maintainers.
Paths in this guide are relative to the project root unless stated otherwise.

**Implementation update, 2026-09-17:** the registry publisher, coherent snapshot
client, action-specific policy hooks and versioned template registry now exist.
Read [the current architecture guide](agent_mvp/docs/ARCHITECTURE_EXTENSIONS.md)
for commands, data contracts and verification. The proposal/status tables below
are historical, not the current implementation status. No Ubuntu registry hosting
service or production authorization system has been deployed automatically.

## Status at a Glance

| Topic | Current implementation | Proposed next step |
| --- | --- | --- |
| Short Windows startup | `agent.cmd` wraps the existing DeepSeek/Vantage6 launcher | No virtual-environment activation needed |
| Execution connection | Windows local SSH forward to the Ubuntu synthetic sandbox | Institution-managed remote access and a separately tested production executor |
| Algorithm knowledge | Windows-local, validated `algorithms.json`, refreshed per research request | Centrally published, versioned metadata releases |
| Hospital/data knowledge | Local deployment configuration and an expiring availability snapshot | Hospital-maintained metadata, centrally assembled and synchronized |
| Synchronization | No Ubuntu-to-Windows automatic synchronization | Publisher, read-only downloader and a unified snapshot loader |
| Policy | Replaceable allow-all checker; actual Vantage6 access controls remain | Authenticated authorization, enforcement and output-privacy review |

The metadata publishing paths, manifest fields and synchronization components
below are a design proposal, not an already deployed service. This update does
not modify Ubuntu, deploy a registry, change network settings or submit a task.

## 1. Start the CLI

Open PowerShell and enter the project folder:

```powershell
cd "C:\Users\p70077044\Desktop\Postdoc_project\Agent_fed"
.\agent.cmd
```

After entering that folder, the only startup command is `.\agent.cmd`.
It uses the isolated `integration/.venv-agents-sdk` environment and
`agent_mvp/start_agent.py`. The model/tool runtime is the OpenAI Agents SDK, with
DeepSeek as the model provider. Add `--agent-runtime json` for the JSON runtime,
which local OpenAI-compatible models use.
It forwards CLI options and exit codes, uses paths relative to its own location,
does not activate an environment, and does not change PowerShell execution policy.
It does not create an SSH tunnel or install dependencies automatically.

The default remains DeepSeek plus the REAL synthetic Vantage6 backend. Keep the
SSH tunnel described below open for execution. Task submission still requires
researcher confirmation and a separate Vantage6 login. Starting the agent alone
does not submit anything.

Useful commands:

```powershell
.\agent.cmd --help
.\agent.cmd --provider demo --backend mock --plain
.\agent.cmd --provider local --llm-endpoint http://127.0.0.1:8000/v1/chat/completions --model YOUR_DEPLOYED_MODEL
```

The second command is an offline demonstration with a limited rule-based planner,
not DeepSeek or a real hospital execution. The local-model example requires an
actual compatible service at the supplied address; it still selects the real
sandbox backend unless `--backend mock` is added.

The existing launcher discovers `~/Desktop/deepseek-api.txt` at runtime. Explicit
`--deepseek-api-key-file` overrides `DEEPSEEK_API_KEY_FILE`, which overrides that
desktop file; `DEEPSEEK_API_KEY` and hidden terminal input are subsequent fallbacks.
An invalid explicitly configured file fails instead of silently switching keys.
The key is not copied into this launcher, metadata or generated artifacts. Do not
enter credentials or patient records at the research prompt.

For example, ask:

```text
Describe age at diagnosis and survival status in hospital 1 and hospital 2 using synthetic_clean.
```

Review the proposed sites, dataset and analysis before choosing submission. Use
`/algorithms` for current catalog capabilities and `/exit` to close the CLI.
Results and session artifacts stay under `integration/agent-demo/` by default.

### One-time setup on another Windows installation

The `.cmd` file is a launcher, not a portable Python installation. If the project
environment is missing, it stops with an English setup message. With Python 3.10
installed, run from the project root:

```powershell
py -3.10 -m venv integration\.venv-vantage6
.\integration\.venv-vantage6\Scripts\python.exe -m pip install -e ".\agent_mvp[live]"
.\agent.cmd --help
```

Recreate the environment on another computer instead of assuming a copied virtual
environment will work. Configure that computer's credentials and network access
separately. A Linux machine uses the Python entry point, not the Windows `.cmd`.

## 2. How Windows Currently Connects to Ubuntu

The configured path is:

```text
Researcher -> Windows CLI -> Vantage6 Python client
                                   |
                         Windows 127.0.0.1:17601
                                   |
                      SSH to 192.168.1.191:22
                                   |
                         Ubuntu 127.0.0.1:7601
                                   |
                           Vantage6 server
                                   |
                     Synthetic hospital nodes 1-3
```

In a separate PowerShell, establish the tunnel:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:17601:127.0.0.1:7601 ericlee@192.168.1.191
```

Verify the SSH host fingerprint through a trusted channel on the first connection.
Enter the Ubuntu SSH password only at the SSH prompt. A connected `ssh -N` process
normally stays quiet; leave that terminal open. Do not disable host-key checking.
The `-L` option forwards the Windows listening port to a destination reached from
Ubuntu; `-N` requests no remote command. [OpenSSH manual](https://man.openbsd.org/ssh)

In another terminal, check the Vantage6 API before starting a real demonstration:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:17601/api/version -TimeoutSec 10
.\agent.cmd
```

The configured and previously verified server version is 4.15.0. This guide does
not assert that the tunnel is currently running. A timeout requires checking the
tunnel, route and server; it is not evidence that an algorithm is unsupported.

Important distinctions:

- `127.0.0.1:17601` belongs to Windows; `127.0.0.1:7601` belongs to Ubuntu.
- The Windows client does not connect directly to each hospital database.
- The three current hospitals are synthetic sites on one Ubuntu sandbox host,
  not three independently governed production hospitals.
- `ericlee` is the SSH account. `sandbox_researcher` is a different Vantage6
  account, whose password is requested only after task confirmation.
- HTTP between the two local tunnel endpoints is protected across the network
  by SSH. That is not the same as Vantage6 payload end-to-end encryption; the
  current synthetic collaboration is explicitly unencrypted.
- DeepSeek is a separate outbound model connection from Windows. Research text
  and public planning metadata can leave Windows for that service; Vantage6
  credentials and returned results are not sent to it by the current workflow.

The Vantage6 server coordinates tasks and results, while nodes run algorithm
containers against local data. The container registry holds executable images;
the algorithm store holds metadata. These are different responsibilities.
[Vantage6 architecture](https://docs.vantage6.ai/en/version-4.15.0/introduction/architecture.html)

## 3. Connecting Outside the Local Network

Being on the same LAN is not a Vantage6 requirement. The requirement is an
authorized network route to the service, valid credentials and a compatible
client. The current private address is not a public Internet endpoint. A hostname
alone does not create routing or open an institutional firewall.

### Recommended production arrangement

Have institutional IT operate a stable, authenticated HTTPS endpoint for the
Vantage6 server. It may be reachable only after connecting to the institution's
VPN, or exposed through an institution-approved external access service.

```text
Windows researcher agent -> VPN when required -> HTTPS service -> Vantage6 server
                                                                  |
                                                   Separate hospital nodes
                                                   with local data sources
```

Nodes need approved connectivity to the coordination service; researchers do not
need an SSH login to every hospital. The platform operator must also arrange any
algorithm-specific networking and container-registry access. Vantage6's optional
algorithm-to-algorithm VPN is distinct from a researcher's institutional access VPN.
[Vantage6 network actors](https://docs.vantage6.ai/en/version-4.15.0/introduction/architecture.html)

| Access option | How the researcher reaches the service | Appropriate use |
| --- | --- | --- |
| Institutional VPN plus private HTTPS | Sign into the approved VPN; connect to the internal service DNS name | A practical default for an institution-restricted research platform |
| Approved external HTTPS endpoint | Connect over HTTPS to an IT-managed entry point with platform authentication | Multi-institution access without individual Ubuntu shell accounts |
| SSH jump host plus forwarding | Connect through an authorized bastion to the private Ubuntu host | Remote sandbox development and administrator access |

For HTTPS, the operator needs DNS, a valid trusted TLS certificate, restricted
firewall rules and an appropriately configured reverse proxy. Vantage6 uses
Socket.IO as well as HTTP; reverse-proxy configuration must support that traffic.
The Vantage6 deployment examples explicitly do not configure TLS, so they are
not complete production security configurations.
[Vantage6 deployment](https://docs.vantage6.ai/en/version-4.15.0/server/deploy.html),
[NGINX HTTPS configuration](https://nginx.org/en/docs/http/configuring_https_servers.html)

For a remote sandbox, OpenSSH `ProxyJump`/`-J` can reach a private target through
an authorized, reachable jump host. IT must provide the actual bastion, target
hostname, accounts and forwarding permissions. It does not make the current
private address reachable without that route. [OpenSSH jump hosts](https://man.openbsd.org/ssh)

Do not solve remote access by exposing hospital databases, Docker control sockets
or all sandbox ports to the Internet. Do not disable TLS certificate checks.
Production identity and access control must protect the API itself, not just a
browser login page.

### Why the current executor is not production-ready

`agent_mvp/src/federated_agent/live_executor.py` intentionally accepts only a
confirmed synthetic environment, Vantage6/client 4.15.0, an HTTP loopback endpoint
and an unencrypted collaboration. Merely replacing `server_url` with an HTTPS
production hostname will be rejected. Do not remove those checks as a deployment
shortcut; implement and test a distinct production profile/execution path.

| Concern | Current sandbox | Production requirement |
| --- | --- | --- |
| Data ownership | Synthetic fixtures on one Ubuntu host | Independent hospital data stewardship and node administration |
| Access | Interactive SSH forwarding and sandbox login | Managed routing, identity lifecycle, authentication and authorization |
| Confidentiality | SSH transport; unencrypted synthetic collaboration | Verified TLS and an explicit payload-encryption/key-management design |
| Policy | Agent checker returns allow; backend controls still operate | Trusted authorization and hospital-side enforcement; OPA can support decisions |
| Results | Local structural validation and plots | Algorithm-specific disclosure controls, audit and scientific validation |
| Operations | Interactive process, bounded wait and local artifacts | Durable task recovery, monitoring, backups and controlled upgrades |
| Model use | Public/synthetic requests with remote DeepSeek | Institution-approved model hosting and rules for metadata/request disclosure |

Connectivity, metadata availability and permission are separate checks. A reachable
server is not an authorization grant, and an aggregate result is not automatically
privacy-safe. TLS, SSH and Vantage6 payload encryption solve different problems.

## 4. Why the Agent Uses Separate JSON Documents

The LLM interprets the research request; it must not invent deployment IDs,
variable meanings, algorithm versions or parameter names. Structured documents
give deterministic tools an explicit contract to validate and a source to record.

| Current file | Question answered | What it does not prove |
| --- | --- | --- |
| `agent_mvp/src/federated_agent/data/catalogs/algorithms.json` | Which analyses are registered, and how is their input constructed? | An image is currently runnable at a hospital |
| `agent_mvp/src/federated_agent/data/config/ubuntu_sandbox.json` | Which collaboration, sites and database labels should this client target? | The researcher has permission or a node is online |
| `agent_mvp/src/federated_agent/data/config/ubuntu_data_availability.json` | What does recent metadata say about variables at each site/dataset? | Patient values, cohort size or live authorization |
| `agent_mvp/src/federated_agent/data/catalogs/AYA_cancer_schema.jsonld` | What does each variable mean, and what semantic class identifies it? | Every hospital has that variable |

JSON is machine-parseable and can be validated, versioned, hashed and archived.
Markdown remains useful documentation but is not a competing executable source
of truth. The JSON-LD schema is the upstream STRONG AYA semantic artifact, not a
new schema invented by this project. It follows the colleague's instruction to
resolve variables through `schema.variables.<name>.class`.
[STRONG AYA semantic map](https://github.com/STRONGAYA/AYA-cancer-semantic-map)

### 4.1 Algorithm contract: `algorithms.json`

| Key or group | Why it exists |
| --- | --- |
| `schema_version` | Version of the document format; determines whether the loader understands it |
| `catalog_version`, `updated_at` | Revision of the catalog contents and maintenance date; not runtime-health evidence |
| `algorithm_id`, `display_name` | Stable machine identity separated from a readable, changeable label |
| `algorithm_version`, `image`, `vantage6_version`, `function` | Identify the intended implementation, runtime compatibility and entry point |
| `status` | `draft`, `enabled` or `disabled`; operational selection state, not user authorization |
| `analysis_types`, `request_keywords` | Help map intent to a registered analysis category; keyword matches do not prove suitability |
| `description`, `capabilities`, `limitations` | State what an algorithm does and must not be claimed to do |
| `constraints`, `supports_stratification` | Machine-readable limits, separate from descriptive prose |
| `accepted_arguments` | Exact keyword names accepted by the selected implementation |
| `source`, `metadata_source`, `contract_warnings` | Trace the implementation and upstream metadata; expose known discrepancies |
| `availability[]` | Scope a contract to a named deployment, participant IDs and database labels, with verification state and evidence |
| `execution.adapter` | Select a trusted compiler adapter; the registry cannot load arbitrary Python or shell code |
| `execution.parameters_schema` | Validate researcher-supplied options and explicitly declared defaults |
| `execution.input_bindings` | Map resolved concepts into exact algorithm argument names |
| `execution.kwargs_schema` | Validate the final compiled arguments, including required keys and forbidden extras |
| `execution.result_adapter` | Choose the correct output parser; unfamiliar results must not inherit an unrelated chart/report format |

For the current algorithm, the exact binding is:

```json
{
  "variables_to_describe": {"source": "variables_by_class"},
  "variables_to_stratify": {"source": "constant", "value": null},
  "organisation_ids": {"source": "organisation_ids"}
}
```

This means the compiler supplies validated semantic classes and resolved site IDs;
the model does not write the executable call itself. `organisation_ids` is the
British spelling required by the pinned algorithm implementation. It is not
interchangeable with the Vantage6 client argument `organizations` or upstream
metadata's `organization_ids`. The repository documents this mismatch explicitly.
[Pinned algorithm implementation](https://github.com/STRONGAYA/v6-descriptive-statistics/blob/v2.1.0/v6-descriptive-statistics/central.py),
[upstream store metadata](https://github.com/STRONGAYA/v6-descriptive-statistics/blob/v2.1.0/algorithm_store.json)

The schema rejects undeclared options with `additionalProperties: false`. Thus a
request for an unsupported filter cannot quietly become an unfiltered analysis.
The current `min_sites_for_numerical: 2` records the reproduced single-site defect,
not a general rule of federated statistics. An algorithm's registration is not
proof of scientific validity or permission. A version tag identifies the intended
release, but an immutable image digest is preferable for production verification.

Compatible new contracts can be added by editing JSON. New input protocols or
output types may require a tested adapter as well. The current adapter does not
automatically support arbitrary model training, iterative orchestration or code
execution. `availability` currently describes a set of sites and a set of labels;
it does not express arbitrary per-site combinations. Data declarations provide
the finer site/dataset check; more complex availability needs a schema extension.

### 4.2 Deployment configuration: `ubuntu_sandbox.json`

| Key | Why it exists |
| --- | --- |
| `name`, `registry_version` | Identify this deployment and its metadata revision; the name joins algorithm and data records |
| `sandbox`, `vantage6_version` | Mark the intended environment and runtime; verified separately by the executor |
| `server_url`, `server_port`, `server_api` | Describe the route used by this Windows client; currently loopback port 17601 and `/api` |
| `collaboration_id` | Select the Vantage6 collaboration, not an arbitrary collection of hospitals |
| `central_organisation_id` | Select the organization running the algorithm's central method |
| `database_label` | Declared default dataset label; not a silent replacement for an explicitly invalid requested label |
| `sites[].name`, `aliases` | Let researchers use readable names while tools resolve canonical identities |
| `sites[].organisation_id` | Exact Vantage6 organization identifier for the selected site |
| `sites[].database_labels` | Dataset labels configured for that site; not local filesystem paths or credentials |

An organization ID is not a node ID. The current central organization is 1, while
the research participants can be hospitals 1 and 2. The client submits the central
task with `organizations=[1]`; the algorithm receives participant IDs in
`kwargs.organisation_ids` and creates its subtasks. The server is a coordinator,
not the process that directly computes the central algorithm.
[Vantage6 task creation](https://docs.vantage6.ai/en/version-4.15.0/user/pyclient.html)

The file currently mixes shared site identities with a Windows-specific connection
route. Before synchronization, split shared deployment metadata from a local
connection profile. Do not overwrite the Windows tunnel port with Ubuntu's local
port or allow downloaded metadata to redirect credentials to an arbitrary server.

### 4.3 Data availability: `ubuntu_data_availability.json`

| Key | Why it exists |
| --- | --- |
| `schema_version`, `version` | Separate format compatibility from the inventory revision |
| `deployment` | Prevent accidentally using evidence from another Vantage6 environment |
| `observed_at`, `expires_at` | Record when the facts were checked and when they stop being adequate evidence |
| `sites[].organisation_id`, `database_label` | Identify a unique site/dataset pair; IDs are scoped to the named deployment |
| `complete_inventory` | Decide whether an omitted variable means absent or simply undocumented |
| `variables` | Map stable semantic class IDs to `available`, `unavailable` or `unknown` |

Example, taken from the current synthetic inventory:

```json
{
  "organisation_id": 1,
  "database_label": "synthetic_clean",
  "complete_inventory": true,
  "variables": {
    "ncit:C156420": "available",
    "ncit:C25717": "available"
  }
}
```

This declares age at initial diagnosis and survival status, not patient values.
If the inventory is complete, an omitted variable is treated as unavailable. If
it is incomplete, omission means unknown. Missing site/dataset records and expired
evidence also mean unknown. These distinctions allow useful researcher feedback
without pretending that the metadata are complete or current.

The current snapshot has 12 site/dataset entries and expires at
`2026-09-14T21:23:30Z`. Its two-variable completeness claim applies only to the
synthetic fixtures. Copying or republishing it must not reset its observation
time. A column marked available does not guarantee non-missing data, a sufficiently
large cohort or correct coding for a particular study.

### 4.4 Semantic definitions: `AYA_cancer_schema.jsonld`

The current loader reads `schema.variables`, using each variable's dictionary key
as its canonical name, `class` as its semantic identifier, and `dataType` plus
`@type` to determine whether this algorithm can treat it as numerical/categorical.
It records `@id` and `version` as source provenance. The JSON-LD `@context` defines
vocabularies, but this MVP loader is not a general-purpose RDF reasoner.

For example, age at initial diagnosis resolves to `ncit:C156420`. The compiler can
then produce `{"ncit:C156420": {"datatype": "numerical"}}`. That mapping explains
the meaning of the requested variable; the separate availability file establishes
whether a particular synthetic site declares it present.

Canonical names are turned into readable aliases; a small extra alias list still
lives in `semantic_catalog.py`. Fuzzy matches require clarification. The semantic
catalog is currently loaded at agent creation, unlike the per-request algorithm
reload. A future synchronized snapshot loader must refresh it and its dependent
result parser together. Simply copying the JSON-LD file during a running session
does not currently hot-reload all semantic knowledge.

### 4.5 Current grounding guarantees and limits

For research/capability requests, the agent reads and validates the local algorithm
file, archives its exact bytes and hash, and uses the same snapshot for planning
and compilation. Deployment and availability are also reread at their check
boundaries. Pure greetings do not require a catalog. Before managed execution,
catalog, deployment, data and policy are checked again and task confirmation must
match. This is not yet a single atomic snapshot across all four documents.

`agent_mvp/src/federated_agent/data/catalogs/algorithm_store.json` is supporting upstream metadata, not a
second executable source of truth. The existing `scripts/sync_catalogs.py`
downloads public upstream inputs; it does not publish Ubuntu state, synchronize
`algorithms.json` or prove local runnability. Vantage6's own algorithm store can
supply algorithm metadata, while its APIs can supply authorized collaboration and
organization information. Our curated input bindings, cross-file validation and
hospital data declarations still need an integration layer.
[Vantage6 algorithm store](https://docs.vantage6.ai/en/version-4.15.0/algorithm_store/introduction.html)

All current runtime metadata copies are on Windows. A local-file reload does not
detect an Ubuntu-side change. Catalog contents must be treated as data, never as
instructions to execute downloaded scripts or override the agent's rules. Only
appropriately public metadata should be included in remote-model prompts.

## 5. Proposed Ubuntu-to-Windows Synchronization

### 5.1 Ownership and transport

Use one authoritative publisher for a deployment, initially on Ubuntu. Algorithm
maintainers provide versioned contracts; hospital stewards provide approved site
and data declarations; the publisher validates and assembles a consistent release.
Windows receives read-only copies. Multiple hospitals can later publish into a
shared metadata service without moving their patient data.

For the two-computer prototype, use an authenticated SFTP read channel to a
dedicated metadata directory. Prefer a read-only registry account instead of a
broad administrator account. The existing TCP tunnel on port 17601 forwards only
the Vantage6 API: it is NOT an SFTP service or an automatic registry endpoint.
SFTP would use a separate authenticated SSH connection to a reachable SSH host,
possibly through the institution's VPN or jump host. Protect host-key verification;
use an approved SSH key/agent or credential flow, never credentials in the catalog.

For production, replace this transport with a read-only authenticated HTTPS
metadata API if appropriate. The release format and validation rules can remain
the same. Partition visible metadata by authorized user/collaboration, and bind
caches to that identity/scope. Do not download a global sensitive inventory and
rely on the LLM to hide unauthorized entries. SSO/API authentication integration
would require explicit engineering; it is not supplied by a JSON file.

### 5.2 Release structure

Proposed publisher directory, not an existing Ubuntu path:

```text
registry/
  latest.json
  releases/
    release-001/
      manifest.json
      algorithms.json
      deployment.json
      data_availability.json
      AYA_cancer_schema.jsonld
    release-002/
      ...
```

| Proposed field | Purpose |
| --- | --- |
| `latest.schema_version` | Pointer format version |
| `latest.release_id`, `latest.sequence` | Current immutable release ID and monotonic sequence for update/rollback checks |
| `latest.manifest_sha256` | Digest of the selected manifest, verified after download |
| `manifest.schema_version`, `release_id`, `sequence`, `deployment` | Identify the format, exact release and environment consistently |
| `manifest.published_at`, `expires_at` | Publication/freshness limits; these do not override individual evidence expiry |
| `manifest.files[]` | Allowlisted relative filename, byte size and SHA-256 for every required component |

These are new proposed fields, not accepted by today's metadata loaders. Hashes
detect mismatches and bind components together; they are not digital signatures.
The authenticated SSH/HTTPS channel and protected publisher establish source
trust. A monotonic sequence helps reject an older pointer; it is not a substitute
for trusted time, authentication or a defined operator rollback procedure.

### 5.3 Ubuntu publish operation

1. Prepare candidate metadata after actually deploying/configuring the relevant
   image, node and dataset. Registration alone is not deployment or permission.
2. Validate supported schemas, unique IDs, compatible adapters, exact input/output
   contracts and references across the semantic, algorithm and site documents.
3. Check relevant synthetic test evidence. Preserve `declared` or unknown states
   when real evidence is unavailable; do not automatically claim verified status.
4. Create a complete new immutable release, computing each file's size and digest.
5. Write its manifest and atomically replace `latest.json` only after the release
   is complete. A failed publication leaves the previous pointer untouched.

The proposed `publish_registry.py` automates validation and publication, not
clinical approval. Keep old releases for audit. A rollback should be an explicit
new publication referencing reviewed content, rather than quietly moving the
pointer backwards or reviving a withdrawn algorithm.

### 5.4 Windows update operation

1. At startup and before each catalog-dependent request, read the trusted pointer.
   A greeting needs no update. Pin its release ID for the entire download.
2. If the selected release is already cached, check its integrity and evidence
   expiry. Otherwise download its manifest and files into a staging directory.
3. Enforce connection timeouts, byte-size limits, allowlisted filenames, no path
   traversal or arbitrary URL redirects, hashes, supported formats and deployment
   identity. Validate cross-file references and future-dated/expired evidence.
4. Build an immutable `RegistrySnapshot` containing all four parsed documents.
   Combine shared deployment identities with a separately trusted local connection
   profile. Do not allow remote metadata to override a credential destination.
5. Atomically activate the complete snapshot. Keep the previous cache if staging
   fails; do not mix old and new components or silently fall back for execution.
6. Give planning and deterministic tools the same snapshot, archive its release
   ID/hashes with the task, and record provenance for automatically filled values.
7. Immediately before submission, check the trusted pointer and evidence again.
   Initially require re-preparation/reconfirmation for any release change. Later
   dependency-aware comparison can avoid unnecessary prompts for unrelated edits.

Changing a variable definition must refresh the semantic resolver and result
parser, not only the algorithm catalog. For longer conversations, retain the
researcher's intent while re-resolving references against the newly selected
snapshot; do not retain stale organization IDs or algorithm assumptions.

### 5.5 Failures and researcher feedback

| Situation | Proposed behavior |
| --- | --- |
| Registry unreachable | Explain connectivity failure; allow explicitly labeled cached exploration only, and pause new real submissions |
| Wrong hash, unsupported schema or partial release | Reject the candidate; preserve the old cache without labeling it current |
| Data evidence expired | Mark availability unknown; request refreshed steward evidence, not a manual date extension |
| Algorithm removed or disabled | Explain that the planned analysis is no longer available; suggest registered alternatives without substituting automatically |
| Release changed after confirmation | Rebuild and ask for fresh task confirmation before submitting |
| No initial cache and registry unreachable | Explain that current capabilities cannot be verified; do not invent a catalog |

Record `source`, `release_id`, `checked_at`, `freshness` and a concrete next action
in diagnostics so researchers can distinguish unsupported analyses from network
or metadata failures. Metadata visibility itself needs authorization in production,
even while the prototype's execution-policy extension returns allow.

Client rechecks still have a check-to-submit race. Immediate revocation guarantees
require server/node-side enforcement at acceptance and/or execution time. A task
already running is not automatically canceled by publishing a new registry version.
Publication expiry must never extend the underlying data-evidence validity.

### 5.6 Acceptance tests for the synchronization work

- A compatible new algorithm becomes visible in a running agent after publication,
  without retraining the LLM, while an unknown adapter remains non-executable.
- An interrupted or corrupt download cannot activate a mixed/partial release.
- A request uses one pinned snapshot even if publication occurs during an LLM call.
- A disabled algorithm or changed release invalidates an older task confirmation.
- Expired, absent and incomplete data evidence produce different, accurate feedback.
- Windows retains its own endpoint and credentials after every update.
- Semantic changes refresh dependent resolution/reporting objects consistently.
- Cache identity isolation, offline behavior and explicit rollback are tested.
- No patient records, secrets or unapproved metadata enter the release or model prompt.

## 6. Implementation Boundary

This update implements the short Windows launcher and documents the design. The
proposed publisher, SFTP/HTTPS sync client, whole-registry snapshot, production
executor and production policy enforcement are still future work. No new real
Vantage6 or DeepSeek execution is claimed by launcher tests.

Launcher tests cover working-directory independence, quoted output paths,
offline overrides, missing runtime and exit codes. Further details are in
`agent_mvp/docs/LIVE_CLI_DEMO.md`, `agent_mvp/docs/ALGORITHM_REGISTRY.md` and
`agent_mvp/docs/RESEARCHER_FEEDBACK.md`.
