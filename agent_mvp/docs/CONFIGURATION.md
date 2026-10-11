# Configuration

Fed Agent runs without a configuration file: the bundled defaults describe the
offline demo deployment. A JSON configuration file lets operators point the agent
at their own metadata, model and backend settings without long command lines.

## Where values come from

For every setting, the first available source wins:

1. a command-line flag;
2. an environment variable (`LLM_PROVIDER`, `LOCAL_LLM_ENDPOINT`, `DEEPSEEK_ENDPOINT`);
3. the configuration file given by `--config` or the `FED_AGENT_CONFIG` variable;
4. the bundled default.

With `--backend vantage6`, missing deployment, data-availability and policy
settings default to the bundled synthetic Ubuntu sandbox files, as before.
`--registry-source` owns all metadata and cannot be combined with individual
metadata paths.

## File format

`examples/fed-agent.example.json` shows a complete file. Relative paths are
resolved against the directory that contains the configuration file.

| Field | Meaning |
| --- | --- |
| `output` | Output directory for responses and saved studies. |
| `metadata.algorithm_catalog` | Algorithm registry (`algorithms.json`). |
| `metadata.semantic_schema` | STRONG AYA semantic schema (JSON-LD). |
| `metadata.deployment` | Deployment registry: server, collaboration, sites and dataset labels. |
| `metadata.data_availability` | Operator-maintained, expiring data availability snapshot. |
| `metadata.execution_templates` | Execution-template descriptors. |
| `metadata.registry_source` | Signed registry source; replaces the individual metadata files. |
| `policy.config` | Policy configuration (`allow_all` or `opa`). |
| `llm.provider` | `demo`, `deepseek` or `local`. |
| `llm.runtime` | `json` (JSON mode over plain HTTP, the default) or `sdk` (Agents SDK). |
| `llm.model`, `llm.endpoint` | Model name and chat-completions endpoint. |
| `llm.api_key_env`, `llm.api_key_file` | Where the key is read from. The key itself never belongs in the file. |
| `llm.stream` | Stream model output and the optional explanation. |
| `llm.tool_lookups`, `llm.max_turns` | Let the model call read-only lookups (variable and hospital names, catalog) for up to `max_turns` turns before it proposes a plan. Off by default; needs the `sdk` runtime. See [MCP and lookups](MCP_SERVER.md). |
| `backend.kind` | `mock` or `vantage6`. |
| `backend.confirm_synthetic_unencrypted_sandbox` | Required for the synthetic sandbox backend. |
| `backend.timeout_seconds` | Wait limit for a real task. |
| `logging.verbosity` | 0 warnings, 1 information, 2 debug (stderr). |
| `logging.file` | Full debug log, including incident tracebacks. |

Unknown fields are rejected, so a typo cannot silently fall back to a default.

## Bundled metadata

The default metadata ships inside the package under
`src/federated_agent/data/` (`catalogs/` and `config/`). An editable checkout can
edit those files directly. An installed package should not be edited in place;
copy the files and point the configuration file at the copies.

## Checking the effective configuration

```bash
python -m federated_agent --describe
python -m federated_agent --config fed-agent.json --describe
```

`--describe` prints the effective paths, planner, backend, policy mode and
algorithms, plus health issues such as an expired data snapshot. It never prints
keys; it reports only which environment variable or key file is configured.

## Logs and incident IDs

Warnings and errors go to stderr. `-v` adds information messages and `-vv`
debug messages; `--log-file` (or `logging.file`) also writes a full debug log.
Credentials are removed from every log line, including tracebacks.

An unexpected failure is logged once with its traceback and an incident ID. The
researcher sees the same ID in the response, for example
`Incident ID: 3f9c2a1b7d40 (the full error is in the agent log)`, so an
administrator can find the matching log entry.
