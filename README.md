# Fed Agent

**Version 2 of the Vantage6 Agent Workflow.** A researcher-facing CLI that turns
natural-language requests into catalog-grounded, reviewable Vantage6 tasks. This is
a research prototype for synthetic/public data, not a production clinical system or
an unrestricted coding agent.

## What It Does

- Parses research intent with an LLM: DeepSeek, or any OpenAI-compatible local
  model such as one served by Ollama. Two runtimes share the same prompts and
  validators: the OpenAI Agents SDK runtime and the JSON runtime (JSON mode over
  plain HTTP).
- Resolves variables, hospitals, datasets and algorithm parameters against local
  versioned metadata, rather than letting an LLM invent backend IDs.
- Asks for missing information, requires explicit dataset selection, and explains
  unsupported requests without silently changing the study.
- Preserves Joshi's fixed researcher-side workflow: authenticate, submit, wait,
  retrieve and save. Generates a reviewable Python script and task artifacts.
- Executes confirmed tasks against a Vantage6 4.15 synthetic sandbox and produces
  local reports. Supports saved sessions, task recovery and report rebuilding
  without repeating a computation.
- Offers a read-only MCP server and read-only lookups for model-driven clients.
- Provides policy extension points and signed registry publication/sync tools. The
  supplied policy is **allow-all development policy**, not authorization.
- Ships an evaluation suite for choosing a local LLM as the planner, with a
  synthetic test set, deterministic scoring and adversarial safeguard tests.

```text
Researcher -> CLI / saved study -> LLM intent proposal
           -> catalog resolution + deterministic validation
           -> policy boundary + researcher confirmation
           -> fixed Vantage6 worker -> result validation -> local report
```

The implemented algorithm contract is STRONG AYA descriptive statistics 2.1.0. Fed
Agent does not implement arbitrary algorithms, cohort filters or survival modelling.
Numerical analysis currently requires at least two sites because of a reproduced
upstream single-site limitation.

## Quick Start: Windows

Use Python 3.10. From this repository's root in PowerShell:

```powershell
py -3.10 -m venv integration\.venv-agents-sdk
.\integration\.venv-agents-sdk\Scripts\python.exe -m pip install -e ".\agent_mvp[sdk,live,registry]" -c .\agent_mvp\requirements-windows-py310.txt
.\agent.cmd --provider demo --backend mock --plain
```

The command above is offline: no API key, hospital connection or model call. Try:

```text
Describe age at diagnosis and survival status in hospital 1 and hospital 2 using default.
```

The demo provider is deterministic, not a test of LLM understanding. Inspect the
prepared task before confirming mock execution. `/algorithms`, `/plan`, `/history`,
`/sessions`, `/new` and `/exit` are available in the CLI.

### DeepSeek With Mock Execution

```powershell
.\agent.cmd --backend mock
```

Enter the API key only at the hidden credential prompt. For unattended local
configuration, `DEEPSEEK_API_KEY_FILE` can point to an external one-line UTF-8 key
file. Do not put keys or patient information into the researcher chat. This mode
uses a remote model and may incur charges, but does not submit to Vantage6.

### A Local Model

```powershell
.\agent.cmd --provider local --llm-endpoint http://localhost:11434/v1/chat/completions --model gemma4:26b --agent-runtime json --backend mock
```

Any OpenAI-compatible endpoint works. Which local models are good enough, and
which safeguards they need, is covered in the
[evaluation results](agent_mvp/evaluation/RESULTS_2026-10.md).

### Real Synthetic Sandbox

`.\agent.cmd` defaults to DeepSeek plus **real Vantage6 execution**. It does not
start a server or SSH tunnel. Before using it:

1. Supply an operator-approved endpoint and site/database IDs in the deployment
   configuration, or select your own file with `--deployment-config`.
2. Establish approved connectivity (for example, an SSH tunnel or managed HTTPS).
3. Reverify data availability and use a current `--data-availability-config`.
4. Review the task and confirm it before entering Vantage6 credentials.

The included `ubuntu_sandbox.json` uses loopback port 17601 and synthetic IDs. Its
data snapshot **expired on 2026-09-18**. It is retained for reproducibility, not as
evidence that any remote hospital is currently available. Never extend timestamps
without rechecking the nodes. Use only synthetic/public data with the unencrypted
sandbox adapter.

## Other Platforms

The `.cmd` launcher is Windows-only. The default demo and sandbox metadata are
bundled inside the package, so both an editable checkout and an installed wheel work:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e './agent_mvp[sdk,live,registry]' -c agent_mvp/requirements-windows-py310.txt
.venv/bin/python agent_mvp/start_agent.py --provider demo --backend mock --plain
```

The pinned constraints install on Linux as well; the test suite passes there. Model
hosting and a real Linux deployment remain separate checks.

## Configuration

For longer setups, put the settings in a JSON file and run
`python -m federated_agent --config fed-agent.json`; `--describe` prints the
effective configuration and health without credentials. See
[configuration](agent_mvp/docs/CONFIGURATION.md) and
`agent_mvp/examples/fed-agent.example.json`.

## Tests

```powershell
.\integration\.venv-agents-sdk\Scripts\python.exe -m unittest discover -s .\agent_mvp\tests
.\integration\.venv-agents-sdk\Scripts\python.exe .\agent_mvp\scripts\check_release.py
```

The suite has 423 tests; a few skip without the optional extras. The release
checker inspects Git's staged/tracked source snapshot, not ignored local files.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `agent_mvp/src/federated_agent/` | Agent, validated task compiler, worker, templates and policy hooks |
| `agent_mvp/src/federated_agent/data/catalogs/` | Algorithms, semantic definitions and template descriptors |
| `agent_mvp/src/federated_agent/data/config/` | Credential-free demo/sandbox configuration and registry examples |
| `agent_mvp/tests/`, `agent_mvp/examples/` | Regression tests and authored synthetic acceptance cases |
| `agent_mvp/scripts/` | Evaluation, catalog synchronization and release checks |
| `agent_mvp/evaluation/` | Planner LLM evaluation: test sets, harness, scoring, safeguards and results |
| `agent_mvp/docs/` | Detailed behavior and architecture documentation |

Local credentials, virtual environments, session outputs, evaluation run outputs,
private node-response evidence and backups are excluded from the repository.

## Further Reading

- [Architecture](agent_mvp/docs/ARCHITECTURE.md)
- [Configuration, logs and incident IDs](agent_mvp/docs/CONFIGURATION.md)
- [Read-only MCP server and model lookups](agent_mvp/docs/MCP_SERVER.md)
- [Evaluating local LLMs as the planner](agent_mvp/docs/LLM_EVALUATION.md), the
  [evaluation suite](agent_mvp/evaluation/README.md) and its
  [results](agent_mvp/evaluation/RESULTS_2026-10.md)
- [Package documentation](agent_mvp/README.md)
- [Fixed execution-template workflow](agent_mvp/docs/FIXED_TEMPLATE_WORKFLOW.md)
- [Algorithm registry](agent_mvp/docs/ALGORITHM_REGISTRY.md)
- [Policy, registry and template extension points](agent_mvp/docs/ARCHITECTURE_EXTENSIONS.md)
- [Study sessions and recovery](agent_mvp/docs/STUDY_SESSIONS.md)

Upstream metadata comes from
[STRONG AYA semantic mapping](https://github.com/STRONGAYA/AYA-cancer-semantic-map)
and [descriptive statistics](https://github.com/STRONGAYA/v6-descriptive-statistics).
Source references remain in the catalogs. Algorithm container images and
third-party dependencies are not vendored. The repository's
[Apache 2.0 license](LICENSE) is retained unchanged; third-party components retain
their own licensing terms.
