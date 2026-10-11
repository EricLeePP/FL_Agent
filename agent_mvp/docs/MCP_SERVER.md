# Read-Only MCP Server and Model Lookups

Fed Agent exposes the same read-only research lookups in two ways:

- an [MCP](https://modelcontextprotocol.io) server, so another agent or an MCP
  client can explore the catalog and check a study plan;
- an optional tool loop for the Agents SDK planner, so the model can look up
  exact names before it proposes a plan.

Both use `lookups.ResearchLookups`, bound to one metadata snapshot. They return the
public facts the planner already sees: algorithm capabilities and limitations,
hospital names, aliases, dataset labels and variable datatypes. They never return
server addresses, organisation IDs, credentials, patient data or results. No tool
writes files, prepares a task for execution or submits anything. Execution always
stays in a researcher's own session, after the task review and confirmation.

## Running the MCP server

Install the extra and start the server over stdio:

```bash
python -m pip install -e "./agent_mvp[mcp]"
python -m federated_agent.mcp_server --config fed-agent.json
```

`federated-agent-mcp` is the same command as an installed script. The server reads
the same configuration file as the command line (see
[configuration](CONFIGURATION.md)) and reads fresh metadata for every tool call.
Logs go to stderr and the optional `--log-file`; stdout carries the protocol.

An MCP client configuration looks like this:

```json
{
  "mcpServers": {
    "fed-agent": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "federated_agent.mcp_server", "--config", "/path/to/fed-agent.json"]
    }
  }
}
```

Only stdio is offered. A network transport would expose these tools to anyone who
can reach the port, so it needs an authentication layer first.

## Tools

| Tool | Returns |
| --- | --- |
| `list_algorithms` | Registered algorithms with capabilities, limitations, constraints and the deployments and dataset labels they are declared for |
| `list_sites` | Configured hospitals with aliases and dataset labels |
| `resolve_variable(name)` | The matching semantic variable and its datatype, or candidates when the name is ambiguous or unknown |
| `resolve_site(name)` | The matching hospital and its dataset labels, or candidates |
| `check_plan(variables, sites, database_label, ...)` | The checked status of a complete plan: `ready`, `needs_clarification`, `unsupported` and so on, with issues and clarification questions |

Every tool is annotated read-only and non-destructive. `check_plan` runs exactly
the checks a researcher's request goes through, including data availability and
the policy's `prepare` decision, but keeps the prepared task in memory. A `ready`
result is not a submission: the researcher must still prepare, review and confirm
the task in the agent. Failures return a status and a researcher-facing message;
an unexpected error returns only an incident ID that matches the server log.

## Lookups in the SDK planner

With `"llm": {"runtime": "sdk", "tool_lookups": true, "max_turns": 6}`, the planner
offers the model four lookup tools (`resolve_variable`, `resolve_site`,
`list_sites`, `list_algorithms`) next to `propose_research_plan`. The model may call
lookups over several turns and must finish with exactly one proposal. If it does
not propose within `max_turns`, planning fails closed. The proposal then passes the
same validators as any other plan, so a lookup can never approve or widen a study.

The option is off by default. It changes how the model works, so evaluate it with
a real model on the request corpus (`scripts/evaluate_requests.py`) before using it
in a study. `--describe` reports the setting and warns when it has no effect.
