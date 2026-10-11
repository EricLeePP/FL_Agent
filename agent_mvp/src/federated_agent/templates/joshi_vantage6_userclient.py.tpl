"""Review copy of the researcher-side Vantage6 workflow for one prepared task.

The agent's trusted worker submits this task with the same submit_task() shown
below. Running this file does not submit anything: it verifies the reviewed
content and prints the agent command that runs the fresh data and policy checks
before submission. It never reads credentials.
"""
import hashlib
import json

confirmation_material = __CONFIRMATION_MATERIAL__
payload = confirmation_material["client_payload"]
CONFIRMATION_ID = __CONFIRMATION_ID__


# The exact submission call made by the trusted worker (federated_agent.runs.submission).
__SUBMISSION_CODE__


actual_confirmation = hashlib.sha256(
    json.dumps(confirmation_material, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
).hexdigest()[:16]
if actual_confirmation != CONFIRMATION_ID:
    raise SystemExit("Generated task content changed after preparation.")
print(json.dumps(payload, indent=2))
raise SystemExit(
    "This task requires fresh data/authorization checks, and this review copy cannot bypass them.\n"
    "Submit it through the agent, from the folder that contains prepared_task.json:\n"
    f"  python -m federated_agent --execute-prepared prepared_task.json --confirm {CONFIRMATION_ID} "
    "--backend vantage6 --confirm-synthetic-unencrypted-sandbox"
)
