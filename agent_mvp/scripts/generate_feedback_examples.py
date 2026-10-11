"""Create inspectable feedback examples with fixed plans, not real LLM/backend calls."""

import argparse
import io
import json
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from federated_agent.app import build_agent
from federated_agent.services.artifacts import write_response
from federated_agent.data_availability import FileDataAvailabilityChecker
from federated_agent.models import PolicyDecision, ResearchPlan


class FixedPlanner:
    def __init__(self, **changes):
        values = dict(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
                      sites=["hospital 1", "hospital 2"], purpose="OFFLINE FEEDBACK EXAMPLE, not a real research request")
        values.update(changes)
        self.plan = ResearchPlan(**values)

    def create_plan(self, request, catalog):
        return self.plan


class ExampleDenyPolicy:
    def evaluate(self, task):
        return PolicyDecision(allowed=False, reasons=["Simulated denial, not a real authorization check"])


def generate(output):
    output.mkdir(parents=True, exist_ok=False)
    scenarios = [
        ("01-unsupported-algorithm", {"analysis_type": "cox", "algorithm_id": "unregistered_cox"}),
        ("02-unknown-variable", {"variables": ["imaginary_marker_xyz"]}),
        ("03-unknown-hospital", {"sites": ["hospital 99"]}),
        ("04-unsupported-filter", {"unsupported_requirements": ["only patients under 25"]}),
        ("05-data-unavailable", {}),
        ("06-permission-denied", {}),
        ("07-ready-but-unverified", {}),
    ]
    for name, changes in scenarios:
        directory = output / name
        directory.mkdir()
        agent = build_agent()
        if name == "05-data-unavailable":
            now = datetime.now(timezone.utc)
            metadata = dict(schema_version="1.0", version="OFFLINE-EXAMPLE-ONLY", deployment="local-demo-sandbox",
                            observed_at=(now - timedelta(minutes=1)).isoformat(),
                            expires_at=(now + timedelta(days=1)).isoformat(), sites=[
                                dict(organisation_id=2, database_label="default", variables={"ncit:C156420": "unavailable"})])
            path = directory / "simulated_metadata.json"
            path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            agent.data_checker = FileDataAvailabilityChecker(path)
        if name == "06-permission-denied":
            agent.policy_checker = ExampleDenyPolicy()
        response = agent.respond("OFFLINE EXAMPLE", FixedPlanner(**changes))
        response.message = "OFFLINE EXAMPLE (simulated inputs, no real LLM or Vantage6 call): " + response.message
        with redirect_stdout(io.StringIO()):
            write_response(response, directory)
        print(f"{name}: {response.status} -> {directory / 'researcher_response.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output-feedback-examples"))
    generate(parser.parse_args().output)
