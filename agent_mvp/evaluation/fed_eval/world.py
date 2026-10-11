"""The fixed evaluation world: deployment, catalogs, data inventory and clock.

The deployment, algorithm catalog and semantic schema are the packaged ones; the
data inventory is evaluation-specific so that requests can reach many variables,
including some that are unavailable or unlisted at particular hospitals.
"""

import hashlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from federated_agent.app import build_agent
from federated_agent.resources import ALGORITHM_CATALOG, SANDBOX_DEPLOYMENT, SEMANTIC_SCHEMA

WORLD = Path(__file__).resolve().parents[1] / "world"
INVENTORY = WORLD / "data_availability.json"
CLOCK = datetime.fromisoformat("2026-10-15T12:00:00+00:00")


def build_world_agent():
    return build_agent(SANDBOX_DEPLOYMENT, ALGORITHM_CATALOG, INVENTORY, semantic_schema_path=SEMANTIC_SCHEMA)


@contextmanager
def world_clock():
    """Evaluate against the inventory's observation window, whatever today's date is."""
    with patch("federated_agent.data_availability.datetime") as clock:
        clock.now.return_value = CLOCK
        yield


def fingerprint():
    files = {"deployment": SANDBOX_DEPLOYMENT, "algorithm_catalog": ALGORITHM_CATALOG,
             "semantic_schema": SEMANTIC_SCHEMA, "inventory": INVENTORY}
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in files.items()}
