"""Default metadata bundled inside the package.

The bundled files describe the offline demo deployment and the synthetic Ubuntu
sandbox. Real deployments should point a configuration file at
operator-maintained copies instead of editing files inside an installed package.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = PACKAGE_ROOT / "data"
# The agent_mvp source checkout. It exists only for editable development installs
# and is used by development scripts and tests, never to locate runtime metadata.
SOURCE_CHECKOUT = PACKAGE_ROOT.parents[1]


def resource(*parts: str) -> Path:
    return RESOURCE_ROOT.joinpath(*parts)


ALGORITHM_CATALOG = resource("catalogs", "algorithms.json")
ALGORITHM_STORE_METADATA = resource("catalogs", "algorithm_store.json")
SEMANTIC_SCHEMA = resource("catalogs", "AYA_cancer_schema.jsonld")
EXECUTION_TEMPLATES = resource("catalogs", "execution_templates.json")
DEMO_DEPLOYMENT = resource("config", "demo_deployment.json")
SANDBOX_DEPLOYMENT = resource("config", "ubuntu_sandbox.json")
SANDBOX_DATA_AVAILABILITY = resource("config", "ubuntu_data_availability.json")
DEFAULT_POLICY = resource("config", "policy.json")
