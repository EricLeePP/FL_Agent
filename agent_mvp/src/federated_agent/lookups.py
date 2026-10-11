"""Read-only research lookups over one metadata snapshot.

The MCP server and the optional SDK tool loop expose these same lookups. They
return the public facts that the planner context already contains: algorithm
capabilities, hospital names, aliases, dataset labels and variable datatypes.
They never return server addresses, organisation IDs, credentials, patient data
or results, and none of them prepares, confirms or submits a task.
"""

from .errors import (
    AmbiguousSiteError,
    AmbiguousVariableError,
    SiteNotFoundError,
    TaskValidationError,
    VariableNotFoundError,
)
from .metadata import MetadataSnapshot

PUBLIC_ALGORITHM_FIELDS = ("algorithm_id", "display_name", "version", "status", "analysis_types", "description",
                           "request_keywords", "capabilities", "limitations", "constraints", "parameters_schema",
                           "adapter_supported", "source")


class ResearchLookups:
    def __init__(self, metadata: MetadataSnapshot, catalog=None):
        self.metadata = metadata
        self.catalog = catalog if catalog is not None else metadata.catalog()

    def list_algorithms(self) -> dict:
        reference = self.catalog.reference
        algorithms = []
        for item in self.catalog.knowledge()["algorithms"]:
            public = {key: item[key] for key in PUBLIC_ALGORITHM_FIELDS if key in item}
            # Where an algorithm is declared, without organisation IDs or internal evidence notes.
            public["availability"] = [{"deployment": scope["deployment"], "database_labels": scope["database_labels"],
                                       "verification": scope["verification"]} for scope in item.get("availability", [])]
            algorithms.append(public)
        return {"catalog_version": reference.catalog_version if reference else None, "algorithms": algorithms}

    def list_sites(self) -> dict:
        profile = self.metadata.deployment_registry.profile
        return {"deployment": profile.name, "dataset_selection_required": True,
                "sites": [{"name": site.name, "aliases": site.aliases, "database_labels": site.database_labels}
                          for site in profile.sites]}

    def resolve_variable(self, name: str) -> dict:
        try:
            variable = self.metadata.semantic_catalog.resolve(name)
        except (AmbiguousVariableError, VariableNotFoundError) as exc:
            return {"status": "needs_confirmation", "requested_name": name, "candidates": list(exc.candidates)}
        except TaskValidationError:
            return {"status": "unsupported", "requested_name": name,
                    "reason": "The variable exists, but no analysis contract supports its datatype."}
        return {"status": "resolved", "requested_name": name, "canonical_name": variable.canonical_name,
                "datatype": variable.datatype, "note": "A schema variable is not proof that data are available."}

    def resolve_site(self, name: str) -> dict:
        try:
            site = self.metadata.deployment_registry.resolve(name)
        except (AmbiguousSiteError, SiteNotFoundError) as exc:
            return {"status": "needs_confirmation", "requested_name": name, "candidates": list(exc.candidates)}
        labels = next(entry.database_labels for entry in self.metadata.deployment_registry.profile.sites
                      if entry.name == site.canonical_name)
        return {"status": "resolved", "requested_name": name, "name": site.canonical_name, "database_labels": labels}
