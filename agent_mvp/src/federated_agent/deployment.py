"""Resolve researcher-facing site names against a checked deployment profile."""

from difflib import SequenceMatcher
from pathlib import Path

from .errors import AmbiguousSiteError, SiteNotFoundError
from .models import (
    DeploymentProfile,
    DeploymentSite,
    ResolvedSite,
    SourceReference,
)
from .semantic_catalog import normalize_text


class DeploymentRegistry:
    def __init__(
        self,
        profile: DeploymentProfile,
        source: SourceReference,
    ) -> None:
        self.profile = profile
        self.source = source

    @classmethod
    def from_json(cls, path: str | Path, *, content: bytes | None = None) -> "DeploymentRegistry":
        path = Path(path)
        profile = DeploymentProfile.model_validate_json(path.read_bytes() if content is None else content)
        source = SourceReference(
            kind="deployment_registry",
            uri=str(path.resolve()),
            version=profile.registry_version,
        )
        return cls(profile=profile, source=source)

    def resolve(self, query: str) -> ResolvedSite:
        normalized_query = normalize_text(query)
        exact = []
        for site in self.profile.sites:
            names = [site.name, *site.aliases]
            if normalized_query in {normalize_text(name) for name in names}:
                exact.append(site)

        if len(exact) > 1:
            raise AmbiguousSiteError(query, [site.name for site in exact])
        if len(exact) == 1:
            return self._to_resolved(query, exact[0])

        scored = [
            (
                max(
                    SequenceMatcher(
                        None, normalized_query, normalize_text(candidate)
                    ).ratio()
                    for candidate in [site.name, *site.aliases]
                ),
                site,
            )
            for site in self.profile.sites
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        candidates = [site.name for score, site in scored if score >= 0.45][:5]
        raise SiteNotFoundError(query, candidates)

    def resolve_many(self, queries: list[str]) -> list[ResolvedSite]:
        resolved = [self.resolve(query) for query in queries]
        unique: dict[int, ResolvedSite] = {}
        for site in resolved:
            unique.setdefault(site.organisation_id, site)
        return list(unique.values())

    def _to_resolved(self, query: str, site: DeploymentSite) -> ResolvedSite:
        return ResolvedSite(
            requested_name=query,
            canonical_name=site.name,
            organisation_id=site.organisation_id,
            source=self.source,
        )
