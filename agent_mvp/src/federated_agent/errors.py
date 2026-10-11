"""Domain exceptions raised by the agent pipeline."""


class AgentError(Exception):
    """Base class for user-facing agent errors."""


class VariableNotFoundError(AgentError):
    def __init__(self, query: str, candidates: list[str] | None = None) -> None:
        self.query = query
        self.candidates = candidates or []
        suffix = (
            f" Possible schema variables: {', '.join(self.candidates)}"
            if self.candidates
            else ""
        )
        super().__init__(f"Unknown schema variable: {query}.{suffix}")


class AmbiguousVariableError(AgentError):
    def __init__(self, query: str, candidates: list[str]) -> None:
        self.query = query
        self.candidates = candidates
        super().__init__(
            f"Variable '{query}' is ambiguous. Candidates: {', '.join(candidates)}"
        )


class SiteNotFoundError(AgentError):
    def __init__(self, query: str, candidates: list[str] | None = None) -> None:
        self.query = query
        self.candidates = candidates or []
        suffix = (
            f" Possible configured sites: {', '.join(self.candidates)}"
            if self.candidates
            else ""
        )
        super().__init__(f"Unknown deployment site: {query}.{suffix}")


class AmbiguousSiteError(AgentError):
    def __init__(self, query: str, candidates: list[str]) -> None:
        self.query = query
        self.candidates = candidates
        super().__init__(
            f"Site '{query}' is ambiguous. Candidates: {', '.join(candidates)}"
        )


class ClarificationRequiredError(AgentError):
    def __init__(self, request: object) -> None:
        self.request = request
        super().__init__("The research request needs clarification before compilation")


class UnsupportedAnalysisError(AgentError):
    pass


class AmbiguousAlgorithmError(AgentError):
    def __init__(self, candidates: list[str]) -> None:
        self.candidates = candidates
        super().__init__("Multiple enabled algorithms match; choose an algorithm ID")


class CatalogError(AgentError):
    pass


class TaskValidationError(AgentError):
    def __init__(self, message: str, *, code: str = "invalid_request", field: str | None = None, constraint: str | None = None):
        self.code = code
        self.field = field
        self.constraint = constraint
        super().__init__(message)


class PolicyDeniedError(AgentError):
    pass


class LLMResponseError(AgentError):
    pass


class DataUnavailableError(AgentError):
    def __init__(self, checks):
        self.checks = checks
        super().__init__("Requested data are unavailable in the checked metadata or backend")


class PreflightError(AgentError):
    def __init__(self, message: str, *, code: str = "preflight_error", incident_id: str | None = None):
        self.code = code
        self.incident_id = incident_id
        super().__init__(message)


class BackendExecutionError(AgentError):
    def __init__(self, code: str, *, task_id: str | None = None, stage: str = "submit", incident_id: str | None = None):
        self.code = code
        self.task_id = task_id
        self.stage = stage
        self.incident_id = incident_id
        super().__init__("The execution backend did not complete the request")


# The server refused the submission request itself, so no remote task exists.
AUTHORIZATION_FAILURES = frozenset({"backend_authentication", "backend_forbidden", "backend_not_found"})


def nothing_submitted(code: str, stage: str) -> bool:
    """True when a backend failure proves that no remote task was created."""
    return stage in {"preflight", "authentication"} or (stage == "submit" and code in AUTHORIZATION_FAILURES)
