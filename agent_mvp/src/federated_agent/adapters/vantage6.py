"""The only module that imports the Vantage6 client SDK.

It accepts already-resolved connection values and exposes the bounded client the
trusted worker uses. Only the confirmed synthetic 4.15 sandbox is supported.
"""

from urllib.parse import urlparse

from ..errors import BackendExecutionError

SUPPORTED_VERSION = "4.15.0"


def validate_sandbox(task, confirmed):
    connection = task.connection
    url = urlparse(connection.server_url)
    if (not confirmed or not connection.sandbox or connection.version != SUPPORTED_VERSION
            or url.scheme != "http" or url.hostname != "127.0.0.1" or url.port
            or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment
            or connection.server_api != "/api" or not 1 <= connection.server_port <= 65535):
        raise BackendExecutionError("backend_configuration", stage="preflight")


def create_client(connection):
    import requests
    from importlib.metadata import version
    from vantage6.client import UserClient

    if version("vantage6-client") != SUPPORTED_VERSION:
        raise BackendExecutionError("backend_configuration", stage="preflight")

    class BoundedClient(UserClient):
        # The pinned SDK otherwise logs HTTP errors and retries writes. Preserve
        # its resource/serialization APIs but make transport failures explicit.
        def request(self, endpoint, json=None, method="get", params=None, headers=None,
                    first_try=True, retry=True, attempts_on_timeout=None, is_for_algorithm_store=False):
            response = requests.request(method, self.generate_path_to(endpoint, is_for_algorithm_store),
                json=json, params=params, headers=(headers or {}) | self.headers,
                timeout=(10, 30), allow_redirects=False)
            code = {401: "backend_authentication", 403: "backend_forbidden", 404: "backend_not_found"}.get(response.status_code)
            if code or not 200 <= response.status_code < 300:
                raise BackendExecutionError(code or "backend_error")
            return response.json()

    return BoundedClient(connection.server_url, connection.server_port, connection.server_api, log_level="warn")
