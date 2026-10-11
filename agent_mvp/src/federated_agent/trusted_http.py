"""Bounded control-plane requests, with no redirects or credential-bearing URLs."""

import json
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def validate_url(url):
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise ValueError("Control-plane URLs must not contain credentials, queries or fragments")
    if not parsed.hostname or (parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    )):
        raise ValueError("HTTPS is required except for a local development endpoint")
    return parsed


def fetch(url, *, body=None, token=None, timeout=5, limit=4 * 1024 * 1024):
    validate_url(url)
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None
    if body is not None:
        data = json.dumps(body, allow_nan=False).encode()
        headers["Content-Type"] = "application/json"
    with build_opener(NoRedirect()).open(Request(url, data=data, headers=headers), timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("Unexpected control-plane response")
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Control-plane response too large")
    return content


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError("Non-finite JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
