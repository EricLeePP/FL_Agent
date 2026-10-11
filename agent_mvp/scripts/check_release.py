"""Check the Git index for accidentally included local files or credential tokens.

This bounded hygiene check prints paths, never matched secret values. It is not
a replacement for a dedicated secret scanner or a manual staged-diff review.
"""

import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
ROOT_FILES = {
    ".gitignore", ".gitattributes", "LICENSE", "README.md",
    "AGENT_OPERATIONS_AND_REGISTRY.md", "agent.cmd",
}
DENIED_PARTS = {
    "integration", "backups", "private", "secrets", "__pycache__", "tmp",
    ".runtimes", ".agents", ".git", "build", "dist",
}
DENIED_NAMES = {"deepseek-api.txt", "vantage6.txt", "ubuntu_data_availability.evidence.json"}
DENIED_SUFFIXES = {".zip", ".log", ".pyc", ".pem", ".key", ".pfx", ".p12", ".pdf", ".png"}
TOKEN_PATTERNS = (
    rb"\bsk-[A-Za-z0-9_-]{24,}\b",
    rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
)
CREDENTIAL_FIELDS = {"password", "api_key", "access_token", "refresh_token", "private_key"}


def config_has_credential(value):
    if isinstance(value, dict):
        return any(
            (key.casefold() in CREDENTIAL_FIELDS and bool(item)) or config_has_credential(item)
            for key, item in value.items()
        )
    return isinstance(value, list) and any(config_has_credential(item) for item in value)


def is_ci_workflow(path):
    return path.parts[:2] == (".github", "workflows") and len(path.parts) == 3 and path.suffix in {".yml", ".yaml"}


def inspect_file(name, content):
    path = PurePosixPath(name)
    parts = tuple(part.casefold() for part in path.parts)
    problems = []
    if len(path.parts) == 1 and name not in ROOT_FILES:
        problems.append("unexpected root file")
    if len(path.parts) > 1 and path.parts[0] != "agent_mvp" and not is_ci_workflow(path):
        problems.append("unexpected source tree")
    if any(part in DENIED_PARTS or part.startswith((".venv", "output")) or part.endswith(".egg-info") for part in parts):
        problems.append("local/generated directory")
    if path.name.casefold() in DENIED_NAMES or path.suffix.casefold() in DENIED_SUFFIXES:
        problems.append("local/generated or private file type")
    if path.name.startswith(".env") and path.name != ".env.example":
        problems.append("environment credentials file")
    if len(content) > 1_000_000:
        problems.append("unexpected file larger than 1 MB")
    if any(re.search(pattern, content) for pattern in TOKEN_PATTERNS):
        problems.append("possible credential token")
    if name.startswith("agent_mvp/src/federated_agent/data/config/") and path.suffix == ".json":
        if config_has_credential(json.loads(content)):
            problems.append("nonempty configuration credential field")
    return problems


def main():
    names = subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=ROOT).decode().split("\0")
    names = [name for name in names if name]
    if not names:
        print("No tracked/staged files found; stage the release first.", file=sys.stderr)
        return 1
    findings, size = [], 0
    for name in names:
        content = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        size += len(content)
        findings.extend((name, reason) for reason in inspect_file(name, content))
    for name, reason in findings:
        print(f"FAIL {name}: {reason}", file=sys.stderr)
    print(f"Checked {len(names)} indexed files ({size:,} bytes); {len(findings)} hygiene findings.")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
