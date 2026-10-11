"""Publish a metadata release; private signing keys never enter its contents."""

import argparse
import base64
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from .registry_sync import ROLES, Release, RegistrySource, canonical, validate_documents
from .storage import write_json
from .trusted_http import strict_json


def publish(config_path, output, *, release_id, sequence, hours=24, signing_key=None, unsigned_local=False):
    config_path = Path(config_path).resolve()
    authored = strict_json(config_path.read_bytes())
    if set(authored) != set(ROLES):
        raise ValueError("Publisher config must map exactly the five registry roles to JSON files")
    documents = {role: strict_json((config_path.parent / path).read_bytes()) for role, path in authored.items()}
    now = datetime.now(timezone.utc)
    inventory_expiry = datetime.fromisoformat(documents["availability"]["expires_at"].replace("Z", "+00:00"))
    expires = min(now + timedelta(hours=hours), inventory_expiry)
    if hours <= 0 or expires <= now:
        raise ValueError("Refresh expired data declarations before publishing")
    profile = documents["deployment"]
    config = RegistrySource(schema_version="1.0", source="local.json", deployment=profile["name"], cache_dir="cache",
        allow_unsigned_local=True, connection={key: profile[key] for key in
            ("server_url", "server_port", "server_api", "collaboration_id")})
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for role, name in ROLES.items():
            (directory / name).write_bytes(canonical(documents[role]))
        validate_documents(directory, config)
    release = Release(schema_version="1.0", release_id=release_id, sequence=sequence,
        deployment=profile["name"], created_at=now, expires_at=expires, documents=documents).model_dump(mode="json")
    signature = None
    if signing_key:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        key = load_pem_private_key(Path(signing_key).read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("Publisher requires an Ed25519 private key")
        signature = base64.b64encode(key.sign(canonical(release))).decode()
    elif not unsigned_local:
        raise ValueError("Supply a signing key or explicitly select unsigned local development")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    from .storage import FileLease
    with closing(FileLease(output.with_suffix(output.suffix + ".publish.lock"))):
        if output.exists():
            previous = strict_json(output.read_bytes())["release"]
            if sequence <= previous["sequence"]:
                raise ValueError("Publisher sequence must increase")
        write_json(output, {"release": release, "signature": signature})
    return {"release_id": release_id, "sequence": sequence, "expires_at": expires.isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--hours", type=float, default=24)
    signing = parser.add_mutually_exclusive_group(required=True)
    signing.add_argument("--signing-key", type=Path)
    signing.add_argument("--unsigned-local", action="store_true")
    args = parser.parse_args()
    try:
        result = publish(args.config, args.output, release_id=args.release_id, sequence=args.sequence,
                         hours=args.hours, signing_key=args.signing_key, unsigned_local=args.unsigned_local)
        print("Published:", result)
    except Exception:
        parser.exit(2, "Registry publication failed. Validate metadata, versions, timestamps and signing configuration.\n")


if __name__ == "__main__":
    main()
