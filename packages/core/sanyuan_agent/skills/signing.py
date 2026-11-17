"""Ed25519 signing + verification for custom skills (PRD §7.5).

Trust model:
- Built-in skills under ``skills/public/`` ship with the reviewed release
  artifact and are trusted by provenance — they need no signature.
- *Custom* (user-uploaded) skills MUST carry a valid Ed25519 ``signature`` in
  their SKILL.md frontmatter; unsigned or tampered skills are rejected (status
  ``pending``) and never loaded.

The signed payload is canonical over (name, version, body-hash) so that
reformatting whitespace in the body or reordering frontmatter keys does not
invalidate a signature, while any change to the prompt body does.
"""

from __future__ import annotations

import base64
import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def canonical_payload(name: str, version: str, body: str) -> bytes:
    """Deterministic bytes signed for a skill.

    Body is hashed (not embedded) so the signature is compact and stable.
    """
    doc = {
        "name": name,
        "version": version,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def generate_keypair() -> tuple[bytes, bytes]:
    """Return (private_raw, public_raw) 32-byte Ed25519 keys."""
    priv = Ed25519PrivateKey.generate()
    return priv.private_bytes_raw(), priv.public_key().public_bytes_raw()


def sign(private_key_raw: bytes, name: str, version: str, body: str) -> str:
    """Sign a skill, returning a base64 signature string."""
    priv = Ed25519PrivateKey.from_private_bytes(private_key_raw)
    sig = priv.sign(canonical_payload(name, version, body))
    return base64.b64encode(sig).decode("ascii")


def verify(
    public_key_raw: bytes,
    name: str,
    version: str,
    body: str,
    signature_b64: str,
) -> bool:
    """Return True iff *signature_b64* is a valid Ed25519 signature for the skill."""
    try:
        pub = Ed25519PublicKey.from_public_bytes(public_key_raw)
        pub.verify(
            base64.b64decode(signature_b64),
            canonical_payload(name, version, body),
        )
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def public_key_from_b64(value: str) -> bytes:
    """Decode a base64-encoded raw public key (e.g. from an env var)."""
    return base64.b64decode(value)


class SkillVerifier:
    """Verifies custom-skill signatures against a trusted public key.

    The public key is the tenant/platform admin key that signs approved skills
    (loaded from ``SANYUAN_AGENT_SKILL_PUBLIC_KEY`` in production, injected via
    Infisical — never committed).
    """

    def __init__(self, public_key_raw: bytes) -> None:
        self._public_key_raw = public_key_raw

    @classmethod
    def from_b64(cls, value: str) -> SkillVerifier:
        return cls(public_key_from_b64(value))

    def verify(self, name: str, version: str, body: str, signature_b64: str) -> bool:
        return verify(self._public_key_raw, name, version, body, signature_b64)
