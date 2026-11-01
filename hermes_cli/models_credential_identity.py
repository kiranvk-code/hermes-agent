"""Credential-identity digests for the provider-models catalog cache.

Local backport of upstream PR #139831 (fix for #139795): key a provider's catalog cache row on
its own slice of the shared ``auth.json`` instead of the file mtime, so pool bookkeeping writes
(``request_count`` / ``last_status`` bumps on every Copilot turn) stop evicting the warmed row and
the non-blocking ``/model`` picker stops flipping between the live list and the curated fallback.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

# Field sets that rotate routinely without changing *which* credential is in use. Mirrors
# upstream ``hermes_cli.auth._POOL_TOKEN_GENERATION_FIELDS`` (not present in older checkouts).
_FALLBACK_TOKEN_GENERATION_FIELDS = (
    "access_token", "refresh_token", "expires_at", "expires_at_ms", "expires_in", "obtained_at",
    "last_refresh", "agent_key", "agent_key_expires_at", "agent_key_expires_in", "agent_key_id",
    "agent_key_obtained_at", "agent_key_reused", "scope", "validated_at",
)


def _auth_store_provider_identity(provider: str) -> Optional[str]:
    """Digest of *provider*'s own credentials in the shared auth.json, for the catalog fingerprint.

    Covers the credential-pool entries and the ``providers.<id>`` state with the routinely rotating
    fields stripped — the same token/status field sets auth treats as generation-coupled, plus the
    pool-row runtime counters — so a token refresh, a pool counter bump or another provider's write
    leaves the digest unchanged, while a real credential switch (entry added/removed, a new static
    key, a different login identity) changes it. ``None`` when the store or its reader is
    unavailable; the caller then falls back to the whole-file mtime, which degrades to a spurious
    cache miss, never a stale hit."""
    try:
        from hermes_cli import auth as _auth
        from hermes_cli.auth import (
            _FLAT_OAUTH_TOKEN_KEYS,
            _POOL_STATUS_FIELDS,
            _load_auth_store,
            _load_provider_state,
            read_credential_pool,
        )
    except Exception:  # health: allow BLE001 -- a missing auth surface must degrade to the mtime key, not kill catalog lookups
        return None
    token_generation_fields = getattr(
        _auth, "_POOL_TOKEN_GENERATION_FIELDS", _FALLBACK_TOKEN_GENERATION_FIELDS)
    volatile = (
        set(_POOL_STATUS_FIELDS)
        | set(token_generation_fields)
        | set(_FLAT_OAUTH_TOKEN_KEYS)
        | {
            # Pool-row runtime state outside the auth field sets: the per-selection counter,
            # model-level cooldowns and the persisted exhaustion classification.
            "request_count",
            "model_cooldowns",
            "failure_reason",
            # Derived from / refreshed with the rotating secrets themselves.
            "anon_token",
            "secret_fingerprint",
            # Store-level write timestamp.
            "updated_at",
        }
    )

    def _stable(value):
        if isinstance(value, dict):
            return {k: _stable(v) for k, v in sorted(value.items()) if k not in volatile}
        if isinstance(value, list):
            return [_stable(v) for v in value]
        return value

    def _blob(value) -> str:
        return json.dumps(_stable(value), sort_keys=True, separators=(",", ":"), default=str)

    try:
        # Pool entries are an unordered set of credentials: digest them sorted so a row re-order
        # (merge, pool rewrite) does not invalidate the catalog; nested lists keep their order.
        pool = sorted(_blob(entry) for entry in read_credential_pool(provider))
        state = _blob(_load_provider_state(_load_auth_store(), provider))
        blob = f"pool={pool}|state={state}"
        return hashlib.blake2b(
            blob.encode("utf-8", errors="replace"), digest_size=8
        ).hexdigest()
    except Exception:  # health: allow BLE001 -- an unreadable store falls back to the caller's mtime key: a spurious cache miss, never a stale hit
        return None
