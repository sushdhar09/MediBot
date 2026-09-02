"""Demo authentication: username/password -> signed, role-tagged session token.

The token is an HMAC-signed payload (no external JWT dependency). It is not a
production identity system - it exists so the backend can derive the role from
the request itself rather than trusting a role sent by the browser.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class DemoUser:
    username: str
    password: str
    role: str
    display_name: str


DEMO_USERS: dict[str, DemoUser] = {
    u.username: u
    for u in (
        DemoUser("dr.mehta", "doctor123", "doctor", "Dr. Anjali Mehta"),
        DemoUser("nurse.priya", "nurse123", "nurse", "Priya Nair"),
        DemoUser("billing.ravi", "billing123", "billing_executive", "Ravi Kulkarni"),
        DemoUser("tech.anand", "tech123", "technician", "Anand Rao"),
        DemoUser("admin.sys", "admin123", "admin", "System Administrator"),
    )
}


class AuthError(Exception):
    """Raised for bad credentials or an invalid/expired token."""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: bytes) -> str:
    return _b64e(hmac.new(config.SECRET_KEY.encode(), payload, hashlib.sha256).digest())


def create_token(user: DemoUser) -> str:
    payload = json.dumps(
        {
            "username": user.username,
            "role": user.role,
            "display_name": user.display_name,
            "exp": int(time.time()) + config.TOKEN_TTL_SECONDS,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"{_b64e(payload)}.{_sign(payload)}"


def decode_token(token: str) -> dict:
    try:
        body, signature = token.split(".", 1)
        payload = _b64d(body)
    except (ValueError, TypeError) as exc:
        raise AuthError("Malformed session token") from exc

    if not hmac.compare_digest(signature, _sign(payload)):
        raise AuthError("Invalid session token signature")

    data = json.loads(payload)
    if data.get("exp", 0) < time.time():
        raise AuthError("Session token has expired")
    return data


def authenticate(username: str, password: str) -> DemoUser:
    user = DEMO_USERS.get(username.strip().lower())
    if user is None or not hmac.compare_digest(user.password, password):
        raise AuthError("Invalid username or password")
    return user
