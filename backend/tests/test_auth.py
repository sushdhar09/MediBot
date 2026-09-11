import json

import pytest

from app import auth


def test_authenticate_accepts_case_and_whitespace():
    user = auth.authenticate("  DR.MEHTA ", "doctor123")
    assert user.role == "doctor"
    assert user.display_name == "Dr. Anjali Mehta"


@pytest.mark.parametrize(
    ("username", "password"),
    [("unknown", "doctor123"), ("dr.mehta", "wrong"), ("", "")],
)
def test_authenticate_rejects_invalid_credentials(username, password):
    with pytest.raises(auth.AuthError, match="Invalid username or password"):
        auth.authenticate(username, password)


def test_token_round_trip():
    user = auth.DEMO_USERS["nurse.priya"]
    payload = auth.decode_token(auth.create_token(user))
    assert payload["username"] == user.username
    assert payload["role"] == "nurse"
    assert payload["display_name"] == user.display_name


def test_token_rejects_tampered_signature():
    token = auth.create_token(auth.DEMO_USERS["dr.mehta"])
    body, signature = token.split(".")
    tampered = f"{body}.{'A' if signature[-1] != 'A' else 'B'}{signature[1:]}"
    with pytest.raises(auth.AuthError, match="signature"):
        auth.decode_token(tampered)


def test_token_rejects_expired_payload(monkeypatch):
    monkeypatch.setattr(auth.config, "TOKEN_TTL_SECONDS", -1)
    token = auth.create_token(auth.DEMO_USERS["admin.sys"])
    with pytest.raises(auth.AuthError, match="expired"):
        auth.decode_token(token)


@pytest.mark.parametrize("token", ["", "not-a-token", "one.two.three"])
def test_token_rejects_malformed_input(token):
    with pytest.raises(auth.AuthError):
        auth.decode_token(token)
