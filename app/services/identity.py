from __future__ import annotations

import os
from dataclasses import dataclass
from hmac import compare_digest

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedActor:
    actor_id: str


def authenticate_operator(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> AuthenticatedActor:
    expected_token = os.getenv("CAREEROPS_OPERATOR_TOKEN", "")
    actor_id = os.getenv("CAREEROPS_OPERATOR_ID", "local-operator").strip()
    if not expected_token or not actor_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="operator authentication is not configured",
        )
    if (
        credentials is None
        or credentials.scheme.casefold() != "bearer"
        or not compare_digest(credentials.credentials, expected_token)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid operator credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthenticatedActor(actor_id=actor_id)
