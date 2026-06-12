import json
import sys
import urllib.error
import urllib.request
import uuid

BASE_URL = "http://localhost:8000"


def _request(method: str, path: str, payload: dict | None = None, token: str | None = None):
    """Send an HTTP request and return (status_code, json_body)."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode() if payload is not None else None

    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, (json.loads(body) if body else {})
        except json.JSONDecodeError:
            return e.code, {}


def run_tests() -> int:
    passed = failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        passed += int(condition)
        failed += int(not condition)
        suffix = f"  ({detail})" if detail else ""
        print(f"{status}  {name}{suffix}")

    email = f"integration_{uuid.uuid4().hex[:10]}@example.com"
    password = "testpass123"

    print("=" * 70)
    print("BACKEND INTEGRATION TESTS")
    print("=" * 70)

    # 1. Register a new user
    status, _ = _request("POST", "/auth/register", {"email": email, "password": password})
    check("POST /auth/register -> 201", status == 201, f"got {status}")

    # 2. Registering the same email again must fail
    status, _ = _request("POST", "/auth/register", {"email": email, "password": password})
    check("POST /auth/register (duplicate email) -> 400", status == 400, f"got {status}")

    # 3. Login with correct credentials
    status, body = _request("POST", "/auth/login", {"email": email, "password": password})
    check("POST /auth/login (correct credentials) -> 200", status == 200, f"got {status}")
    token = body.get("access_token")
    check("Login response contains access_token", bool(token))

    # 4. Login with wrong password
    status, _ = _request("POST", "/auth/login", {"email": email, "password": "wrongpassword"})
    check("POST /auth/login (wrong password) -> 401", status == 401, f"got {status}")

    # 5. Protected endpoint without token
    status, _ = _request("GET", "/analyze/history")
    check("GET /analyze/history (no token) -> 401", status == 401, f"got {status}")

    # 6. Protected endpoint with token
    status, _ = _request("GET", "/analyze/history", token=token)
    check("GET /analyze/history (with token) -> 200", status == 200, f"got {status}")

    # 7. /auth/me returns the logged-in user
    status, body = _request("GET", "/auth/me", token=token)
    check("GET /auth/me -> 200", status == 200, f"got {status}")
    check("GET /auth/me returns correct email", body.get("email") == email)

    # 8. Lexical analysis with a valid URL
    status, body = _request("POST", "/analyze/lexical", {"url": "https://www.google.com"}, token=token)
    check("POST /analyze/lexical (valid URL) -> 200", status == 200, f"got {status}")
    check("Lexical response contains 'score'", "score" in body)
    n_features = len(body.get("features", {}))
    check("Lexical response contains 21 features", n_features == 21, f"got {n_features}")

    # 9. Lexical analysis with an invalid URL
    status, _ = _request("POST", "/analyze/lexical", {"url": "not-a-url"}, token=token)
    check("POST /analyze/lexical (invalid URL) -> 422", status == 422, f"got {status}")

    # 10. Account deletion (cascade cleanup)
    status, _ = _request("DELETE", "/auth/me", token=token)
    check("DELETE /auth/me -> 204", status == 204, f"got {status}")

    # 11. Login must fail after the account has been deleted
    status, _ = _request("POST", "/auth/login", {"email": email, "password": password})
    check("POST /auth/login (after account deletion) -> 401", status == 401, f"got {status}")

    print("-" * 70)
    print(f"TOTAL: {passed} passed, {failed} failed")
    print("=" * 70)
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_tests() else 0)