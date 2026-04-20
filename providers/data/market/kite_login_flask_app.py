from pathlib import Path

from dotenv import load_dotenv, set_key
import os
from flask import Flask, jsonify, request
from kiteconnect import KiteConnect

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.getenv("KITE_API_KEY", "")
API_SECRET = os.getenv("KITE_API_SECRET", "")

app = Flask(__name__)


def _validate_config() -> None:
    if not API_KEY:
        raise RuntimeError("KITE_API_KEY is missing in .env")
    if not API_SECRET:
        raise RuntimeError("KITE_API_SECRET is missing in .env")


@app.get("/")
def index() -> tuple[str, int]:
    try:
        _validate_config()
    except RuntimeError as exc:
        return str(exc), 500

    kite = KiteConnect(api_key=API_KEY)
    return (
        "Open this URL in browser, login, then ensure redirect URI points to /login: \n"
        + kite.login_url(),
        200,
    )


@app.get("/login")
def login() -> tuple[object, int]:
    try:
        _validate_config()
    except RuntimeError as exc:
        return {"error": str(exc)}, 500

    request_token = request.args.get("request_token", "")
    if not request_token:
        return {"error": "Missing request_token in callback query params"}, 400

    kite = KiteConnect(api_key=API_KEY)
    print (API_KEY, API_SECRET, request_token)
    session_data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = session_data["access_token"]

    print(ENV_PATH, API_KEY, API_SECRET, request_token, access_token)
    set_key(str(ENV_PATH), "KITE_ACCESS_TOKEN", access_token)

    return (
        jsonify(
            {
                "status": "ok",
                "message": "Access token generated and saved to .env",
                "access_token": access_token,
                "user_id": session_data.get("user_id"),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
