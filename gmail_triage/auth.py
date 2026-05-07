"""
Gmail OAuth flow.

The first time you run this, it opens a browser for Google's consent screen and
saves the resulting refresh token to `token.json`. After that, all subsequent
calls use the cached token and silently refresh it when it expires.

Two files live next to your repo root:
- credentials.json — the OAuth client downloaded from Google Cloud Console.
                     Never commit this. Gitignored.
- token.json       — the refresh token saved after consent. Never commit this. Gitignored.

Required scopes:
- gmail.modify       — read messages, apply/remove labels, create drafts.
                       (Strictly less powerful than gmail.send: we never send mail.)

If you need to switch accounts or change scopes, delete token.json and rerun.
"""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

REPO_ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS_PATH = REPO_ROOT / "credentials.json"
TOKEN_PATH = REPO_ROOT / "token.json"

# gmail.modify lets us read, label, and draft. We deliberately do NOT request
# gmail.send — drafts are saved to your Drafts folder; you press send.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class CredentialsMissingError(RuntimeError):
    """Raised when credentials.json is not present and we can't authenticate."""


def _setup_message() -> str:
    return (
        "\n"
        "credentials.json not found at " + str(CREDENTIALS_PATH) + ".\n\n"
        "To set up Gmail access:\n"
        "  1. Open https://console.cloud.google.com/ and create a new project (free).\n"
        "  2. APIs & Services → Library → search 'Gmail API' → Enable.\n"
        "  3. APIs & Services → OAuth consent screen → External → fill the required\n"
        "     fields. Add yourself as a test user.\n"
        "  4. APIs & Services → Credentials → Create credentials → OAuth client ID.\n"
        "     Application type: Desktop app.\n"
        "  5. Download the JSON, rename to credentials.json, place at the repo root.\n"
        "\n"
        "Then re-run this command. credentials.json is already gitignored.\n"
    )


def get_credentials(*, interactive: bool = True) -> Credentials:
    """Load (or create) Gmail credentials.

    Args:
        interactive: If True, run the browser consent flow when no token exists.
                     If False, raise instead — useful for non-TTY runs.
    """
    creds: Credentials | None = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
        return creds

    # No valid creds — need to do the OAuth flow.
    if not CREDENTIALS_PATH.exists():
        raise CredentialsMissingError(_setup_message())

    if not interactive:
        raise RuntimeError(
            "No valid token and interactive=False. "
            "Run `python -m cli gmail setup` once to authorize."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
    # run_local_server opens a browser, listens on localhost for the callback,
    # and exchanges the code for a token automatically.
    creds = flow.run_local_server(port=0)
    TOKEN_PATH.write_text(creds.to_json())
    return creds


def gmail_service():
    """Return an authorised Gmail API client (`users.messages`, `users.labels`, etc.)."""
    creds = get_credentials()
    return build("gmail", "v1", credentials=creds, cache_discovery=False)
