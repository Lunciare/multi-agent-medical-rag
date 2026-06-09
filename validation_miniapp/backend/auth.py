import hashlib
import hmac
import time
from urllib.parse import parse_qsl, urlencode


class InitDataError(Exception):
    """Raised when initData is missing, malformed, or fails the HMAC check."""


def _data_check_string(pairs):
    return "\n".join(f"{k}={v}" for k, v in sorted(pairs, key=lambda kv: kv[0]))


def validate_init_data(init_data, bot_token, max_age_hours=None, now=None):
    """Validate a raw ``initData`` query string. Returns the parsed field dict.

    Raises ``InitDataError`` on any problem. ``max_age_hours`` (if not None) also
    rejects data whose ``auth_date`` is older than that many hours.
    """
    if not init_data:
        raise InitDataError("empty initData")
    if not bot_token:
        raise InitDataError("server misconfigured: no BOT_TOKEN")

    pairs = parse_qsl(init_data, keep_blank_values=True)
    fields = dict(pairs)

    provided_hash = fields.get("hash")
    if not provided_hash:
        raise InitDataError("missing hash field")

    check_pairs = [(k, v) for k, v in pairs if k not in ("hash", "signature")]
    dcs = _data_check_string(check_pairs)

    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    expected = hmac.new(secret_key, dcs.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, provided_hash):
        raise InitDataError("hash mismatch")

    if max_age_hours is not None:
        auth_date = fields.get("auth_date")
        if auth_date is None:
            raise InitDataError("missing auth_date")
        try:
            auth_ts = int(auth_date)
        except ValueError:
            raise InitDataError("non-integer auth_date")
        now_ts = time.time() if now is None else now
        if now_ts - auth_ts > max_age_hours * 3600:
            raise InitDataError("initData expired")

    return fields


def build_init_data(bot_token, fields):
    """Build a correctly-signed ``initData`` string (for tests / local tooling).

    ``fields`` is a dict of the data fields (e.g. auth_date, query_id, user) WITHOUT
    a hash. Returns the URL-encoded string with a valid ``hash`` appended.
    """
    pairs = [(k, str(v)) for k, v in fields.items() if k not in ("hash", "signature")]
    dcs = _data_check_string(pairs)
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    h = hmac.new(secret_key, dcs.encode("utf-8"), hashlib.sha256).hexdigest()
    return urlencode(pairs + [("hash", h)])
