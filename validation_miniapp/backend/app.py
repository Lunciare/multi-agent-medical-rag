import hashlib
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from . import auth, db, items_loader

_BASE = Path(__file__).resolve().parent.parent
DEFAULT_ITEMS_PATH = _BASE / "data" / "items.dummy.json"
FRONTEND_DIR = _BASE / "frontend"

SURVEY_OPEN = "open"
SURVEY_CLOSED = "closed"


def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def hash_rater_id(raw_id, salt=None):
    """Hash a raw rater id so the DB never stores the raw Telegram user id."""
    salt = salt if salt is not None else os.getenv("RID_HASH_SALT", "validation-stage1")
    return hashlib.sha256(f"{salt}:{raw_id}".encode("utf-8")).hexdigest()[:16]


class SubmitBody(BaseModel):
    item_id: str
    preference: str
    safety_flag_opt1: bool
    safety_flag_opt2: bool
    routing_judgment: str
    client_ts: str | None = None

    @field_validator("preference")
    @classmethod
    def _check_pref(cls, v):
        if v not in db.PREFERENCE_VALUES:
            raise ValueError(f"preference must be one of {db.PREFERENCE_VALUES}")
        return v

    @field_validator("routing_judgment")
    @classmethod
    def _check_routing(cls, v):
        if v not in db.ROUTING_VALUES:
            raise ValueError(f"routing_judgment must be one of {db.ROUTING_VALUES}")
        return v


def create_app(db_path=None, items_path=None, dev_mode=None, bot_token=None,
               admin_token=None, default_survey_state=None, initdata_max_age_hours=None,
               rid_salt=None):
    """Build a FastAPI app. Tests inject explicit config; production reads env."""
    db_path = str(db_path if db_path is not None
                  else os.getenv("DB_PATH") or db.DEFAULT_DB_PATH)
    items_path = items_path if items_path is not None else os.getenv("ITEMS_PATH", DEFAULT_ITEMS_PATH)
    dev_mode = _env_bool("DEV_MODE", False) if dev_mode is None else dev_mode
    bot_token = bot_token if bot_token is not None else os.getenv("BOT_TOKEN", "")
    admin_token = admin_token if admin_token is not None else os.getenv("ADMIN_TOKEN", "")
    default_survey_state = (default_survey_state if default_survey_state is not None
                            else os.getenv("SURVEY_STATE", SURVEY_OPEN))
    if initdata_max_age_hours is None:
        _raw = os.getenv("INITDATA_MAX_AGE_HOURS")
        initdata_max_age_hours = int(_raw) if _raw else None
    rid_salt = rid_salt if rid_salt is not None else os.getenv("RID_HASH_SALT", "validation-stage1")

    if default_survey_state not in (SURVEY_OPEN, SURVEY_CLOSED):
        raise ValueError(f"SURVEY_STATE must be 'open' or 'closed', got {default_survey_state!r}")

    items_list, items_by_id = items_loader.load_and_validate(items_path)

    app = FastAPI(title="Validation Mini App (Stage 2)")
    app.state.db_path = db_path
    app.state.items_list = items_list
    app.state.items_by_id = items_by_id
    app.state.dev_mode = dev_mode

    def get_conn():
        return db.connect(app.state.db_path)

    _seed = get_conn()
    try:
        if db.get_config(_seed, "survey_state") is None:
            db.set_config(_seed, "survey_state", default_survey_state)
    finally:
        _seed.close()

    def current_state(conn):
        return db.get_config(conn, "survey_state", SURVEY_OPEN)

    def resolve_rater(request: Request) -> str:
        """Return the hashed rater id, or raise 401. DEV_MODE accepts ?rid=."""
        if dev_mode:
            rid = request.query_params.get("rid")
            if not rid:
                raise HTTPException(status_code=401, detail="DEV_MODE: missing ?rid")
            return hash_rater_id(rid, rid_salt)
        init_data = request.headers.get("X-Init-Data") or request.query_params.get("init_data")
        try:
            fields = auth.validate_init_data(init_data, bot_token,
                                             max_age_hours=initdata_max_age_hours)
            user = json.loads(fields["user"])
            raw_id = str(user["id"])
        except (auth.InitDataError, KeyError, ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=401, detail=f"invalid initData: {e}")
        return hash_rater_id(raw_id, rid_salt)

    def require_admin(request: Request):
        token = request.headers.get("X-Admin-Token")
        if not admin_token or not token or not _consteq(token, admin_token):
            raise HTTPException(status_code=401, detail="admin auth failed")

    # ---- public status ------------------------------------------------------
    @app.get("/status")
    def status():
        conn = get_conn()
        try:
            return {"state": current_state(conn)}
        finally:
            conn.close()

    # ---- items --------------------------------------------------------------
    @app.get("/items")
    def get_items(request: Request):
        conn = get_conn()
        try:
            state = current_state(conn)
            if state == SURVEY_CLOSED:
                return {"state": SURVEY_CLOSED, "items": []}
            rater_id = resolve_rater(request)
            now = time.time()
            ordered_ids = db.get_or_assign_order(
                conn, rater_id, [it["item_id"] for it in app.state.items_list], now)
            total = len(ordered_ids)
            out = []
            for idx, iid in enumerate(ordered_ids):
                item = app.state.items_by_id[iid]
                pos = db.get_or_assign_position(conn, rater_id, iid, now)
                opt1_text = (item["answer_rag_ru"] if pos["option_1_arm"] == "rag"
                             else item["answer_vanilla_ru"])
                opt2_text = (item["answer_rag_ru"] if pos["option_2_arm"] == "rag"
                             else item["answer_vanilla_ru"])
                out.append({
                    "item_id": iid,
                    "index": idx,
                    "total": total,
                    "case_ru": item["case_ru"],
                    "option_1_text": opt1_text,
                    "option_2_text": opt2_text,
                    "routed_specialty": item["routed_specialty"],
                    "available_specialties": item["available_specialties"],
                    "already_done": db.is_done(conn, rater_id, iid),
                })
            return {"state": SURVEY_OPEN, "rater_id": rater_id, "items": out}
        finally:
            conn.close()

    # ---- submit -------------------------------------------------------------
    @app.post("/submit")
    def submit(body: SubmitBody, request: Request):
        conn = get_conn()
        try:
            if current_state(conn) == SURVEY_CLOSED:
                raise HTTPException(status_code=409, detail="survey closed")
            rater_id = resolve_rater(request)
            if body.item_id not in app.state.items_by_id:
                raise HTTPException(status_code=404, detail="unknown item_id")
            # Ensure a position map exists (rater could POST without a prior GET).
            db.get_or_assign_position(conn, rater_id, body.item_id, time.time())
            db.upsert_submission(
                conn,
                rater_id=rater_id,
                item_id=body.item_id,
                preference=body.preference,
                safety_flag_opt1=body.safety_flag_opt1,
                safety_flag_opt2=body.safety_flag_opt2,
                routing_judgment=body.routing_judgment,
                client_ts=body.client_ts,
                server_ts=time.time(),
            )
            return {"ok": True, "item_id": body.item_id}
        finally:
            conn.close()

    # ---- progress -----------------------------------------------------------
    @app.get("/progress")
    def progress():
        conn = get_conn()
        try:
            counts = db.completed_count_by_item(conn)
            out = [
                {"item_id": item["item_id"], "completed_raters": counts.get(item["item_id"], 0)}
                for item in app.state.items_list
            ]
            return {"items": out, "target_per_item": 3}
        finally:
            conn.close()

    # ---- admin toggle -------------------------------------------------------
    @app.post("/admin/open")
    def admin_open(request: Request):
        require_admin(request)
        conn = get_conn()
        try:
            db.set_config(conn, "survey_state", SURVEY_OPEN)
            return {"state": SURVEY_OPEN}
        finally:
            conn.close()

    @app.post("/admin/close")
    def admin_close(request: Request):
        require_admin(request)
        conn = get_conn()
        try:
            db.set_config(conn, "survey_state", SURVEY_CLOSED)
            return {"state": SURVEY_CLOSED}
        finally:
            conn.close()

    # ---- frontend (mounted last so API routes win) --------------------------
    if FRONTEND_DIR.is_dir():
        @app.get("/")
        def index():
            return FileResponse(str(FRONTEND_DIR / "index.html"))

        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    return app


def _consteq(a, b):
    import hmac as _hmac
    return _hmac.compare_digest(a, b)


app = create_app()
