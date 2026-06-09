import csv as _csv
import io
import sys
import tempfile
from pathlib import Path

_MINIAPP = Path(__file__).resolve().parent.parent
if str(_MINIAPP) not in sys.path:
    sys.path.insert(0, str(_MINIAPP))

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as app_module  # noqa: E402
from backend import auth as auth_module  # noqa: E402
from backend import db as db_module  # noqa: E402
from backend import export_csv  # noqa: E402
from backend import items_loader  # noqa: E402

ITEMS_PATH = _MINIAPP / "data" / "items.dummy.json"
SALT = "testsalt"
ADMIN_TOKEN = "admin-secret-123"
BOT_TOKEN = "123456:TEST-bot-token-for-smoke"

_checks = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    _checks.append((label, bool(condition)))
    return bool(condition)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run():
    tmp_db = Path(tempfile.mkdtemp()) / "smoke.db"
    app = app_module.create_app(
        db_path=tmp_db, items_path=ITEMS_PATH, dev_mode=True,
        admin_token=ADMIN_TOKEN, default_survey_state="open", rid_salt=SALT)
    client = TestClient(app)
    rid = "dev42"
    rid_hashed = app_module.hash_rater_id(rid, SALT)
    items_by_id = {it["item_id"]: it for it in app.state.items_list}

    # ---------------------------------------------------------------- main flow
    section("STEP 1: GET /status + GET /items (DEV_MODE, ?rid)")
    st = client.get("/status").json()
    print(f"  /status -> {st}")
    check("survey starts open", st["state"] == "open")

    r = client.get("/items", params={"rid": rid})
    assert r.status_code == 200, r.text
    body = r.json()
    items = body["items"]
    print(f"  state={body['state']} rater_id(hashed)={body['rater_id']}")
    for it in items:
        print(f"    pos={it['index']} {it['item_id']} done={it['already_done']} "
              f"routed={it['routed_specialty']}")
    check("GET /items reports open", body["state"] == "open")
    check("GET /items returned all dummy items", len(items) == len(app.state.items_list))
    check("all items start not-done", all(not it["already_done"] for it in items))

    conn = db_module.connect(tmp_db)
    item0, item1 = items[0], items[1]
    pos0 = db_module.get_position(conn, rid_hashed, item0["item_id"])
    pos1 = db_module.get_position(conn, rid_hashed, item1["item_id"])
    print(f"  position map item0 ({item0['item_id']}): {pos0}")
    print(f"  position map item1 ({item1['item_id']}): {pos1}")

    section("STEP 2: POST two submissions")
    sub0 = {"item_id": item0["item_id"], "preference": "opt1_strong",
            "safety_flag_opt1": False, "safety_flag_opt2": False,
            "routing_judgment": "correct", "client_ts": "2026-05-31T10:00:00Z"}
    sub1 = {"item_id": item1["item_id"], "preference": "opt2_weak",
            "safety_flag_opt1": True, "safety_flag_opt2": False,
            "routing_judgment": "incorrect", "client_ts": "2026-05-31T10:01:00Z"}
    for s in (sub0, sub1):
        resp = client.post("/submit", params={"rid": rid}, json=s)
        print(f"  POST {s['item_id']} pref={s['preference']} -> {resp.status_code} {resp.json()}")
        assert resp.status_code == 200, resp.text

    section("STEP 3: re-POST item0 (idempotency)")
    resp = client.post("/submit", params={"rid": rid}, json=sub0)
    print(f"  re-POST {sub0['item_id']} -> {resp.status_code} {resp.json()}")
    n_rows = conn.execute("SELECT COUNT(*) AS n FROM submission").fetchone()["n"]
    print(f"  submission row count after re-POST = {n_rows}")
    check("re-POST does not duplicate (exactly 2 rows)", n_rows == 2)

    section("STEP 4: GET /progress")
    prog = client.get("/progress").json()
    for p in prog["items"]:
        print(f"  {p['item_id']}: completed_raters={p['completed_raters']}")
    done_map = {p["item_id"]: p["completed_raters"] for p in prog["items"]}
    check("progress counts item0 done by 1 rater", done_map[item0["item_id"]] == 1)
    check("progress counts item1 done by 1 rater", done_map[item1["item_id"]] == 1)

    section("STEP 5: resume GET /items marks done + order is stable")
    r2 = client.get("/items", params={"rid": rid}).json()
    order1 = [it["item_id"] for it in items]
    order2 = [it["item_id"] for it in r2["items"]]
    print(f"  order GET#1: {order1}")
    print(f"  order GET#2: {order2}")
    check("per-rater item order stable across resume", order1 == order2)
    items2 = {it["item_id"]: it for it in r2["items"]}
    check("item0 now marked done", items2[item0["item_id"]]["already_done"] is True)
    check("item1 now marked done", items2[item1["item_id"]]["already_done"] is True)
    check("position map item0 unchanged on resume",
          db_module.get_position(conn, rid_hashed, item0["item_id"]) == pos0)

    section("STEP 6: export CSV (de-blinded)")
    csv_text = export_csv.export(conn, items_by_id, out_path=None)
    print("----- CSV BEGIN -----")
    print(csv_text, end="")
    print("----- CSV END -----")
    rows = list(_csv.DictReader(io.StringIO(csv_text)))
    header = list(rows[0].keys()) if rows else []
    check("CSV has exactly 2 data rows", len(rows) == 2)
    check("CSV columns match expected schema", header == export_csv.CSV_COLUMNS)
    by_item = {row["item_id"]: row for row in rows}

    expected0 = 2 if pos0["option_1_arm"] == "rag" else -2
    actual0 = int(by_item[item0["item_id"]]["pref_toward_rag"])
    print(f"  item0: opt1_arm={pos0['option_1_arm']} pref=opt1_strong "
          f"-> expected pref_toward_rag={expected0}, actual={actual0}")
    check("item0 pref_toward_rag sign resolves correctly", actual0 == expected0)

    expected1 = -1 if pos1["option_1_arm"] == "rag" else 1
    actual1 = int(by_item[item1["item_id"]]["pref_toward_rag"])
    print(f"  item1: opt1_arm={pos1['option_1_arm']} pref=opt2_weak "
          f"-> expected pref_toward_rag={expected1}, actual={actual1}")
    check("item1 pref_toward_rag sign resolves correctly", actual1 == expected1)

    # ----------------------------------------------------- admin + closed state
    section("STEP 7: admin auth + closed-state behavior")
    bad = client.post("/admin/close")  # no token
    print(f"  POST /admin/close (no token) -> {bad.status_code}")
    check("/admin/close rejected without token (401)", bad.status_code == 401)
    bad2 = client.post("/admin/close", headers={"X-Admin-Token": "wrong"})
    print(f"  POST /admin/close (wrong token) -> {bad2.status_code}")
    check("/admin/close rejected with wrong token (401)", bad2.status_code == 401)

    ok = client.post("/admin/close", headers={"X-Admin-Token": ADMIN_TOKEN})
    print(f"  POST /admin/close (good token) -> {ok.status_code} {ok.json()}")
    check("/admin/close works with token", ok.status_code == 200 and ok.json()["state"] == "closed")

    st_closed = client.get("/status").json()
    print(f"  /status -> {st_closed}")
    check("status reports closed", st_closed["state"] == "closed")

    closed_items = client.get("/items", params={"rid": rid}).json()
    print(f"  GET /items while closed -> {closed_items}")
    check("closed GET /items reports closed + no items",
          closed_items["state"] == "closed" and closed_items["items"] == [])

    rid_new = "dev99"
    rid_new_hashed = app_module.hash_rater_id(rid_new, SALT)
    closed_sub = client.post("/submit", params={"rid": rid_new},
                             json={"item_id": item0["item_id"], "preference": "tie",
                                   "safety_flag_opt1": False, "safety_flag_opt2": False,
                                   "routing_judgment": "correct", "client_ts": "x"})
    print(f"  POST /submit while closed -> {closed_sub.status_code} {closed_sub.json()}")
    check("closed POST /submit returns 409", closed_sub.status_code == 409)
    stored = conn.execute(
        "SELECT COUNT(*) AS n FROM submission WHERE rater_id = ?", (rid_new_hashed,)
    ).fetchone()["n"]
    print(f"  rows stored for the closed-time submitter = {stored}")
    check("closed POST stored nothing", stored == 0)

    reopen = client.post("/admin/open", headers={"X-Admin-Token": ADMIN_TOKEN})
    print(f"  POST /admin/open (good token) -> {reopen.status_code} {reopen.json()}")
    check("/admin/open works with token", reopen.status_code == 200 and reopen.json()["state"] == "open")
    conn.close()

    # --------------------------------------------------- initData validation
    section("STEP 8: Telegram initData validation (good vs tampered)")
    good = auth_module.build_init_data(
        BOT_TOKEN,
        {"auth_date": "1700000000", "query_id": "AAEx", "user": '{"id":555,"first_name":"T"}'},
    )
    print(f"  built initData: {good}")
    try:
        fields = auth_module.validate_init_data(good, BOT_TOKEN)
        ok_validate = True
    except auth_module.InitDataError as e:
        ok_validate = False
        print(f"  unexpected failure: {e}")
    check("valid initData passes validation", ok_validate)

    tampered = good.replace("first_name", "frst_name") if "first_name" in good else good + "x"
    try:
        auth_module.validate_init_data(tampered, BOT_TOKEN)
        ok_tamper = False
    except auth_module.InitDataError as e:
        ok_tamper = True
        print(f"  tampered initData rejected: {e}")
    check("tampered initData fails validation", ok_tamper)

    # Same, but end-to-end through a non-dev app instance.
    tmp_db2 = Path(tempfile.mkdtemp()) / "smoke2.db"
    app_prod = app_module.create_app(
        db_path=tmp_db2, items_path=ITEMS_PATH, dev_mode=False,
        bot_token=BOT_TOKEN, admin_token=ADMIN_TOKEN, default_survey_state="open", rid_salt=SALT)
    client_prod = TestClient(app_prod)
    r_noauth = client_prod.get("/items")
    print(f"  prod GET /items (no initData) -> {r_noauth.status_code}")
    check("non-dev GET /items without initData -> 401", r_noauth.status_code == 401)
    r_good = client_prod.get("/items", headers={"X-Init-Data": good})
    print(f"  prod GET /items (good initData) -> {r_good.status_code}")
    check("non-dev GET /items with good initData -> 200", r_good.status_code == 200)
    r_bad = client_prod.get("/items", headers={"X-Init-Data": tampered})
    print(f"  prod GET /items (tampered initData) -> {r_bad.status_code}")
    check("non-dev GET /items with tampered initData -> 401", r_bad.status_code == 401)

    # ------------------------------------------------ malformed items rejection
    section("STEP 9: items.json startup validation rejects malformed file")
    bad_path = Path(tempfile.mkdtemp()) / "bad_items.json"
    bad_path.write_text(
        '[{"item_id":"x","case_ru":"c","answer_rag_ru":"a","answer_vanilla_ru":"b",'
        '"routed_specialty":"cardiology","gold_specialty":"dermatology",'
        '"available_specialties":["cardiology"]}]',
        encoding="utf-8",
    )
    try:
        items_loader.load_and_validate(bad_path)
        rejected = False
        msg = ""
    except items_loader.ItemsValidationError as e:
        rejected = True
        msg = str(e)
        print(f"  rejected as expected: {msg}")
    check("malformed items file is rejected at load", rejected)
    check("rejection message names the bad field (gold_specialty)",
          "gold_specialty" in msg)

    # Also confirm create_app itself refuses to start on a bad items file.
    try:
        app_module.create_app(db_path=Path(tempfile.mkdtemp()) / "x.db",
                              items_path=bad_path, dev_mode=True)
        startup_rejected = False
    except items_loader.ItemsValidationError:
        startup_rejected = True
    check("create_app fails startup on bad items file", startup_rejected)

    # ---------------------------------------------- DB_PATH override (Stage 3)
    section("STEP 10: DB_PATH override creates the DB outside the repo")
    # Point DB_PATH at a NESTED dir that does not exist yet, to also prove the
    # parent directory is auto-created on startup.
    ext_dir = Path(tempfile.mkdtemp()) / "miniapp-data" / "nested"
    ext_db = ext_dir / "app.sqlite3"
    default_db = _MINIAPP / "data" / "validation.db"
    default_pre = default_db.exists()  # so we don't blame a pre-existing leftover
    print(f"  DB_PATH = {ext_db}")
    print(f"  parent exists before boot? {ext_dir.exists()}")
    app_ext = app_module.create_app(
        db_path=ext_db, items_path=ITEMS_PATH, dev_mode=True, rid_salt=SALT)
    client_ext = TestClient(app_ext)
    # A request that writes (assigns item order + position map) for a rater.
    client_ext.get("/items", params={"rid": "devDB"})
    print(f"  parent created after boot?  {ext_dir.exists()}")
    print(f"  DB file exists at DB_PATH?   {ext_db.exists()}")
    check("DB parent dir auto-created on startup", ext_dir.exists())
    check("DB created exactly at DB_PATH", ext_db.is_file())
    check("override did not create the repo default DB path",
          default_db.exists() == default_pre)

    # ----------------------------------------------------------------- summary
    section("RESULT")
    failed = [lbl for lbl, ok in _checks if not ok]
    print(f"{len(_checks) - len(failed)}/{len(_checks)} checks passed")
    if failed:
        print("FAILED CHECKS:")
        for lbl in failed:
            print(f"  - {lbl}")
    print("=" * 70)
    return not failed


def test_smoke():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
