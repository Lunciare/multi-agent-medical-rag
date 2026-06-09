"use strict";

// ---- Telegram WebApp wiring (with plain-browser fallback) -------------------
const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;

function applyTheme() {
  if (!tg || !tg.themeParams) return;
  const root = document.documentElement;
  for (const [k, v] of Object.entries(tg.themeParams)) {
    // themeParams keys look like "bg_color"; expose as --tg-theme-bg-color.
    root.style.setProperty(`--tg-theme-${k.replace(/_/g, "-")}`, v);
  }
}

// Auth wiring. In Telegram, the server validates the signed initData string and
// derives the rater id from it; the client just forwards initData on every request.
// Outside Telegram (DEV_MODE on the server), we fall back to ?rid=devNN.
function devRid() {
  const q = new URLSearchParams(window.location.search);
  return q.get("rid");
}
function authQuery() {
  const rid = devRid();
  return rid ? `rid=${encodeURIComponent(rid)}` : "";
}
function authHeaders() {
  const h = {};
  if (tg && tg.initData) h["X-Init-Data"] = tg.initData;
  return h;
}
function withQuery(path) {
  const q = authQuery();
  return q ? `${path}${path.includes("?") ? "&" : "?"}${q}` : path;
}

// ---- App state --------------------------------------------------------------
const state = {
  items: [],        // all items from server (ordered)
  queue: [],        // indices of not-yet-done items
  cursor: 0,        // position within queue
  screen: "welcome",
};

const $ = (id) => document.getElementById(id);

// Canonical specialty -> Russian display label (display only; stored data stays canonical).
// Mirrors backend/specialties.py RU_DISPLAY. The routing screen shows the routed specialist
// and the available choice set to Russian-speaking raters.
const SPECIALTY_RU = {
  cardiology: "Кардиолог",
  endocrinology: "Эндокринолог",
  gastroenterology: "Гастроэнтеролог",
  infectious_diseases: "Инфекционист",
};
const ruSpecialty = (canonical) => SPECIALTY_RU[canonical] || canonical;

function show(screenId) {
  for (const s of document.querySelectorAll(".screen")) s.classList.add("hidden");
  $(screenId).classList.remove("hidden");
}

function showError(msg) {
  const el = $("error");
  el.textContent = msg;
  el.classList.remove("hidden");
}
function clearError() { $("error").classList.add("hidden"); }

// ---- MainButton helpers -----------------------------------------------------
function setMainButton(text, handler) {
  if (tg && tg.MainButton) {
    tg.MainButton.offClick(mainHandler);
    mainHandler = handler;
    tg.MainButton.setText(text);
    tg.MainButton.onClick(mainHandler);
    tg.MainButton.show();
  } else {
    // Plain-browser fallback: a fixed bottom button.
    let btn = $("fallback-main");
    if (!btn) {
      btn = document.createElement("button");
      btn.id = "fallback-main";
      btn.style.cssText =
        "position:fixed;left:16px;right:16px;bottom:16px;padding:14px;font-size:1rem;" +
        "border:none;border-radius:12px;background:var(--button);color:var(--button-text);";
      document.body.appendChild(btn);
    }
    btn.textContent = text;
    btn.onclick = handler;
    btn.style.display = "block";
  }
}
let mainHandler = () => {};

// ---- Survey state + data load ----------------------------------------------
function showClosed() {
  show("screen-closed");
  if (tg && tg.MainButton) tg.MainButton.hide();
  else { const b = $("fallback-main"); if (b) b.style.display = "none"; }
}

async function fetchState() {
  const res = await fetch("/status");
  if (!res.ok) throw new Error(`GET /status failed: ${res.status}`);
  return (await res.json()).state;
}

// Returns "closed" if the survey is closed; otherwise loads items and returns "open".
async function loadItems() {
  const res = await fetch(withQuery("/items"), { headers: authHeaders() });
  if (!res.ok) throw new Error(`GET /items failed: ${res.status}`);
  const data = await res.json();
  if (data.state === "closed") return "closed";
  state.items = data.items;
  state.queue = state.items
    .map((it, i) => (it.already_done ? null : i))
    .filter((i) => i !== null);
  state.cursor = 0;
  return "open";
}

// ---- Screens ----------------------------------------------------------------
function startWelcome() {
  show("screen-welcome");
  const remaining = state.queue.length;
  const total = state.items.length;
  $("welcome-status").textContent =
    remaining === total
      ? `Случаев для оценки: ${total}.`
      : `Осталось оценить: ${remaining} из ${total} (часть уже выполнена).`;
  if (remaining === 0) {
    setMainButton("Всё уже оценено", () => show("screen-done"));
  } else {
    setMainButton("Начать", () => nextCase());
  }
}

// Holds the in-progress judgment for the current item across compare->routing.
let current = null;

function nextCase() {
  clearError();
  if (state.cursor >= state.queue.length) {
    show("screen-done");
    if (tg && tg.MainButton) tg.MainButton.hide();
    else { const b = $("fallback-main"); if (b) b.style.display = "none"; }
    return;
  }
  const item = state.items[state.queue[state.cursor]];
  current = { item, preference: null, safety1: false, safety2: false, routing: null };
  renderCompare(item);
}

function renderCompare(item) {
  show("screen-compare");
  $("case-i").textContent = item.index + 1;
  $("case-n").textContent = item.total;
  $("case-text").textContent = item.case_ru;
  $("answer-1").textContent = item.option_1_text;
  $("answer-2").textContent = item.option_2_text;
  $("safety-1").checked = false;
  $("safety-2").checked = false;
  for (const r of document.querySelectorAll('input[name="preference"]')) r.checked = false;
  setupToggle();
  setMainButton("Далее", onCompareNext);
}

function onCompareNext() {
  const pref = document.querySelector('input[name="preference"]:checked');
  if (!pref) { showError("Пожалуйста, выберите один из вариантов."); return; }
  clearError();
  current.preference = pref.value;
  current.safety1 = $("safety-1").checked;
  current.safety2 = $("safety-2").checked;
  renderRouting(current.item);
}

function renderRouting(item) {
  show("screen-routing");
  // Display in Russian; the stored/canonical values are unchanged on the server.
  $("routed-specialty").textContent = ruSpecialty(item.routed_specialty);
  $("available-specialties").textContent =
    (item.available_specialties || []).map(ruSpecialty).join(", ");
  for (const r of document.querySelectorAll('input[name="routing"]')) r.checked = false;
  setMainButton("Отправить", onSubmit);
}

async function onSubmit() {
  const routing = document.querySelector('input[name="routing"]:checked');
  if (!routing) { showError("Пожалуйста, оцените выбор специалиста."); return; }
  clearError();
  current.routing = routing.value;

  const body = {
    item_id: current.item.item_id,
    preference: current.preference,
    safety_flag_opt1: current.safety1,
    safety_flag_opt2: current.safety2,
    routing_judgment: current.routing,
    client_ts: new Date().toISOString(),
  };
  let res;
  try {
    res = await fetch(withQuery("/submit"), {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    showError("Не удалось отправить. Проверьте соединение и попробуйте снова.");
    return;
  }
  // Survey closed mid-session: route to the closed screen. Already-submitted
  // items are safe on the server; only this in-progress item is dropped.
  if (res.status === 409) { showClosed(); return; }
  if (!res.ok) {
    showError("Не удалось отправить. Проверьте соединение и попробуйте снова.");
    return;
  }
  state.cursor += 1;
  nextCase();
}

// ---- Narrow-screen A/B toggle ----------------------------------------------
function setupToggle() {
  const answers = $("answers");
  const toggle = $("answers-toggle");
  const narrow = window.matchMedia("(max-width: 560px)").matches;
  if (!narrow) {
    answers.classList.remove("toggled");
    toggle.classList.add("hidden");
    return;
  }
  toggle.classList.remove("hidden");
  answers.classList.add("toggled");
  const setActive = (target) => {
    $("col-1").classList.toggle("show", target === "1");
    $("col-2").classList.toggle("show", target === "2");
    for (const b of toggle.querySelectorAll(".toggle-btn"))
      b.classList.toggle("active", b.dataset.target === target);
  };
  for (const b of toggle.querySelectorAll(".toggle-btn"))
    b.onclick = () => setActive(b.dataset.target);
  setActive("1");
}

// ---- Boot -------------------------------------------------------------------
async function boot() {
  if (tg) { applyTheme(); tg.ready(); tg.expand(); }
  try {
    // Check survey state first so a closed survey shows only the closed screen.
    if ((await fetchState()) === "closed") { showClosed(); return; }
    if ((await loadItems()) === "closed") { showClosed(); return; }
  } catch (e) {
    showError("Не удалось загрузить случаи. " + e.message);
    return;
  }
  startWelcome();
}

boot();
