"""
AquaLedger — Water Footprint Accounting & Conservation Planner
Single-file Streamlit application. No API keys. 100% open-source dependencies.

Run:
    streamlit run app.py

Architecture (all in this file, cleanly separated):
  * Domain engine  : pure functions, no Streamlit calls (balance, cooling-tower,
                     NPV/payback, Monte Carlo, portfolio optimizer, validator).
  * Data layer     : session-scoped tables (records/sites/processes/projects/
                     scenarios/audit/attachments/config) with stable IDs,
                     timestamps, provenance, schema & rule versions.
                     Production path: SQLite -> PostgreSQL (see Governance).
  * UI layer       : Overview (live telemetry simulation), Water Ledger,
                     Import & Validate, Conservation Lab, Review & Approve,
                     Reports & Share, Governance (roles, audit, dictionary,
                     lineage, backup/restore, live self-check suite).

All data is SYNTHETIC and labeled as such. Decision support only —
every material action is audited and requires human approval.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html as html_mod
import io
import json
import math
import urllib.parse
import zipfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    import qrcode
    HAS_QR = True
except Exception:
    HAS_QR = False

APP_NAME = "AquaLedger"
TAGLINE = "Water Footprint Accounting & Conservation Planner"
VERSION = "1.0.0"
RULES_VERSION = "WBA-1.0"
SCHEMA_VERSION = "1.0"

COLORWAY = ["#22d3ee", "#3b82f6", "#34d399", "#fbbf24", "#e879f9", "#a78bfa", "#fb7185", "#2dd4bf"]
SRC_COLORS = {"Municipal": "#3b82f6", "Groundwater": "#22d3ee", "Surface": "#2dd4bf", "Harvested": "#34d399"}
FATE_COLORS = {"Consumed": "#fbbf24", "Discharged": "#60a5fa", "Storage": "#a78bfa"}
STATUS_META = {
    "Draft": ("#94a3b8", "◔"), "Validated": ("#22d3ee", "◑"), "In Review": ("#fbbf24", "◕"),
    "Approved": ("#34d399", "●"), "Rejected": ("#fb7185", "✕"), "Archived": ("#64748b", "▫"),
}
STRESS_META = {"Low": ("#34d399", 1.0), "Medium": ("#fbbf24", 1.6), "High": ("#fb923c", 2.6), "Extremely high": ("#fb7185", 4.0)}
METER_UNC = {"metered": 2.0, "calculated": 8.0, "estimated": 15.0}

NAV = [("overview", "🌊 Overview"), ("ledger", "📒 Water Ledger"), ("import", "📥 Import & Validate"),
       ("lab", "🧪 Conservation Lab"), ("review", "✅ Review & Approve"), ("reports", "📤 Reports & Share"),
       ("governance", "🛡️ Governance")]
NAV_LABELS = [l for _, l in NAV]
NAV_BY_KEY = dict(NAV)

ROLES = ["Administrator", "Analyst", "Reviewer", "Stakeholder"]
ROLE_ICON = {"Administrator": "👑", "Analyst": "🧑‍🔬", "Reviewer": "🧾", "Stakeholder": "👁️"}
PERMS = {
    "Administrator": {"view", "import", "edit", "submit", "review", "approve", "override", "archive", "export", "share", "scenario", "admin"},
    "Analyst":       {"view", "import", "edit", "submit", "export", "share", "scenario"},
    "Reviewer":      {"view", "review", "approve", "export", "share"},
    "Stakeholder":   {"view", "export", "share"},
}
PERM_ROWS = ["view", "import", "edit", "submit", "review", "approve", "override", "archive", "export", "share", "scenario", "admin"]

TEMPLATE_COLUMNS = ["site_id", "process_id", "period", "municipal_m3", "groundwater_m3", "surface_m3",
                    "harvested_m3", "recycled_m3", "consumption_m3", "discharge_m3", "storage_change_m3",
                    "output_units", "metering", "notes"]
NUM_COLS = ["municipal_m3", "groundwater_m3", "surface_m3", "harvested_m3", "recycled_m3",
            "consumption_m3", "discharge_m3", "storage_change_m3", "output_units"]

LAB_DEFAULTS = {"lab_budget": 1_600_000, "lab_tariff": 1.85, "lab_rate": 8, "lab_horizon": 12,
                "lab_n": 2000, "lab_seed": 42, "lab_unc": 1.0, "lab_target": 15,
                "lab_obj": "Max NPV", "lab_risk": "P50 (median)",
                "proj_P-COOL": True, "proj_P-ADX": False, "proj_P-LEAK": True, "proj_P-RO": False,
                "proj_P-RAIN": True, "proj_P-CIP": True, "proj_P-LOOP": False, "proj_P-METER": True}

# ----------------------------------------------------------------------------------
# Compatibility helpers (graceful degradation on older Streamlit builds)
# ----------------------------------------------------------------------------------
def rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()

def goto(page_key):
    # The "nav" radio is already instantiated when buttons run, so defer the change to the next run.
    st.session_state.nav_goto = NAV_BY_KEY[page_key]
    rerun()

def toggle_w(label, key, help=None):
    fn = getattr(st, "toggle", None)
    if fn:
        return fn(label, key=key, help=help)
    return st.checkbox(label, key=key, help=help)

def link_btn(label, url, wide=True):
    fn = getattr(st, "link_button", None)
    if fn:
        fn(label, url, use_container_width=wide)
    else:
        st.markdown(f'<a class="ak-link" href="{html_mod.escape(url)}" target="_blank">{label}</a>', unsafe_allow_html=True)

@contextmanager
def popover_w(label):
    fn = getattr(st, "popover", None)
    if fn:
        with fn(label, use_container_width=True) as p:
            yield p
    else:
        with st.expander(label) as p:
            yield p

def query_params() -> dict:
    try:
        return {k: st.query_params[k] for k in st.query_params}
    except Exception:
        try:
            return {k: v[0] for k, v in st.experimental_get_query_params().items()}
        except Exception:
            return {}

# ----------------------------------------------------------------------------------
# Formatting / small utilities
# ----------------------------------------------------------------------------------
def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def fmt_n(x): return f"{float(x):,.0f}"
def fmt_1(x): return f"{float(x):,.1f}"
def money(x): return f"${float(x):,.0f}"
def pctf(x): return f"{float(x):.1f}%"

def compact(x):
    x = float(x)
    if abs(x) >= 1e6: return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.1f}k"
    return f"{x:,.0f}"

def badge(text, color):
    return (f'<span class="badge" style="background:{color}1e;border-color:{color}66;color:{color}">'
            f'{html_mod.escape(str(text))}</span>')

def status_badge(s):
    c, i = STATUS_META.get(s, ("#94a3b8", "•"))
    return badge(f"{i} {s}", c)

def spark_svg(values, color, gid, w=150, h=34, fill=True):
    vals = [float(v) for v in values] or [0.0]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = max(len(vals) - 1, 1)
    pts = [(i / n * w, h - 4 - (v - lo) / rng * (h - 8)) for i, v in enumerate(vals)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"M0,{h} L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f" L{w},{h} Z"
    fp = f'<path d="{area}" fill="url(#g{gid})"/>' if fill else ""
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><linearGradient id="g{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity="0.35"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'{fp}<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')

def fig_layout(fig, height=340, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cfe6f5", size=12),
        margin=dict(l=10, r=10, t=36, b=10), colorway=COLORWAY,
        showlegend=legend, legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor="#0a2236", bordercolor="#22d3ee", font=dict(color="#e8f4fb")),
    )
    fig.update_xaxes(gridcolor="rgba(148,190,220,0.12)", zerolinecolor="rgba(148,190,220,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,190,220,0.12)", zerolinecolor="rgba(148,190,220,0.2)")
    return fig

PLOT_CFG = {"displayModeBar": False}

def copy_button(text, label, key):
    payload = json.dumps(str(text)).replace("</", "<\\/")
    tpl = """
<div style="font-family:Inter,sans-serif">
<button id="cb" style="width:100%;padding:9px 12px;border-radius:10px;cursor:pointer;
 border:1px solid rgba(34,211,238,.45);background:linear-gradient(135deg,rgba(14,165,233,.22),rgba(37,99,235,.22));
 color:#dff6ff;font-weight:600;font-size:13px">__LABEL__</button>
<script>
const b=document.getElementById('cb');const original='__LABEL__';
b.onclick=async()=>{const t=__PAYLOAD__;
 try{await navigator.clipboard.writeText(t);}catch(e){
  const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);
  a.select();document.execCommand('copy');a.remove();}
 b.textContent='✓ Copied to clipboard';b.style.borderColor='#34d399';
 setTimeout(()=>{b.textContent=original;b.style.borderColor='rgba(34,211,238,.45)';},1800);};
</script></div>"""
    components.html(tpl.replace("__PAYLOAD__", payload).replace("__LABEL__", label), height=48)

# ----------------------------------------------------------------------------------
# DOMAIN ENGINE — pure, deterministic, UI-independent (rules version WBA-1.0)
# ----------------------------------------------------------------------------------
def balance_tolerance(w: float) -> float:
    """Accept |W − (C + D + ΔS)| ≤ max(2% of withdrawal, 5 m³)."""
    return max(0.02 * abs(float(w)), 5.0)

def balance_check(rec: dict):
    r = float(rec["withdrawal_m3"]) - (float(rec["consumption_m3"]) + float(rec["discharge_m3"]) + float(rec["storage_change_m3"]))
    tol = balance_tolerance(rec["withdrawal_m3"])
    state = "balanced" if abs(r) <= tol else ("warning" if abs(r) <= 2 * tol else "error")
    return r, tol, state

def cooling_savings(evap_m3: float, coc_old: float = 3.0, coc_new: float = 6.0) -> float:
    """Blowdown = E/(CoC−1). Savings = E·(1/(old−1) − 1/(new−1)).
    Hand check: E=10,000, 3→6 cycles => 10,000·(0.5−0.2) = 3,000 m³/yr."""
    if coc_old <= 1 or coc_new <= 1:
        raise ValueError("Cycles of concentration must be > 1")
    if evap_m3 < 0:
        raise ValueError("Evaporation must be ≥ 0")
    return float(evap_m3) * (1.0 / (coc_old - 1.0) - 1.0 / (coc_new - 1.0))

def annuity(rate: float, years: float) -> float:
    if years <= 0: raise ValueError("years must be > 0")
    return float(years) if abs(rate) < 1e-12 else (1.0 - (1.0 + rate) ** (-years)) / rate

def npv(capex: float, net_annual: float, years: float, rate: float) -> float:
    return -float(capex) + float(net_annual) * annuity(rate, years)

def simple_payback(capex: float, net_annual: float):
    return None if net_annual <= 0 else float(capex) / float(net_annual)

def m3_to_ml(x: float) -> float:
    return float(x) / 1000.0

def pctl(arr):
    a = np.asarray(arr, dtype=float)
    return {"p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "mean": float(np.mean(a))}

def active_df(db) -> pd.DataFrame:
    """Non-archived records within the trailing 12 periods present in the ledger."""
    rows = [r for r in db["records"] if r["status"] != "Archived"]
    if not rows:
        return pd.DataFrame(columns=["period", "site_id", "process_id", "withdrawal_m3",
                                     "consumption_m3", "discharge_m3", "recycled_m3",
                                     "storage_change_m3", "output_units", "metering", "status"])
    df = pd.DataFrame(rows)
    per = sorted(df["period"].unique())[-12:]
    return df[df["period"].isin(per)].copy()

def project_bases(db):
    """Derive each project's base annual saving from the live ledger (traceable)."""
    df = active_df(db)
    proc = {p["id"]: p for p in db["processes"]}
    def ksum(kinds, col):
        ids = [pid for pid, p in proc.items() if p["kind"] in kinds]
        sub = df[df["process_id"].isin(ids)] if len(df) else df
        return float(sub[col].sum()) if len(sub) else 0.0, len(sub)
    E, nE = ksum({"cooling"}, "consumption_m3")
    losses, nL = ksum({"distribution"}, "consumption_m3")
    roW, nR = ksum({"blending", "washing"}, "withdrawal_m3")
    cipW, nC = ksum({"cip"}, "withdrawal_m3")
    machW, nM = ksum({"paper_machine"}, "withdrawal_m3")
    totalW = float(df["withdrawal_m3"].sum()) if len(df) else 0.0
    out = []
    for p in PROJECTS_DEF:
        pid = p["id"]
        if pid == "P-COOL":
            base = cooling_savings(E, 3, 6) if E > 0 else 0.0
            basis = f"Cooling evaporation {fmt_n(E)} m³/yr × (1/(3−1) − 1/(6−1)) = 0.30·E"
            n = nE
        elif pid == "P-ADX":
            base = 0.55 * E
            basis = f"Adiabatic hybrid dry-cooling displaces 55% of cooling evaporation ({fmt_n(E)} m³/yr); energy penalty in co-costs"
            n = nE
        elif pid == "P-LEAK":
            base = 0.60 * losses
            basis = f"District consumptive losses {fmt_n(losses)} m³/yr × 60% detection-and-repair effectiveness"
            n = nL
        elif pid == "P-RO":
            base = 0.60 * 0.25 * roW
            basis = f"RO/blending feed {fmt_n(roW)} m³/yr × 25% reject fraction × 60% recovery"
            n = nR
        elif pid == "P-RAIN":
            base = 18000 * 0.62 * 0.85 * 0.90
            basis = "18,000 m² roof × 0.62 m/yr rainfall × 0.85 runoff × 0.90 capture (synthetic climate inputs)"
            n = len(df)
        elif pid == "P-CIP":
            base = 0.65 * 0.60 * cipW
            basis = f"CIP withdrawal {fmt_n(cipW)} m³/yr × 60% rinse share × 65% reuse"
            n = nC
        elif pid == "P-LOOP":
            base = 0.35 * machW
            basis = f"Paper-machine fresh water {fmt_n(machW)} m³/yr × 35% white-water closure"
            n = nM
        else:  # P-METER
            base = 0.06 * totalW
            basis = f"Submetering + analytics: 6% behavioral/ops saving on {fmt_n(totalW)} m³/yr (halved when ≥5 other projects stack)"
            n = len(df)
        out.append({**p, "base_m3": float(base), "basis": basis, "applicable": base > 0, "n_recs": int(n)})
    return out

def build_sig(projects, selected):
    sel = set(selected)
    sig = []
    for p in sorted(projects, key=lambda x: x["id"]):
        if p["id"] not in sel or not p["applicable"]:
            continue
        overlap = 0.5 if (p["id"] == "P-METER" and len(sel) >= 6) else 1.0
        sig.append((p["id"], p["base_m3"], float(p["capex"]), float(p["opex"]), float(p["life"]),
                    p["sav_lo"], p["sav_hi"], p["cap_lo"], p["cap_hi"], float(p["co"]), overlap))
    return tuple(sig)

def _mc_core(sig, n, seed, tariff, rate, horizon, unc, target_pct, baseline_annual):
    """Triangular Monte Carlo. Savings independent per project; water tariff is one
    shared draw per iteration (correlated across portfolio). Fully seeded."""
    rng = np.random.default_rng(int(seed))
    t = rng.triangular(tariff * 0.85, tariff, tariff * 1.25, n)
    sav = np.zeros(n); cap = np.zeros(n); net = np.zeros(n); npv_a = np.zeros(n)
    per = {}
    base_total = 0.0
    for (pid, base, capex, opex, life, slo, shi, clo, chi, co, ov) in sig:
        b = base * ov
        base_total += b
        lo = max(b * (1 - slo * unc), 0.0)
        hi = max(b * (1 + shi * unc), b + 1e-9)
        s = rng.triangular(lo, max(b, 1e-9), hi, n)
        c = rng.triangular(capex * (1 - clo), capex, capex * (1 + chi), n)
        an = s * t + co - opex
        af = annuity(rate, min(life, horizon))
        pj_npv = -c + an * af
        sav += s; cap += c; net += an; npv_a += pj_npv
        pb = np.where(an > 0, c / np.where(an > 0, an, 1), np.nan)
        pb50 = float(np.nanmedian(pb)) if np.isfinite(np.nanmedian(pb)) else None
        per[pid] = {"sav": pctl(s), "npv": pctl(pj_npv), "capex_p50": float(np.percentile(c, 50)),
                    "payback_p50": pb50, "base": b}
    pb_all = np.where(net > 0, cap / np.where(net > 0, net, 1), np.nan)
    target_m3 = baseline_annual * target_pct / 100.0
    return {"sav": sav, "npv": npv_a, "net": net, "cap": cap,
            "sav_p": pctl(sav), "npv_p": pctl(npv_a), "cap_p": pctl(cap), "net_p": pctl(net),
            "payback_p50": float(np.nanmedian(pb_all)) if np.isfinite(np.nanmedian(pb_all)) else None,
            "prob_target": float((sav >= target_m3).mean()) if baseline_annual > 0 else 0.0,
            "per": per, "base_total": base_total, "target_m3": target_m3}

@st.cache_data(show_spinner=False)
def run_mc(sig, n, seed, tariff, rate, horizon, unc, target_pct, baseline_annual):
    return _mc_core(sig, n, seed, tariff, rate, horizon, unc, target_pct, baseline_annual)

def det_portfolio(projects, selected, tariff, rate, horizon, meter_overlap=1.0):
    rows = []
    for p in projects:
        if p["id"] not in selected or not p["applicable"]:
            continue
        ov = meter_overlap if p["id"] == "P-METER" else 1.0
        s = p["base_m3"] * ov
        an = s * tariff + p["co"] - p["opex"]
        rows.append({"id": p["id"], "name": p["name"], "sav": s, "capex": p["capex"],
                     "net": an, "npv": npv(p["capex"], an, min(p["life"], horizon), rate),
                     "payback": simple_payback(p["capex"], an)})
    return rows

def tornado_data(projects, selected, tariff, rate, horizon, unc, ov_meter):
    base_rows = det_portfolio(projects, selected, tariff, rate, horizon, ov_meter)
    base_npv = sum(r["npv"] for r in base_rows)
    out = []
    for p in projects:
        if p["id"] not in selected or not p["applicable"]:
            continue
        ov = ov_meter if p["id"] == "P-METER" else 1.0
        others = sum(r["npv"] for r in base_rows if r["id"] != p["id"])
        def at(mult):
            s = p["base_m3"] * ov * mult
            an = s * tariff + p["co"] - p["opex"]
            return others + npv(p["capex"], an, min(p["life"], horizon), rate) - base_npv
        out.append({"factor": p["name"], "low": at(1 - p["sav_lo"] * unc), "high": at(1 + p["sav_hi"] * unc)})
    def tar_at(tf):
        return sum(r["npv"] - (-r["capex"] + r["net"] * annuity(rate, 1)) * 0 for r in [])  # placeholder avoided
    # tariff swing on the whole portfolio
    def port_npv(tf):
        return sum(npv(r["capex"], r["sav"] * tf + next((p["co"] - p["opex"] for p in projects if p["id"] == r["id"]), 0),
                       min(next(p["life"] for p in projects if p["id"] == r["id"]), horizon), rate) for r in base_rows)
    out.append({"factor": "Water tariff ($/m³)", "low": port_npv(tariff * 0.85) - base_npv,
                "high": port_npv(tariff * 1.25) - base_npv})
    out.sort(key=lambda r: abs(r["high"] - r["low"]), reverse=True)
    return pd.DataFrame(out), base_npv

def optimize_portfolio(cands, budget, objective, risk):
    """Greedy ratio heuristic under hard constraints (budget, exclusivity groups).
    Every returned solution is verified against all hard constraints."""
    suffix = "p10" if risk.startswith("P10") else "p50"
    pool = [c for c in cands if c["applicable"]]
    if not pool or budget < min(c["capex"] for c in pool):
        return {"ids": [], "feasible": False,
                "reason": "Infeasible: budget is below the cheapest applicable project.", "checks": []}
    if objective == "Max NPV":
        pool = [c for c in pool if c["npv"][suffix] > 0]
        pool.sort(key=lambda c: c["npv"][suffix] / c["capex"], reverse=True)
    elif objective == "Max water saved":
        pool.sort(key=lambda c: c["sav"][suffix] / c["capex"], reverse=True)
    else:
        pool = [c for c in pool if c["payback_p50"] is not None]
        pool.sort(key=lambda c: c["payback_p50"])
    sel, groups, remaining = [], set(), float(budget)
    for c in pool:
        g = c.get("group")
        if g and g in groups:
            continue
        if c["capex"] <= remaining + 1e-6:
            sel.append(c); groups.add(g) if g else None; remaining -= c["capex"]
    spent = sum(c["capex"] for c in sel)
    gs = [c["group"] for c in sel if c.get("group")]
    checks = [("Budget", spent <= budget + 1e-6, f"{money(spent)} ≤ {money(budget)}"),
              ("Exclusivity groups", len(gs) == len(set(gs)), "≤1 project per competing group"),
              ("Applicability", all(c["applicable"] for c in sel), "all selected projects have a live basis")]
    feasible = all(ok for _, ok, _ in checks)
    obj_val = (sum(c["npv"][suffix] for c in sel) if objective == "Max NPV"
               else sum(c["sav"][suffix] for c in sel) if objective == "Max water saved"
               else (max((c["payback_p50"] or 0) for c in sel) if sel else 0))
    return {"ids": [c["id"] for c in sel], "feasible": feasible, "spent": spent,
            "remaining": remaining, "objective_value": obj_val, "checks": checks,
            "reason": "" if feasible else "Constraint verification failed"}

def projection_data(db, mc, selected_projects, target_pct):
    df = active_df(db)
    if not len(df):
        return pd.DataFrame()
    monthly = df.groupby("period")["withdrawal_m3"].sum()
    by_m = {int(p[5:7]): float(v) for p, v in monthly.items()}
    avg_by_m = {m: by_m.get(m, float(monthly.mean())) for m in range(1, 13)}
    start = pd.Period(sorted(df["period"].unique())[-1], freq="M") + 1
    fwd = [str(start + i) for i in range(12)]
    base_total = max(mc["base_total"], 1e-9)
    r10, r50, r90 = mc["sav_p"]["p10"] / base_total, mc["sav_p"]["p50"] / base_total, mc["sav_p"]["p90"] / base_total
    rows = []
    for i, per in enumerate(fwd):
        m = int(per[5:7])
        b = avg_by_m[m]
        ramp = 0.0
        for p in selected_projects:
            ramp += (p["base_m3"] / 12.0) * min(1.0, (i + 1) / p["ramp"])
        rows.append({"period": per, "baseline": b,
                     "mid": b - ramp * r50, "upper": b - ramp * r10, "lower": b - ramp * r90,
                     "target": float(monthly.mean()) * (1 - target_pct / 100.0)})
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------------
# IMPORT VALIDATOR — pure function, row-level actionable errors
# ----------------------------------------------------------------------------------
def validate_import(df: pd.DataFrame, db: dict, mode: str):
    fatal = []
    if df is None or len(df) == 0:
        return {"fatal": ["File is empty — no data rows found."], "rows": [], "issues": pd.DataFrame()}
    missing = [c for c in TEMPLATE_COLUMNS if c not in df.columns]
    if missing:
        fatal.append("Missing required columns: " + ", ".join(missing) + ". Download the template below.")
    extra = [c for c in df.columns if c not in TEMPLATE_COLUMNS]
    if extra:
        fatal.append("NOTE (ignored columns): " + ", ".join(map(str, extra)))
    if any("Missing required" in f for f in fatal):
        return {"fatal": fatal, "rows": [], "issues": pd.DataFrame()}
    sites = {s["id"]: s for s in db["sites"]}
    procs = {p["id"]: p for p in db["processes"]}
    existing = {(r["site_id"], r["process_id"], r["period"]): r["id"]
                for r in db["records"] if r["status"] != "Archived"}
    lock = db["config"]["lock_period"]
    seen, rows, issues = {}, [], []
    for i, raw in df.iterrows():
        rowno = int(i) + 2  # header is row 1
        errs, warns = [], []
        def err(col, msg): errs.append(msg); issues.append({"row": rowno, "column": col, "severity": "ERROR", "message": msg})
        def warn(col, msg): warns.append(msg); issues.append({"row": rowno, "column": col, "severity": "WARNING", "message": msg})
        sid = str(raw.get("site_id", "")).strip()
        pid = str(raw.get("process_id", "")).strip()
        period = str(raw.get("period", "")).strip()
        if sid not in sites: err("site_id", f"Unknown site '{sid}'. Allowed: {', '.join(sites)}")
        if pid not in procs: err("process_id", f"Unknown process '{pid}'. See reference catalog in Governance.")
        elif sid in sites and procs[pid]["site_id"] != sid: err("process_id", f"Process '{pid}' does not belong to site '{sid}'.")
        import re
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", period): err("period", f"'{period}' is not YYYY-MM with month 01–12.")
        vals = {}
        for c in NUM_COLS:
            v = pd.to_numeric(raw.get(c), errors="coerce")
            if pd.isna(v): err(c, f"'{raw.get(c)}' is not a number (required).")
            else:
                v = float(v)
                if c != "storage_change_m3" and v < 0: err(c, f"{c} must be ≥ 0 (got {v:,.1f}).")
                if abs(v) > 1e7: err(c, f"{c} = {v:,.0f} m³ is implausibly large (> 10,000,000).")
                vals[c] = v
        meter = str(raw.get("metering", "")).strip().lower()
        if meter not in METER_UNC: err("metering", f"'{meter}' invalid. Allowed: metered / calculated / estimated.")
        action, rec = "create", None
        key = (sid, pid, period)
        if not errs:
            W = vals["municipal_m3"] + vals["groundwater_m3"] + vals["surface_m3"] + vals["harvested_m3"]
            if W <= 0: err("withdrawal", "Total withdrawal (sum of 4 source columns) must be > 0.")
            if vals.get("output_units", 0) <= 0: err("output_units", "output_units must be > 0 (intensity denominator).")
            if key in seen: err("duplicate", f"Duplicate of file row {seen[key]} for {sid}/{pid}/{period}.")
            seen.setdefault(key, rowno)
            if not errs:
                residual = W - (vals["consumption_m3"] + vals["discharge_m3"] + vals["storage_change_m3"])
                tol = balance_tolerance(W)
                if abs(residual) > 2 * tol:
                    err("balance", f"Balance mismatch: W−(C+D+ΔS) = {residual:,.0f} m³ exceeds 2× tolerance ({2*tol:,.0f}). Fix volumes.")
                elif abs(residual) > tol:
                    warn("balance", f"Balance residual {residual:,.0f} m³ above tolerance {tol:,.0f} — will import flagged.")
                if period < lock:
                    warn("period", f"Late entry: {period} is before reporting lock {lock}. Reviewer note required.")
                if sid in sites and vals["discharge_m3"] > sites[sid]["permit_m3_month"]:
                    warn("discharge_m3", f"Discharge exceeds site monthly permit ({fmt_n(sites[sid]['permit_m3_month'])} m³).")
                if key in existing:
                    if mode.startswith("Skip"):
                        action = "skip"
                        warn("duplicate", f"Key already exists as {existing[key]} — row skipped (safe re-import).")
                    else:
                        action = f"supersede {existing[key]}"
                        warn("duplicate", f"Correction mode: will supersede {existing[key]} (old version archived, never overwritten).")
                rec = {"site_id": sid, "process_id": pid, "period": period,
                       "sources": {"Municipal": vals["municipal_m3"], "Groundwater": vals["groundwater_m3"],
                                   "Surface": vals["surface_m3"], "Harvested": vals["harvested_m3"]},
                       "withdrawal_m3": W, "recycled_m3": vals["recycled_m3"],
                       "consumption_m3": vals["consumption_m3"], "discharge_m3": vals["discharge_m3"],
                       "storage_change_m3": vals["storage_change_m3"], "output_units": vals["output_units"],
                       "metering": meter, "uncertainty_pct": METER_UNC[meter],
                       "notes": str(raw.get("notes", "") or ""), "flags": [],
                       "balance_flag": abs(residual) > tol}
        rows.append({"row": rowno, "errors": errs, "warnings": warns, "record": rec, "action": action,
                     "raw": {c: raw.get(c) for c in TEMPLATE_COLUMNS}})
    return {"fatal": [f for f in fatal if not f.startswith("NOTE")], "notes": fatal,
            "rows": rows, "issues": pd.DataFrame(issues)}

# ----------------------------------------------------------------------------------
# SYNTHETIC SEED DATA (clearly labeled — never claim real-world performance)
# ----------------------------------------------------------------------------------
SITES_DEF = [
    {"id": "S1", "name": "Riverside Beverage Plant", "sector": "Manufacturing", "basin": "Cedar River Basin",
     "stress": "High", "tariff": 1.85, "permit_m3_month": 16000, "output_unit": "kL product",
     "prod_base": 2800, "prod_amp": 0.15, "prod_peak": 7, "lat": 39.1, "lon": -84.5,
     "src": {"Municipal": 0.70, "Groundwater": 0.28, "Surface": 0.0, "Harvested": 0.02}},
    {"id": "S2", "name": "Northgate Paper Mill", "sector": "Manufacturing", "basin": "Northgate Watershed",
     "stress": "Medium", "tariff": 2.40, "permit_m3_month": 45000, "output_unit": "tonnes paper",
     "prod_base": 4200, "prod_amp": 0.05, "prod_peak": 3, "lat": 44.0, "lon": -79.4,
     "src": {"Municipal": 0.10, "Groundwater": 0.28, "Surface": 0.62, "Harvested": 0.0}},
    {"id": "S3", "name": "Harborview Municipal Utility", "sector": "Utility", "basin": "Harborview Aquifer",
     "stress": "Low", "tariff": 1.20, "permit_m3_month": 105000, "output_unit": "ML delivered",
     "prod_base": 96, "prod_amp": 0.12, "prod_peak": 7, "lat": 36.8, "lon": -121.8,
     "src": {"Municipal": 0.0, "Groundwater": 0.18, "Surface": 0.82, "Harvested": 0.0}},
    {"id": "S4", "name": "Mesa Verde Foods", "sector": "Manufacturing", "basin": "Mesa Verde Arid Basin",
     "stress": "Extremely high", "tariff": 3.10, "permit_m3_month": 15000, "output_unit": "tonnes food",
     "prod_base": 1500, "prod_amp": 0.12, "prod_peak": 8, "lat": 33.4, "lon": -112.0,
     "src": {"Municipal": 0.17, "Groundwater": 0.80, "Surface": 0.0, "Harvested": 0.03}},
]
PROC_DEF = [
    ("S1", "Blending & Product", "blending", 9000, 0.85, 0.05, 0.14, 7),
    ("S1", "Bottle & Crate Washing", "washing", 6500, 0.15, 0.15, 0.10, 7),
    ("S1", "Cooling Towers", "cooling", 14000, 0.75, 0.40, 0.35, 7),
    ("S1", "CIP & Sanitation", "cip", 4200, 0.20, 0.10, 0.05, 7),
    ("S1", "Utilities & Landscape", "landscape", 1800, 0.60, 0.00, 0.45, 7),
    ("S2", "Pulping & Stock Prep", "pulping", 30000, 0.25, 0.35, 0.04, 3),
    ("S2", "Paper Machine", "paper_machine", 22000, 0.30, 0.60, 0.04, 3),
    ("S2", "Cooling & Vacuum", "cooling", 16000, 0.75, 0.40, 0.32, 7),
    ("S2", "Boiler & Steam", "boiler", 5000, 0.80, 0.10, 0.18, 1),
    ("S3", "Treatment & Distribution", "distribution", 120000, 0.18, 0.02, 0.10, 7),
    ("S3", "Depot & Facilities", "facilities", 2500, 0.50, 0.05, 0.20, 7),
    ("S4", "Raw Washing & Prep", "washing", 8000, 0.20, 0.10, 0.12, 8),
    ("S4", "Cooking & Blanching", "cooking", 5000, 0.45, 0.05, 0.08, 8),
    ("S4", "Cooling & Refrigeration", "cooling", 9000, 0.75, 0.40, 0.38, 7),
    ("S4", "CIP & Hygiene", "cip", 3000, 0.20, 0.10, 0.05, 8),
]
PROJECTS_DEF = [
    {"id": "P-COOL", "icon": "🧊", "name": "Cooling tower efficiency (CoC 3→6)", "group": "cooling",
     "capex": 320000, "opex": 8000, "life": 10, "ramp": 4, "co": 12000,
     "sav_lo": 0.25, "sav_hi": 0.25, "cap_lo": 0.20, "cap_hi": 0.20,
     "desc": "Raise cycles of concentration; less blowdown, same evaporation duty.",
     "formula": "Savings = E·(1/(CoC_old−1) − 1/(CoC_new−1))",
     "assumptions": "Evaporation duty unchanged; water chemistry supports 6 cycles; co-savings from treatment chemicals."},
    {"id": "P-ADX", "icon": "🌬️", "name": "Adiabatic hybrid dry-cooling retrofit", "group": "cooling",
     "capex": 950000, "opex": 55000, "life": 12, "ramp": 9, "co": -15000,
     "sav_lo": 0.20, "sav_hi": 0.20, "cap_lo": 0.25, "cap_hi": 0.25,
     "desc": "Displace evaporative duty with dry/adiabatic heat rejection. Competes with CoC upgrade.",
     "formula": "Savings = 0.55·E; energy penalty booked as negative co-savings",
     "assumptions": "Mutually exclusive with P-COOL (same assets); fan energy penalty included."},
    {"id": "P-LEAK", "icon": "🛰️", "name": "AI leak detection & district repair", "group": None,
     "capex": 520000, "opex": 24000, "life": 8, "ramp": 6, "co": 0,
     "sav_lo": 0.35, "sav_hi": 0.35, "cap_lo": 0.20, "cap_hi": 0.25,
     "desc": "Acoustic/log-noise analytics + targeted mains repair on the distribution network.",
     "formula": "Savings = consumptive losses × 60% effectiveness",
     "assumptions": "Losses approximated by distribution consumption; repair crew capacity assumed."},
    {"id": "P-RO", "icon": "🔁", "name": "RO reject recovery (blending/washing)", "group": None,
     "capex": 480000, "opex": 28000, "life": 12, "ramp": 6, "co": 8000,
     "sav_lo": 0.20, "sav_hi": 0.20, "cap_lo": 0.20, "cap_hi": 0.20,
     "desc": "Second-pass recovery on reverse-osmosis reject streams.",
     "formula": "Savings = feed × 25% reject × 60% recovery",
     "assumptions": "Reject quality suitable for reuse after polishing; membrane replacement in opex."},
    {"id": "P-RAIN", "icon": "🌧️", "name": "Rainwater harvesting (roof retrofit)", "group": None,
     "capex": 260000, "opex": 4000, "life": 20, "ramp": 5, "co": 0,
     "sav_lo": 0.45, "sav_hi": 0.45, "cap_lo": 0.25, "cap_hi": 0.30,
     "desc": "Capture roof runoff for wash-down and landscape demand across sites.",
     "formula": "Savings = area × rainfall × runoff × capture",
     "assumptions": "Rainfall is synthetic (0.62 m/yr) — widest uncertainty band in the portfolio."},
    {"id": "P-CIP", "icon": "🚿", "name": "CIP rinse reuse & conductivity control", "group": None,
     "capex": 180000, "opex": 7000, "life": 8, "ramp": 3, "co": 5000,
     "sav_lo": 0.20, "sav_hi": 0.20, "cap_lo": 0.15, "cap_hi": 0.20,
     "desc": "Recover final rinse water as pre-rinse; conductivity-triggered dosing.",
     "formula": "Savings = CIP withdrawal × 60% rinse share × 65% reuse",
     "assumptions": "Food-safety review required before reuse (human approval boundary)."},
    {"id": "P-LOOP", "icon": "📜", "name": "Paper-machine white-water closure", "group": None,
     "capex": 1450000, "opex": 65000, "life": 15, "ramp": 12, "co": 20000,
     "sav_lo": 0.30, "sav_hi": 0.30, "cap_lo": 0.30, "cap_hi": 0.35,
     "desc": "Deep closure of the paper-machine water loop with kidney filtration.",
     "formula": "Savings = machine fresh water × 35%",
     "assumptions": "High capex; marginal at low tariffs — optimizer will usually defer it."},
    {"id": "P-METER", "icon": "📡", "name": "Submetering + analytics-driven ops", "group": None,
     "capex": 140000, "opex": 12000, "life": 6, "ramp": 3, "co": 0,
     "sav_lo": 0.50, "sav_hi": 0.50, "cap_lo": 0.15, "cap_hi": 0.15,
     "desc": "Process-level submeters, dashboards and leak alarms; behavioral savings.",
     "formula": "Savings = 6% of total withdrawal (×0.5 overlap when ≥5 other projects stack)",
     "assumptions": "Behavioral savings are the least certain — widest relative band; overlap haircut documented."},
]

def build_seed():
    rng = np.random.default_rng(16113)
    periods = [str(p) for p in pd.period_range("2025-11", "2026-10", freq="M")]
    file_name = "demo_scada_export_2026.csv"
    file_sha = sha256_bytes((file_name + "|synthetic-seed-16113").encode())
    sites = [{**s, "stress_factor": STRESS_META[s["stress"]][1], "schema_version": SCHEMA_VERSION,
              "created_at": utcnow(), "updated_at": utcnow()} for s in SITES_DEF]
    smap = {s["id"]: s for s in sites}
    processes, records = [], []
    seq = 0
    prod_cache = {}
    for k, (sid, name, kind, base, cf, rf, amp, peak) in enumerate(PROC_DEF, start=1):
        pid = f"{sid}-P{k}"
        processes.append({"id": pid, "site_id": sid, "name": name, "kind": kind,
                          "schema_version": SCHEMA_VERSION, "created_at": utcnow()})
        for i, per in enumerate(periods):
            m = int(per[5:7])
            s = smap[sid]
            if (sid, per) not in prod_cache:
                season_p = 1 + s["prod_amp"] * math.cos((m - s["prod_peak"]) / 12 * 2 * math.pi)
                prod_cache[(sid, per)] = float(s["prod_base"] * season_p * (1 + rng.normal(0, 0.03)))
            season = 1 + amp * math.cos((m - peak) / 12 * 2 * math.pi)
            W = float(max(base * season * (1 + rng.normal(0, 0.05)), base * 0.4))
            recycled = float(max(W * rf * (1 + rng.normal(0, 0.08)), 0.0))
            storage = float(W * rng.uniform(-0.008, 0.008))
            D = float(min(max(W * (1 - cf) * (1 + rng.normal(0, 0.04)), 0.0), W - storage))
            C = float(W - D - storage)
            fr = {k2: max(v * (1 + rng.normal(0, 0.05)), 0.0) for k2, v in s["src"].items()}
            tot = sum(fr.values()) or 1.0
            sources = {k2: float(v / tot * W) for k2, v in fr.items()}
            meter = str(rng.choice(["metered", "estimated", "calculated"], p=[0.78, 0.15, 0.07]))
            status = "Approved" if i < 10 else str(rng.choice(["In Review", "Draft", "Validated"], p=[0.55, 0.30, 0.15]))
            seq += 1
            created = f"{per}-28T09:{seq % 60:02d}:00Z"
            records.append({
                "id": f"REC-{seq:04d}", "site_id": sid, "process_id": pid, "period": per,
                "sources": sources, "withdrawal_m3": W, "recycled_m3": recycled,
                "consumption_m3": C, "discharge_m3": D, "storage_change_m3": storage,
                "output_units": prod_cache[(sid, per)], "output_unit": s["output_unit"],
                "metering": meter, "uncertainty_pct": METER_UNC[meter], "status": status,
                "flags": [], "notes": "", "version": 1, "supersedes": None, "superseded_by": None,
                "provenance": {"kind": "synthetic_seed", "file": file_name, "row": seq + 1, "checksum": file_sha},
                "created_at": created, "updated_at": created,
                "entered_by": "A. Rivera (Analyst)" if seq % 2 else "J. Chen (Analyst)",
                "reviewed_by": "R. Okafor (Reviewer)" if status == "Approved" else "",
                "review_comment": "", "schema_version": SCHEMA_VERSION,
            })
    rmap = {(r["site_id"], r["process_id"], r["period"]): r for r in records}
    pmap = {p["id"]: p for p in processes}
    cool4 = [pid for pid, p in pmap.items() if p["site_id"] == "S4" and p["kind"] == "cooling"][0]
    r = rmap[("S4", cool4, "2026-08")]
    r["consumption_m3"] *= 1.22; r["status"] = "Draft"; r["flags"].append("balance_mismatch")
    r["notes"] = "SCADA tag drift suspected — consumption inconsistent with withdrawal."
    mach2 = [pid for pid, p in pmap.items() if p["site_id"] == "S2" and p["kind"] == "paper_machine"][0]
    r = rmap[("S2", mach2, "2026-09")]
    r["discharge_m3"] = smap["S2"]["permit_m3_month"] * 0.62
    r["consumption_m3"] = r["withdrawal_m3"] - r["discharge_m3"] - r["storage_change_m3"]
    r["status"] = "In Review"; r["flags"].append("permit_exceedance")
    r["notes"] = "Storm event bypass — site-month discharge above permit."
    fac3 = [pid for pid, p in pmap.items() if p["site_id"] == "S3" and p["kind"] == "facilities"][0]
    r = rmap[("S3", fac3, "2026-04")]
    r["status"] = "In Review"; r["flags"].append("late_entry")
    r["created_at"] = "2026-10-28T14:05:00Z"; r["updated_at"] = r["created_at"]
    r["notes"] = "Backfilled from paper logbooks after meter replacement."
    cip1 = [pid for pid, p in pmap.items() if p["site_id"] == "S1" and p["kind"] == "cip"][0]
    r = rmap[("S1", cip1, "2026-09")]
    r["status"] = "Rejected"; r["flags"].append("evidence_missing")
    r["reviewed_by"] = "R. Okafor (Reviewer)"
    r["review_comment"] = "Meter calibration certificate missing — resubmit with evidence."
    audit = [
        {"ts": "2026-11-01T08:00:00Z", "actor": "System", "action": "seed", "entity": "database",
         "detail": f"Synthetic demo seeded: {len(records)} records, 4 sites, 15 processes (rng seed 16113)."},
        {"ts": "2026-11-01T08:00:05Z", "actor": "System", "action": "import", "entity": file_name,
         "detail": f"Generated seed file • sha256 {file_sha[:16]}… • {len(records)} rows."},
        {"ts": "2026-11-05T10:22:00Z", "actor": "R. Okafor (Reviewer)", "action": "approve_batch", "entity": "records",
         "detail": "Approved Nov 2025 – Aug 2026 closing balance (150 records)."},
        {"ts": "2026-11-07T15:40:00Z", "actor": "R. Okafor (Reviewer)", "action": "reject", "entity": r["id"],
         "detail": r["review_comment"]},
        {"ts": "2026-11-10T09:12:00Z", "actor": "A. Rivera (Analyst)", "action": "export", "entity": "monthly_balance.csv",
         "detail": "Draft board pack exported (synthetic demo)."},
    ]
    attachments = [{"id": "ATT-0001", "filename": file_name, "sha256": file_sha, "size_bytes": 18432,
                    "rows": len(records), "imported_at": "2026-11-01T08:00:05Z",
                    "actor": "System", "kind": "synthetic_seed", "bytes_stored": False}]
    db = {"schema_version": SCHEMA_VERSION, "records": records, "sites": sites, "processes": processes,
          "projects": [dict(p) for p in PROJECTS_DEF], "scenarios": [], "audit": audit,
          "attachments": attachments,
          "config": {"lock_period": "2026-07", "retention_months": 24, "ref_version": "REF-1.0",
                     "rules_version": RULES_VERSION, "synthetic": True},
          "seq": {"rec": seq, "sc": 0, "att": 1}}
    return db

def default_scenario(db):
    projects = project_bases(db)
    df = active_df(db)
    baseline = float(df["withdrawal_m3"].sum())
    sel = [k.replace("proj_", "") for k, v in LAB_DEFAULTS.items() if k.startswith("proj_") and v]
    sig = build_sig(projects, sel)
    mc = _mc_core(sig, 2000, 42, 1.85, 0.08, 12, 1.0, 15, baseline)
    db["seq"]["sc"] += 1
    return {"id": f"SC-{db['seq']['sc']:04d}", "name": "FY27 Conservation Portfolio (recommended)",
            "status": "Approved", "params": {k: v for k, v in LAB_DEFAULTS.items() if not k.startswith("proj_")},
            "selected": sel,
            "results": {"sav": mc["sav_p"], "npv": mc["npv_p"], "capex": mc["cap_p"],
                        "payback_p50": mc["payback_p50"], "prob_target": mc["prob_target"],
                        "baseline_annual": baseline, "reduction_pct": mc["sav_p"]["p50"] / baseline * 100 if baseline else 0},
            "basis_record_ids": [r["id"] for r in df.to_dict("records")],
            "created_at": utcnow(), "created_by": "A. Rivera (Analyst)",
            "reviewed_by": "R. Okafor (Reviewer)", "review_comment": "Basis and uncertainty reviewed against WBA-1.0.",
            "rules_version": RULES_VERSION, "ref_version": db["config"]["ref_version"], "seed": 42, "n": 2000}

# ----------------------------------------------------------------------------------
# LIVE SELF-CHECK SUITE — real assertions executed in-session (observable pass/fail)
# ----------------------------------------------------------------------------------
def run_self_checks(db):
    results = []
    def check(name, fn):
        try:
            ok, detail = fn()
            results.append({"Check": name, "Result": "✅ PASS" if ok else "❌ FAIL", "Detail": detail})
        except Exception as e:
            results.append({"Check": name, "Result": "❌ FAIL", "Detail": f"{type(e).__name__}: {e}"})
    check("Cooling-tower hand example (E=10,000, 3→6 ⇒ 3,000 m³)",
          lambda: (abs(cooling_savings(10000, 3, 6) - 3000) < 1e-9, "computed = 3,000.000 m³"))
    def _coc():
        try:
            cooling_savings(1000, 1, 6); return (False, "invalid CoC accepted")
        except ValueError:
            return (True, "ValueError raised for CoC ≤ 1")
    check("Engine rejects invalid boundary (CoC ≤ 1)", _coc)
    def _bal():
        r = next(r for r in db["records"] if r["status"] == "Approved")
        res, tol, stt = balance_check(r)
        return (stt == "balanced", f"{r['id']}: residual {res:,.2f} m³ ≤ tol {tol:,.1f}")
    check("Seeded approved record balances (W = C + D + ΔS)", _bal)
    check("NPV hand example (100 capex, 30/yr, 5y, 10% ⇒ 13.72)",
          lambda: (abs(npv(100, 30, 5, 0.10) - 13.7236) < 0.001, f"computed = {npv(100,30,5,0.10):.4f}"))
    check("Zero-discount annuity edge (rate 0 ⇒ years)", lambda: (annuity(0, 5) == 5, "annuity(0,5) = 5"))
    check("Payback guards non-positive net", lambda: (simple_payback(100, 0) is None and simple_payback(100, 25) == 4.0, "None / 4.0 y"))
    check("Unit conversion m³ → ML", lambda: (m3_to_ml(2500) == 2.5, "2,500 m³ = 2.5 ML"))
    sig = (("T1", 10000.0, 100000.0, 5000.0, 10.0, 0.25, 0.25, 0.2, 0.2, 0.0, 1.0),)
    a = _mc_core(sig, 300, 7, 1.85, 0.08, 12, 1.0, 15, 100000)
    b = _mc_core(sig, 300, 7, 1.85, 0.08, 12, 1.0, 15, 100000)
    check("Monte Carlo reproducible with same seed", lambda: (bool(np.array_equal(a["sav"], b["sav"])), "identical arrays across runs"))
    c = _mc_core(sig, 300, 8, 1.85, 0.08, 12, 1.0, 15, 100000)
    check("Monte Carlo differs with new seed", lambda: (not bool(np.array_equal(a["sav"], c["sav"])), "seed 7 ≠ seed 8"))
    check("Percentiles ordered P10 ≤ P50 ≤ P90",
          lambda: (a["sav_p"]["p10"] <= a["sav_p"]["p50"] <= a["sav_p"]["p90"], "ordering holds"))
    cands = [{"id": "A", "capex": 100, "group": "g", "applicable": True, "npv": {"p50": 90, "p10": 60}, "sav": {"p50": 1000, "p10": 800}, "payback_p50": 2},
             {"id": "B", "capex": 80, "group": "g", "applicable": True, "npv": {"p50": 70, "p10": 50}, "sav": {"p50": 900, "p10": 700}, "payback_p50": 3},
             {"id": "C", "capex": 60, "group": None, "applicable": True, "npv": {"p50": 40, "p10": 20}, "sav": {"p50": 500, "p10": 400}, "payback_p50": 4}]
    opt = optimize_portfolio(cands, 170, "Max NPV", "P50 (median)")
    check("Optimizer satisfies budget + exclusivity",
          lambda: (opt["feasible"] and "B" not in opt["ids"] and opt["spent"] <= 170, f"selected {opt['ids']}, spent {opt['spent']}"))
    inf = optimize_portfolio(cands, 10, "Max NPV", "P50 (median)")
    check("Optimizer reports infeasible budget safely", lambda: (not inf["feasible"] and inf["ids"] == [], inf["reason"]))
    good = pd.DataFrame([{**{c: 0 for c in TEMPLATE_COLUMNS}, "site_id": "S1", "process_id": "S1-P3",
                          "period": "2030-01", "municipal_m3": 1000, "consumption_m3": 700,
                          "discharge_m3": 295, "storage_change_m3": 5, "output_units": 300,
                          "metering": "metered", "notes": "self-check"}])
    v = validate_import(good, db, "Skip duplicates (safe)")
    check("Validator accepts a clean row", lambda: (len(v["rows"]) == 1 and not v["rows"][0]["errors"], "0 errors, action=create"))
    bad = good.copy(); bad.loc[0, "consumption_m3"] = -5
    vb = validate_import(bad, db, "Skip duplicates (safe)")
    check("Validator rejects negative volume", lambda: (any("must be ≥ 0" in e for e in vb["rows"][0]["errors"]), "actionable message emitted"))
    dup = pd.concat([good, good], ignore_index=True)
    vd = validate_import(dup, db, "Skip duplicates (safe)")
    check("Validator flags within-file duplicate key", lambda: (any("Duplicate" in e for e in vd["rows"][1]["errors"]), "second occurrence blocked"))
    check("Ledger integrity: unique IDs, non-empty",
          lambda: (len(db["records"]) > 0 and len({r["id"] for r in db["records"]}) == len(db["records"]), f"{len(db['records'])} records"))
    def _stress():
        df = active_df(db)
        sm = {s["id"]: s["stress_factor"] for s in db["sites"]}
        sw = float((df["withdrawal_m3"] * df["site_id"].map(sm)).sum())
        return (sw >= float(df["withdrawal_m3"].sum()) - 1e-6, f"{sw:,.0f} m³-eq ≥ plain withdrawal")
    check("Stress-weighting never reduces withdrawal", _stress)
    return results

# ----------------------------------------------------------------------------------
# SESSION STATE / DATA ACCESS / AUDIT / PERMISSIONS
# ----------------------------------------------------------------------------------
def init_state():
    if "db" not in st.session_state:
        db = build_seed()
        db["scenarios"].append(default_scenario(db))
        st.session_state.db = db
        st.session_state.selfcheck = run_self_checks(db)
        st.session_state.selfcheck_ts = utcnow()
        st.session_state.telemetry = {}
        rng = np.random.default_rng()
        for s in db["sites"]:
            annual = sum(r["withdrawal_m3"] for r in db["records"] if r["site_id"] == s["id"])
            base_rate = annual / 8760.0
            hist = [max(base_rate * (1 + rng.normal(0, 0.06)), 0.0) for _ in range(40)]
            st.session_state.telemetry[s["id"]] = {"rate": hist[-1], "base": base_rate,
                                                   "hist": hist, "today": base_rate * 9.5}
        st.session_state.sim_clock = dt.datetime.now()
    defaults = {"nav": NAV_LABELS[0], "role": "Administrator", "sim_running": True, "sim_speed": 1,
                "selected_record": None, "loaded_scenario": None, "undo": None,
                "import_preview": None, "last_report": None, "share_link": None,
                "share_base": "http://localhost:8501", "dl_done": False, "row_select_ok": True}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    for k, v in LAB_DEFAULTS.items():
        st.session_state.setdefault(k, v)

def db(): return st.session_state.db
def role(): return st.session_state.role
def can(perm): return perm in PERMS.get(role(), set())

def audit(action, entity, detail):
    d = db()
    d["audit"].append({"ts": utcnow(), "actor": f"{ROLE_ICON[role()]} {role()} (demo)",
                       "action": action, "entity": str(entity), "detail": detail})
    if len(d["audit"]) > 2000:
        d["audit"] = d["audit"][-2000:]

def on_role_change():
    audit("role_switch", "session", f"Demo role switched to {st.session_state.role}. Production enforces server-side auth.")

def on_sim_toggle():
    audit("simulation", "telemetry", "Live telemetry " + ("resumed" if st.session_state.sim_running else "paused"))

def new_rec_id():
    d = db(); d["seq"]["rec"] += 1
    return f"REC-{d['seq']['rec']:04d}"

def find_rec(rid):
    return next((r for r in db()["records"] if r["id"] == rid), None)

def site_name(sid): return next((s["name"] for s in db()["sites"] if s["id"] == sid), sid)
def proc_name(pid): return next((p["name"] for p in db()["processes"] if p["id"] == pid), pid)

def sanitize_select(key, options):
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]

def sanitize_multi(key, options):
    if key in st.session_state:
        st.session_state[key] = [x for x in st.session_state[key] if x in options]

def sanitize_range(key, options):
    if key in st.session_state:
        v = st.session_state[key]
        if not (isinstance(v, (tuple, list)) and len(v) == 2 and v[0] in options and v[1] in options):
            del st.session_state[key]

def periods_list():
    return sorted({r["period"] for r in db()["records"]})

def quality_flags(rec):
    """Recomputed live: balance + permit + late + metering flags."""
    flags = list(rec.get("flags", []))
    _, _, bstate = balance_check(rec)
    if bstate == "error" and "balance_mismatch" not in flags:
        flags.append("balance_mismatch")
    return flags

def all_exceptions():
    out = []
    d = db()
    permits = {s["id"]: s["permit_m3_month"] for s in d["sites"]}
    month_d = {}
    for r in d["records"]:
        if r["status"] == "Archived":
            continue
        month_d[(r["site_id"], r["period"])] = month_d.get((r["site_id"], r["period"]), 0.0) + r["discharge_m3"]
        for f in quality_flags(r):
            sev = "error" if f in ("balance_mismatch",) else "warning"
            msg = {"balance_mismatch": "Consumption/discharge contradict withdrawal (balance error)",
                   "permit_exceedance": "Discharge above site monthly permit",
                   "late_entry": "Late backfill after reporting lock — reviewer note required",
                   "evidence_missing": "Calibration / evidence document missing",
                   "manual_override": "Manual override applied — see audit trail",
                   "corrected": "Superseding correction — re-review required"}.get(f, f)
            out.append({"sev": sev, "flag": f, "rid": r["id"],
                        "label": f"{site_name(r['site_id'])} · {proc_name(r['process_id'])} · {r['period']}", "msg": msg})
    for (sid, per), v in month_d.items():
        if v > permits.get(sid, 1e18) and not any(e["flag"] == "permit_exceedance" and e["rid"] in
                [r["id"] for r in d["records"] if r["site_id"] == sid and r["period"] == per] for e in out):
            rid = next(r["id"] for r in d["records"] if r["site_id"] == sid and r["period"] == per and r["status"] != "Archived")
            out.append({"sev": "warning", "flag": "permit_exceedance", "rid": rid,
                        "label": f"{site_name(sid)} · site-month {per}",
                        "msg": f"Site discharge {fmt_n(v)} m³ exceeds permit {fmt_n(permits[sid])} m³"})
    order = {"error": 0, "warning": 1}
    out.sort(key=lambda e: order.get(e["sev"], 2))
    return out

# ----------------------------------------------------------------------------------
# RECORD LIFECYCLE ACTIONS (server-side-style permission checks + audit)
# ----------------------------------------------------------------------------------
def act_submit(rid):
    if not can("submit"): return
    r = find_rec(rid)
    if r and r["status"] in ("Draft", "Validated"):
        r["status"] = "In Review"; r["updated_at"] = utcnow()
        audit("submit", rid, f"Submitted for review ({r['period']}).")
        st.toast(f"{rid} submitted for review", icon="📨")

def act_approve(rid, comment):
    if not can("approve"): return
    r = find_rec(rid)
    if not r or r["status"] != "In Review": return
    if quality_flags(r) and not comment.strip():
        st.error("This record carries data-quality flags — a reviewer comment is required to approve.")
        return
    r["status"] = "Approved"; r["reviewed_by"] = f"{role()} (demo)"; r["review_comment"] = comment.strip()
    r["updated_at"] = utcnow()
    audit("approve", rid, comment.strip() or "Approved — balance and provenance verified.")
    st.toast(f"{rid} approved", icon="✅")

def act_reject(rid, comment):
    if not can("approve"): return
    if not comment.strip():
        st.error("A rejection reason is required (actionable feedback for the analyst).")
        return
    r = find_rec(rid)
    if not r or r["status"] != "In Review": return
    r["status"] = "Rejected"; r["reviewed_by"] = f"{role()} (demo)"; r["review_comment"] = comment.strip()
    r["updated_at"] = utcnow()
    audit("reject", rid, comment.strip())
    st.toast(f"{rid} rejected — returned for correction", icon="↩️")

def act_archive(rid):
    if not can("archive"): return
    r = find_rec(rid)
    if not r or r["status"] == "Archived": return
    st.session_state.undo = {"type": "archive", "id": rid, "prev": r["status"]}
    r["status"] = "Archived"; r["updated_at"] = utcnow()
    audit("archive", rid, "Archived (undo available).")
    st.toast(f"{rid} archived — Undo available on the ledger", icon="🗄️")

def act_undo():
    u = st.session_state.undo
    if u and u["type"] == "archive":
        r = find_rec(u["id"])
        if r:
            r["status"] = u["prev"]; r["updated_at"] = utcnow()
            audit("undo_archive", r["id"], f"Restored to {u['prev']}.")
            st.toast(f"{r['id']} restored", icon="↩️")
    st.session_state.undo = None

def act_correction(rid, fields, reason):
    if not can("edit"): return
    if not reason.strip():
        st.error("A correction reason is required (audit trail)."); return
    old = find_rec(rid)
    if not old: return
    new = dict(old)
    new.update(fields)
    new["sources"] = dict(fields.get("sources", old["sources"]))
    new["withdrawal_m3"] = float(sum(new["sources"].values()))
    new["id"] = new_rec_id(); new["version"] = old["version"] + 1
    new["supersedes"] = old["id"]; new["superseded_by"] = None
    new["status"] = "Draft"; new["flags"] = ["corrected"]; new["notes"] = reason.strip()
    new["created_at"] = utcnow(); new["updated_at"] = utcnow()
    new["entered_by"] = f"{role()} (demo)"; new["reviewed_by"] = ""; new["review_comment"] = ""
    new["provenance"] = {"kind": "manual_correction", "file": None, "row": None,
                         "checksum": sha256_bytes(json.dumps(fields, sort_keys=True, default=str).encode())}
    old["status"] = "Archived"; old["superseded_by"] = new["id"]; old["updated_at"] = utcnow()
    db()["records"].append(new)
    audit("correction", new["id"], f"v{new['version']} supersedes {old['id']}: {reason.strip()}")
    st.session_state.selected_record = new["id"]
    st.toast(f"Correction saved as {new['id']} (v{new['version']}) — submit for re-review", icon="📝")

def act_override(rid, field, value, reason):
    if not can("override"): return
    if not reason.strip():
        st.error("Overrides require a written reason (audit + explanation panel)."); return
    r = find_rec(rid)
    if not r: return
    oldv = r[field]
    r[field] = float(value)
    if "manual_override" not in r["flags"]:
        r["flags"].append("manual_override")
    r["updated_at"] = utcnow()
    audit("override", rid, f"{field}: {oldv:,.2f} → {float(value):,.2f}. Reason: {reason.strip()}")
    st.toast(f"Override applied to {rid} — audited", icon="🛡️")

def act_create(fields):
    if not can("edit"): return None
    d = db()
    key = (fields["site_id"], fields["process_id"], fields["period"])
    if any((r["site_id"], r["process_id"], r["period"]) == key and r["status"] != "Archived" for r in d["records"]):
        st.error("Duplicate key: a live record already exists for this site/process/period. Use a correction instead.")
        return None
    W = float(sum(fields["sources"].values()))
    errs = []
    if W <= 0: errs.append("Total withdrawal must be > 0.")
    if fields["output_units"] <= 0: errs.append("Output units must be > 0.")
    if any(v < 0 for k, v in fields["sources"].items()) or fields["consumption_m3"] < 0 or fields["discharge_m3"] < 0 or fields["recycled_m3"] < 0:
        errs.append("Volumes must be ≥ 0 (storage change may be negative).")
    import re
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", fields["period"]): errs.append("Period must be YYYY-MM.")
    if errs:
        for e in errs: st.error(e)
        return None
    r = {"id": new_rec_id(), **fields, "withdrawal_m3": W,
         "uncertainty_pct": METER_UNC[fields["metering"]],
         "output_unit": next(s["output_unit"] for s in d["sites"] if s["id"] == fields["site_id"]),
         "status": "Draft", "flags": [], "version": 1, "supersedes": None, "superseded_by": None,
         "provenance": {"kind": "manual_entry", "file": None, "row": None,
                        "checksum": sha256_bytes(json.dumps(fields, sort_keys=True, default=str).encode())},
         "created_at": utcnow(), "updated_at": utcnow(), "entered_by": f"{role()} (demo)",
         "reviewed_by": "", "review_comment": "", "schema_version": SCHEMA_VERSION}
    _, _, bstate = balance_check(r)
    if bstate != "balanced":
        r["flags"].append("balance_mismatch")
        r["notes"] = (r.get("notes", "") + " [Created with balance exception — contradiction path]").strip()
    else:
        r["status"] = "Validated"
    d["records"].append(r)
    audit("create", r["id"], f"Manual entry {r['site_id']}/{r['process_id']} {r['period']} ({r['status']}).")
    st.session_state.selected_record = r["id"]
    st.toast(f"{r['id']} created as {r['status']}", icon="✨")
    return r["id"]

# ----------------------------------------------------------------------------------
# SAMPLE FILES / IMPORT COMMIT
# ----------------------------------------------------------------------------------
def sample_valid_csv():
    rows = [
        {"site_id": "S1", "process_id": "S1-P3", "period": "2026-11", "municipal_m3": 9100, "groundwater_m3": 3600,
         "surface_m3": 0, "harvested_m3": 250, "recycled_m3": 5200, "consumption_m3": 9700, "discharge_m3": 3200,
         "storage_change_m3": 50, "output_units": 2780, "metering": "metered", "notes": "November closing"},
        {"site_id": "S2", "process_id": "S2-P9", "period": "2026-11", "municipal_m3": 520, "groundwater_m3": 1400,
         "surface_m3": 3300, "harvested_m3": 0, "recycled_m3": 500, "consumption_m3": 4180, "discharge_m3": 1030,
         "storage_change_m3": 10, "output_units": 4230, "metering": "metered", "notes": "Boiler November"},
        {"site_id": "S4", "process_id": "S4-P12", "period": "2026-11", "municipal_m3": 1400, "groundwater_m3": 6600,
         "surface_m3": 0, "harvested_m3": 240, "recycled_m3": 820, "consumption_m3": 1660, "discharge_m3": 6540,
         "storage_change_m3": 40, "output_units": 1520, "metering": "calculated", "notes": "Wash line November"},
    ]
    return pd.DataFrame(rows)[TEMPLATE_COLUMNS].to_csv(index=False).encode()

def sample_invalid_csv():
    base = json.loads(pd.DataFrame([{c: 0 for c in TEMPLATE_COLUMNS}]).to_json())  # noop guard
    good = {"site_id": "S3", "process_id": "S3-P11", "period": "2026-11", "municipal_m3": 0, "groundwater_m3": 1800,
            "surface_m3": 250, "harvested_m3": 0, "recycled_m3": 120, "consumption_m3": 1030, "discharge_m3": 1015,
            "storage_change_m3": 5, "output_units": 60, "metering": "metered", "notes": "valid row"}
    rows = [
        {**good},
        {**good, "period": "2026-13", "notes": "bad month"},
        {**good, "site_id": "S9", "notes": "unknown site"},
        {**good, "consumption_m3": -40, "notes": "negative volume"},
        {**good, "consumption_m3": 5000, "discharge_m3": 4000, "notes": "balance mismatch"},
        {**good, "notes": "duplicate of row 2"},
        {**good, "output_units": "", "notes": "missing denominator"},
        {**good, "period": "2026-05", "notes": "existing key (skip or supersede)"},
        {**good, "metering": "guess", "notes": "bad enum"},
    ]
    return pd.DataFrame(rows)[TEMPLATE_COLUMNS].to_csv(index=False).encode()

def commit_import(preview, valid_rows, filename, checksum, raw_bytes):
    if not can("import"): return
    d = db()
    created, superseded, skipped = [], [], 0
    new_records = []
    for row in valid_rows:
        if row["action"] == "skip" or row["record"] is None:
            skipped += 1
            continue
        f = row["record"]
        r = {"id": new_rec_id(), **f,
             "output_unit": next(s["output_unit"] for s in d["sites"] if s["id"] == f["site_id"]),
             "status": "Draft" if f.get("balance_flag") else "Validated",
             "flags": (["balance_mismatch"] if f.get("balance_flag") else []) +
                      (["late_entry"] if f["period"] < d["config"]["lock_period"] else []),
             "version": 1, "supersedes": None, "superseded_by": None,
             "provenance": {"kind": "import", "file": filename, "row": row["row"], "checksum": checksum},
             "created_at": utcnow(), "updated_at": utcnow(), "entered_by": f"{role()} (demo)",
             "reviewed_by": "", "review_comment": "", "schema_version": SCHEMA_VERSION}
        r.pop("balance_flag", None)
        if row["action"].startswith("supersede"):
            old_id = row["action"].split()[1]
            old = find_rec(old_id)
            if old:
                old["status"] = "Archived"; old["superseded_by"] = r["id"]; old["updated_at"] = utcnow()
                r["supersedes"] = old_id; r["version"] = old["version"] + 1
                superseded.append(old_id)
        new_records.append(r)
        created.append(r["id"])
    d["records"].extend(new_records)  # atomic-ish: constructed fully before mutation
    d["seq"]["att"] += 1
    store = raw_bytes is not None and len(raw_bytes) <= 2_000_000
    att = {"id": f"ATT-{d['seq']['att']:04d}", "filename": filename, "sha256": checksum,
           "size_bytes": len(raw_bytes) if raw_bytes else 0, "rows": len(created),
           "imported_at": utcnow(), "actor": f"{role()} (demo)", "kind": "upload", "bytes_stored": store}
    d["attachments"].append(att)
    if store:
        st.session_state.setdefault("att_bytes", {})[checksum] = raw_bytes
    audit("import", filename, f"sha256 {checksum[:16]}… • created {len(created)}, superseded {len(superseded)}, skipped {skipped}. Original preserved.")
    st.session_state.import_preview = None
    st.toast(f"Imported {len(created)} records ({len(superseded)} corrections, {skipped} skipped)", icon="📥")
    st.balloons()

# ----------------------------------------------------------------------------------
# CHARTS
# ----------------------------------------------------------------------------------
def hex_rgba(h, a):
    # Plotly rejects 8-digit hex (#rrggbbaa); convert to rgba().
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

def sankey_fig(df):
    srcs = list(SRC_COLORS)
    sites = sorted(df["site_id"].unique())
    fates = list(FATE_COLORS)
    nodes = srcs + [site_name(s) for s in sites] + fates
    idx = {n: i for i, n in enumerate(nodes)}
    colors = [SRC_COLORS[s] for s in srcs] + [COLORWAY[(i + 1) % len(COLORWAY)] for i in range(len(sites))] + [FATE_COLORS[f] for f in fates]
    links = {"s": [], "t": [], "v": [], "c": []}
    for s in srcs:
        col = {"Municipal": "municipal", "Groundwater": "groundwater", "Surface": "surface", "Harvested": "harvested"}[s]
        for sid in sites:
            sub = df[df["site_id"] == sid]
            v = float(sub["sources"].apply(lambda d: d.get(s, 0.0)).sum()) if len(sub) else 0.0
            if v > 1:
                links["s"].append(idx[s]); links["t"].append(idx[site_name(sid)]); links["v"].append(v)
                links["c"].append(hex_rgba(SRC_COLORS[s], 0.33))
    for sid in sites:
        sub = df[df["site_id"] == sid]
        for f, col in [("Consumed", "consumption_m3"), ("Discharged", "discharge_m3"), ("Storage", "storage_change_m3")]:
            v = float(sub[col].sum())
            if abs(v) > 1:
                links["s"].append(idx[site_name(sid)]); links["t"].append(idx[f]); links["v"].append(abs(v))
                links["c"].append(hex_rgba(FATE_COLORS[f], 0.33))
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=colors, pad=16, thickness=16, line=dict(width=0),
                  hovertemplate="%{label}<br>%{value:,.0f} m³<extra></extra>"),
        link=dict(source=links["s"], target=links["t"], value=links["v"], color=links["c"],
                  hovertemplate="%{source.label} → %{target.label}<br>%{value:,.0f} m³<extra></extra>")))
    return fig_layout(fig, 420, legend=False)

def monthly_fig(df):
    fig = go.Figure()
    g = df.groupby("period")
    for s in SRC_COLORS:
        vals = g.apply(lambda x: float(x["sources"].apply(lambda d: d.get(s, 0.0)).sum()), include_groups=False)
        fig.add_bar(x=vals.index, y=vals.values, name=s, marker_color=SRC_COLORS[s])
    d = g["discharge_m3"].sum()
    fig.add_scatter(x=d.index, y=d.values, name="Discharge (total)", mode="lines+markers",
                    line=dict(color="#e8f4fb", dash="dot", width=2), marker=dict(size=5))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="m³ / month")
    return fig_layout(fig, 340)

def bubble_fig(df):
    sm = {s["id"]: s for s in db()["sites"]}
    fig = go.Figure()
    prod = df.groupby(["site_id", "period"])["output_units"].first().groupby("site_id").sum()
    for stress, (color, _) in STRESS_META.items():
        xs, ys, sz, tx = [], [], [], []
        for sid in df["site_id"].unique():
            if sm[sid]["stress"] != stress: continue
            sub = df[df["site_id"] == sid]
            W = float(sub["withdrawal_m3"].sum())
            outp = float(prod.get(sid, 0.0)) or 1.0
            xs.append(W / outp)
            ys.append(float(sub["recycled_m3"].sum()) / max(W + float(sub["recycled_m3"].sum()), 1) * 100)
            sz.append(max(W / 12000, 12)); tx.append(f"{sm[sid]['name']}<br>{sm[sid]['output_unit']}")
        if xs:
            fig.add_scatter(x=xs, y=ys, mode="markers+text", name=stress,
                            text=[t.split("<br>")[0].split(" ")[0] for t in tx], textposition="top center",
                            textfont=dict(size=10, color="#9db8cc"),
                            marker=dict(size=sz, color=color, opacity=0.8, line=dict(width=1, color="#04101c")),
                            customdata=tx, hovertemplate="%{customdata}<br>Intensity %{x:.2f} m³/unit<br>Reuse %{y:.1f}%<extra></extra>")
    fig.update_xaxes(title="Withdrawal intensity (m³ / output unit)")
    fig.update_yaxes(title="Reuse rate (%)")
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.04, text="⚠ high intensity · low reuse → prioritize",
                       showarrow=False, font=dict(size=10, color="#fb7185"), align="right")
    return fig_layout(fig, 340)

def top_processes_fig(df):
    g = df.groupby(["site_id", "process_id"])[["withdrawal_m3", "consumption_m3", "discharge_m3"]].sum()
    g = g.sort_values("withdrawal_m3", ascending=True).tail(8)
    labels = [f"{site_name(s).split(' ')[0]} · {proc_name(p)}" for s, p in g.index]
    fig = go.Figure()
    fig.add_bar(y=labels, x=g["consumption_m3"], name="Consumed", orientation="h", marker_color=FATE_COLORS["Consumed"])
    fig.add_bar(y=labels, x=g["discharge_m3"], name="Discharged", orientation="h", marker_color=FATE_COLORS["Discharged"])
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="m³ (trailing 12 periods)")
    return fig_layout(fig, 330)

def metering_fig(df):
    counts = df["metering"].value_counts()
    labels = ["metered", "calculated", "estimated"]
    vals = [int(counts.get(l, 0)) for l in labels]
    fig = go.Figure(go.Pie(labels=[l.capitalize() for l in labels], values=vals, hole=0.62,
                           marker=dict(colors=["#22d3ee", "#a78bfa", "#fbbf24"], line=dict(color="#04101c", width=2)),
                           textinfo="label+percent", textfont=dict(size=11)))
    mp = vals[0] / max(sum(vals), 1) * 100
    fig.add_annotation(text=f"<b>{mp:.0f}%</b><br>metered", showarrow=False, font=dict(size=14, color="#e8f4fb"))
    return fig_layout(fig, 230, legend=False)

def mc_hist_fig(mc):
    fig = go.Figure(go.Histogram(x=mc["sav"] / 1000, nbinsx=42,
                                 marker=dict(color="#22d3ee", opacity=0.85, line=dict(color="#0a2236", width=1))))
    for k, c, lab in [("p10", "#fb7185", "P10"), ("p50", "#34d399", "P50"), ("p90", "#fbbf24", "P90")]:
        fig.add_vline(x=mc["sav_p"][k] / 1000, line_color=c, line_dash="dash",
                      annotation_text=f"{lab} {mc['sav_p'][k]/1000:,.0f}k", annotation_font_color=c)
    if mc["target_m3"] > 0:
        fig.add_vline(x=mc["target_m3"] / 1000, line_color="#e879f9", line_width=2,
                      annotation_text="Target", annotation_font_color="#e879f9")
    fig.update_xaxes(title="Annual water saved (thousand m³)")
    fig.update_yaxes(title="Iterations")
    return fig_layout(fig, 330, legend=False)

def tornado_fig(tdf):
    fig = go.Figure()
    fig.add_bar(y=tdf["factor"], x=tdf["low"], orientation="h", name="Downside", marker_color="#fb7185")
    fig.add_bar(y=tdf["factor"], x=tdf["high"], orientation="h", name="Upside", marker_color="#34d399")
    fig.update_layout(barmode="relative")
    fig.update_xaxes(title="Δ portfolio NPV vs base case ($)")
    fig.update_yaxes(autorange="reversed")
    return fig_layout(fig, 330)

def projection_fig(pdf):
    fig = go.Figure()
    fig.add_scatter(x=pdf["period"], y=pdf["baseline"], name="Baseline (no projects)", mode="lines",
                    line=dict(color="#94a3b8", dash="dash", width=2))
    fig.add_scatter(x=pdf["period"], y=pdf["lower"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=pdf["period"], y=pdf["upper"], name="P10–P90 band", mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(34,211,238,0.14)")
    fig.add_scatter(x=pdf["period"], y=pdf["mid"], name="With portfolio (P50)", mode="lines+markers",
                    line=dict(color="#22d3ee", width=3), marker=dict(size=5))
    fig.add_scatter(x=pdf["period"], y=pdf["target"], name="Target", mode="lines",
                    line=dict(color="#e879f9", dash="dot", width=2))
    fig.update_yaxes(title="Withdrawal m³ / month")
    return fig_layout(fig, 340)

# ----------------------------------------------------------------------------------
# CSS — deep-water design system (custom scrollbars, gradients, motion)
# ----------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 750px at 88% -10%, rgba(59,130,246,0.16), transparent 60%),
    radial-gradient(950px 700px at -8% 108%, rgba(45,212,191,0.13), transparent 55%),
    radial-gradient(700px 500px at 55% 115%, rgba(232,121,249,0.06), transparent 60%),
    linear-gradient(180deg, #04101c 0%, #051524 100%);
  color: #e8f4fb;
}
[data-testid="stAppViewContainer"]::before {
  content:""; position: fixed; inset:0; pointer-events:none; opacity:.035; z-index:0;
  background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160"><filter id="n"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2"/></filter><rect width="160" height="160" filter="url(%23n)" opacity="0.5"/></svg>');
}
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1500px; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #061826 0%, #05202f 100%);
  border-right: 1px solid rgba(56,189,248,0.16);
}
[data-testid="stSidebar"] * { color: #d7ecf8; }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: #061624; }
*::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#155e75,#1d4ed8); border-radius: 8px; border: 2px solid #061624; }
*::-webkit-scrollbar-thumb:hover { background: #22d3ee; }
* { scrollbar-width: thin; scrollbar-color: #1d4ed8 #061624; }
h1,h2,h3,.ak-title { font-family:'Space Grotesk',sans-serif !important; }
.grad-anim {
  background: linear-gradient(90deg,#7dd3fc,#22d3ee,#2dd4bf,#7dd3fc);
  background-size: 220% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: hueshift 9s linear infinite;
}
@keyframes hueshift { 0%{background-position:0% 0} 100%{background-position:220% 0} }
@keyframes rise { from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:translateY(0)} }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(52,211,153,.55)} 70%{box-shadow:0 0 0 9px rgba(52,211,153,0)} 100%{box-shadow:0 0 0 0 rgba(52,211,153,0)} }
.ak-card {
  background: linear-gradient(160deg, rgba(15,38,60,0.85), rgba(9,24,40,0.92));
  border: 1px solid rgba(56,189,248,0.16); border-radius: 14px; padding: 14px 16px;
  position: relative; overflow: hidden; animation: rise .5s ease both; margin-bottom: 6px;
}
.ak-card::before { content:""; position:absolute; inset:0 0 auto 0; height:2px;
  background: linear-gradient(90deg,#22d3ee,#3b82f6,#2dd4bf); }
.ak-card:hover { border-color: rgba(34,211,238,0.45); transform: translateY(-1px); transition: all .2s; }
.kpi-label { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #8fb3c9; font-weight: 600; }
.kpi-val { font-family:'Space Grotesk'; font-size: 27px; font-weight: 700; color: #f2fbff; letter-spacing: -.5px; line-height:1.15; margin-top:2px;}
.kpi-unit { font-size: 12px; color: #7fa6bd; font-weight: 500; }
.kpi-sub { font-size: 11px; color: #7fa6bd; margin-top: 2px; }
.chip { display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; margin-top:4px;}
.chip-good { background: rgba(52,211,153,.14); color:#34d399; border:1px solid rgba(52,211,153,.4);}
.chip-bad { background: rgba(251,113,133,.13); color:#fb7185; border:1px solid rgba(251,113,133,.4);}
.chip-info { background: rgba(34,211,238,.12); color:#22d3ee; border:1px solid rgba(34,211,238,.4);}
.badge { display:inline-flex; align-items:center; gap:5px; padding:2px 10px; border-radius:999px;
  font-size:11.5px; font-weight:600; border:1px solid; margin-right:4px; white-space:nowrap;}
.ak-hr { height:1px; border:0; margin: 8px 0 16px;
  background: linear-gradient(90deg, transparent, rgba(34,211,238,.55), rgba(59,130,246,.55), transparent); }
.sec-title { font-family:'Space Grotesk'; font-size: 21px; font-weight: 700; margin: 6px 0 0; }
.sec-sub { color:#8fb3c9; font-size: 12.5px; margin-bottom: 8px; }
.dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:6px;}
.dot-live { background:#34d399; animation: pulse 1.6s infinite; }
.dot-paused { background:#fbbf24; }
.tele-val { font-family:'Space Grotesk'; font-size: 24px; font-weight:700; color:#f2fbff; }
.alert-row { display:flex; gap:10px; align-items:flex-start; padding:9px 12px; border-radius:10px;
  border:1px solid; margin-bottom:7px; font-size:12.5px; animation: rise .4s ease both;}
.eq { font-family:'Space Grotesk'; font-size: 17px; color:#dff6ff; }
.stButton > button, [data-testid="stDownloadButton"] button {
  border-radius: 10px; border: 1px solid rgba(34,211,238,0.4);
  background: rgba(13,34,54,0.6); color: #dff6ff; font-weight: 600; transition: all .18s;
}
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  border-color:#22d3ee; box-shadow: 0 0 14px rgba(34,211,238,.25); color:#fff; transform: translateY(-1px);}
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg,#0ea5e9,#2563eb) !important; border: none !important; color: #fff !important;}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  box-shadow: 0 4px 18px rgba(37,99,235,.45) !important;}
button:disabled { opacity:.45 !important; }
[data-testid="stLinkButton"] a {
  border-radius:10px !important; border:1px solid rgba(34,211,238,.4) !important;
  background:rgba(13,34,54,.6) !important; color:#dff6ff !important; font-weight:600 !important;}
.ak-link { display:block; text-align:center; padding:8px 10px; border-radius:10px; margin-bottom:6px;
  border:1px solid rgba(34,211,238,.4); background:rgba(13,34,54,.6); color:#dff6ff !important;
  font-weight:600; text-decoration:none; font-size:13.5px;}
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 5px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
  background: rgba(13,34,54,0.5); border:1px solid transparent; border-radius:10px;
  padding: 7px 12px; width:100%; transition: all .15s; margin:0;}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
  border-color: rgba(34,211,238,.35); background: rgba(34,211,238,.08);}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has([aria-checked="true"]) {
  background: linear-gradient(135deg, rgba(34,211,238,.18), rgba(59,130,246,.18));
  border-color: rgba(34,211,238,.55);}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display:none; }
[data-testid="stSidebar"] div[role="radiogroup"] p { font-size: 14px; font-weight:600; }
[data-testid="stExpander"] { background: rgba(9,26,42,0.55); border:1px solid rgba(56,189,248,.14); border-radius:12px; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(9,26,42,0.45); border:1px solid rgba(56,189,248,.16); border-radius:14px; }
[data-testid="stDataFrame"] { border:1px solid rgba(56,189,248,.18); border-radius:12px; }
.stTabs [data-baseweb="tab"] { color:#8fb3c9; font-weight:600; font-size:13.5px;}
.stTabs [aria-selected="true"] { color:#22d3ee !important; }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg,#22d3ee,#3b82f6) !important; }
[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea, .stNumberInput input {
  background:#0a2236 !important; color:#e8f4fb !important; border-color: rgba(56,189,248,.28) !important; border-radius:10px !important;}
[data-testid="stSlider"] [role="slider"] { background:#22d3ee !important; border-color:#22d3ee !important;}
[data-testid="stAlert"] { border-radius: 12px; border:1px solid rgba(56,189,248,.25); }
.ak-foot { color:#5f7f95; font-size:11px; text-align:center; margin-top:26px; }
[data-testid="stPopover"] button { width:100%; }
</style>
"""

LOGO_SVG = """<svg width="38" height="38" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#22d3ee"/><stop offset="0.55" stop-color="#3b82f6"/><stop offset="1" stop-color="#2dd4bf"/>
</linearGradient></defs><rect x="2" y="2" width="44" height="44" rx="12" fill="url(#lg)" opacity="0.18" stroke="url(#lg)" stroke-width="1.5"/>
<path d="M8 20c4-5 8-5 12 0s8 5 12 0 8-5 8 0" fill="none" stroke="url(#lg)" stroke-width="3" stroke-linecap="round"/>
<path d="M8 29c4-5 8-5 12 0s8 5 12 0 8-5 8 0" fill="none" stroke="#22d3ee" stroke-width="3" stroke-linecap="round" opacity="0.7"/>
<path d="M8 38c4-5 8-5 12 0s8 5 12 0 8-5 8 0" fill="none" stroke="#2dd4bf" stroke-width="3" stroke-linecap="round" opacity="0.45"/></svg>"""

def sec(icon, title, sub=""):
    st.markdown(f'<div class="sec-title">{icon} <span class="grad-anim">{title}</span></div>'
                f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)

def kpi_card(label, value, unit, spark=None, color="#22d3ee", delta=None, good_down=False, sub="", gid="k"):
    chip = ""
    if delta is not None:
        good = (delta <= 0) if good_down else (delta >= 0)
        cls = "chip-good" if good else "chip-bad"
        arrow = "▼" if delta < 0 else "▲"
        chip = f'<span class="chip {cls}">{arrow} {abs(delta):.1f}% vs prior 6 mo</span>'
    sp = spark_svg(spark, color, gid) if spark is not None and len(spark) else ""
    st.markdown(f"""<div class="ak-card"><div class="kpi-label">{label}</div>
<div class="kpi-val">{value} <span class="kpi-unit">{unit}</span></div>
{chip}<div class="kpi-sub">{sub}</div>{sp}</div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# LIVE TELEMETRY SIMULATION (fragment auto-refresh; starts on load)
# ----------------------------------------------------------------------------------
def _telemetry_body():
    d = db()
    running = st.session_state.sim_running
    speed = st.session_state.sim_speed
    tele = st.session_state.telemetry
    rng = np.random.default_rng()
    if running:
        st.session_state.sim_clock += dt.timedelta(seconds=2 * speed)
        for s in d["sites"]:
            t = tele.get(s["id"])
            if t is None: continue
            shock = rng.normal(0, 0.035 * speed)
            revert = 0.08 * (t["base"] - t["rate"])
            t["rate"] = max(t["rate"] * (1 + shock) + revert, t["base"] * 0.25)
            t["hist"] = (t["hist"] + [t["rate"]])[-48:]
            t["today"] += t["rate"] * (2 * speed / 3600.0)
    clock = st.session_state.sim_clock.strftime("%H:%M:%S")
    state_chip = ('<span class="dot dot-live"></span>LIVE' if running else '<span class="dot dot-paused"></span>PAUSED')
    total_today = sum(t["today"] for t in tele.values())
    st.markdown(f'<div class="sec-sub" style="margin-top:2px">{state_chip} &nbsp;·&nbsp; simulated SCADA feed · '
                f'plant time <b>{clock}</b> · speed {speed}× · refresh 2 s · '
                f'fleet today <b>{fmt_n(total_today)} m³</b> '
                f'&nbsp;{badge("SYNTHETIC SIMULATION", "#e879f9")}</div>', unsafe_allow_html=True)
    cols = st.columns(len(d["sites"]))
    for c, s in zip(cols, d["sites"]):
        t = tele.get(s["id"], {"rate": 0, "base": 1, "hist": [0], "today": 0})
        ratio = t["rate"] / max(t["base"], 1e-9)
        col = "#fb7185" if ratio > 1.18 else ("#fbbf24" if ratio > 1.08 else "#22d3ee")
        trend = "▲" if len(t["hist"]) > 1 and t["hist"][-1] >= t["hist"][-2] else "▼"
        stress_c = STRESS_META[s["stress"]][0]
        with c:
            st.markdown(f"""<div class="ak-card">
<div class="kpi-label">{html_mod.escape(s['name'])}</div>
<div class="tele-val">{t['rate']:,.1f} <span class="kpi-unit">m³/h {trend}</span></div>
<div class="kpi-sub">today {fmt_n(t['today'])} m³ · normal {t['base']:,.0f} m³/h</div>
{spark_svg(t['hist'], col, 'tele' + s['id'], w=210, h=40)}
<div>{badge(s['stress'] + ' stress', stress_c)}</div></div>""", unsafe_allow_html=True)
    if not running:
        st.caption("Telemetry paused — resume from the sidebar. Analytics below use the audited ledger, not the live feed.")

_frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if _frag:
    @_frag(run_every=2)
    def live_telemetry():
        _telemetry_body()
else:
    def live_telemetry():
        _telemetry_body()
        if st.button("↻ Refresh telemetry", key="tele_refresh"):
            rerun()

# ----------------------------------------------------------------------------------
# SHARING (visible on every page via the top bar + full hub on Reports & Share)
# ----------------------------------------------------------------------------------
def best_scenario():
    scs = db()["scenarios"]
    approved = [s for s in scs if s["status"] == "Approved"]
    return (approved or scs or [None])[-1]

def build_share_link():
    base = st.session_state.share_base.rstrip("/")
    key = next((k for k, l in NAV if l == st.session_state.nav), "overview")
    params = {"page": key}
    sites = st.session_state.get("ov_sites")
    if sites and len(sites) < len(db()["sites"]):
        params["sites"] = ",".join(sites)
    sc = st.session_state.get("loaded_scenario") or (best_scenario()["id"] if best_scenario() else None)
    if sc:
        params["sc"] = sc
    return base + "/?" + urllib.parse.urlencode(params)

def exec_summary():
    d = db()
    df = active_df(d)
    W = float(df["withdrawal_m3"].sum()) if len(df) else 0.0
    C = float(df["consumption_m3"].sum()) if len(df) else 0.0
    D = float(df["discharge_m3"].sum()) if len(df) else 0.0
    R = float(df["recycled_m3"].sum()) if len(df) else 0.0
    sm = {s["id"]: s["stress_factor"] for s in d["sites"]}
    SW = float((df["withdrawal_m3"] * df["site_id"].map(sm)).sum()) if len(df) else 0.0
    pend = sum(1 for r in d["records"] if r["status"] == "In Review")
    exc = len(all_exceptions())
    lines = [f"🌊 {APP_NAME} — water balance snapshot (synthetic demo)",
             f"• {len(d['sites'])} sites · {len(df)} records · trailing 12 months",
             f"• Withdrawal {compact(W)} m³ | Consumed {C/max(W,1)*100:.0f}% | Discharged {D/max(W,1)*100:.0f}% | Reuse {R/max(W+R,1)*100:.0f}%",
             f"• Stress-weighted withdrawal {compact(SW)} m³-eq"]
    sc = best_scenario()
    if sc:
        r = sc["results"]
        pb = f"{r['payback_p50']:.1f}y" if r.get("payback_p50") else "—"
        lines.append(f"• Portfolio “{sc['name']}” [{sc['status']}]: {compact(r['sav']['p50'])} m³/yr saved "
                     f"(P10–P90 {compact(r['sav']['p10'])}–{compact(r['sav']['p90'])}), payback {pb}, "
                     f"{r['prob_target']*100:.0f}% chance of the reduction target")
    lines.append(f"• {pend} records awaiting review · {exc} data-quality flags · rules {RULES_VERSION}")
    lines.append("Decision support only — synthetic data; human approval required.")
    if st.session_state.share_link:
        lines.append(st.session_state.share_link)
    return "\n".join(lines)

def share_buttons(link, summary):
    enc = urllib.parse.quote(link, safe="")
    short = summary.split("\n")[0] + " — live interactive demo"
    c1, c2 = st.columns(2)
    with c1:
        link_btn("💼 LinkedIn", f"https://www.linkedin.com/sharing/share-offsite/?url={enc}")
        link_btn("💬 WhatsApp", f"https://wa.me/?text={urllib.parse.quote(summary[:900], safe='')}")
    with c2:
        link_btn("𝕏 Post", f"https://twitter.com/intent/tweet?text={urllib.parse.quote(short, safe='')}&url={enc}")
        link_btn("✉️ Email", f"mailto:?subject={urllib.parse.quote(APP_NAME + ' — water balance demo')}&body={urllib.parse.quote(summary[:1500], safe='')}")
    st.caption("LinkedIn opens its link-share dialog — paste the copied summary for the full text. "
               "For recipients outside your machine, deploy the app and set its public URL in Reports & Share.")

def share_hub(compact_mode=False):
    if st.session_state.share_link is None:
        st.session_state.share_link = build_share_link()
    link = st.session_state.share_link
    if "share_summary" not in st.session_state:
        st.session_state.share_summary = exec_summary()
    summary = st.session_state.share_summary
    if compact_mode:
        copy_button(summary, "📋 Copy executive summary", key="cp_sum_top")
        copy_button(link, "🔗 Copy link to this view", key="cp_link_top")
        share_buttons(link, summary)
        if st.button("📤 Open full Share hub", key="open_share_hub", use_container_width=True):
            goto("reports")
        return
    c1, c2 = st.columns([0.42, 0.58])
    with c1:
        st.markdown("**🔗 Shareable link** — restores this page, site filter and scenario")
        st.text_input("Public app URL", key="share_base",
                      help="localhost works for you; set your deployed URL for external sharing.")
        if st.button("🔄 Rebuild link from current view", key="rebuild_link", use_container_width=True):
            st.session_state.share_link = build_share_link()
            st.session_state.share_summary = exec_summary()
            audit("share_link", st.session_state.share_link, "Share link prepared from current view.")
            rerun()
        st.code(st.session_state.share_link, language=None)
        copy_button(st.session_state.share_link, "🔗 Copy link", key="cp_link_full")
        if HAS_QR:
            buf = io.BytesIO()
            qrcode.make(st.session_state.share_link, box_size=6, border=2).save(buf, format="PNG")
            st.image(buf.getvalue(), width=170, caption="Scan to open this view")
    with c2:
        st.markdown("**📝 Executive summary** — editable, clipboard-ready")
        st.text_area("Summary", key="share_summary", height=190, label_visibility="collapsed")
        cc1, cc2 = st.columns(2)
        with cc1:
            copy_button(st.session_state.share_summary, "📋 Copy summary", key="cp_sum_full")
        with cc2:
            if st.button("♻️ Regenerate from live data", key="regen_sum", use_container_width=True):
                st.session_state.share_summary = exec_summary()
                audit("share_summary", "clipboard", "Executive summary regenerated from live ledger.")
                rerun()
        share_buttons(st.session_state.share_link, st.session_state.share_summary)

# ----------------------------------------------------------------------------------
# TOP BAR + SIDEBAR
# ----------------------------------------------------------------------------------
def topbar():
    c1, c2, c3, c4 = st.columns([0.46, 0.18, 0.18, 0.18])
    with c1:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:12px">
{LOGO_SVG}<div><div class="ak-title" style="font-size:24px;font-weight:700">
<span class="grad-anim">{APP_NAME}</span></div>
<div style="font-size:11.5px;color:#8fb3c9">{TAGLINE} · {badge('SYNTHETIC DEMO', '#e879f9')}</div></div></div>""",
                    unsafe_allow_html=True)
    with c2:
        live = st.session_state.sim_running
        dot = "dot-live" if live else "dot-paused"
        st.markdown(f'<div class="ak-card" style="padding:9px 12px;text-align:center">'
                    f'<span class="dot {dot}"></span><b>{"LIVE SIM" if live else "PAUSED"}</b>'
                    f'<div class="kpi-sub">telemetry · {st.session_state.sim_speed}×</div></div>', unsafe_allow_html=True)
    with c3:
        r = role()
        st.markdown(f'<div class="ak-card" style="padding:9px 12px;text-align:center">'
                    f'<b>{ROLE_ICON[r]} {r}</b><div class="kpi-sub">{len(PERMS[r])} permissions</div></div>',
                    unsafe_allow_html=True)
    with c4:
        with popover_w("📤 Share"):
            st.markdown("**Share this view**")
            share_hub(compact_mode=True)
    st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
{LOGO_SVG}<div><b class="ak-title" style="font-size:19px">AquaLedger</b>
<div style="font-size:10.5px;color:#7fa6bd">v{VERSION} · rules {RULES_VERSION}</div></div></div>""",
                    unsafe_allow_html=True)
        st.radio("Navigate", NAV_LABELS, key="nav", label_visibility="collapsed")
        st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)
        st.selectbox("Demo role", ROLES, key="role", format_func=lambda r: f"{ROLE_ICON[r]} {r}",
                     on_change=on_role_change,
                     help="Demo switcher. Production enforces authorization server-side; the UI mirrors the permission matrix in Governance.")
        st.caption("Switch to **Stakeholder** to see read-only gating, or **Reviewer** for the approval queue.")
        st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)
        toggle_w("▶ Live telemetry", key="sim_running", help="Auto-starts on load; pause any time.")
        st.select_slider("Simulation speed", options=[1, 2, 5], key="sim_speed", format_func=lambda x: f"{x}×")
        d = db()
        pend = sum(1 for r in d["records"] if r["status"] == "In Review")
        exc = len(all_exceptions())
        st.markdown(f"""<div class="ak-card" style="padding:10px 12px">
<div class="kpi-sub">📒 {len(d['records'])} records · 🕳 {pend} in review</div>
<div class="kpi-sub">⚠ {exc} quality flags · 🧪 {len(d['scenarios'])} scenarios</div></div>""", unsafe_allow_html=True)
        sc = st.session_state.selfcheck
        passed = sum(1 for x in sc if "PASS" in x["Result"])
        color = "#34d399" if passed == len(sc) else "#fb7185"
        st.markdown(f'<div style="text-align:center;margin-top:8px">{badge(f"🧪 Self-check {passed}/{len(sc)}", color)}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ak-foot">100% open-source · no API keys<br>session-scoped demo database</div>',
                    unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# RECORD DETAIL (shared by Ledger + Review)
# ----------------------------------------------------------------------------------
def record_detail(rec, review_mode=False):
    d = db()
    site = next(s for s in d["sites"] if s["id"] == rec["site_id"])
    res, tol, bstate = balance_check(rec)
    bcolor = {"balanced": "#34d399", "warning": "#fbbf24", "error": "#fb7185"}[bstate]
    flags = quality_flags(rec)
    flag_html = " ".join(badge(f.replace("_", " "), "#fb7185" if f == "balance_mismatch" else "#fbbf24") for f in flags) or badge("no flags", "#34d399")
    st.markdown(f"""<div class="ak-card"><div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px">
<div><b class="ak-title" style="font-size:17px">{rec['id']}</b>
<span class="kpi-sub"> v{rec['version']} · {html_mod.escape(site['name'])} · {html_mod.escape(proc_name(rec['process_id']))} · {rec['period']}</span></div>
<div>{status_badge(rec['status'])} {flag_html}</div></div></div>""", unsafe_allow_html=True)
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        src_txt = " · ".join(f"{k} {fmt_n(v)}" for k, v in rec["sources"].items() if v > 0)
        reuse = rec["recycled_m3"] / max(rec["withdrawal_m3"] + rec["recycled_m3"], 1) * 100
        intensity = rec["withdrawal_m3"] / max(rec["output_units"], 1e-9)
        st.markdown(f"""<div class="ak-card">
<div class="kpi-label">Volumes (m³)</div>
<div class="kpi-sub" style="font-size:13px;color:#dff6ff">
Withdrawal <b>{fmt_n(rec['withdrawal_m3'])}</b> &nbsp;·&nbsp; Recycled <b>{fmt_n(rec['recycled_m3'])}</b><br>
Consumption <b>{fmt_n(rec['consumption_m3'])}</b> &nbsp;·&nbsp; Discharge <b>{fmt_n(rec['discharge_m3'])}</b> &nbsp;·&nbsp; ΔStorage <b>{fmt_1(rec['storage_change_m3'])}</b></div>
<div class="kpi-sub">Sources: {src_txt}</div>
<div class="kpi-sub">Output {fmt_1(rec['output_units'])} {rec['output_unit']} → intensity <b>{intensity:.2f} m³/unit</b> · reuse <b>{reuse:.1f}%</b></div>
<div class="kpi-sub">Metering: {badge(rec['metering'], '#22d3ee')} ±{rec['uncertainty_pct']:.0f}% measurement uncertainty</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="ak-card"><div class="kpi-label">Balance check · {RULES_VERSION}</div>
<div class="eq">{fmt_n(rec['withdrawal_m3'])} = {fmt_n(rec['consumption_m3'])} + {fmt_n(rec['discharge_m3'])} + {fmt_1(rec['storage_change_m3'])}</div>
<div class="kpi-sub">W = C + D + ΔS → residual <b style="color:{bcolor}">{fmt_1(res)} m³</b> · tolerance ±{fmt_1(tol)} m³ {badge(bstate, bcolor)}</div>
<div class="kpi-sub">Site permit: {fmt_n(site['permit_m3_month'])} m³/month discharge · basin stress {badge(site['stress'], STRESS_META[site['stress']][0])}</div>
</div>""", unsafe_allow_html=True)
    with st.expander("🔍 Explanation panel — inputs, rules, lineage, uncertainty, review"):
        st.markdown(f"""
- **Inputs**: the volumes above; output denominator from site production; uncertainty by metering class (metered ±2%, calculated ±8%, estimated ±15%).
- **Rules**: balance `W = C + D + ΔS`, tolerance `max(2%·W, 5 m³)`; reuse `R/(W+R)`; intensity `W/output`; stress-weighting `W × site factor` — engine **{RULES_VERSION}**, reference data **{d['config']['ref_version']}**.
- **Factor provenance**: stress multipliers and tariffs are **synthetic demo values** — replace with licensed WRI Aqueduct stress layers and local tariff schedules before real use.
- **Uncertainty**: ±{rec['uncertainty_pct']:.0f}% on withdrawal propagates to site totals and scenario baselines.
- **Data quality**: {', '.join(flags) if flags else 'no flags'} · reviewer status **{rec['status']}**{' — ' + html_mod.escape(rec['review_comment']) if rec['review_comment'] else ''}.
- **Boundary**: decision support only; approvals require an authorized human. No causality is implied by any analytics.""")
        prov = rec["provenance"]
        st.markdown(f"**Lineage** — kind `{prov['kind']}` · file `{prov['file'] or '—'}` · row `{prov['row'] or '—'}` · checksum `{(prov['checksum'] or '')[:20]}…`")
        st.markdown(f"Created `{rec['created_at']}` by {rec['entered_by']} · updated `{rec['updated_at']}`"
                    + (f" · reviewed by {rec['reviewed_by']}" if rec["reviewed_by"] else "")
                    + (f" · supersedes `{rec['supersedes']}`" if rec["supersedes"] else "")
                    + (f" · superseded by `{rec['superseded_by']}`" if rec.get("superseded_by") else ""))
        if rec["notes"]:
            st.info("Note: " + rec["notes"])
    # ---- lifecycle actions ----
    if review_mode or rec["status"] == "In Review":
        comment = st.text_input("Reviewer comment (required for rejection or flagged approval)",
                                key=f"rev_c_{rec['id']}")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("✅ Approve", key=f"ap_{rec['id']}", type="primary", disabled=not can("approve"),
                         use_container_width=True):
                act_approve(rec["id"], comment); rerun()
        with b2:
            if st.button("↩️ Reject", key=f"rj_{rec['id']}", disabled=not can("approve"), use_container_width=True):
                act_reject(rec["id"], comment); rerun()
        with b3:
            if st.button("🗄️ Archive", key=f"ar_{rec['id']}", disabled=not can("archive"), use_container_width=True):
                act_archive(rec["id"]); rerun()
        if not can("approve"):
            st.caption(f"{ROLE_ICON[role()]} {role()} cannot approve — switch to Reviewer or Administrator.")
    else:
        b1, b2, b3 = st.columns(3)
        with b1:
            submittable = rec["status"] in ("Draft", "Validated")
            if st.button("📨 Submit for review", key=f"sb_{rec['id']}", disabled=not (can("submit") and submittable),
                         use_container_width=True):
                act_submit(rec["id"]); rerun()
        with b2:
            if rec["status"] == "Archived":
                if st.button("♻️ Restore", key=f"rs_{rec['id']}", disabled=not can("archive"), use_container_width=True):
                    rec["status"] = "Draft"; rec["updated_at"] = utcnow()
                    audit("restore", rec["id"], "Restored from archive as Draft.")
                    st.toast(f"{rec['id']} restored as Draft", icon="♻️"); rerun()
            else:
                if st.button("🗄️ Archive", key=f"ar2_{rec['id']}", disabled=not can("archive"), use_container_width=True):
                    act_archive(rec["id"]); rerun()
        with b3:
            st.caption("Lifecycle: Draft → Validated → In Review → Approved / Rejected → correction (new version) → Archived.")
    # ---- correction editor ----
    if can("edit") and rec["status"] != "Archived":
        with st.expander("📝 Correct record (creates a new version — never overwrites)"):
            k = rec["id"]
            cc = st.columns(4)
            src_vals = {}
            for i, sname in enumerate(["Municipal", "Groundwater", "Surface", "Harvested"]):
                src_vals[sname] = cc[i].number_input(f"{sname} m³", value=float(rec["sources"].get(sname, 0.0)),
                                                     min_value=0.0, step=10.0, key=f"corr_{k}_{sname}")
            cc2 = st.columns(4)
            newC = cc2[0].number_input("Consumption m³", value=float(rec["consumption_m3"]), min_value=0.0, step=10.0, key=f"corr_{k}_C")
            newD = cc2[1].number_input("Discharge m³", value=float(rec["discharge_m3"]), min_value=0.0, step=10.0, key=f"corr_{k}_D")
            newS = cc2[2].number_input("ΔStorage m³", value=float(rec["storage_change_m3"]), step=1.0, key=f"corr_{k}_S")
            newR = cc2[3].number_input("Recycled m³", value=float(rec["recycled_m3"]), min_value=0.0, step=10.0, key=f"corr_{k}_R")
            newO = st.number_input(f"Output ({rec['output_unit']})", value=float(rec["output_units"]), min_value=0.0, step=1.0, key=f"corr_{k}_O")
            Wp = sum(src_vals.values())
            resp = Wp - (newC + newD + newS)
            okp = abs(resp) <= balance_tolerance(Wp)
            st.markdown(f"Live preview: W {fmt_n(Wp)} − (C+D+ΔS) = **{fmt_1(resp)} m³** "
                        + (badge("balanced", "#34d399") if okp else badge("mismatch", "#fb7185")), unsafe_allow_html=True)
            reason = st.text_input("Correction reason (required, audited)", key=f"corr_{k}_reason")
            if st.button("💾 Save as new version", key=f"corr_{k}_save", type="primary"):
                act_correction(rec["id"], {"sources": src_vals, "consumption_m3": newC, "discharge_m3": newD,
                                           "storage_change_m3": newS, "recycled_m3": newR, "output_units": newO}, reason)
                rerun()
    # ---- admin override ----
    if can("override") and rec["status"] != "Archived":
        with st.expander("🛡️ Manual override (Administrator — reason + audit required)"):
            f = st.selectbox("Field", ["consumption_m3", "uncertainty_pct"], key=f"ovr_{rec['id']}_f")
            v = st.number_input("New value", value=float(rec[f]), step=0.5, key=f"ovr_{rec['id']}_v")
            why = st.text_input("Override reason", key=f"ovr_{rec['id']}_why")
            if st.button("Apply override", key=f"ovr_{rec['id']}_go"):
                act_override(rec["id"], f, v, why); rerun()

# ----------------------------------------------------------------------------------
# PAGE: OVERVIEW
# ----------------------------------------------------------------------------------
def page_overview():
    d = db()
    site_ids = [s["id"] for s in d["sites"]]
    pers = periods_list()
    sanitize_multi("ov_sites", site_ids)
    sanitize_range("ov_periods", pers)
    f1, f2 = st.columns([0.5, 0.5])
    with f1:
        st.multiselect("Sites", site_ids, default=site_ids, key="ov_sites", format_func=site_name)
    with f2:
        st.select_slider("Periods", options=pers, value=(pers[0], pers[-1]), key="ov_periods")
    lo, hi = st.session_state.ov_periods
    df = active_df(d)
    df = df[df["site_id"].isin(st.session_state.ov_sites) & df["period"].between(lo, hi)]
    if not len(df):
        st.warning("No records in the current filter. Widen the site/period selection.")
        return
    sec("🌊", "Live Water Balance", "Withdrawals, consumption and discharge by site & process — every number traceable to an audited record.")
    g = df.groupby("period")
    Wseries = g["withdrawal_m3"].sum().sort_index()
    Cseries = g["consumption_m3"].sum().sort_index()
    Dseries = g["discharge_m3"].sum().sort_index()
    Rseries = (g["recycled_m3"].sum() / (g["withdrawal_m3"].sum() + g["recycled_m3"].sum()) * 100).sort_index()
    W, C, D, R = float(df["withdrawal_m3"].sum()), float(df["consumption_m3"].sum()), float(df["discharge_m3"].sum()), float(df["recycled_m3"].sum())
    sm = {s["id"]: s["stress_factor"] for s in d["sites"]}
    SW = float((df["withdrawal_m3"] * df["site_id"].map(sm)).sum())
    appr = float((df["status"] == "Approved").mean() * 100)
    def delta6(series):
        v = series.values
        if len(v) < 4: return None
        h = max(len(v) // 2, 1)
        prev, cur = float(np.mean(v[:h])), float(np.mean(v[h:]))
        return (cur - prev) / prev * 100 if prev else None
    cols = st.columns(6)
    with cols[0]: kpi_card("Withdrawal", compact(W), "m³", Wseries.values, "#22d3ee", delta6(Wseries), True, "sum of 4 sources", "w")
    with cols[1]: kpi_card("Consumed", compact(C), "m³", Cseries.values, "#fbbf24", delta6(Cseries), True, f"{C/max(W,1)*100:.0f}% of withdrawal · evap/product", "c")
    with cols[2]: kpi_card("Discharged", compact(D), "m³", Dseries.values, "#60a5fa", delta6(Dseries), True, f"{D/max(W,1)*100:.0f}% of withdrawal · permit-checked", "d")
    with cols[3]: kpi_card("Reuse rate", f"{R/max(W+R,1)*100:.1f}", "%", Rseries.values, "#34d399", delta6(Rseries), False, "recycled / total demand", "r")
    with cols[4]: kpi_card("Stress-weighted", compact(SW), "m³-eq", None, "#fb7185", None, False, "W × basin stress factor (synthetic)", "s")
    with cols[5]: kpi_card("Approved share", f"{appr:.0f}", "%", None, "#a78bfa", None, False, f"{int((df['status']=='Approved').sum())}/{len(df)} records approved", "a")
    st.markdown("")
    live_telemetry()
    st.markdown("")
    c1, c2 = st.columns([0.62, 0.38])
    with c1:
        sec("🔀", "Source → Site → Fate", "Where every cubic metre comes from and where it ends up. Hover flows for volumes.")
        st.plotly_chart(sankey_fig(df), use_container_width=True, config=PLOT_CFG, key="ov_sankey")
        top_site = df.groupby("site_id")["withdrawal_m3"].sum().idxmax()
        st.caption(f"💡 {site_name(top_site)} drives {df[df['site_id']==top_site]['withdrawal_m3'].sum()/W*100:.0f}% of filtered withdrawal — start conservation screening there.")
    with c2:
        sec("🚨", "Exceptions & data quality", "Contradictory, late, out-of-range or missing-evidence records — the exception path.")
        exc = [e for e in all_exceptions() if find_rec(e["rid"]) and find_rec(e["rid"])["site_id"] in st.session_state.ov_sites]
        if not exc:
            st.success("No open data-quality exceptions in scope. 🎉")
        for i, e in enumerate(exc[:4]):
            color = "#fb7185" if e["sev"] == "error" else "#fbbf24"
            cc1, cc2 = st.columns([0.78, 0.22])
            with cc1:
                st.markdown(f'<div class="alert-row" style="border-color:{color}55;background:{color}0d">'
                            f'<span>{"🛑" if e["sev"]=="error" else "⚠️"}</span>'
                            f'<div><b>{html_mod.escape(e["label"])}</b><br>{html_mod.escape(e["msg"])}</div></div>',
                            unsafe_allow_html=True)
            with cc2:
                if st.button("Open →", key=f"jump_{i}_{e['rid']}"):
                    st.session_state.selected_record = e["rid"]
                    goto("ledger")
        st.plotly_chart(metering_fig(df), use_container_width=True, config=PLOT_CFG, key="ov_meter")
        est = float((df["metering"] == "estimated").mean() * 100)
        st.caption(f"💡 {est:.0f}% of records are estimates (±15%) — prioritize metering to tighten scenario confidence.")
    c3, c4 = st.columns([0.6, 0.4])
    with c3:
        sec("📅", "Monthly balance by source", "Seasonality drives cooling and irrigation peaks — the conservation timing window.")
        st.plotly_chart(monthly_fig(df), use_container_width=True, config=PLOT_CFG, key="ov_month")
    with c4:
        sec("🎯", "Intensity vs reuse", "Bubble = withdrawal volume. Bottom-right quadrant = highest conservation priority.")
        st.plotly_chart(bubble_fig(df), use_container_width=True, config=PLOT_CFG, key="ov_bubble")
        if st.button("🧪 Model conservation projects →", key="go_lab", type="primary", use_container_width=True):
            goto("lab")
    sec("🏭", "Top processes by withdrawal", "Consumed vs discharged split — consumptive losses and recoverable discharge are different levers.")
    st.plotly_chart(top_processes_fig(df), use_container_width=True, config=PLOT_CFG, key="ov_top")

# ----------------------------------------------------------------------------------
# PAGE: WATER LEDGER
# ----------------------------------------------------------------------------------
def page_records():
    d = db()
    sec("📒", "Water Ledger", "Site × process × month records with lifecycle status, balance checks and full lineage. Click a row to open it.")
    if st.session_state.undo:
        u = st.session_state.undo
        c1, c2, c3 = st.columns([0.6, 0.2, 0.2])
        c1.warning(f"{u['id']} was archived.")
        if c2.button("↩️ Undo", key="undo_btn", use_container_width=True):
            act_undo(); rerun()
        if c3.button("Dismiss", key="undo_dismiss", use_container_width=True):
            st.session_state.undo = None; rerun()
    site_ids = [s["id"] for s in d["sites"]]
    pers = periods_list()
    sanitize_multi("rec_sites", site_ids)
    sanitize_range("rec_periods", pers)
    f = st.columns([0.28, 0.2, 0.16, 0.36])
    f[0].multiselect("Sites", site_ids, default=site_ids, key="rec_sites", format_func=site_name)
    f[1].multiselect("Status", list(STATUS_META), default=list(STATUS_META), key="rec_status")
    f[2].multiselect("Metering", list(METER_UNC), default=list(METER_UNC), key="rec_meter")
    f[3].select_slider("Periods", options=pers, value=(pers[0], pers[-1]), key="rec_periods")
    s1, s2 = st.columns([0.6, 0.4])
    search = s1.text_input("Search", key="rec_search", placeholder="Search id, site, process, notes…")
    sort = s2.selectbox("Sort by", ["Period (newest)", "Withdrawal (high→low)", "Balance residual (worst)", "Status"], key="rec_sort")
    lo, hi = st.session_state.rec_periods
    rows = []
    for r in d["records"]:
        if r["site_id"] not in st.session_state.rec_sites: continue
        if r["status"] not in st.session_state.rec_status: continue
        if r["metering"] not in st.session_state.rec_meter: continue
        if not (lo <= r["period"] <= hi): continue
        hay = f"{r['id']} {site_name(r['site_id'])} {proc_name(r['process_id'])} {r['notes']}".lower()
        if search and search.lower() not in hay: continue
        res, tol, bst = balance_check(r)
        rows.append({"ID": r["id"], "Site": site_name(r["site_id"]), "Process": proc_name(r["process_id"]),
                     "Period": r["period"], "Withdrawal m³": r["withdrawal_m3"], "Consumed m³": r["consumption_m3"],
                     "Discharged m³": r["discharge_m3"], "Reuse %": r["recycled_m3"] / max(r["withdrawal_m3"] + r["recycled_m3"], 1) * 100,
                     "Residual m³": res, "Balance": bst, "Meter": r["metering"], "Status": r["status"],
                     "Flags": ", ".join(quality_flags(r)) or "—"})
    rdf = pd.DataFrame(rows)
    if sort == "Period (newest)": rdf = rdf.sort_values(["Period", "ID"], ascending=False)
    elif sort == "Withdrawal (high→low)": rdf = rdf.sort_values("Withdrawal m³", ascending=False)
    elif sort == "Balance residual (worst)": rdf = rdf.sort_values("Residual m³", key=lambda s: s.abs(), ascending=False)
    else: rdf = rdf.sort_values("Status")
    rdf = rdf.reset_index(drop=True)
    st.caption(f"{len(rdf)} records match · {sum(1 for r in d['records'] if r['status']=='In Review')} awaiting review")
    cfg = {"Withdrawal m³": st.column_config.NumberColumn(format="%.0f"),
           "Consumed m³": st.column_config.NumberColumn(format="%.0f"),
           "Discharged m³": st.column_config.NumberColumn(format="%.0f"),
           "Reuse %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
           "Residual m³": st.column_config.NumberColumn(format="%.1f")}
    picked = None
    if st.session_state.row_select_ok:
        try:
            ev = st.dataframe(rdf, use_container_width=True, height=360, hide_index=True,
                              column_config=cfg, on_select="rerun", selection_mode="single-row", key="rec_table")
            if ev and ev.selection and ev.selection.rows:
                picked = rdf.iloc[ev.selection.rows[0]]["ID"]
        except Exception:
            st.session_state.row_select_ok = False
            rerun()
    if not st.session_state.row_select_ok:
        st.dataframe(rdf, use_container_width=True, height=360, hide_index=True, column_config=cfg)
        opts = ["—"] + rdf["ID"].tolist()
        sel = st.selectbox("Open record", opts, key="rec_open")
        picked = sel if sel != "—" else None
    if picked:
        st.session_state.selected_record = picked
    st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)
    rid = st.session_state.selected_record
    rec = find_rec(rid) if rid else None
    if rec:
        record_detail(rec)
    else:
        st.info("Select a row above to open its balance check, explanation panel, lineage and lifecycle actions.")
    with st.expander("➕ Add a manual record (Analyst / Administrator)"):
        if not can("edit"):
            st.caption(f"{ROLE_ICON[role()]} {role()} has read-only data access.")
        else:
            c1, c2, c3 = st.columns(3)
            sid = c1.selectbox("Site", site_ids, format_func=site_name, key="add_site")
            popts = [p["id"] for p in d["processes"] if p["site_id"] == sid]
            sanitize_select("add_proc", popts)
            pid = c2.selectbox("Process", popts, format_func=proc_name, key="add_proc")
            per = c3.text_input("Period (YYYY-MM)", value="2026-11", key="add_per")
            v = st.columns(4)
            src = {s: v[i].number_input(f"{s} m³", min_value=0.0, step=10.0, key=f"add_src_{s}",
                                        value={"Municipal": 1000.0}.get(s, 0.0)) for i, s in enumerate(["Municipal", "Groundwater", "Surface", "Harvested"])}
            v2 = st.columns(5)
            Cn = v2[0].number_input("Consumption m³", min_value=0.0, value=700.0, step=10.0, key="add_C")
            Dn = v2[1].number_input("Discharge m³", min_value=0.0, value=295.0, step=10.0, key="add_D")
            Sn = v2[2].number_input("ΔStorage m³", value=5.0, step=1.0, key="add_S")
            Rn = v2[3].number_input("Recycled m³", min_value=0.0, value=100.0, step=10.0, key="add_R")
            On = v2[4].number_input("Output units", min_value=0.0, value=300.0, step=1.0, key="add_O")
            m = st.selectbox("Metering class", list(METER_UNC), key="add_meter")
            note = st.text_input("Notes", key="add_note")
            Wp = sum(src.values()); resp = Wp - (Cn + Dn + Sn)
            ok = abs(resp) <= balance_tolerance(Wp)
            st.markdown(f"Live balance: **{fmt_1(resp)} m³** residual " +
                        (badge("balanced → will create as Validated", "#34d399") if ok else
                         badge("mismatch → will create as Draft, flagged (exception path)", "#fbbf24")),
                        unsafe_allow_html=True)
            if st.button("✨ Create record", key="add_go", type="primary", disabled=Wp <= 0):
                act_create({"site_id": sid, "process_id": pid, "period": per.strip(), "sources": src,
                            "consumption_m3": Cn, "discharge_m3": Dn, "storage_change_m3": Sn,
                            "recycled_m3": Rn, "output_units": On, "metering": m, "notes": note})
                rerun()

# ----------------------------------------------------------------------------------
# PAGE: IMPORT & VALIDATE
# ----------------------------------------------------------------------------------
def page_import():
    d = db()
    sec("📥", "Import & Validate", "CSV / Excel → structural checks → row-level validation → preview → safe commit. Originals are preserved; validated data is never silently overwritten.")
    st.markdown(badge("1 Upload", "#22d3ee") + badge("2 Validate", "#3b82f6") + badge("3 Preview", "#fbbf24") + badge("4 Commit", "#34d399"), unsafe_allow_html=True)
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        up = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"], key="imp_uploader")
        st.radio("Re-import strategy", ["Skip duplicates (safe)", "Import as correction (supersedes)"], key="imp_mode",
                 help="Correction mode archives the old version and links a new one — full lineage, no silent overwrite.")
        if up is not None:
            raw = up.getvalue()
            if len(raw) > 5_000_000:
                st.error("File exceeds the 5 MB upload limit.")
            else:
                try:
                    df_in = pd.read_excel(io.BytesIO(raw), engine="openpyxl") if up.name.lower().endswith(".xlsx") else pd.read_csv(io.BytesIO(raw))
                    st.session_state.import_preview = {"df": df_in, "filename": up.name,
                                                       "sha": sha256_bytes(raw), "bytes": raw}
                except Exception as e:
                    st.error(f"Could not parse file: {e}. Use the template's column names and UTF-8 CSV or a single-sheet XLSX.")
    with c2:
        st.markdown("**No file handy? The demo works instantly:**")
        kind = st.radio("Generated sample", ["Valid file (3 new November rows)", "Invalid file (8 deliberate problems)"], key="imp_sample_kind")
        if st.button("⚡ Load sample into validator", key="imp_load_sample", type="primary", use_container_width=True):
            valid = kind.startswith("Valid")
            raw = sample_valid_csv() if valid else sample_invalid_csv()
            name = "sample_valid_2026-11.csv" if valid else "sample_invalid_demo.csv"
            st.session_state.import_preview = {"df": pd.read_csv(io.BytesIO(raw)), "filename": name,
                                               "sha": sha256_bytes(raw), "bytes": raw}
            audit("import_preview", name, "Generated sample loaded into validator.")
            rerun()
        t1, t2, t3 = st.columns(3)
        t1.download_button("⬇ Template", pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(index=False).encode(),
                           "aqualedger_template.csv", "text/csv", use_container_width=True)
        t2.download_button("⬇ Valid sample", sample_valid_csv(), "sample_valid.csv", "text/csv", use_container_width=True)
        t3.download_button("⬇ Invalid sample", sample_invalid_csv(), "sample_invalid.csv", "text/csv", use_container_width=True)
        with st.expander("Expected columns & rules"):
            st.markdown("`site_id, process_id, period(YYYY-MM), municipal_m3, groundwater_m3, surface_m3, harvested_m3, recycled_m3, consumption_m3, discharge_m3, storage_change_m3, output_units, metering(metered|calculated|estimated), notes`")
            st.markdown("Checks: required columns · numeric types · non-negative volumes · valid period · referential integrity (site/process) · duplicate keys (file + ledger) · balance `W=C+D+ΔS` within `max(2%·W, 5 m³)` · permit caps · late entries vs lock period · implausible magnitudes.")
    prev = st.session_state.import_preview
    if prev:
        st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)
        result = validate_import(prev["df"], d, st.session_state.imp_mode)
        st.markdown(f"**Preview:** `{prev['filename']}` · sha256 `{prev['sha'][:20]}…`")
        hard_fatal = [f for f in result["fatal"] if not f.startswith("NOTE")]
        for fmsg in result["fatal"]:
            (st.error if not fmsg.startswith("NOTE") else st.info)(fmsg)
        if not hard_fatal:
            rows = result["rows"]
            n_err = sum(1 for r in rows if r["errors"])
            n_warn = sum(1 for r in rows if r["warnings"] and not r["errors"])
            n_ok = sum(1 for r in rows if not r["errors"] and r["action"] != "skip")
            n_skip = sum(1 for r in rows if r["action"] == "skip" and not r["errors"])
            m = st.columns(5)
            m[0].metric("Rows", len(rows)); m[1].metric("✅ Importable", n_ok)
            m[2].metric("⚠ With warnings", n_warn); m[3].metric("🛑 Rejected", n_err); m[4].metric("⏭ Skipped", n_skip)
            preview_rows = []
            for r in rows:
                status = "🛑 " + "; ".join(r["errors"]) if r["errors"] else ("⏭ skipped (exists)" if r["action"] == "skip" else ("⚠ " + "; ".join(r["warnings"]) if r["warnings"] else "✅ " + r["action"]))
                preview_rows.append({**r["raw"], "_validation": status[:300], "_action": r["action"]})
            pdf = pd.DataFrame(preview_rows)
            def row_style(row):
                s = row["_validation"]
                if s.startswith("🛑"): return ["background-color: rgba(251,113,133,0.14)"] * len(row)
                if s.startswith("⚠") or s.startswith("⏭"): return ["background-color: rgba(251,191,36,0.10)"] * len(row)
                return ["background-color: rgba(52,211,153,0.08)"] * len(row)
            st.dataframe(pdf.head(100).style.apply(row_style, axis=1), use_container_width=True, height=300, hide_index=True)
            if len(pdf) > 100: st.caption("Showing first 100 rows.")
            ic1, ic2, ic3 = st.columns(3)
            if len(result["issues"]):
                ic1.download_button("⬇ Error report (CSV)", result["issues"].to_csv(index=False).encode(),
                                    f"errors_{prev['filename']}.csv", "text/csv", use_container_width=True)
            clean = [r for r in rows if r["record"] and not r["errors"] and r["action"] != "skip"]
            if clean:
                clean_df = pd.DataFrame([{**r["raw"]} for r in clean])
                ic2.download_button("⬇ Clean preview (CSV)", clean_df.to_csv(index=False).encode(),
                                    "clean_preview.csv", "text/csv", use_container_width=True)
            with ic3:
                if st.button(f"📥 Commit {len(clean)} rows", key="imp_commit", type="primary",
                             disabled=not can("import") or not clean, use_container_width=True):
                    commit_import(prev, rows, prev["filename"], prev["sha"], prev["bytes"])
                    rerun()
            if not can("import"):
                st.caption(f"{ROLE_ICON[role()]} {role()} cannot commit imports — switch to Analyst or Administrator.")
            st.caption("Commit is all-or-nothing for valid rows: rejected rows never touch the ledger; corrections archive the prior version with a supersede link.")
    st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)
    sec("🗂️", "Import history & preserved originals", "Checksums + metadata live in the ledger; file bytes are kept outside relational rows.")
    att = pd.DataFrame(d["attachments"])
    if len(att):
        st.dataframe(att[["id", "filename", "sha256", "rows", "imported_at", "actor", "kind", "bytes_stored"]],
                     use_container_width=True, hide_index=True)
        stored = st.session_state.get("att_bytes", {})
        for a in d["attachments"]:
            if a["bytes_stored"] and a["sha256"] in stored:
                st.download_button(f"⬇ Original: {a['filename']}", stored[a["sha256"]], a["filename"],
                                   "text/csv", key=f"att_dl_{a['id']}")

# ----------------------------------------------------------------------------------
# PAGE: CONSERVATION LAB
# ----------------------------------------------------------------------------------
def current_lab_state():
    p = {k: st.session_state[k] for k in LAB_DEFAULTS if not k.startswith("proj_")}
    sel = [k.replace("proj_", "") for k in LAB_DEFAULTS if k.startswith("proj_") and st.session_state[k]]
    return p, sel

def load_scenario(sc, silent=False):
    for k, v in sc["params"].items():
        if k in LAB_DEFAULTS:
            st.session_state[k] = v
    for k in LAB_DEFAULTS:
        if k.startswith("proj_"):
            st.session_state[k] = k.replace("proj_", "") in sc["selected"]
    st.session_state.loaded_scenario = sc["id"]
    if not silent:
        audit("scenario_load", sc["id"], f"Loaded '{sc['name']}' into the Lab.")
        st.toast(f"Scenario “{sc['name']}” loaded", icon="🧪")

def page_lab():
    d = db()
    sec("🧪", "Conservation Lab", "Deterministic project models + seeded Monte Carlo uncertainty + budget-constrained portfolio optimization. Every control applies live.")
    projects = project_bases(d)
    df = active_df(d)
    baseline = float(df["withdrawal_m3"].sum()) if len(df) else 0.0
    params, selected = current_lab_state()
    left, right = st.columns([0.34, 0.66])
    with left:
        with st.container(border=True):
            st.markdown("**🎛 Portfolio controls** — applied live")
            st.slider("Capital budget ($)", 50_000, 3_000_000, step=25_000, key="lab_budget")
            st.slider("Water tariff ($/m³)", 0.5, 6.0, step=0.05, key="lab_tariff")
            st.slider("Discount rate (%)", 2, 15, key="lab_rate")
            st.slider("Analysis horizon (years, caps asset life)", 5, 20, key="lab_horizon")
            st.select_slider("Monte Carlo iterations", options=[500, 1000, 2000, 5000], key="lab_n")
            st.number_input("Random seed (reproducibility)", min_value=0, max_value=99999, key="lab_seed")
            st.slider("Uncertainty scale ×", 0.5, 2.0, step=0.1, key="lab_unc")
            st.slider("Reduction target (% of baseline)", 5, 50, key="lab_target")
            st.selectbox("Optimizer objective", ["Max NPV", "Max water saved", "Fastest payback"], key="lab_obj")
            st.selectbox("Risk basis", ["P50 (median)", "P10 (conservative)"], key="lab_risk")
            st.markdown("**Projects** (toggle to include)")
            for p in projects:
                c1, c2 = st.columns([0.86, 0.14])
                with c1:
                    toggle_w(f"{p['icon']} {p['name']}", key=f"proj_{p['id']}",
                             help=f"{p['desc']}\nBase: {fmt_n(p['base_m3'])} m³/yr · Capex {money(p['capex'])}")
                with c2:
                    st.caption(f"{compact(p['base_m3'])}")
            b1, b2, b3 = st.columns(3)
            if b1.button("🎯 Optimize", key="lab_opt", type="primary", use_container_width=True, disabled=not can("scenario")):
                all_sig = tuple((p["id"], p["base_m3"], float(p["capex"]), float(p["opex"]), float(p["life"]),
                                 p["sav_lo"], p["sav_hi"], p["cap_lo"], p["cap_hi"], float(p["co"]),
                                 0.5 if p["id"] == "P-METER" else 1.0)
                                for p in sorted(projects, key=lambda x: x["id"]) if p["applicable"])
                mc_all = run_mc(all_sig, 1000, int(params["lab_seed"]), float(params["lab_tariff"]),
                                params["lab_rate"] / 100, params["lab_horizon"], float(params["lab_unc"]),
                                params["lab_target"], baseline)
                cands = [{"id": p["id"], "capex": p["capex"], "group": p["group"], "applicable": p["applicable"],
                          "npv": mc_all["per"][p["id"]]["npv"], "sav": mc_all["per"][p["id"]]["sav"],
                          "payback_p50": mc_all["per"][p["id"]]["payback_p50"]}
                         for p in projects if p["applicable"] and p["id"] in mc_all["per"]]
                opt = optimize_portfolio(cands, params["lab_budget"], params["lab_obj"], params["lab_risk"])
                if not opt["feasible"] and not opt["ids"]:
                    st.error(opt["reason"])
                else:
                    for p in projects:
                        st.session_state[f"proj_{p['id']}"] = p["id"] in opt["ids"]
                    audit("scenario_optimize", "portfolio",
                          f"{params['lab_obj']} / {params['lab_risk']} under {money(params['lab_budget'])} → {opt['ids']} (objective {opt['objective_value']:,.0f}).")
                    st.toast(f"Optimizer selected {len(opt['ids'])} projects · objective {opt['objective_value']:,.0f}", icon="🎯")
                    rerun()
            if b2.button("🎲 Randomize", key="lab_rand", use_container_width=True, disabled=not can("scenario")):
                rng = np.random.default_rng()
                ids = [p["id"] for p in projects if p["applicable"]]
                pick = list(rng.choice(ids, size=int(rng.integers(2, 6)), replace=False))
                if "P-COOL" in pick and "P-ADX" in pick:
                    pick.remove(str(rng.choice(["P-COOL", "P-ADX"])))
                for p in projects:
                    st.session_state[f"proj_{p['id']}"] = p["id"] in pick
                st.session_state.lab_budget = int(rng.choice([600_000, 1_000_000, 1_600_000, 2_400_000]))
                st.session_state.lab_tariff = float(round(rng.uniform(1.0, 3.5), 2))
                st.session_state.lab_target = int(rng.choice([10, 15, 20, 25]))
                st.session_state.lab_unc = float(round(rng.uniform(0.7, 1.5), 1))
                audit("scenario_randomize", "portfolio", f"Randomized exploration → {pick}.")
                rerun()
            if b3.button("↺ Reset", key="lab_reset", use_container_width=True):
                for k, v in LAB_DEFAULTS.items():
                    st.session_state[k] = v
                st.session_state.loaded_scenario = None
                rerun()
    params, selected = current_lab_state()
    with right:
        if not selected:
            st.info("No projects selected. Toggle projects on the left or press **🎯 Optimize** to build a portfolio from the live ledger.")
            return
        sig = build_sig(projects, selected)
        mc = run_mc(sig, int(params["lab_n"]), int(params["lab_seed"]), float(params["lab_tariff"]),
                    params["lab_rate"] / 100, params["lab_horizon"], float(params["lab_unc"]),
                    params["lab_target"], baseline)
        red = mc["sav_p"]["p50"] / baseline * 100 if baseline else 0
        k = st.columns(4)
        with k[0]: kpi_card("Portfolio capex", money(mc["cap_p"]["p50"]), "P50", None, "#3b82f6", None, False,
                            f"budget {money(params['lab_budget'])}", "l1")
        with k[1]: kpi_card("Water saved / yr", compact(mc["sav_p"]["p50"]), "m³ P50", None, "#22d3ee", None, False,
                            f"P10–P90 {compact(mc['sav_p']['p10'])}–{compact(mc['sav_p']['p90'])}", "l2")
        with k[2]: kpi_card("Portfolio NPV", money(mc["npv_p"]["p50"]), "P50", None, "#34d399" if mc["npv_p"]["p50"] > 0 else "#fb7185",
                            None, False, f"{params['lab_rate']}% · ≤{params['lab_horizon']}y lives", "l3")
        with k[3]:
            pb = f"{mc['payback_p50']:.1f}" if mc["payback_p50"] else "—"
            kpi_card("Simple payback", pb, "years P50", None, "#fbbf24", None, False,
                     f"net {money(mc['net_p']['p50'])}/yr incl. co-costs", "l4")
        c1, c2 = st.columns([0.5, 0.5])
        with c1:
            st.markdown(f"**Baseline reduction (P50): {red:.1f}%** of {compact(baseline)} m³/yr")
            st.progress(min(red / max(params["lab_target"], 1), 1.0))
        with c2:
            prob = mc["prob_target"] * 100
            color = "#34d399" if prob >= 70 else ("#fbbf24" if prob >= 40 else "#fb7185")
            st.markdown(f"**Probability of ≥{params['lab_target']}% target:** " + badge(f"{prob:.0f}%", color), unsafe_allow_html=True)
            st.progress(mc["prob_target"])
        used = mc["cap_p"]["p50"] / max(params["lab_budget"], 1)
        if used > 1:
            st.warning(f"Selection exceeds budget by {money(mc['cap_p']['p50'] - params['lab_budget'])} (P50). The optimizer enforces the hard budget constraint.")
        tabs = st.tabs(["📊 Savings distribution", "🌪 Sensitivity", "📈 12-month projection", "🧱 Constraints & basis", "🔍 Explanation"])
        sel_projects = [p for p in projects if p["id"] in selected and p["applicable"]]
        with tabs[0]:
            st.plotly_chart(mc_hist_fig(mc), use_container_width=True, config=PLOT_CFG, key="lab_hist")
            tbl = []
            for p in sel_projects:
                pp = mc["per"].get(p["id"])
                if not pp: continue
                tbl.append({"Project": f"{p['icon']} {p['name']}", "Base m³/yr": pp["base"],
                            "Saved P50": pp["sav"]["p50"], "Saved P10": pp["sav"]["p10"], "Saved P90": pp["sav"]["p90"],
                            "NPV P50 ($)": pp["npv"]["p50"], "Capex P50 ($)": pp["capex_p50"],
                            "Payback P50 (y)": pp["payback_p50"] if pp["payback_p50"] else float("nan")})
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True,
                         column_config={"Base m³/yr": st.column_config.NumberColumn(format="%.0f"),
                                        "Saved P50": st.column_config.NumberColumn(format="%.0f"),
                                        "Saved P10": st.column_config.NumberColumn(format="%.0f"),
                                        "Saved P90": st.column_config.NumberColumn(format="%.0f"),
                                        "NPV P50 ($)": st.column_config.NumberColumn(format="%.0f"),
                                        "Capex P50 ($)": st.column_config.NumberColumn(format="%.0f"),
                                        "Payback P50 (y)": st.column_config.NumberColumn(format="%.1f")})
        with tabs[1]:
            ov_meter = 0.5 if ("P-METER" in selected and len(selected) >= 6) else 1.0
            tdf, base_npv = tornado_data(projects, selected, params["lab_tariff"], params["lab_rate"] / 100,
                                         params["lab_horizon"], params["lab_unc"], ov_meter)
            st.plotly_chart(tornado_fig(tdf), use_container_width=True, config=PLOT_CFG, key="lab_torn")
            st.caption(f"One-at-a-time swings around the deterministic base NPV of {money(base_npv)}. "
                       "Ranges are assumption-based (synthetic), not fitted to real outcomes.")
        with tabs[2]:
            pdf = projection_data(d, mc, sel_projects, params["lab_target"])
            if len(pdf):
                st.plotly_chart(projection_fig(pdf), use_container_width=True, config=PLOT_CFG, key="lab_proj")
                st.caption("Baseline keeps the observed month-of-year shape; projects ramp linearly over 3–12 months. "
                           "Band = portfolio P10–P90 scaled onto monthly savings (documented approximation).")
        with tabs[3]:
            ov_meter = 0.5 if ("P-METER" in selected and len(selected) >= 6) else 1.0
            det = det_portfolio(projects, selected, params["lab_tariff"], params["lab_rate"] / 100, params["lab_horizon"], ov_meter)
            spent = sum(r["capex"] for r in det)
            groups = [p["group"] for p in sel_projects if p["group"]]
            checks = [("Hard: budget (optimizer-enforced)", spent <= params["lab_budget"],
                       f"{money(spent)} vs {money(params['lab_budget'])}"),
                      ("Hard: exclusivity (cooling group ≤ 1)", len(groups) == len(set(groups)),
                       "P-COOL and P-ADX compete for the same assets"),
                      ("Applicability basis exists", all(p["applicable"] for p in sel_projects), "all bases derived from live ledger"),
                      ("Soft: positive net annual benefit", mc["net_p"]["p50"] > 0, money(mc["net_p"]["p50"]) + "/yr")]
            st.dataframe(pd.DataFrame([{"Constraint": n, "Status": "✅ satisfied" if ok else "❌ violated", "Detail": dtl}
                                       for n, ok, dtl in checks]), use_container_width=True, hide_index=True)
            for p in sel_projects:
                with st.expander(f"{p['icon']} {p['name']} — basis & formula"):
                    st.markdown(f"**Basis (traceable):** {p['basis']} · from {p['n_recs']} ledger records")
                    st.markdown(f"**Formula:** `{p['formula']}`")
                    st.markdown(f"**Assumptions:** {p['assumptions']}")
                    st.markdown(f"Capex {money(p['capex'])} (±{p['cap_lo']*100:.0f}/{p['cap_hi']*100:.0f}%) · opex {money(p['opex'])}/yr · "
                                f"life {p['life']}y · ramp {p['ramp']}mo · savings band ±{p['sav_lo']*100:.0f}% · co-savings {money(p['co'])}/yr")
        with tabs[4]:
            loaded = next((s for s in d["scenarios"] if s["id"] == st.session_state.loaded_scenario), None)
            est_share = float((df["metering"] == "estimated").mean() * 100) if len(df) else 0
            st.markdown(f"""
**Inputs**: budget {money(params['lab_budget'])} · tariff ${params['lab_tariff']:.2f}/m³ (triangular −15%/+25%, one shared draw per iteration) ·
discount {params['lab_rate']}% · horizon ≤{params['lab_horizon']}y · target −{params['lab_target']}% of baseline {compact(baseline)} m³/yr.

**Method**: seeded triangular Monte Carlo, n={params['lab_n']:,}, seed={params['lab_seed']} — fully reproducible.
Savings independent across projects; tariff correlated portfolio-wide. NPV = −capex + net·annuity(rate, min(life, horizon)).
Optimizer = greedy ratio heuristic with post-hoc verification of every hard constraint (not an exact solver).

**Versions**: engine {RULES_VERSION} · reference {d['config']['ref_version']} · baseline from {len(df)} active records (trailing 12 periods).

**Data-quality warnings**: {est_share:.0f}% of baseline records are estimates (±15%); open ledger flags flow into baseline risk.
Rainfall and stress factors are **synthetic** — no real-world predictive performance is claimed.

**Reviewer status**: {'“' + loaded['name'] + '” — ' + loaded['status'] + (' · ' + loaded['review_comment'] if loaded.get('review_comment') else '') if loaded else 'unsaved working selection (submit a snapshot for review)'}.

**Human boundary**: food-safety reuse (CIP) and permit-affecting projects require authorized approval before implementation.
""")
        st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)
        s1, s2, s3 = st.columns([0.5, 0.25, 0.25])
        name = s1.text_input("Scenario name", value=(loaded["name"] if loaded else "Working portfolio"), key="lab_name")
        with s2:
            if st.button("💾 Save snapshot", key="lab_save", type="primary", use_container_width=True, disabled=not can("scenario")):
                d["seq"]["sc"] += 1
                sc = {"id": f"SC-{d['seq']['sc']:04d}", "name": name.strip() or "Untitled scenario", "status": "Draft",
                      "params": dict(params), "selected": list(selected),
                      "results": {"sav": mc["sav_p"], "npv": mc["npv_p"], "capex": mc["cap_p"],
                                  "payback_p50": mc["payback_p50"], "prob_target": mc["prob_target"],
                                  "baseline_annual": baseline, "reduction_pct": red},
                      "basis_record_ids": df["id"].tolist(), "created_at": utcnow(),
                      "created_by": f"{role()} (demo)", "reviewed_by": "", "review_comment": "",
                      "rules_version": RULES_VERSION, "ref_version": d["config"]["ref_version"],
                      "seed": int(params["lab_seed"]), "n": int(params["lab_n"])}
                d["scenarios"].append(sc)
                st.session_state.loaded_scenario = sc["id"]
                audit("scenario_save", sc["id"], f"'{sc['name']}' saved (P50 {fmt_n(mc['sav_p']['p50'])} m³/yr, seed {params['lab_seed']}).")
                st.toast(f"Scenario {sc['id']} saved — send it to review", icon="💾")
                rerun()
        with s3:
            sc_json = json.dumps({"params": params, "selected": selected,
                                  "results": {"sav": mc["sav_p"], "npv": mc["npv_p"], "prob_target": mc["prob_target"]}},
                                 indent=2, default=str)
            st.download_button("⬇ Export JSON", sc_json.encode(), "scenario.json", "application/json", use_container_width=True)
        if loaded and loaded["status"] == "Draft" and can("submit"):
            if st.button(f"📨 Send “{loaded['name']}” to review", key="lab_send"):
                loaded["status"] = "In Review"
                audit("scenario_submit", loaded["id"], "Scenario submitted for review.")
                st.toast("Scenario submitted for review", icon="📨"); rerun()
    st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)
    sec("🗃️", "Saved scenarios", "Versioned snapshots with reviewer status — load, review or delete.")
    if not d["scenarios"]:
        st.info("No saved scenarios yet.")
    for sc in d["scenarios"]:
        with st.expander(f"{sc['id']} · {sc['name']} · {sc['status']} · P50 {compact(sc['results']['sav']['p50'])} m³/yr"):
            c1, c2, c3, c4 = st.columns([0.22, 0.22, 0.34, 0.22])
            r = sc["results"]
            c1.caption(f"Capex {money(r['capex']['p50'])}\nProb target {r['prob_target']*100:.0f}%")
            c2.caption(f"Seed {sc['seed']} · n {sc['n']}\n{sc['rules_version']} / {sc['ref_version']}")
            if c3.button("📂 Load into Lab", key=f"sc_load_{sc['id']}", use_container_width=True):
                load_scenario(sc); goto("lab")
            confirm = c4.checkbox("confirm delete", key=f"sc_delok_{sc['id']}")
            if c4.button("🗑 Delete", key=f"sc_del_{sc['id']}", disabled=not (can("edit") and confirm and sc["status"] in ("Draft", "Rejected")),
                         use_container_width=True):
                d["scenarios"] = [s for s in d["scenarios"] if s["id"] != sc["id"]]
                audit("scenario_delete", sc["id"], f"Deleted '{sc['name']}'.")
                st.toast(f"{sc['id']} deleted", icon="🗑"); rerun()

# ----------------------------------------------------------------------------------
# PAGE: REVIEW & APPROVE
# ----------------------------------------------------------------------------------
def page_review():
    d = db()
    sec("✅", "Review & Approve", "The human approval boundary: pending records, flagged exceptions and scenario sign-off — all audited.")
    pending = [r for r in d["records"] if r["status"] == "In Review"]
    flagged = [r for r in pending if quality_flags(r)]
    appr_month = sum(1 for r in d["records"] if r["status"] == "Approved" and r["updated_at"] >= "2026-11")
    m = st.columns(3)
    m[0].metric("🕳 Pending review", len(pending))
    m[1].metric("⚠ Flagged in queue", len(flagged))
    m[2].metric("✅ Approved (recent)", appr_month)
    if pending:
        opts = [r["id"] for r in pending]
        sanitize_select("rev_pick", opts)
        def fmt_rid(rid):
            r = find_rec(rid)
            fl = " ⚠" if quality_flags(r) else ""
            return f"{rid}{fl} · {site_name(r['site_id'])} · {proc_name(r['process_id'])} · {r['period']}"
        pick = st.selectbox("Record under review", opts, format_func=fmt_rid, key="rev_pick")
        rec = find_rec(pick)
        prev = next((r for r in d["records"] if r["site_id"] == rec["site_id"] and r["process_id"] == rec["process_id"]
                     and r["period"] < rec["period"] and r["status"] != "Archived"), None)
        if prev is None:
            same = sorted([r for r in d["records"] if r["site_id"] == rec["site_id"] and r["process_id"] == rec["process_id"]
                           and r["period"] < rec["period"]], key=lambda r: r["period"])
            prev = same[-1] if same else None
        if prev:
            swing = (rec["withdrawal_m3"] - prev["withdrawal_m3"]) / max(prev["withdrawal_m3"], 1) * 100
            chip = "chip-bad" if abs(swing) > 30 else "chip-info"
            st.markdown(f'Prior period comparison: withdrawal <span class="chip {chip}">{"▲" if swing>=0 else "▼"} {abs(swing):.1f}% vs {prev["period"]}</span>'
                        + (" — large swing, verify meter/evidence." if abs(swing) > 30 else ""), unsafe_allow_html=True)
        record_detail(rec, review_mode=True)
        clean = [r for r in pending if not quality_flags(r)]
        if st.button(f"⚡ Bulk-approve {len(clean)} clean pending records", key="bulk_ap",
                     disabled=not can("approve") or not clean):
            for r in clean:
                r["status"] = "Approved"; r["reviewed_by"] = f"{role()} (demo)"
                r["review_comment"] = "Bulk approval: balance verified, no flags."; r["updated_at"] = utcnow()
            audit("approve_batch", "records", f"Bulk-approved {len(clean)} clean records: {[r['id'] for r in clean][:12]}…")
            st.toast(f"{len(clean)} records approved", icon="✅"); rerun()
    else:
        st.success("Review queue is empty — all caught up. 🎉")
    st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        sec("🧪", "Scenario sign-off", "Approve conservation portfolios before they enter board reports.")
        scs = [s for s in d["scenarios"] if s["status"] in ("Draft", "In Review")]
        if not scs:
            st.info("No scenarios awaiting review.")
        for sc in scs:
            with st.expander(f"{sc['id']} · {sc['name']} · {sc['status']}"):
                r = sc["results"]
                st.caption(f"P50 {fmt_n(r['sav']['p50'])} m³/yr (P10–P90 {fmt_n(r['sav']['p10'])}–{fmt_n(r['sav']['p90'])}) · "
                           f"capex {money(r['capex']['p50'])} · prob target {r['prob_target']*100:.0f}% · basis {len(sc['basis_record_ids'])} records")
                comment = st.text_input("Reviewer comment", key=f"screv_c_{sc['id']}")
                b1, b2, b3 = st.columns(3)
                if b1.button("📨 Submit", key=f"screv_s_{sc['id']}", disabled=not (can("submit") and sc["status"] == "Draft"), use_container_width=True):
                    sc["status"] = "In Review"; audit("scenario_submit", sc["id"], "Submitted for review."); rerun()
                if b2.button("✅ Approve", key=f"screv_a_{sc['id']}", type="primary", disabled=not (can("approve") and sc["status"] == "In Review"), use_container_width=True):
                    sc["status"] = "Approved"; sc["reviewed_by"] = f"{role()} (demo)"; sc["review_comment"] = comment.strip() or "Approved."
                    audit("scenario_approve", sc["id"], sc["review_comment"]); st.toast(f"{sc['id']} approved", icon="✅"); rerun()
                if b3.button("↩️ Reject", key=f"screv_r_{sc['id']}", disabled=not (can("approve") and sc["status"] == "In Review"), use_container_width=True):
                    if not comment.strip():
                        st.error("Rejection requires a reason.")
                    else:
                        sc["status"] = "Rejected"; sc["reviewed_by"] = f"{role()} (demo)"; sc["review_comment"] = comment.strip()
                        audit("scenario_reject", sc["id"], comment.strip()); rerun()
    with c2:
        sec("🧾", "Recent audit events", "Every material change is recorded.")
        for e in db()["audit"][-10:][::-1]:
            st.markdown(f'<div class="alert-row" style="border-color:rgba(56,189,248,.25);background:rgba(13,34,54,.5)">'
                        f'<div><b>{html_mod.escape(e["action"])}</b> · {html_mod.escape(e["entity"])}'
                        f'<div class="kpi-sub">{e["ts"]} · {html_mod.escape(e["actor"])}</div>'
                        f'<div style="font-size:12px">{html_mod.escape(e["detail"][:160])}</div></div></div>',
                        unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# REPORTS
# ----------------------------------------------------------------------------------
def df_html_table(df, max_rows=40):
    cols = list(df.columns)
    head = "".join(f"<th>{html_mod.escape(str(c))}</th>" for c in cols)
    body = ""
    for _, row in df.head(max_rows).iterrows():
        tds = ""
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                v = f"{v:,.1f}" if abs(v) < 1000 else f"{v:,.0f}"
            tds += f"<td>{html_mod.escape(str(v))}</td>"
        body += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

def build_report(sites_sel, lo, hi, sections, scenario):
    d = db()
    df = active_df(d)
    df = df[df["site_id"].isin(sites_sel) & df["period"].between(lo, hi)]
    W = float(df["withdrawal_m3"].sum()) if len(df) else 0.0
    C = float(df["consumption_m3"].sum()) if len(df) else 0.0
    D = float(df["discharge_m3"].sum()) if len(df) else 0.0
    R = float(df["recycled_m3"].sum()) if len(df) else 0.0
    sm = {s["id"]: s for s in d["sites"]}
    SW = float((df["withdrawal_m3"] * df["site_id"].map(lambda x: sm[x]["stress_factor"])).sum()) if len(df) else 0.0
    by_site = []
    for sid in sites_sel:
        sub = df[df["site_id"] == sid]
        if not len(sub): continue
        w = float(sub["withdrawal_m3"].sum())
        by_site.append({"Site": sm[sid]["name"], "Basin stress": sm[sid]["stress"],
                        "Withdrawal m³": w, "Consumed m³": float(sub["consumption_m3"].sum()),
                        "Discharged m³": float(sub["discharge_m3"].sum()),
                        "Reuse %": float(sub["recycled_m3"].sum()) / max(w + float(sub["recycled_m3"].sum()), 1) * 100,
                        "Records": len(sub)})
    monthly = df.groupby("period")[["withdrawal_m3", "consumption_m3", "discharge_m3"]].sum().reset_index()
    exc = [e for e in all_exceptions() if find_rec(e["rid"]) and find_rec(e["rid"])["site_id"] in sites_sel]
    checksum = sha256_bytes(json.dumps(df.drop(columns=["sources"]).to_dict("records"), sort_keys=True, default=str).encode())
    return {"meta": {"app": APP_NAME, "version": VERSION, "rules": RULES_VERSION,
                     "ref": d["config"]["ref_version"], "generated": utcnow(), "generated_by": f"{role()} (demo)",
                     "periods": f"{lo} → {hi}", "sites": [sm[s]["name"] for s in sites_sel],
                     "synthetic": True, "records": len(df), "checksum": checksum},
            "kpis": {"withdrawal": W, "consumed": C, "discharged": D, "reuse_pct": R / max(W + R, 1) * 100,
                     "stress_weighted": SW, "approved_pct": float((df["status"] == "Approved").mean() * 100) if len(df) else 0},
            "by_site": pd.DataFrame(by_site), "monthly": monthly,
            "exceptions": exc, "metering": df["metering"].value_counts().to_dict() if len(df) else {},
            "scenario": scenario, "sections": sections,
            "audit_tail": d["audit"][-25:], "record_ids": df["id"].tolist()}

def report_html(rep):
    m, k = rep["meta"], rep["kpis"]
    css = """body{font-family:Segoe UI,Arial,sans-serif;color:#0b2437;margin:34px auto;max-width:900px;line-height:1.45}
h1{color:#0e7490;margin-bottom:2px}h2{color:#1d4ed8;border-bottom:2px solid #bae6fd;padding-bottom:4px;margin-top:28px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}th{background:#e0f2fe;text-align:left;padding:6px 8px}
td{padding:5px 8px;border-bottom:1px solid #e2e8f0}tr:nth-child(even){background:#f0f9ff}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:11px;font-weight:700}
.warn{background:#fef3c7;color:#92400e;border:1px solid #f59e0b}.kpis{display:flex;gap:12px;flex-wrap:wrap}
.kpi{background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:10px 16px;min-width:130px}
.kpi b{display:block;font-size:20px;color:#0e7490}.bar{height:14px;background:linear-gradient(90deg,#22d3ee,#3b82f6);border-radius:4px}
.foot{margin-top:30px;font-size:11px;color:#64748b;border-top:1px solid #cbd5e1;padding-top:10px}"""
    parts = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>",
             f"<h1>🌊 {APP_NAME} — Water Balance & Conservation Report</h1>",
             f"<span class='badge warn'>SYNTHETIC DEMO — NOT FOR REGULATORY USE</span>",
             f"<p>Generated {m['generated']} by {html_mod.escape(m['generated_by'])} · periods {m['periods']} · "
             f"{m['records']} records · engine {m['rules']} / refs {m['ref']} · data checksum <code>{m['checksum'][:20]}…</code></p>",
             f"<p>Sites: {html_mod.escape(', '.join(m['sites']))}</p>",
             "<div class='kpis'>",
             f"<div class='kpi'><b>{compact(k['withdrawal'])}</b>Withdrawal m³</div>",
             f"<div class='kpi'><b>{compact(k['consumed'])}</b>Consumed m³</div>",
             f"<div class='kpi'><b>{compact(k['discharged'])}</b>Discharged m³</div>",
             f"<div class='kpi'><b>{k['reuse_pct']:.1f}%</b>Reuse rate</div>",
             f"<div class='kpi'><b>{compact(k['stress_weighted'])}</b>Stress-weighted m³-eq</div>",
             f"<div class='kpi'><b>{k['approved_pct']:.0f}%</b>Approved records</div></div>",
             "<h2>1 · Balance by site</h2>", df_html_table(rep["by_site"])]
    mx = rep["monthly"]["withdrawal_m3"].max() if len(rep["monthly"]) else 1
    bars = "".join(f"<tr><td>{r['period']}</td><td style='width:55%'><div class='bar' style='width:{r['withdrawal_m3']/max(mx,1)*100:.0f}%'></div></td><td>{r['withdrawal_m3']:,.0f} m³</td></tr>"
                   for _, r in rep["monthly"].iterrows())
    parts.append(f"<h2>2 · Monthly withdrawal</h2><table>{bars}</table>")
    if rep["sections"].get("quality"):
        parts.append("<h2>3 · Data quality & exceptions</h2>")
        parts.append(f"<p>Metering mix: {html_mod.escape(json.dumps(rep['metering']))} · open exceptions: {len(rep['exceptions'])}</p>")
        if rep["exceptions"]:
            edf = pd.DataFrame([{"Severity": e["sev"], "Record": e["rid"], "Where": e["label"], "Issue": e["msg"]} for e in rep["exceptions"]])
            parts.append(df_html_table(edf))
        else:
            parts.append("<p>No open exceptions in scope.</p>")
    sc = rep["scenario"]
    if rep["sections"].get("scenario") and sc:
        r = sc["results"]
        pb = f"{r['payback_p50']:.1f} y" if r.get("payback_p50") else "—"
        parts.append(f"<h2>4 · Conservation scenario — {html_mod.escape(sc['name'])} <span class='badge warn'>{sc['status']}</span></h2>")
        parts.append(f"<p>Selected: {html_mod.escape(', '.join(sc['selected']))} · seed {sc['seed']} · n={sc['n']} · {sc['rules_version']}</p>")
        parts.append(df_html_table(pd.DataFrame([
            {"Metric": "Annual water saved P50", "Value": f"{r['sav']['p50']:,.0f} m³"},
            {"Metric": "P10–P90 band", "Value": f"{r['sav']['p10']:,.0f} – {r['sav']['p90']:,.0f} m³"},
            {"Metric": "Portfolio NPV P50", "Value": f"${r['npv']['p50']:,.0f}"},
            {"Metric": "Capex P50", "Value": f"${r['capex']['p50']:,.0f}"},
            {"Metric": "Simple payback P50", "Value": pb},
            {"Metric": "Baseline reduction", "Value": f"{r['reduction_pct']:.1f}%"},
            {"Metric": "Probability of target", "Value": f"{r['prob_target']*100:.0f}%"}])))
        if sc.get("reviewed_by"):
            parts.append(f"<p>Reviewed by {html_mod.escape(sc['reviewed_by'])}: {html_mod.escape(sc.get('review_comment',''))}</p>")
    if rep["sections"].get("method"):
        parts.append("<h2>5 · Methodology, lineage & boundaries</h2><ul>"
                     "<li>Balance: W = C + D + ΔS, tolerance max(2%·W, 5 m³); reuse R/(W+R); intensity W/output; stress-weighting W×factor.</li>"
                     "<li>Cooling: savings = E·(1/(CoC_old−1) − 1/(CoC_new−1)). NPV = −capex + net·annuity(r, min(life,horizon)).</li>"
                     "<li>Uncertainty: seeded triangular Monte Carlo; independent savings, shared tariff draw; optimizer is a verified greedy heuristic.</li>"
                     f"<li>Lineage: {m['records']} records; sources preserved with sha256 checksums; corrections supersede (never overwrite).</li>"
                     "<li>Stress factors, tariffs and rainfall are synthetic demo values — replace with licensed references before real use.</li>"
                     "<li>Decision support only. Implementation and regulated decisions require authorized human approval.</li></ul>")
    if rep["sections"].get("audit"):
        parts.append("<h2>6 · Audit trail (latest 25)</h2>")
        parts.append(df_html_table(pd.DataFrame(rep["audit_tail"])[["ts", "actor", "action", "entity", "detail"]]))
    parts.append("<div class='foot'>AquaLedger v" + VERSION + " · open-source demo · synthetic data · "
                 "no certification or regulatory compliance is claimed.</div></body></html>")
    return "".join(parts)

def report_md(rep):
    m, k = rep["meta"], rep["kpis"]
    lines = [f"# 🌊 {APP_NAME} — Water Balance Report", f"**SYNTHETIC DEMO — NOT FOR REGULATORY USE**",
             f"Generated {m['generated']} · {m['periods']} · {m['records']} records · {m['rules']}/{m['ref']}",
             f"Checksum `{m['checksum'][:20]}…`", "",
             f"- Withdrawal **{compact(k['withdrawal'])} m³** · Consumed {compact(k['consumed'])} · Discharged {compact(k['discharged'])}",
             f"- Reuse **{k['reuse_pct']:.1f}%** · Stress-weighted {compact(k['stress_weighted'])} m³-eq · Approved {k['approved_pct']:.0f}%", "",
             "## Balance by site", rep["by_site"].to_markdown(index=False) if len(rep["by_site"]) else "_no rows_", "",
             "## Monthly withdrawal", rep["monthly"].to_markdown(index=False) if len(rep["monthly"]) else "_no rows_"]
    sc = rep["scenario"]
    if rep["sections"].get("scenario") and sc:
        r = sc["results"]
        lines += ["", f"## Scenario: {sc['name']} ({sc['status']})",
                  f"- Saved P50 **{r['sav']['p50']:,.0f} m³/yr** (P10–P90 {r['sav']['p10']:,.0f}–{r['sav']['p90']:,.0f})",
                  f"- NPV P50 ${r['npv']['p50']:,.0f} · capex ${r['capex']['p50']:,.0f} · target probability {r['prob_target']*100:.0f}%"]
    lines += ["", "_Decision support only — synthetic data; authorized human approval required._"]
    return "\n".join(lines)

def page_reports():
    d = db()
    sec("📤", "Reports & Share", "Generate an auditable report, then share it — link, QR, clipboard, LinkedIn, WhatsApp, X or email.")
    site_ids = [s["id"] for s in d["sites"]]
    pers = periods_list()
    sanitize_multi("rep_sites", site_ids)
    sanitize_range("rep_periods", pers)
    c1, c2 = st.columns([0.38, 0.62])
    with c1:
        with st.container(border=True):
            st.markdown("**🧾 Report builder**")
            st.multiselect("Sites", site_ids, default=site_ids, key="rep_sites", format_func=site_name)
            st.select_slider("Periods", options=pers, value=(pers[0], pers[-1]), key="rep_periods")
            sec_q = st.checkbox("Data quality & exceptions", value=True, key="rep_sec_q")
            sec_s = st.checkbox("Conservation scenario", value=True, key="rep_sec_s")
            sec_m = st.checkbox("Methodology & lineage", value=True, key="rep_sec_m")
            sec_a = st.checkbox("Audit trail", value=False, key="rep_sec_a")
            sc_opts = ["current"] + [s["id"] for s in d["scenarios"]]
            sanitize_select("rep_scen", sc_opts)
            approved = best_scenario()
            if "rep_scen" not in st.session_state and approved:
                st.session_state.rep_scen = approved["id"]
            sc_pick = st.selectbox("Scenario", sc_opts, key="rep_scen",
                                   format_func=lambda x: "Current Lab selection (unsaved)" if x == "current"
                                   else next(f"{s['id']} · {s['name']} [{s['status']}]" for s in d["scenarios"] if s["id"] == x))
            fmt = st.radio("Format", ["HTML (styled, printable)", "Markdown", "CSV bundle (ZIP)", "JSON"], key="rep_fmt")
            go_btn = st.button("⚙️ Generate report", key="rep_go", type="primary", use_container_width=True, disabled=not can("export"))
    if st.session_state.last_report is None or go_btn:
        lo, hi = st.session_state.rep_periods
        if sc_pick == "current":
            params, selected = current_lab_state()
            projects = project_bases(d)
            dfb = active_df(d)
            baseline = float(dfb["withdrawal_m3"].sum())
            mc = run_mc(build_sig(projects, selected), int(params["lab_n"]), int(params["lab_seed"]),
                        float(params["lab_tariff"]), params["lab_rate"] / 100, params["lab_horizon"],
                        float(params["lab_unc"]), params["lab_target"], baseline) if selected else None
            scenario = {"id": "unsaved", "name": "Current Lab selection", "status": "Unsaved",
                        "selected": selected, "seed": params["lab_seed"], "n": params["lab_n"],
                        "rules_version": RULES_VERSION, "reviewed_by": "", "review_comment": "",
                        "results": {"sav": mc["sav_p"], "npv": mc["npv_p"], "capex": mc["cap_p"],
                                    "payback_p50": mc["payback_p50"], "prob_target": mc["prob_target"],
                                    "baseline_annual": baseline,
                                    "reduction_pct": mc["sav_p"]["p50"] / max(baseline, 1) * 100}} if mc else None
        else:
            scenario = next(s for s in d["scenarios"] if s["id"] == sc_pick)
        rep = build_report(st.session_state.rep_sites, lo, hi,
                           {"quality": sec_q, "scenario": sec_s, "method": sec_m, "audit": sec_a}, scenario)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        if fmt.startswith("HTML"):
            data, mime, fname, kind = report_html(rep).encode(), "text/html", f"aqualedger_report_{stamp}.html", "html"
        elif fmt.startswith("Markdown"):
            data, mime, fname, kind = report_md(rep).encode(), "text/markdown", f"aqualedger_report_{stamp}.md", "md"
        elif fmt.startswith("CSV"):
            buf = io.BytesIO()
            files = {"report.html": report_html(rep).encode(),
                     "records.csv": active_df(d).drop(columns=["sources"]).to_csv(index=False).encode(),
                     "monthly_summary.csv": rep["monthly"].to_csv(index=False).encode(),
                     "by_site.csv": rep["by_site"].to_csv(index=False).encode(),
                     "audit.csv": pd.DataFrame(d["audit"]).to_csv(index=False).encode()}
            manifest = {"generated": rep["meta"]["generated"], "rules": RULES_VERSION,
                        "checksums": {n: sha256_bytes(b) for n, b in files.items()}}
            files["manifest.json"] = json.dumps(manifest, indent=2).encode()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for n, b in files.items():
                    z.writestr(n, b)
            data, mime, fname, kind = buf.getvalue(), "application/zip", f"aqualedger_bundle_{stamp}.zip", "zip"
        else:
            data, mime, fname, kind = json.dumps(rep, indent=2, default=str).encode(), "application/json", f"aqualedger_report_{stamp}.json", "json"
        st.session_state.last_report = {"data": data, "mime": mime, "fname": fname, "kind": kind,
                                        "rep": rep, "sha": sha256_bytes(data)}
        audit("export", fname, f"Report prepared ({kind}) · sha256 {sha256_bytes(data)[:16]}… · {rep['meta']['records']} records.")
    lr = st.session_state.last_report
    with c2:
        st.markdown(f"**Preview** · `{lr['fname']}` · sha256 `{lr['sha'][:16]}…`")
        if lr["kind"] == "html":
            components.html(lr["data"].decode(), height=560, scrolling=True)
        elif lr["kind"] == "md":
            with st.container(height=560):
                st.markdown(lr["rep"]["by_site"].to_markdown(index=False) if len(lr["rep"]["by_site"]) else "")
                st.markdown(report_md(lr["rep"]))
        elif lr["kind"] == "zip":
            with st.container(height=560):
                st.markdown("**Bundle contents** — report.html, records.csv, monthly_summary.csv, by_site.csv, audit.csv, manifest.json (with per-file checksums).")
                st.json(json.loads(zipfile.ZipFile(io.BytesIO(lr["data"])).read("manifest.json")))
        else:
            with st.container(height=560):
                st.json(lr["rep"]["meta"])
                st.json(lr["rep"]["kpis"])
        st.download_button("⬇ Download report", lr["data"], lr["fname"], lr["mime"],
                           type="primary", use_container_width=True, key="rep_dl")
    st.markdown('<hr class="ak-hr">', unsafe_allow_html=True)
    sec("🔗", "Share hub", "Everything you need to circulate this demo — recipients land on the same page, filters and scenario.")
    share_hub(compact_mode=False)
    with st.expander("📦 Exchange a full data snapshot (JSON)"):
        st.markdown("Send a colleague the entire demo database; they restore it in **Governance → Backup & Restore**.")
        payload = json.dumps(d, default=str).encode()
        st.download_button("⬇ Download snapshot", payload, "aqualedger_snapshot.json", "application/json", key="snap_dl")
        st.caption(f"sha256 {sha256_bytes(payload)[:24]}… · {len(payload)/1024:.0f} KB")

# ----------------------------------------------------------------------------------
# PAGE: GOVERNANCE
# ----------------------------------------------------------------------------------
DATA_DICT = [
    ("records", "id", "text", "—", "Stable record identifier", "REC-####", "system", "Internal", "unique, immutable"),
    ("records", "site_id / process_id", "FK", "—", "Owning site and process", "catalog ids", "import/manual", "Internal", "must exist; process belongs to site"),
    ("records", "period", "text", "month", "Accounting month (site local time)", "YYYY-MM", "import/manual", "Internal", "regex; late if < lock period"),
    ("records", "sources{Municipal,Groundwater,Surface,Harvested}", "float", "m³", "Withdrawal by source", "≥ 0", "meter/import", "Internal", "sum > 0; ≤ 1e7"),
    ("records", "withdrawal_m3", "float", "m³", "Total fresh withdrawal", "> 0", "derived Σ sources", "Internal", "Σ sources"),
    ("records", "recycled_m3", "float", "m³", "Internal recirculation (reuse numerator)", "≥ 0", "meter/import", "Internal", "≥ 0"),
    ("records", "consumption_m3", "float", "m³", "Evaporated / incorporated / lost (not returned)", "≥ 0", "meter/estimate", "Internal", "balance rule"),
    ("records", "discharge_m3", "float", "m³", "Returned to sewer/surface/land", "≥ 0", "meter/import", "Confidential", "balance + site permit cap"),
    ("records", "storage_change_m3", "float", "m³", "Δ tank/reservoir storage", "any", "estimate", "Internal", "|Δ| small vs W"),
    ("records", "output_units", "float", "site unit", "Production denominator for intensity", "> 0", "ERP/import", "Confidential", "> 0"),
    ("records", "metering / uncertainty_pct", "enum/float", "%", "Measurement class → uncertainty", "metered±2 / calculated±8 / estimated±15", "catalog", "Internal", "enum map"),
    ("records", "status", "enum", "—", "Lifecycle state", "Draft/Validated/In Review/Approved/Rejected/Archived", "workflow", "Internal", "transition rules"),
    ("records", "provenance{kind,file,row,checksum}", "json", "—", "Lineage to source file/row", "sha256", "system", "Internal", "immutable per version"),
    ("records", "version / supersedes / superseded_by", "int/FK", "—", "Correction chain (never overwrite)", "≥1", "system", "Internal", "linked list"),
    ("records", "created_at / updated_at", "ts", "UTC ISO", "Record timestamps", "ISO-8601 Z", "system", "Internal", "auto"),
    ("sites", "stress / stress_factor", "enum/float", "×", "Basin water-stress weighting (SYNTHETIC)", "Low 1 / Med 1.6 / High 2.6 / Ext 4", "reference REF-x", "Public-demo", "replace with licensed Aqueduct data"),
    ("sites", "tariff", "float", "$/m³", "Water+sewer marginal tariff (demo USD)", "> 0", "reference", "Confidential", "admin-editable, versioned"),
    ("sites", "permit_m3_month", "float", "m³/mo", "Discharge permit cap", "> 0", "reference", "Confidential", "site-month exceedance flag"),
    ("projects", "capex/opex/life/ramp/co", "num", "$ / y / mo", "Project economics (synthetic)", "ranges in catalog", "reference", "Internal", "admin review"),
    ("scenarios", "results{sav,npv,capex,prob_target}", "json", "m³/$", "MC snapshot P10/P50/P90", "seeded", "engine", "Internal", "reproducible via seed+n"),
    ("audit", "ts/actor/action/entity/detail", "json", "—", "Material-change log", "append-only", "system", "Confidential", "never edited"),
    ("attachments", "sha256/size/rows/bytes_stored", "json", "—", "Original-file metadata; bytes outside rows", "sha256", "system", "Confidential", "checksum verified"),
]

def page_governance():
    d = db()
    sec("🛡️", "Governance", "Roles, audit, data dictionary, reference versioning, lineage, backup/restore and the live self-check suite.")
    tabs = st.tabs(["👥 Roles & access", "🧾 Audit log", "📚 Data dictionary", "🗺 Reference data",
                    "🧬 Lineage", "💾 Backup & restore", "🧪 Self-check", "📖 Method & demo script"])
    with tabs[0]:
        mat = pd.DataFrame([{"Permission": p, **{r: ("✅" if p in PERMS[r] else "—") for r in ROLES}} for p in PERM_ROWS])
        st.dataframe(mat, use_container_width=True, hide_index=True)
        st.info("Demo role-switcher lives in the sidebar. **Production boundary:** authorization is enforced in the action "
                "functions here (mirroring a server), passwords would be hashed with a recognized library (e.g., Argon2), "
                "and the UI is never the only gate. Stakeholder exports are limited to approved outputs.")
        st.markdown("**Record lifecycle:** Draft → Validated (auto balance pass) → In Review → Approved / Rejected → "
                    "Correction (new version, old archived) → Archived → export. Exceptions: missing, contradictory "
                    "(balance error), late (post-lock backfill) and out-of-range (permit/magnitude) all route to the review queue with flags.")
    with tabs[1]:
        adf = pd.DataFrame(d["audit"])
        actions = sorted(adf["action"].unique())
        c1, c2 = st.columns([0.4, 0.6])
        fa = c1.multiselect("Action types", actions, default=actions, key="gov_aud_f")
        q = c2.text_input("Search detail/entity", key="gov_aud_q")
        view = adf[adf["action"].isin(fa)]
        if q:
            view = view[view.apply(lambda r: q.lower() in (str(r["entity"]) + str(r["detail"])).lower(), axis=1)]
        st.dataframe(view.iloc[::-1], use_container_width=True, hide_index=True, height=420)
        st.download_button("⬇ Export audit CSV", view.to_csv(index=False).encode(), "audit_log.csv", "text/csv", key="gov_aud_dl")
    with tabs[2]:
        ent = st.selectbox("Entity", sorted({r[0] for r in DATA_DICT}), key="gov_dict_ent")
        ddf = pd.DataFrame([r for r in DATA_DICT if r[0] == ent],
                           columns=["Entity", "Field", "Type", "Unit", "Meaning", "Allowed values", "Source", "Sensitivity", "Validation rule"])
        st.dataframe(ddf, use_container_width=True, hide_index=True, height=420)
        st.caption("Timezone policy: timestamps stored UTC; accounting periods are site-local months. Currency: demo USD. "
                   "Sensitivity classes drive export gating and logging redaction in production.")
    with tabs[3]:
        st.markdown(f"Reference version **{d['config']['ref_version']}** · reporting lock **{d['config']['lock_period']}** · retention **{d['config']['retention_months']} months**")
        sdf = pd.DataFrame(d["sites"])[["id", "name", "stress", "stress_factor", "tariff", "permit_m3_month", "output_unit"]]
        if can("admin"):
            edited = st.data_editor(sdf, disabled=["id", "name", "stress", "stress_factor", "output_unit"],
                                    hide_index=True, key="ref_editor", use_container_width=True)
            if st.button("💾 Save reference changes (bumps version + audit)", key="ref_save"):
                n = int(d["config"]["ref_version"].split("-")[1].split(".")[1]) + 1
                changed = []
                for _, row in edited.iterrows():
                    s = next(x for x in d["sites"] if x["id"] == row["id"])
                    for f in ["tariff", "permit_m3_month"]:
                        if abs(float(row[f]) - float(s[f])) > 1e-9:
                            changed.append(f"{s['id']}.{f}: {s[f]} → {row[f]}")
                            s[f] = float(row[f]); s["updated_at"] = utcnow()
                if changed:
                    d["config"]["ref_version"] = f"REF-1.{n}"
                    audit("reference_update", d["config"]["ref_version"], "; ".join(changed))
                    st.toast(f"Reference data bumped to {d['config']['ref_version']}", icon="🗺"); rerun()
                else:
                    st.info("No changes detected.")
        else:
            st.dataframe(sdf, use_container_width=True, hide_index=True)
            st.caption("Administrator role required to edit reference data.")
        st.markdown("**Process catalog**")
        st.dataframe(pd.DataFrame(d["processes"])[["id", "site_id", "name", "kind"]], use_container_width=True, hide_index=True, height=260)
        st.markdown("**Dependency & license inventory (SBOM — names only)**")
        st.dataframe(pd.DataFrame([
            {"Library": "streamlit", "License": "Apache-2.0", "Role": "UI framework"},
            {"Library": "pandas", "License": "BSD-3-Clause", "Role": "tabular data"},
            {"Library": "numpy", "License": "BSD-3-Clause", "Role": "numerics / Monte Carlo"},
            {"Library": "plotly", "License": "MIT", "Role": "charts"},
            {"Library": "openpyxl", "License": "MIT", "Role": "Excel import"},
            {"Library": "qrcode", "License": "BSD", "Role": "share QR codes"},
            {"Library": "pillow", "License": "MIT-CMU", "Role": "QR imaging"},
        ]), use_container_width=True, hide_index=True)
        st.caption("All permissive licenses, commercially compatible. Pin exact versions in your deployment lockfile; no paid APIs.")
    with tabs[4]:
        opts = [r["id"] for r in d["records"]]
        sanitize_select("lin_pick", opts)
        rid = st.selectbox("Trace a record", opts, key="lin_pick",
                           format_func=lambda x: f"{x} · {site_name(find_rec(x)['site_id'])} · {find_rec(x)['period']}")
        r = find_rec(rid)
        prov = r["provenance"]
        used_sc = [s["id"] for s in d["scenarios"] if rid in s.get("basis_record_ids", [])]
        used_rep = st.session_state.last_report and rid in st.session_state.last_report["rep"]["record_ids"]
        steps = [(f"Source · {prov['kind']}", f"file {prov['file'] or 'manual entry'} · row {prov['row'] or '—'} · sha256 {(prov['checksum'] or '')[:20]}…"),
                 (f"Record {r['id']} v{r['version']}", f"status {r['status']} · created {r['created_at']}" + (f" · supersedes {r['supersedes']}" if r["supersedes"] else "")),
                 (f"Engine {RULES_VERSION}", "balance check, reuse, intensity, stress-weighting (documented formulas)"),
                 ("Derived outputs", f"scenarios: {', '.join(used_sc) or 'none yet'} · latest report: {'included' if used_rep else 'not included'}")]
        for i, (t, sub) in enumerate(steps):
            st.markdown(f'<div class="alert-row" style="border-color:rgba(34,211,238,.35);background:rgba(13,34,54,.55)">'
                        f'<span>{"📄🧾⚙️📊"[i]}</span><div><b>{html_mod.escape(t)}</b><div class="kpi-sub">{html_mod.escape(sub)}</div></div></div>',
                        unsafe_allow_html=True)
            if i < len(steps) - 1:
                st.markdown('<div style="text-align:center;color:#22d3ee">↓</div>', unsafe_allow_html=True)
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Backup** — full database as JSON (attachments as metadata + checksums)")
            payload = json.dumps(d, default=str).encode()
            st.download_button("⬇ Download backup", payload, f"aqualedger_backup_{dt.date.today()}.json",
                               "application/json", key="bk_dl", disabled=not can("export"))
            st.caption(f"sha256 {sha256_bytes(payload)[:24]}… · {len(payload)/1024:.0f} KB")
            months = st.number_input("Retention (months, archived records)", 6, 120, value=d["config"]["retention_months"], key="ret_m")
            cutoff = str(pd.Period(dt.datetime.now(), freq="M") - int(months))
            purgeable = [r for r in d["records"] if r["status"] == "Archived" and r["period"] < cutoff]
            st.caption(f"{len(purgeable)} archived records older than {cutoff} are purge-eligible.")
            okp = st.checkbox("I confirm purge (export a backup first)", key="purge_ok")
            if st.button("🧹 Purge eligible archived records", key="purge_go", disabled=not (can("admin") and okp and purgeable)):
                ids = [r["id"] for r in purgeable]
                d["records"] = [r for r in d["records"] if r["id"] not in set(ids)]
                d["config"]["retention_months"] = int(months)
                audit("retention_purge", "records", f"Purged {len(ids)} archived records older than {cutoff}.")
                st.toast(f"{len(ids)} records purged", icon="🧹"); rerun()
        with c2:
            st.markdown("**Restore** — validates schema before touching state")
            up = st.file_uploader("Upload backup/snapshot JSON", type=["json"], key="restore_up")
            if st.button("♻️ Restore uploaded backup", key="restore_go", disabled=not (can("admin") and up is not None)):
                try:
                    cand = json.loads(up.getvalue())
                    need = {"schema_version", "records", "sites", "processes", "projects", "scenarios", "audit", "attachments", "config", "seq"}
                    if not need.issubset(cand.keys()) or cand["schema_version"] != SCHEMA_VERSION:
                        raise ValueError(f"Schema mismatch — expected keys {sorted(need)} at version {SCHEMA_VERSION}.")
                    if not isinstance(cand["records"], list):
                        raise ValueError("records must be a list.")
                    st.session_state.db = cand
                    st.session_state.db["audit"].append({"ts": utcnow(), "actor": f"{role()} (demo)", "action": "restore",
                                                         "entity": up.name, "detail": f"Restored {len(cand['records'])} records from uploaded backup."})
                    st.session_state.selected_record = None
                    st.session_state.loaded_scenario = None
                    st.session_state.import_preview = None
                    st.session_state.last_report = None
                    st.session_state.selfcheck = run_self_checks(cand)
                    st.session_state.telemetry = {}
                    rng = np.random.default_rng()
                    for s in cand["sites"]:
                        annual = sum(r["withdrawal_m3"] for r in cand["records"] if r["site_id"] == s["id"])
                        base_rate = max(annual, 1) / 8760.0
                        st.session_state.telemetry[s["id"]] = {"rate": base_rate, "base": base_rate,
                                                               "hist": [base_rate] * 40, "today": base_rate * 9}
                    st.toast("Backup restored and re-validated", icon="♻️"); rerun()
                except Exception as e:
                    st.error(f"Restore rejected — existing data untouched. {e}")
            st.markdown("**Reset demo**")
            okreset = st.checkbox("Confirm full reset to synthetic seed", key="reset_ok")
            if st.button("🔄 Reset demo database", key="reset_go", disabled=not (can("admin") and okreset)):
                del st.session_state["db"]
                for k in ["selected_record", "loaded_scenario", "import_preview", "last_report", "share_link"]:
                    st.session_state[k] = None
                init_state()
                st.toast("Fresh synthetic demo seeded (reproducible, rng 16113)", icon="🔄"); rerun()
        st.info("**Persistence & migration path (honest scope):** this demo keeps state in the browser session so it runs "
                "anywhere with zero setup. The production design maps these tables 1:1 to **SQLite** (local MVP, WAL mode, "
                "file attachments outside rows) and then **PostgreSQL** for the team edition (same schema via migration "
                "scripts, row-level transactions, concurrent-review tests). Backup = DB dump + attachment folder + checksums; "
                "restore is tested by re-running the self-check suite, exactly as this page does after a JSON restore.")
    with tabs[6]:
        sc = st.session_state.selfcheck
        passed = sum(1 for x in sc if "PASS" in x["Result"])
        m = st.columns(3)
        m[0].metric("Checks", len(sc)); m[1].metric("Passed", passed); m[2].metric("Failed", len(sc) - passed)
        st.caption(f"Executed live in this session at {st.session_state.selfcheck_ts} — hand-checkable domain examples, "
                   "validator behavior, optimizer hard constraints, Monte Carlo reproducibility and ledger integrity.")
        st.dataframe(pd.DataFrame(sc), use_container_width=True, hide_index=True, height=430)
        if st.button("▶ Re-run self-check suite", key="sc_rerun", type="primary"):
            st.session_state.selfcheck = run_self_checks(d)
            st.session_state.selfcheck_ts = utcnow()
            audit("self_check", "suite", f"{sum(1 for x in st.session_state.selfcheck if 'PASS' in x['Result'])}/{len(sc)} passed.")
            rerun()
    with tabs[7]:
        with st.expander("🎬 Demo walkthrough (record this as your live demo video)", expanded=True):
            st.markdown("""
1. **Overview** — telemetry is already running; pause/resume and change speed from the sidebar. Hover the Sankey.
2. **Import & Validate** — press *Load sample into validator* with the **invalid** file: 8 rejected/skipped rows with actionable errors; download the error report. Switch to the **valid** file and commit (balloons = new records).
3. **Water Ledger** — open the flagged Mesa Verde cooling record: see the balance contradiction, then *Correct record* with a reason → new version, old archived. Try **Undo** after archiving.
4. **Review & Approve** — switch role to **Reviewer**, approve the correction, bulk-approve the clean queue.
5. **Conservation Lab** — move the budget/tariff sliders (everything updates live), press **🎯 Optimize**, save a scenario, send to review.
6. **Reports & Share** — generate the HTML report, download it, then use the top-bar **📤 Share** → copy summary / QR / LinkedIn.
7. **Governance → Self-check** — show the live pass/fail suite and the audit log of everything you just did.""")
        with st.expander("📐 Methodology, assumptions & risk register"):
            st.markdown(f"""
**Formulas (engine {RULES_VERSION})** — balance `W = C + D + ΔS` (tol `max(2%·W, 5 m³)`); reuse `R/(W+R)`;
intensity `W/output`; stress-weighted `W×factor`; cooling `E·(1/(CoC_o−1)−1/(CoC_n−1))`;
`NPV = −capex + net·annuity(r, min(life,horizon))`; payback `capex/net` (None if net ≤ 0).

**Assumptions** — synthetic stress factors/tariffs/rainfall; triangular uncertainty bands are expert-style assumptions,
not fitted; tariff correlated across portfolio; savings independent; METER overlap haircut ×0.5 when ≥5 other projects;
linear ramp 3–12 months; greedy optimizer (verified, not exact).

**Limitations** — no real-world predictive performance is claimed; synthetic data tests plumbing and edge cases only.
Real model claims require representative, permissioned, independently evaluated data with time/site-split validation.

**Risk register** — *Domain validity*: synthetic factors → replace with licensed references + qualified review.
*Data quality*: estimated meters → flagging, tolerance rules, review queue. *Security*: demo roles → server-side auth,
Argon2 hashing, HTTPS, secrets in env vars in production. *Adoption*: spreadsheet habits → one-click import + share.
*Operational failure*: session state → SQLite/Postgres with backups, restore tests (this page), pinned dependencies.

**Human approval boundary** — decision support only; no certification or regulatory compliance is claimed;
implementation of any project (especially food-safety reuse and permit-affecting changes) requires authorized sign-off.""")

# ----------------------------------------------------------------------------------
# DEEP LINKS + MAIN
# ----------------------------------------------------------------------------------
def apply_deep_link():
    if st.session_state.dl_done:
        return
    st.session_state.dl_done = True
    qp = query_params()
    page = qp.get("page")
    if page in NAV_BY_KEY:
        st.session_state.nav = NAV_BY_KEY[page]
    sites = qp.get("sites")
    if sites:
        ids = [s for s in sites.split(",") if s in {x["id"] for x in db()["sites"]}]
        if ids:
            st.session_state.ov_sites = ids
    sc_id = qp.get("sc")
    if sc_id:
        sc = next((s for s in db()["scenarios"] if s["id"] == sc_id), None)
        if sc:
            load_scenario(sc, silent=True)

def main():
    st.set_page_config(page_title=f"{APP_NAME} — {TAGLINE}", page_icon="🌊", layout="wide",
                       initial_sidebar_state="expanded")
    init_state()
    pending = st.session_state.pop("nav_goto", None)
    if pending:
        st.session_state.nav = pending
    st.markdown(CSS, unsafe_allow_html=True)
    apply_deep_link()
    sidebar()
    topbar()
    page = st.session_state.nav
    if page == NAV_BY_KEY["overview"]:
        page_overview()
    elif page == NAV_BY_KEY["ledger"]:
        page_records()
    elif page == NAV_BY_KEY["import"]:
        page_import()
    elif page == NAV_BY_KEY["lab"]:
        page_lab()
    elif page == NAV_BY_KEY["review"]:
        page_review()
    elif page == NAV_BY_KEY["reports"]:
        page_reports()
    else:
        page_governance()
    st.markdown(f'<div class="ak-foot">{APP_NAME} v{VERSION} · engine {RULES_VERSION} · synthetic demo data · '
                f'decision support only — authorized human approval required · no API keys</div>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()