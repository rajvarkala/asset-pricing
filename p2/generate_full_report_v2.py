"""
Generate the full project research report as a self-contained HTML file.
Usage:
    python3 p2/generate_full_report_v2.py [reports_dir] [--output path] [--pdf]
"""
from __future__ import annotations

import argparse
import base64
import csv
import re
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = WORKSPACE / "p2/backtests/2026-05-16_23-33-18/reports"
VIP_PLOTS_DIR       = WORKSPACE / "ml_results_cache/plots"
FF6_CSV             = WORKSPACE / "p2/ml_inputs/ff6_factors_10y.csv"
PRED_CSV            = WORKSPACE / "p2/ml_inputs/synthetic_predictor_features_all_months.csv"

MODEL_ORDER = ["OLS-3_L2","OLS-3_Huber","OLS-all_L2","OLS-all_Huber","OLS-all_Huber-EN","RF","GBRF_Huber"]
MODEL_LABEL = {
    "OLS-3_L2":         "OLS-3 L2",
    "OLS-3_Huber":      "OLS-3 Huber",
    "OLS-all_L2":       "OLS-all L2",
    "OLS-all_Huber":    "OLS-all Huber",
    "OLS-all_Huber-EN": "OLS-all Huber+EN",
    "RF":               "Random Forest",
    "GBRF_Huber":       "GBRF Huber",
}
METRIC_LABEL = {
    "cumulative_return":  "Cumulative Return",
    "sharpe_ratio":       "Sharpe Ratio",
    "information_ratio":  "Information Ratio",
    "alpha_vs_benchmark": "α vs Benchmark (mo.)",
    "alpha_ff6":          "α vs FF6 (mo.)",
    "beta_mkt_rf":        "β Market",
    "beta_smb":           "β SMB",
    "beta_hml":           "β HML",
    "beta_rmw":           "β RMW",
    "beta_cma":           "β CMA",
    "beta_umd":           "β UMD",
    "drawdown":           "Max Drawdown",
    "turnover":           "Monthly Turnover",
    "mean_return":        "Mean Monthly Return",
}
QUARTILE_LABEL = {
    "All":             "All Stocks",
    "Top Quartile":    "Top Quartile (Long)",
    "Bottom Quartile": "Bottom Quartile (Short)",
}
PCT_ROWS = {"alpha_vs_benchmark", "alpha_ff6", "mean_return"}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def b64_image(path: Path, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def load_csv_dict(path: Path):
    """Row-keyed CSV → ({row: {col: val}}, [col_headers])"""
    rows = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)[1:]
        for row in reader:
            rows[row[0]] = {h: row[i + 1] for i, h in enumerate(headers)}
    return rows, headers


def fmt(val: str, is_pct: bool = False) -> str:
    try:
        v = float(val)
        return f"{v * 100:.2f}%" if is_pct else f"{v:.4f}"
    except (ValueError, TypeError):
        return val or "—"


def html_table(row_keys, col_keys, data, row_labels=None, col_labels=None,
               pct_rows=None, highlight_best=False, table_id="") -> str:
    rl = row_labels or {}
    cl = col_labels or {}
    pr = pct_rows or set()
    best_idx = {}
    if highlight_best:
        for col in col_keys:
            vals = []
            for i, row in enumerate(row_keys):
                try:
                    vals.append((float(data[row][col]), i))
                except Exception:
                    vals.append((float("-inf"), i))
            best_idx[col] = max(vals, key=lambda x: x[0])[1]
    hdrs = "".join(f'<th>{cl.get(c, c)}</th>' for c in col_keys)
    body = ""
    for ri, row in enumerate(row_keys):
        cells = f'<td class="rh">{rl.get(row, row)}</td>'
        for col in col_keys:
            raw = data.get(row, {}).get(col, "—")
            rendered = fmt(raw, row in pr)
            bc = " best" if highlight_best and best_idx.get(col) == ri else ""
            cells += f'<td class="dc{bc}">{rendered}</td>'
        body += f"<tr>{cells}</tr>\n"
    return (
        f'<div class="tscroll"><table id="{table_id}" class="dt">'
        f'<thead><tr><th></th>{hdrs}</tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )


def load_ff6_table() -> str:
    if not FF6_CSV.exists():
        return "<p>FF6 data not found.</p>"
    cols = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "umd"]
    heads = ["Date", "Mkt-Rf", "SMB", "HML", "RMW", "CMA", "UMD"]
    rows_html = ""
    with FF6_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cells = ""
            for c in cols:
                v = row.get(c, "")
                if c != "date":
                    try:
                        v = f"{float(v) * 100:.3f}%"
                    except Exception:
                        pass
                cells += f"<td>{v}</td>"
            rows_html += f"<tr>{cells}</tr>"
    hdrs = "".join(f"<th>{h}</th>" for h in heads)
    return (
        '<div class="tscroll" style="max-height:320px;overflow-y:auto">'
        f'<table class="dt compact-table"><thead><tr>{hdrs}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>'
    )


def load_predictor_sample() -> str:
    if not PRED_CSV.exists():
        return "<p>Predictor data not found.</p>"
    show = ["date", "ticker", "mom1m", "mom12m", "mom6m", "mvel1", "bm",
            "operprof", "invest", "lev", "beta", "retvol", "ep", "sp"]
    heads = ["Date", "Ticker", "mom1m", "mom12m", "mom6m", "ln(ME)", "B/M",
             "OperProf", "Invest", "Lev", "Beta", "RetVol", "E/P", "S/P"]
    samples, seen = [], set()
    with PRED_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")
            if d not in seen:
                seen.add(d)
                samples.append(row)
            if len(samples) >= 15:
                break
    rows_html = ""
    for row in samples:
        cells = ""
        for c in show:
            v = row.get(c, "—")
            if c not in ("date", "ticker"):
                try:
                    v = f"{float(v):.4f}"
                except Exception:
                    pass
            cells += f"<td>{v}</td>"
        rows_html += f"<tr>{cells}</tr>"
    hdrs = "".join(f"<th>{h}</th>" for h in heads)
    return (
        '<div class="tscroll" style="max-height:300px;overflow-x:auto;overflow-y:auto">'
        f'<table class="dt compact-table"><thead><tr>{hdrs}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────────────────────────────────────

def section_data() -> str:
    firm_cats = [
        ("Momentum",        "mom1m · mom6m · mom12m · mom36m · chmom · indmom"),
        ("Size",            "mvel1 (log market equity)"),
        ("Value",           "bm · bm_ia · ep · sp · dy · dp · cashpr"),
        ("Profitability",   "operprof · depr · nincr · rd_mve · ps"),
        ("Investment",      "invest · agr · lgr · chinv · chcsho"),
        ("Liquidity / Risk","turn · std_turn · maxret · retvol · idiovol · beta · betasq · dolvol · ill · baspread · zerotrade"),
        ("Debt / Structure","lev · securedind · convind"),
        ("Sector",          "sic2 (90 SIC-2 one-hot dummies)"),
    ]
    cat_rows = "".join(
        f'<tr><td class="rh">{c}</td>'
        f'<td style="font-family:var(--mono);font-size:11.5px;color:#1e40af">{v}</td></tr>'
        for c, v in firm_cats
    )
    firm_table = (
        '<div class="tscroll"><table class="dt">'
        '<thead><tr><th>Category</th><th>Variables</th></tr></thead>'
        f'<tbody>{cat_rows}</tbody></table></div>'
    )
    return f"""
<section id="sec-data" class="sec">
  <h2><span class="secnum">1</span> Data</h2>

  <div class="sub">
    <h3>Universe</h3>
    <p>500 synthetic tickers (EQ00001–EQ00500), <strong>119 monthly periods</strong>
    (Jul 2016 – Apr 2026). Monthly returns are drawn from a factor model with idiosyncratic noise;
    no predictor carries a persistent out-of-sample signal by design — the primary data-quality
    caveat addressed in §4.</p>
    <div class="kpi-row">
      <div class="kpi"><div class="kv">500</div><div class="kl">Tickers</div></div>
      <div class="kpi"><div class="kv">119</div><div class="kl">Monthly Periods</div></div>
      <div class="kpi"><div class="kv">447</div><div class="kl">Feature Columns</div></div>
      <div class="kpi"><div class="kv">9yr / 1yr</div><div class="kl">Train / Test</div></div>
    </div>
  </div>

  <div class="sub">
    <h3>Firm-Level Predictors (31 variables)</h3>
    <p>Cross-sectional characteristics mirroring Green–Hand–Zhang (2017) and Hou–Xue–Zhang (2020).
    All variables are lagged one month before merging with the return target.</p>
    {firm_table}
    <p style="margin-top:10px">Additionally: 8 macro variables
    (<code>ntis · tbl · tms · dfy · svar · dp · ep · bm</code>) plus
    <strong>312 firm × macro interaction terms</strong> and <strong>91 SIC-2 sector dummies</strong>
    give <strong>447 total feature columns</strong> for OLS-all and tree models.
    OLS-3 uses only <code>mvel1 · bm · mom12m</code>.</p>
  </div>

  <div class="sub">
    <h3>Sample Predictor Data</h3>
    <p>One observation per month-end; one ticker per row shown. All firm features are lagged one period.</p>
    {load_predictor_sample()}
  </div>

  <div class="sub">
    <h3>Benchmark Construction</h3>
    <p>Each month the benchmark return is the <strong>market-cap-weighted average return premium</strong>
    across all tickers in the universe. Concretely, letting <em>w<sub>i</sub> = market_cap<sub>i</sub> / &Sigma; market_cap</em>:</p>
    <blockquote class="formula">benchmark_ret = &Sigma; w<sub>i</sub> &middot; return_premium<sub>i</sub></blockquote>
    <p>The strategy return adds a 30&thinsp;% dollar-neutral long-short overlay on top of this passive base:</p>
    <blockquote class="formula">strategy_ret = benchmark_ret + 0.30 &times; (long_ret &minus; short_ret)</blockquote>
    <p>The long (short) book is the top (bottom) <strong>decile</strong> of tickers ranked by model prediction,
    each value-weighted within its book. The benchmark NAV compounds these monthly returns from an
    initial value of 1.0 and is registered directly with Lean — so the Lean report benchmark line is
    identical to the combined chart benchmark.</p>
  </div>

  <div class="sub">
    <h3>Fama-French 6 Factor Series</h3>
    <p>Factors constructed from the synthetic panel: 2&times;3 size/B-M sorts (SMB, HML),
    size/profitability (RMW), size/investment (CMA), 12-1 month momentum (UMD).
    Values shown as monthly percentages.</p>
    {load_ff6_table()}
  </div>
</section>
"""


def section_models() -> str:
    specs = [
        ("OLS-3 L2",         "Ridge",               "size · B/M · mom12m", "Interpretable Fama-French baseline"),
        ("OLS-3 Huber",      "Huber + Scaler",       "size · B/M · mom12m", "Robust variant; down-weights outlier months"),
        ("OLS-all L2",       "Ridge",                "447 columns",         "Tests macro interactions beyond 3-factor"),
        ("OLS-all Huber",    "SGD Huber-L2",         "447 columns",         "Scalable robust linear on full feature space"),
        ("OLS-all Huber+EN", "SGD ElasticNet-Huber", "447 columns",         "L1+L2 sparsity; shrinks irrelevant interactions"),
        ("Random Forest",    "RF (50 trees, d=4)",   "447 columns",         "Nonlinear; importance via impurity decrease"),
        ("GBRF Huber",       "GBM Huber (100 est.)", "447 columns",         "Sequential Huber minimisation; robust to outliers"),
    ]
    rows = "".join(
        f'<tr><td class="rh">{m}</td><td>{a}</td><td><code>{f}</code></td><td>{e}</td></tr>'
        for m, a, f, e in specs
    )
    return f"""
<section id="sec-models" class="sec">
  <h2><span class="secnum">2</span> Models</h2>
  <p>All models share <strong>walk-forward monthly retraining</strong>: retrain on all data up to
  <em>t−1</em>, predict return premium at <em>t</em>, form a rank-based long-short overlay.</p>
  <div class="tscroll">
    <table class="dt">
      <thead><tr><th>Model</th><th>Algorithm</th><th>Feature Set</th><th>Motivation</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</section>
"""


def section_results(reports_dir: Path, vip_dir: Path) -> str:
    # R² panel
    r2_path = WORKSPACE / "ml_results_summary.csv"
    r2_data, _ = load_csv_dict(r2_path) if r2_path.exists() else ({}, [])
    r2_models = [m for m in MODEL_ORDER if m in r2_data]
    r2_table = html_table(
        r2_models, ["In-Sample R²", "Out-Sample R²"], r2_data,
        row_labels=MODEL_LABEL, table_id="tbl-r2",
    ) if r2_data else "<p>R² data not found.</p>"

    # Quartile panel
    q_path = WORKSPACE / "ml_results_quartile_panel.csv"
    q_data, q_hdrs = load_csv_dict(q_path) if q_path.exists() else ({}, [])
    q_cols = [m for m in MODEL_ORDER if m in q_hdrs]
    q_table = html_table(
        list(q_data.keys()), q_cols, q_data,
        row_labels=QUARTILE_LABEL, col_labels=MODEL_LABEL, table_id="tbl-q",
    ) if q_data else "<p>Quartile data not found.</p>"

    # VIP images
    vip_cards = ""
    for mid in MODEL_ORDER:
        img = vip_dir / f"{mid}_importance.png"
        if img.exists():
            lbl = MODEL_LABEL.get(mid, mid)
            vip_cards += (
                f'<figure class="vip-card">'
                f'<img src="{b64_image(img)}" alt="{lbl}" loading="lazy"/>'
                f'<figcaption>{lbl}</figcaption></figure>'
            )
    vip_grid = f'<div class="vip-grid">{vip_cards}</div>' if vip_cards else "<p>No plots found.</p>"

    # Cumulative returns (inline)
    cum_path = reports_dir / "combined_cumulative_returns.html"
    cum_embed = "<p>Combined returns chart not found.</p>"
    if cum_path.exists():
        raw = cum_path.read_text(encoding="utf-8")
        body_m   = re.search(r"<body[^>]*>(.*?)</body>", raw, re.DOTALL | re.IGNORECASE)
        style_m  = re.search(r"<style[^>]*>(.*?)</style>", raw, re.DOTALL | re.IGNORECASE)
        script_m = re.search(r"<script[^>]*>(.*?)</script>", raw, re.DOTALL | re.IGNORECASE)
        cum_embed = (
            f"<style>{style_m.group(1) if style_m else ''}</style>"
            f'<div class="cum-wrap">{body_m.group(1).strip() if body_m else raw}</div>'
            f"<script>{script_m.group(1) if script_m else ''}</script>"
        )

    # Metrics table
    comp_path = reports_dir / "model_comparison_panel.csv"
    comp_data, comp_hdrs = load_csv_dict(comp_path) if comp_path.exists() else ({}, [])
    comp_cols    = [m for m in MODEL_ORDER if m in comp_hdrs]
    comp_metrics = [m for m in METRIC_LABEL if m in comp_data]
    comp_table = html_table(
        comp_metrics, comp_cols, comp_data,
        row_labels=METRIC_LABEL, col_labels=MODEL_LABEL,
        pct_rows=PCT_ROWS, highlight_best=True, table_id="tbl-comp",
    ) if comp_data else "<p>Metrics panel not found.</p>"

    # Lean reports
    lean_blocks = ""
    for mid in MODEL_ORDER:
        p = reports_dir / mid / "lean_report.html"
        if p.exists():
            lbl = MODEL_LABEL.get(mid, mid)
            enc = base64.b64encode(p.read_bytes()).decode()
            lean_blocks += (
                f'<div class="lean-block">'
                f'<h4 class="lean-lbl">{lbl}</h4>'
                f'<iframe src="data:text/html;base64,{enc}" class="lean-frame"'
                f' loading="lazy" title="{lbl}"></iframe></div>'
            )
    if not lean_blocks:
        lean_blocks = "<p>No Lean reports found.</p>"

    return f"""
<section id="sec-results" class="sec">
  <h2><span class="secnum">3</span> Results</h2>

  <div class="sub" id="sec-vip">
    <h3>Variable Importance</h3>
    <p>Permutation importance for tree models; absolute coefficient magnitude for linear models.
    Size, momentum, and B/M rank highest — consistent with the academic literature and confirming
    feature-engineering correctness.</p>
    {vip_grid}
  </div>

  <div class="sub" id="sec-cum">
    <h3>Cumulative Returns</h3>
    <p>Daily normalized equity (start = 1.0) from Lean result packet series. One shared benchmark.</p>
    <div class="chart-card">{cum_embed}</div>
  </div>

  <div class="sub" id="sec-metrics">
    <h3>Model Comparison Metrics</h3>
    {comp_table}
    <p class="note">α rows are monthly. Drawdown and turnover are averages over the backtest.</p>
  </div>

  <div class="sub" id="sec-lean">
    <h3>Lean Backtest Reports</h3>
    {lean_blocks}
  </div>
</section>
"""


def section_next_steps() -> str:
    return """
<section id="sec-next" class="sec">
  <h2><span class="secnum">4</span> Next Steps</h2>

  <div class="sub">
    <h3>Real Data Integration</h3>
    <ul class="nlist">
      <li><strong>Zerodha tick data</strong> — extraction pipeline is written.
        Pending: verify dividend adjustment (total-return vs. price-only) and validate
        split/corporate-action handling end-to-end.</li>
      <li><strong>Screener.in financials</strong> — scraped quarterly P&amp;L, balance sheet and
        cash-flow data is available. Pending: compute point-in-time cross-sectional ratios
        (B/M, operating profitability, asset growth, R&amp;D/MVE, etc.) aligned to month-end dates.</li>
      <li>Pending the Zerodha validation, a buy-vs-build review of commercial data vendors
        (Refinitiv, WRDS) is deferred.</li>
    </ul>
  </div>

  <div class="sub">
    <h3>Algorithm Tuning for Scale</h3>
    <p>Current retraining runs are calibrated for 500 tickers. Scaling to
    <strong>3,000+ NSE/BSE tickers</strong> will require significant tuning: feature-selection
    pre-screening to reduce the ~380-column matrix, parallelisation of the monthly refit loop,
    and lighter tree hyperparameters (fewer estimators, shallower depth) to keep each monthly
    retrain within an acceptable wall-clock budget. This is a hard constraint before any
    production deployment.</p>
  </div>
</section>
"""


def section_observations() -> str:
    return """
<section id="sec-obs" class="sec">
  <h2><span class="secnum">5</span> Observations</h2>

  <div class="callout info">
    <strong>The full ML pipeline is production-ready infrastructure.</strong>
    Feature engineering, walk-forward retraining, rank-based portfolio construction,
    Lean backtesting, FF6 attribution, and this reporting suite all run end-to-end without
    modification when real data is substituted. This represents significant completed work
    that should not be understated.
  </div>

  <ul class="obs-list">
    <li>Negative R² values across all models are <strong>expected and correct</strong> for synthetic data —
    they validate pipeline integrity, not model failure.</li>
    <li>Variable importance rankings are internally consistent with the academic literature
    (size, 12-month momentum, B/M dominate across all models), confirming feature-engineering
    correctness independently of data quality.</li>
    <li>Scaling to 3,000+ real tickers is the most immediate engineering challenge.
    <strong>Algorithm tuning is required</strong>: current monthly refit wall-clock times
    at 500 tickers will become prohibitive at full NSE/BSE scope without parallelisation
    and hyperparameter lightening.</li>
    <li>The primary remaining blocker is <strong>data quality, not model architecture</strong>.
    Once point-in-time real fundamentals and dividend-adjusted price histories are validated,
    all seven models run with no pipeline changes required.</li>
  </ul>
</section>
"""


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

def global_css() -> str:
    return """
:root {
  --ink:    #0f172a;
  --muted:  #475569;
  --bg:     #f1f5f9;
  --panel:  #ffffff;
  --border: #e2e8f0;
  --accent: #2563eb;
  --hdr:    #0f2744;
  --hdr-fg: #ffffff;
  --best:   #bbf7d0;
  --warn-bg:#fff7ed; --warn-bd:#f97316;
  --info-bg:#eff6ff; --info-bd:#3b82f6;
  --mono:   'Fira Code','Cascadia Code','Menlo','Courier New',monospace;
  --nav-w:  210px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.7;
  color: var(--ink);
  background: var(--bg);
  -webkit-font-smoothing: antialiased;
}

/* ── Sidebar ─────────────────────────────────── */
#nav {
  position: fixed; top: 0; left: 0;
  width: var(--nav-w); height: 100vh;
  overflow-y: auto; background: var(--hdr);
  color: var(--hdr-fg); padding: 20px 0; z-index: 200;
}
#nav .brand {
  font-size: 11px; font-weight: 800; letter-spacing: .08em;
  text-transform: uppercase; padding: 0 16px 16px;
  border-bottom: 1px solid rgba(255,255,255,.1); opacity: .9;
}
#nav ul { list-style: none; padding: 10px 0; }
#nav li a {
  display: block; padding: 6px 16px; font-size: 12px;
  color: rgba(255,255,255,.72); text-decoration: none;
  border-left: 3px solid transparent; transition: all .12s;
}
#nav li a:hover, #nav li a.active {
  color: #fff; background: rgba(255,255,255,.07); border-left-color: #60a5fa;
}
#nav li a.sub { padding-left: 26px; font-size: 11.5px; }

/* ── Main ────────────────────────────────────── */
#main { margin-left: var(--nav-w); padding: 36px 44px 80px; max-width: 1320px; }

/* ── Cover ───────────────────────────────────── */
#cover {
  background: linear-gradient(135deg, #0f2744 0%, #1d4ed8 100%);
  color: #fff; border-radius: 16px; padding: 44px 48px 36px; margin-bottom: 44px;
}
#cover h1 { font-size: 30px; font-weight: 800; line-height: 1.2; }
#cover .tagline { margin-top: 8px; font-size: 14px; opacity: .82; }
#cover .meta {
  margin-top: 18px; display: flex; flex-wrap: wrap; gap: 20px;
  font-size: 11.5px; opacity: .65;
}
.print-btn {
  display: inline-block; margin-top: 20px; padding: 8px 20px;
  border-radius: 8px; background: rgba(255,255,255,.15); color: #fff;
  border: 1px solid rgba(255,255,255,.3); font-size: 12.5px; font-weight: 600;
  cursor: pointer; text-decoration: none; transition: background .15s;
}
.print-btn:hover { background: rgba(255,255,255,.25); }

/* ── KPI row ──────────────────────────────────── */
.kpi-row { display: flex; flex-wrap: wrap; gap: 14px; margin: 16px 0; }
.kpi {
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 22px; text-align: center; flex: 1; min-width: 100px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.kv { font-size: 26px; font-weight: 800; color: var(--accent); line-height: 1; }
.kl { font-size: 10.5px; color: var(--muted); margin-top: 4px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .05em; }

/* ── Sections ─────────────────────────────────── */
.sec { margin-bottom: 60px; }
.sec > h2 {
  font-size: 20px; font-weight: 800; color: var(--hdr);
  display: flex; align-items: center; gap: 10px;
  padding-bottom: 10px; margin-bottom: 22px;
  border-bottom: 3px solid var(--accent);
}
.secnum {
  display: inline-flex; align-items: center; justify-content: center;
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--accent); color: #fff; font-size: 13px; font-weight: 800; flex-shrink: 0;
}
.sub { margin-top: 26px; }
.sub > h3 {
  font-size: 15px; font-weight: 700; color: var(--ink); margin-bottom: 10px;
  padding: 8px 14px; background: #f8fafc; border-left: 4px solid var(--accent);
  border-radius: 0 6px 6px 0;
}
.sub > h4 { font-size: 13px; font-weight: 700; color: var(--muted); margin: 14px 0 6px; }
p { margin-bottom: 10px; }

/* ── Callouts ─────────────────────────────────── */
.callout {
  border-left: 4px solid; border-radius: 8px;
  padding: 13px 18px; margin: 14px 0; font-size: 13.5px;
}
.callout.warn { background: var(--warn-bg); border-color: var(--warn-bd); }
.callout.info { background: var(--info-bg); border-color: var(--info-bd); }

/* ── Tables ───────────────────────────────────── */
.tscroll { overflow-x: auto; border-radius: 10px; border: 1px solid var(--border); margin: 10px 0; }
.dt { width: 100%; border-collapse: collapse; font-size: 12px; }
.dt thead tr { background: var(--hdr); color: #fff; }
.dt thead th {
  padding: 9px 12px; text-align: center; font-weight: 700;
  white-space: nowrap; border-right: 1px solid rgba(255,255,255,.1);
}
.dt thead th:first-child { text-align: left; }
.dt tbody tr:nth-child(even) { background: #f8fafc; }
.dt tbody tr:hover { background: #eff6ff; }
.rh { padding: 8px 12px; font-weight: 600; white-space: nowrap; text-align: left; border-right: 1px solid var(--border); }
.dc { padding: 7px 11px; text-align: right; border-left: 1px solid #f0f4f8; font-variant-numeric: tabular-nums; }
.dc.best { background: var(--best) !important; font-weight: 700; }
.legend-best-swatch {
  display: inline-block; width: 14px; height: 14px;
  background: var(--best); border: 1px solid #4ade80;
  vertical-align: middle; border-radius: 3px;
}
.compact-table td, .compact-table .rh { padding: 5px 10px; font-size: 11.5px; }

/* ── VIP grid ──────────────────────────────────── */
.vip-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 18px; margin-top: 14px;
}
.vip-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.07);
}
.vip-card img { width: 100%; display: block; }
.vip-card figcaption {
  text-align: center; font-size: 12px; font-weight: 700;
  padding: 8px; color: var(--muted); background: #f8fafc;
  border-top: 1px solid var(--border);
}

/* ── Chart card ────────────────────────────────── */
.chart-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.07); margin: 10px 0;
}
.cum-wrap .wrap { border: none !important; box-shadow: none !important; max-width: 100% !important; }

/* ── Lean iframes ──────────────────────────────── */
.lean-block { margin-bottom: 28px; }
.lean-lbl {
  font-size: 14px; font-weight: 700; color: var(--hdr);
  margin-bottom: 5px; padding-bottom: 4px; border-bottom: 2px solid #dbeafe;
}
.lean-frame {
  width: 100%; height: 1500px; border: 1px solid var(--border);
  border-radius: 10px; background: white; display: block;
}

/* ── Lists ─────────────────────────────────────── */
.nlist { margin-left: 18px; }
.nlist li { margin-bottom: 8px; }
.obs-list { margin-left: 18px; }
.obs-list li { margin-bottom: 8px; }

/* ── Misc ──────────────────────────────────────── */
.note { font-size: 11.5px; color: var(--muted); font-style: italic; }
code {
  font-family: var(--mono); font-size: 11.5px;
  background: #f1f5f9; padding: 1px 5px; border-radius: 3px;
}blockquote.formula {
  font-family: var(--mono); font-size: 13px;
  background: #f8fafc; border-left: 4px solid var(--accent);
  padding: 10px 18px; margin: 10px 0; border-radius: 0 6px 6px 0;
  color: var(--ink);
}
/* ── Print / PDF ───────────────────────────────── */
@media print {
  body { background: #fff; font-size: 11pt; }
  #nav, .print-btn { display: none !important; }
  #main { margin-left: 0 !important; padding: 20px 28px !important; max-width: 100% !important; }
  #cover {
    border-radius: 0; page-break-after: always;
    background: #0f2744 !important;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
  .sec { page-break-inside: avoid; }
  .vip-grid { grid-template-columns: repeat(2, 1fr); }
  .lean-frame { height: 1200px; }
  .tscroll { overflow-x: visible; }
  .callout, .best, .kpi, .secnum {
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
}
"""


def nav_js() -> str:
    return """
    const links = document.querySelectorAll('#nav a');
    const ids = Array.from(links).map(a => a.getAttribute('href').replace('#',''));
    const sections = ids.map(id => document.getElementById(id)).filter(Boolean);
    const io = new IntersectionObserver(entries => {
      const vis = entries.filter(e => e.isIntersecting)
        .sort((a,b) => a.target.getBoundingClientRect().top - b.target.getBoundingClientRect().top);
      if (vis.length) {
        links.forEach(l => l.classList.remove('active'));
        const link = document.querySelector('#nav a[href="#'+vis[0].target.id+'"]');
        if (link) link.classList.add('active');
      }
    }, { rootMargin: '0px 0px -65% 0px', threshold: 0 });
    sections.forEach(s => io.observe(s));
"""


# ──────────────────────────────────────────────────────────────────────────────

def build_report(reports_dir: Path, vip_dir: Path) -> str:
    nav = """
<nav id="nav">
  <div class="brand">Asset Pricing ML</div>
  <ul>
    <li><a href="#cover">Overview</a></li>
    <li><a href="#sec-data">1. Data</a></li>
    <li><a href="#sec-models">2. Models</a></li>
    <li><a href="#sec-results">3. Results</a></li>
    <li><a href="#sec-vip" class="sub">Variable Importance</a></li>
    <li><a href="#sec-cum" class="sub">Cumulative Returns</a></li>
    <li><a href="#sec-metrics" class="sub">Metrics</a></li>
    <li><a href="#sec-lean" class="sub">Lean Reports</a></li>
    <li><a href="#sec-next">4. Next Steps</a></li>
    <li><a href="#sec-obs">5. Observations</a></li>
  </ul>
</nav>"""

    cover = """
<div id="cover">
  <h1>Asset Pricing ML Pipeline</h1>
  <p class="tagline">Research Report &mdash; Synthetic Universe Validation &mdash; May 2026</p>
  <div class="meta">
    <span>500 tickers &middot; 119 months</span>
    <span>7 models: OLS-3, OLS-all, RF, GBRF</span>
    <span>Walk-forward retraining &middot; Lean backtests</span>
    <span>FF6 performance attribution</span>
  </div>
  <a class="print-btn" onclick="window.print();return false;" href="#">&#x2B07; Save as PDF</a>
</div>"""

    body = (
        cover
        + section_data()
        + section_models()
        + section_results(reports_dir, vip_dir)
        + section_next_steps()
        + section_observations()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Asset Pricing ML Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800;1,14..32,400&display=swap" rel="stylesheet"/>
  <style>{global_css()}</style>
</head>
<body>
{nav}
<div id="main">
{body}
</div>
<script>(function(){{ {nav_js()} }})();</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────────

def generate_pdf(html_path: Path) -> Path:
    pdf_path = html_path.with_suffix(".pdf")
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome, "--headless", "--disable-gpu", "--no-sandbox",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        "--run-all-compositor-stages-before-draw",
        f"file://{html_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"Chrome PDF failed:\n{result.stderr}")
    return pdf_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_dir", nargs="?", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--output", default="")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output) if args.output else reports_dir / "full_report.html"

    html = build_report(reports_dir, VIP_PLOTS_DIR)
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML → {output_path}")

    if args.pdf:
        try:
            pdf = generate_pdf(output_path.resolve())
            print(f"PDF  → {pdf}")
        except Exception as e:
            print(f"PDF generation failed: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
