"""
streamlit_app.py  —  RevInsight · Light Theme (Tabler/Linear inspired)
Run with:  streamlit run streamlit_app.py
"""

import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

sys.path.insert(0, os.path.dirname(__file__))
from analysis_engine import run_full_analysis, normalize_sp_name

st.set_page_config(
    page_title="Staffing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette  (Tabler-inspired light theme) ─────────────────────────────────────
BG        = "#f0f2f5"
SIDEBAR   = "#ffffff"
WHITE     = "#ffffff"
BORDER    = "#e2e8f0"
TEXT      = "#1e293b"
MUTED     = "#64748b"
BLUE      = "#2563eb"
BLUE_LT   = "#eff6ff"
BLUE_MID  = "#bfdbfe"
GREEN     = "#16a34a"
GREEN_LT  = "#f0fdf4"
RED       = "#dc2626"
RED_LT    = "#fef2f2"
AMBER     = "#d97706"
AMBER_LT  = "#fffbeb"
PURPLE    = "#7c3aed"
PURPLE_LT = "#f5f3ff"

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp {{ background:{BG} !important; color:{TEXT}; font-family:'Inter',sans-serif; }}
.stApp header {{ background:transparent !important; box-shadow:none !important; }}
section[data-testid="stSidebar"] {{
  background:{SIDEBAR} !important;
  border-right:1px solid {BORDER} !important;
  box-shadow:2px 0 8px rgba(0,0,0,0.04);
}}

#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:1.6rem !important; padding-bottom:2rem !important; }}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption {{ color:{MUTED} !important; font-size:13px; }}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color:{TEXT} !important; }}

.hero {{
  background: linear-gradient(135deg, {BLUE} 0%, #1d4ed8 50%, #1e40af 100%);
  border-radius:16px; padding:36px 40px 32px;
  margin-bottom:28px; position:relative; overflow:hidden;
}}
.hero::after {{
  content:''; position:absolute; top:-80px; right:-80px;
  width:260px; height:260px;
  background:radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
  pointer-events:none;
}}
.hero-h1   {{ font-size:30px; font-weight:800; color:#fff; line-height:1.2; margin-bottom:10px; }}
.hero-sub  {{ font-size:14px; color:rgba(255,255,255,0.75); max-width:560px; line-height:1.7; margin-bottom:20px; }}
.hero-pill {{
  display:inline-block; background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25);
  color:#fff; font-size:12px; font-weight:500; border-radius:100px;
  padding:4px 14px; margin-right:8px; margin-bottom:6px; backdrop-filter:blur(4px);
}}

.sec-label {{
  font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase;
  color:{MUTED}; margin:28px 0 14px; display:flex; align-items:center; gap:10px;
}}
.sec-label::after {{ content:''; flex:1; height:1px; background:{BORDER}; }}

.kpi-grid {{ display:flex; gap:14px; margin-bottom:24px; flex-wrap:wrap; }}
.kpi {{
  flex:1; min-width:150px;
  background:{WHITE}; border:1px solid {BORDER}; border-radius:12px;
  padding:18px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.05);
  position:relative; overflow:hidden;
}}
.kpi::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:12px 12px 0 0; }}
.kpi.blue::before   {{ background:{BLUE}; }}
.kpi.green::before  {{ background:{GREEN}; }}
.kpi.red::before    {{ background:{RED}; }}
.kpi.amber::before  {{ background:{AMBER}; }}
.kpi.purple::before {{ background:{PURPLE}; }}

.kpi-label {{ font-size:11px; font-weight:600; color:{MUTED}; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }}
.kpi-val   {{ font-size:26px; font-weight:800; color:{TEXT}; font-family:'JetBrains Mono',monospace; line-height:1; }}
.kpi-val.blue   {{ color:{BLUE}; }}
.kpi-val.green  {{ color:{GREEN}; }}
.kpi-val.red    {{ color:{RED}; }}
.kpi-val.amber  {{ color:{AMBER}; }}
.kpi-val.purple {{ color:{PURPLE}; }}
.kpi-sub  {{ font-size:12px; color:{MUTED}; margin-top:6px; }}

.verdict {{
  border-radius:14px; padding:28px 32px; margin:4px 0 24px;
  border:1px solid; position:relative; overflow:hidden;
}}
.verdict.yes {{ background:{GREEN_LT}; border-color:#bbf7d0; }}
.verdict.no  {{ background:{RED_LT};   border-color:#fecaca; }}
.verdict::before {{
  content:''; position:absolute; left:0; top:0; bottom:0; width:5px; border-radius:14px 0 0 14px;
}}
.verdict.yes::before {{ background:{GREEN}; }}
.verdict.no::before  {{ background:{RED}; }}

.verdict-badge {{
  display:inline-flex; align-items:center; gap:6px;
  font-size:11px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;
  padding:4px 12px; border-radius:100px; margin-bottom:12px;
}}
.verdict-badge.yes {{ background:#dcfce7; color:{GREEN}; }}
.verdict-badge.no  {{ background:#fee2e2; color:{RED}; }}

.verdict-h      {{ font-size:22px; font-weight:800; color:{TEXT}; margin-bottom:6px; }}
.verdict-reason {{ font-size:13px; font-weight:500; color:{TEXT}; margin-bottom:10px;
                   padding:8px 14px; background:rgba(0,0,0,0.04); border-radius:8px;
                   display:inline-block; }}
.verdict-txt {{ font-size:14px; color:{MUTED}; line-height:1.7; max-width:660px; }}
.verdict-stats {{ display:flex; gap:28px; margin-top:20px; flex-wrap:wrap; }}
.vs-val  {{ font-size:20px; font-weight:700; font-family:'JetBrains Mono',monospace; }}
.vs-val.pos {{ color:{GREEN}; }}
.vs-val.neg {{ color:{RED}; }}
.vs-val.neu {{ color:{BLUE}; }}
.vs-lab  {{ font-size:11px; color:{MUTED}; margin-top:3px; }}

.info-box {{
  background:{BLUE_LT}; border:1px solid {BLUE_MID}; border-radius:10px;
  padding:14px 18px; font-size:13px; color:{TEXT}; line-height:1.7; margin:10px 0 18px;
}}
.tip-box {{
  background:{AMBER_LT}; border:1px solid #fde68a; border-radius:10px;
  padding:13px 18px; font-size:13px; color:#92400e; line-height:1.65; margin:10px 0 16px;
}}

.card {{
  background:{WHITE}; border:1px solid {BORDER}; border-radius:12px;
  padding:22px 24px; margin-bottom:16px;
  box-shadow:0 1px 4px rgba(0,0,0,0.05);
}}

.stTabs [data-baseweb="tab-list"] {{
  background:{WHITE}; border:1px solid {BORDER}; border-radius:10px;
  padding:4px; gap:2px;
}}
.stTabs [data-baseweb="tab"] {{
  color:{MUTED}; border-radius:8px; font-weight:500; font-size:13px; padding:8px 16px;
}}
.stTabs [aria-selected="true"] {{
  background:{BLUE_LT} !important; color:{BLUE} !important; font-weight:600 !important;
}}

.stButton > button {{
  background:{BLUE} !important; color:#fff !important;
  font-weight:600 !important; border:none !important; border-radius:8px !important;
  font-size:14px !important; padding:10px 20px !important;
  box-shadow:0 2px 8px rgba(37,99,235,0.25) !important;
  transition:all .2s ease;
}}
.stButton > button:hover {{ background:#1d4ed8 !important; transform:translateY(-1px); }}

.stDataFrame {{ border:1px solid {BORDER} !important; border-radius:10px; overflow:hidden; }}

[data-testid="stMetricValue"] {{ color:{TEXT} !important; font-weight:700 !important; }}
[data-testid="stMetricLabel"] {{ color:{MUTED} !important; }}

hr {{ border-color:{BORDER} !important; }}

[data-testid="stFileUploader"] {{
  border:2px dashed {BLUE_MID} !important; border-radius:10px !important;
  background:{BLUE_LT} !important;
}}

[data-baseweb="select"] {{ border-color:{BORDER} !important; }}

/* ── Page footer ── */
.page-footer {{
  text-align:center; color:{MUTED}; font-size:12px;
  padding:32px 0 12px; border-top:1px solid {BORDER}; margin-top:40px;
}}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px;">
      <div style="background:{BLUE};width:36px;height:36px;border-radius:9px;
                  display:flex;align-items:center;justify-content:center;font-size:18px;">📊</div>
      <div>
        <div style="font-size:11px;color:{MUTED};">Staffing Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown(f'<div style="font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{MUTED};margin-bottom:8px;">Data Source</div>', unsafe_allow_html=True)
    st.caption("CSV with columns: Month, Sales, sale sp1, sale sp2 …")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.divider()
    st.markdown(f'<div style="font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{MUTED};margin-bottom:8px;">Financial Parameters</div>', unsafe_allow_html=True)
    salary   = st.number_input("Monthly salary / person", value=3.0, step=0.5, min_value=0.1)
    comm_pct = st.slider("Commission rate (%)", 0, 20, 5) / 100
    gm_pct   = st.slider("Gross margin rate (%)", 10, 90, 50) / 100

    st.divider()
    st.markdown(f'<div style="font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{MUTED};margin-bottom:8px;">Forecast Settings</div>', unsafe_allow_html=True)
    horizon       = st.slider("Projection horizon (months)", 3, 24, 6)
    season_period = st.selectbox("Seasonal cycle (0 = auto)", [0, 6, 12, 18], index=0)

    st.divider()
    show_raw = st.checkbox("Show raw data table", value=False)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-h1">Staffing Intelligence Platform</div>
  <div class="hero-sub">
    Upload your store's sales history and get a statistically rigorous hiring recommendation —
    with profit forecasts, skill rankings, and scenario analysis, in seconds.
  </div>
  <span class="hero-pill">📈 Profit Forecasting</span>
  <span class="hero-pill">🏆 Skill Ranking</span>
  <span class="hero-pill">📅 Seasonality Detection</span>
  <span class="hero-pill">🎯 Scenario Modelling</span>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.markdown(f"""
    <div class="info-box">
      👈 &nbsp;<strong>Start by uploading your sales CSV in the sidebar.</strong>
      One row per month, with individual salesperson columns named <code>sale sp1</code>, <code>sale sp2</code>, etc.
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📂 What should my CSV look like?"):
            st.markdown("""
| Month | Sales | sale sp1 | sale sp2 |
|-------|-------|----------|----------|
| 1     | 23.1  | 7.2      | 6.3      |
| 2     | 41.6  | 8.9      | 7.8      |
            """)
    with col2:
        with st.expander("🖥️ Installation"):
            st.code("pip install streamlit pandas numpy statsmodels matplotlib plotly\nstreamlit run streamlit_app.py")
    st.stop()


# ── Load CSV ───────────────────────────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = [c.strip() for c in df_raw.columns]
except Exception as e:
    st.error(f"Couldn't read file: {e}"); st.stop()

sp_cols = [c for c in df_raw.columns if re.match(r"^sale\s+sp\d+$", c.strip(), re.IGNORECASE)]
if not sp_cols:
    st.error("No salesperson columns found. Name them 'sale sp1', 'sale sp2', etc."); st.stop()

df_raw[sp_cols] = df_raw[sp_cols].fillna(0)
last_month  = int(df_raw["Month"].max())
last_row    = df_raw[df_raw["Month"] == last_month].iloc[0]
active_now  = [sp for sp in sp_cols if last_row[sp] > 0]
total_sales = df_raw["Sales"].sum()

st.markdown('<div class="sec-label">Departing Employee</div>', unsafe_allow_html=True)

matched     = [sp for sp in active_now if "sp12" in sp.lower()]
default_sp  = matched[0] if matched else active_now[0]
default_idx = active_now.index(default_sp)

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    leaving_sp_label = st.selectbox("Select Departing Employee", options=active_now, index=default_idx)
with col_btn:
    st.write(""); st.write("")
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

if not run_btn:
    st.markdown(f'<div class="tip-box">💡 Select the departing employee and click <strong>Run Analysis</strong>.</div>', unsafe_allow_html=True)
    st.stop()


# ── Analysis ───────────────────────────────────────────────────────────────────
with st.spinner("Running statistical models…"):
    try:
        results = run_full_analysis(
            df_raw, leaving_name=leaving_sp_label,
            horizon=horizon, salary=salary,
            commission_rate=comm_pct, gross_margin_rate=gm_pct,
            season_period=season_period,
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}"); st.stop()

leaving_sp     = results["leaving_sp"]
params         = results["params"]
ceiling        = results["ceiling"]
projection     = results["projection"]
skill_table    = results["skill_table"]
margin_summary = results["margin_summary"]
df             = results["df"]

nr_total   = projection["scenarios"]["noreplace"]["margin"].sum()
rm_total   = projection["scenarios"]["replace_mean"]["margin"].sum()
diff_mean  = rm_total - nr_total
recommend  = diff_mean > 0

leaving_row   = skill_table[skill_table["Salesperson"] == leaving_sp]
leaving_rank  = int(leaving_row["Rank"].values[0])  if not leaving_row.empty else "?"
leaving_skill = float(leaving_row["Skill Effect"].values[0]) if not leaving_row.empty else 0.0
n_total       = len(skill_table)
r2_pct        = params["r2"] * 100
all_skills    = projection["all_skills"]
below_count   = sum(1 for s in all_skills if s < leaving_skill)
upgrade_pct   = int(100 * (len(all_skills) - below_count - 1) / len(all_skills))
gain_sign     = "+" if diff_mean >= 0 else ""


# ═══ VERDICT (shown first — above the fold) ═══════════════════════════════════
st.markdown('<div class="sec-label">Recommendation</div>', unsafe_allow_html=True)

if recommend:
    vcls    = "yes"; vbadge = "yes"; vicon = "✅"; vword = "Hire a Replacement"
    vhead   = f"Replace {leaving_sp}"
    vreason = (f"{leaving_sp} ranks #{leaving_rank} of {n_total} — replacing them is projected to generate "
               f"₹{diff_mean:+.2f} more profit over {horizon} months ({r2_pct:.0f}% model confidence).")
    vtxt    = (f"Based on {last_month} months of store data, hiring a replacement is projected to generate "
               f"<strong style='color:{GREEN};'>₹{diff_mean:+.2f} more profit</strong> over {horizon} months "
               f"vs. leaving the seat empty. {leaving_sp} ranks #{leaving_rank} of {n_total} — "
               f"{upgrade_pct}% of all historical hires would outperform them.")
else:
    vcls    = "no"; vbadge = "no"; vicon = "⛔"; vword = "Hold Off"
    vhead   = f"Don't rush to replace {leaving_sp}"
    vreason = (f"{leaving_sp} ranks #{leaving_rank} of {n_total} — leaving the seat empty saves "
               f"₹{abs(diff_mean):.2f} over {horizon} months ({r2_pct:.0f}% model confidence).")
    vtxt    = (f"Leaving the role empty is projected to save "
               f"<strong style='color:{RED};'>₹{abs(diff_mean):.2f}</strong> over {horizon} months. "
               f"Only proceed if you have a strong candidate already identified.")

stat_cls = "pos" if diff_mean >= 0 else "neg"

st.markdown(f"""
<div class="verdict {vcls}">
  <div class="verdict-badge {vbadge}">{vicon} &nbsp;{vword}</div>
  <div class="verdict-h">{vhead}</div>
  <div class="verdict-reason">💬 {vreason}</div>
  <div class="verdict-txt">{vtxt}</div>
  <div class="verdict-stats">
    <div>
      <div class="vs-val {stat_cls}">{gain_sign}₹{diff_mean:.2f}</div>
      <div class="vs-lab">Profit delta over {horizon} months</div>
    </div>
    <div>
      <div class="vs-val neu">#{leaving_rank} / {n_total}</div>
      <div class="vs-lab">All-time performance rank</div>
    </div>
    <div>
      <div class="vs-val neu">{upgrade_pct}%</div>
      <div class="vs-lab">Chance any hire is an upgrade</div>
    </div>
    <div>
      <div class="vs-val neu">{r2_pct:.0f}%</div>
      <div class="vs-lab">Forecast confidence (R²)</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══ KPI BAR (below verdict) ══════════════════════════════════════════════════
st.markdown('<div class="sec-label">Key Metrics</div>', unsafe_allow_html=True)

gain_cls = "green" if recommend else "red"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi blue">
    <div class="kpi-label">Total Historical Sales</div>
    <div class="kpi-val blue">₹{total_sales:.1f}</div>
    <div class="kpi-sub">{last_month} months of data</div>
  </div>
  <div class="kpi blue">
    <div class="kpi-label">Active Staff</div>
    <div class="kpi-val blue">{len(active_now)}</div>
    <div class="kpi-sub">of {len(sp_cols)} ever hired</div>
  </div>
  <div class="kpi purple">
    <div class="kpi-label">Performance Rank</div>
    <div class="kpi-val purple">#{leaving_rank} / {n_total}</div>
    <div class="kpi-sub">Tenure & season adjusted</div>
  </div>
  <div class="kpi {gain_cls}">
    <div class="kpi-label">Expected Profit Gain</div>
    <div class="kpi-val {gain_cls}">{gain_sign}₹{diff_mean:.1f}</div>
    <div class="kpi-sub">if you hire · {horizon} months</div>
  </div>
  <div class="kpi blue">
    <div class="kpi-label">Forecast Confidence</div>
    <div class="kpi-val blue">{r2_pct:.0f}%</div>
    <div class="kpi-sub">Model R²</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══ TABS ═════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Profit Forecast", "🎯 Hire Quality", "🏆 Team Rankings",
    "📅 Seasonal Patterns", "🏪 Overlap & Ceiling", "📋 Tips & Rules", "📌 Assumptions",
])


def plotly_layout(fig, title="", xlab="", ylab=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=TEXT, family="Inter"), x=0, pad=dict(b=8)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=MUTED, size=11),
        xaxis=dict(title=xlab, gridcolor=BORDER, linecolor=BORDER,
                   tickcolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(title=ylab, gridcolor=BORDER, linecolor=BORDER,
                   tickcolor=BORDER, tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    return fig


# ── TAB 1: Profit Forecast ─────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-label">Month-by-Month Profit Forecast</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
      <strong>Blue line</strong> — hire a replacement (average-quality hire) &nbsp;·&nbsp;
      <strong>Red line</strong> — leave the role empty.<br>
      The shaded gap is your expected profit difference each month.
    </div>
    """, unsafe_allow_html=True)

    mc = projection["monthly_compare"]

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mc["month"], y=mc["margin_noreplace"],
            mode="lines+markers", name="Leave empty",
            line=dict(color=RED, width=2.5), marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=mc["month"], y=mc["margin_replace_mean"],
            mode="lines+markers", name="Hire replacement",
            line=dict(color=BLUE, width=2.5), marker=dict(size=5, symbol="square"),
        ))
        fig.add_trace(go.Scatter(
            x=list(mc["month"]) + list(mc["month"])[::-1],
            y=list(mc["margin_replace_mean"]) + list(mc["margin_noreplace"])[::-1],
            fill="toself", fillcolor="rgba(37,99,235,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="Profit gap", showlegend=True,
        ))
        plotly_layout(fig, xlab="Projected month", ylab="Gross profit (₹)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig_m, ax_m = plt.subplots(figsize=(10, 3.5), facecolor="white")
        ax_m.set_facecolor("white")
        ax_m.plot(mc["month"], mc["margin_noreplace"],  color=RED,  marker="o", lw=2, ms=4, label="Leave empty")
        ax_m.plot(mc["month"], mc["margin_replace_mean"], color=BLUE, marker="s", lw=2, ms=4, label="Hire replacement")
        ax_m.fill_between(mc["month"], mc["margin_noreplace"], mc["margin_replace_mean"], alpha=0.1, color=BLUE)
        ax_m.spines[["top","right"]].set_visible(False)
        ax_m.spines[["bottom","left"]].set_color(BORDER)
        ax_m.tick_params(colors=MUTED); ax_m.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig_m, use_container_width=True)

    st.markdown(f'<div class="tip-box">💡 Over {horizon} months, hiring is expected to earn <strong>₹{diff_mean:+.2f} more</strong> than leaving the seat empty.</div>', unsafe_allow_html=True)


# ── TAB 2: Hire Quality ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-label">Sensitivity by Hire Quality</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
      Extra profit vs. leaving empty — from the worst hire your store has ever made to the best.
      Even a below-average hire usually beats an empty seat.
    </div>
    """, unsafe_allow_html=True)

    scens  = projection["scenarios"]
    labels = [("Worst ever","replace_worst"),("Bottom 25%","replace_p25"),
              ("Median",    "replace_median"),("Average",   "replace_mean"),
              ("Top 25%",   "replace_p75"),  ("Best ever", "replace_best")]
    names  = [l for l, _ in labels]
    deltas = [scens[k]["margin"].sum() - nr_total for _, k in labels]

    if HAS_PLOTLY:
        fig2 = go.Figure(go.Bar(
            x=names, y=deltas,
            marker_color=[BLUE if d >= 0 else RED for d in deltas],
            marker_line_color="white", marker_line_width=1.5,
            text=[f"₹{d:+.1f}" for d in deltas], textposition="outside",
            textfont=dict(color=TEXT, size=11, family="JetBrains Mono"),
        ))
        fig2.add_hline(y=0, line_color=MUTED, line_width=1)
        plotly_layout(fig2, title=f"Extra profit vs. empty seat over {horizon} months (₹)", ylab="₹")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        fig2m, ax2m = plt.subplots(figsize=(8, 3.5), facecolor="white")
        ax2m.set_facecolor("white")
        ax2m.bar(names, deltas, color=[BLUE if d >= 0 else RED for d in deltas], edgecolor="white", width=0.55)
        ax2m.axhline(0, color=MUTED, linewidth=1)
        ax2m.spines[["top","right"]].set_visible(False); ax2m.spines[["bottom","left"]].set_color(BORDER)
        ax2m.tick_params(colors=MUTED); plt.xticks(rotation=20, ha="right", fontsize=9)
        plt.tight_layout(); st.pyplot(fig2m, use_container_width=True)

    st.markdown('<div class="sec-label">Scenario Breakdown</div>', unsafe_allow_html=True)
    ms = margin_summary.copy()
    ms.columns = ["Scenario", "Skill (₹/mo)", f"Total profit ({horizon} mo)", "vs. empty seat"]
    st.dataframe(ms, use_container_width=True, hide_index=True)


# ── TAB 3: Team Rankings ───────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-label">Team Performance Ranking</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
      Monthly sales contribution vs. a baseline hire — adjusted for tenure & seasonality.
      <strong style="color:{AMBER};">Amber</strong> = leaving &nbsp;·&nbsp;
      <strong style="color:{BLUE};">Blue</strong> = active &nbsp;·&nbsp;
      <strong style="color:{MUTED};">Grey</strong> = departed
    </div>
    """, unsafe_allow_html=True)

    col_t, col_c = st.columns([1, 1.5])
    with col_t:
        def hl(row):
            if row["Salesperson"] == leaving_sp:
                return [f"background:{AMBER_LT};color:{AMBER};font-weight:600"] * len(row)
            if row["Status"] == "Active":
                return [f"background:{BLUE_LT};color:{BLUE}"] * len(row)
            return [f"color:{MUTED}"] * len(row)
        st.dataframe(
            skill_table[["Rank","Salesperson","Tenure (mo)","Skill Effect","Status"]].style.apply(hl, axis=1),
            use_container_width=True, hide_index=True
        )
    with col_c:
        sk = skill_table.sort_values("Skill Effect")
        bar_cs = [AMBER if r["Salesperson"]==leaving_sp else (BLUE if r["Status"]=="Active" else "#cbd5e1")
                  for _, r in sk.iterrows()]
        if HAS_PLOTLY:
            fig3 = go.Figure(go.Bar(
                x=sk["Skill Effect"], y=sk["Salesperson"], orientation="h",
                marker_color=bar_cs, marker_line_color="white", marker_line_width=1,
                text=[f"{v:+.2f}" for v in sk["Skill Effect"]], textposition="outside",
                textfont=dict(color=TEXT, size=10, family="JetBrains Mono"),
            ))
            fig3.add_vline(x=0, line_color=MUTED, line_width=1, line_dash="dash")
            plotly_layout(fig3, xlab="Monthly contribution vs. average (₹)")
            fig3.update_layout(height=max(320, len(skill_table)*28), margin=dict(l=8,r=8,t=12,b=8))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            fig3m, ax3m = plt.subplots(figsize=(6, max(4, len(skill_table)*0.38)), facecolor="white")
            ax3m.set_facecolor("white")
            ax3m.barh(sk["Salesperson"], sk["Skill Effect"], color=bar_cs, edgecolor="white")
            ax3m.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
            ax3m.spines[["top","right"]].set_visible(False); ax3m.spines[["bottom","left"]].set_color(BORDER)
            ax3m.tick_params(colors=MUTED); plt.tight_layout(); st.pyplot(fig3m, use_container_width=True)

    st.markdown('<div class="sec-label">Skill Distribution — All Historical Hires</div>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=all_skills, nbinsx=8,
            marker_color=BLUE, marker_line_color="white", marker_line_width=1.5,
            opacity=0.75, name="All hires",
        ))
        for val, col, lbl in [
            (np.mean(all_skills),   TEXT,  f"Avg ({np.mean(all_skills):.2f})"),
            (np.median(all_skills), AMBER, f"Median ({np.median(all_skills):.2f})"),
            (leaving_skill,         RED,   f"{leaving_sp} ({leaving_skill:+.2f})"),
        ]:
            fig_h.add_vline(x=val, line_color=col, line_width=2,
                            annotation_text=lbl, annotation_font_color=col,
                            annotation_position="top right")
        plotly_layout(fig_h, xlab="Monthly contribution vs. baseline (₹)", ylab="Count")
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        fig_hm, ax_hm = plt.subplots(figsize=(8, 3), facecolor="white")
        ax_hm.set_facecolor("white")
        ax_hm.hist(all_skills, bins=8, color=BLUE, edgecolor="white", alpha=0.75)
        ax_hm.axvline(np.mean(all_skills),   color=TEXT,  lw=1.8, linestyle="--")
        ax_hm.axvline(np.median(all_skills), color=AMBER, lw=1.8, linestyle=":")
        ax_hm.axvline(leaving_skill,          color=RED,   lw=2.2)
        ax_hm.spines[["top","right"]].set_visible(False); ax_hm.spines[["bottom","left"]].set_color(BORDER)
        ax_hm.tick_params(colors=MUTED); plt.tight_layout(); st.pyplot(fig_hm, use_container_width=True)

    st.markdown(f'<div class="tip-box">💡 Only <strong>{below_count} of {len(all_skills)}</strong> hires ever performed worse. Any random replacement has a <strong>{upgrade_pct}%+ chance</strong> of being an upgrade.</div>', unsafe_allow_html=True)


# ── TAB 4: Seasonal Patterns ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-label">Seasonal Sales Cycle</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">Repeating seasonal rhythm detected in your data. <strong>Blue bars</strong> = above-average months · <strong>Red bars</strong> = slower months. Baked into every forecast.</div>', unsafe_allow_html=True)

    if params["season_period"] > 1:
        sp_len  = params["season_period"]
        offsets = list(range(sp_len))
        effects = [params["season_effects"].get(o, 0.0) for o in offsets]
        if HAS_PLOTLY:
            fig5 = go.Figure(go.Bar(
                x=offsets, y=effects,
                marker_color=[BLUE if e >= 0 else RED for e in effects],
                marker_line_color="white", marker_line_width=1,
                text=[f"{e:+.2f}" for e in effects], textposition="outside",
                textfont=dict(color=TEXT, size=10),
            ))
            fig5.add_hline(y=0, line_color=MUTED)
            plotly_layout(fig5, title=f"{sp_len}-month cycle", xlab="Month in cycle", ylab="vs. average (₹)")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            fig5m, ax5m = plt.subplots(figsize=(8, 3), facecolor="white")
            ax5m.set_facecolor("white")
            ax5m.bar(offsets, effects, color=[BLUE if e >= 0 else RED for e in effects], edgecolor="white")
            ax5m.axhline(0, color=MUTED, linewidth=0.8)
            ax5m.spines[["top","right"]].set_visible(False); ax5m.spines[["bottom","left"]].set_color(BORDER)
            ax5m.tick_params(colors=MUTED); plt.tight_layout(); st.pyplot(fig5m, use_container_width=True)
    else:
        st.info("No clear seasonal pattern detected — sales are fairly consistent year-round.")

    st.markdown('<div class="sec-label">Full Sales History</div>', unsafe_allow_html=True)
    hc = (df[sp_cols] > 0).sum(axis=1)

    if HAS_PLOTLY:
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=df["Month"], y=df["Sales"], name="Total Sales",
            marker_color=BLUE, opacity=0.55,
        ))
        fig6.add_trace(go.Scatter(
            x=df["Month"], y=hc, name="Headcount",
            mode="lines+markers", line=dict(color=AMBER, width=2),
            yaxis="y2",
        ))
        fig6.update_layout(
            yaxis2=dict(overlaying="y", side="right",
                        title=dict(text="Staff count", font=dict(color=AMBER)),
                        gridcolor="rgba(0,0,0,0)",
                        tickfont=dict(color=AMBER)),
        )
        plotly_layout(fig6, xlab="Month", ylab="Total sales (₹)")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        fig6m, ax6a = plt.subplots(figsize=(10, 3), facecolor="white")
        ax6b = ax6a.twinx()
        ax6a.set_facecolor("white"); fig6m.set_facecolor("white")
        ax6a.bar(df["Month"], df["Sales"], color=BLUE, alpha=0.45, edgecolor="white")
        ax6b.plot(df["Month"], hc, color=AMBER, linewidth=1.8)
        ax6a.spines[["top","right"]].set_color(BORDER); ax6a.tick_params(colors=MUTED)
        ax6b.tick_params(colors=MUTED); plt.tight_layout(); st.pyplot(fig6m, use_container_width=True)


# ── TAB 5: Overlap & Ceiling ───────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="sec-label">Capacity Ceiling Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">Does running an extra person during handover grow sales — or just split the same customers? Compares normal team size vs. 8-staff months.</div>', unsafe_allow_html=True)

    if ceiling["n_above"] >= 2:
        normal_resid = ceiling["normal_mean_resid"] or 0
        above_resid  = ceiling["above_mean_resid"]  or 0
        has_ceiling  = above_resid < -2

        c1, c2, c3 = st.columns(3)
        c1.metric("Normal team size", f"{ceiling['modal_headcount']} people", f"{ceiling['n_normal']} months")
        c2.metric("Sales gap — normal months", f"{normal_resid:+.2f}", "actual vs. model")
        c3.metric("Sales gap — 8-staff months", f"{above_resid:+.2f}", "negative = ceiling", delta_color="inverse")

        if has_ceiling:
            st.markdown(f"""
            <div class="tip-box">
              ⚠️ <strong>Capacity ceiling detected.</strong>
              During {ceiling['n_above']} months with 8 staff, sales ran <strong>{abs(above_resid):.1f} units below</strong> what individual track records predict.
              Time the new hire's start close to the departing employee's last day.
            </div>""", unsafe_allow_html=True)
        else:
            st.success("No ceiling found — the 8th person does appear to grow total sales.")

        if HAS_PLOTLY:
            vals = [normal_resid, above_resid]
            cats = [f"Normal ({ceiling['modal_headcount']} staff)", "Overlap (8 staff)"]
            fig7 = go.Figure(go.Bar(
                x=cats, y=vals, width=0.35,
                marker_color=[BLUE if v >= 0 else RED for v in vals],
                marker_line_color="white", marker_line_width=1.5,
                text=[f"{v:+.2f}" for v in vals], textposition="outside",
                textfont=dict(color=TEXT, size=13, family="JetBrains Mono"),
            ))
            fig7.add_hline(y=0, line_color=MUTED)
            plotly_layout(fig7, ylab="Avg sales gap (₹)")
            fig7.update_layout(height=300)
            st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("Not enough months with 8 staff to test the capacity ceiling.")


# ── TAB 6: Tips & Rules ────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="sec-label">Staffing Playbook</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">Practical rules derived from {last_month} months of your store\'s data.</div>', unsafe_allow_html=True)

    tips = [
        ("✅  Usually replace when someone leaves",
         "Hiring beats an empty seat in almost every scenario — even a below-average hire. The only exception: a specific reason to expect an unusually weak candidate pool."),
        ("⏱️  Avoid full-month overlap if possible",
         "8-staff months show lower-than-expected total sales. Time the new hire's start to coincide with the departing person's last day."),
        ("🔄  Re-run this tool every time someone leaves",
         "The answer depends on who's leaving, the season, and who's staying. Fresh data takes a minute."),
        ("📊  Use the skill table to manage performance",
         "The ranking adjusts for tenure and seasonality — use it for 6- and 12-month new hire reviews."),
        ("⚠️  Capacity ceiling around 7 staff",
         "An 8th person doesn't grow total sales proportionally. Customer footfall is the real constraint, not headcount."),
    ]
    for title_r, body_r in tips:
        with st.expander(title_r, expanded=True):
            st.write(body_r)

    st.markdown('<div class="sec-label">Model Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Months of data | {last_month} |
| Total salespeople | {n_total} |
| Seasonal cycle | {params['season_period']} months |
| Model R² | {params['r2']:.3f} ({r2_pct:.1f}%) |
| Salary | ₹{salary}/mo |
| Commission | {comm_pct:.0%} |
| Gross margin | {gm_pct:.0%} |
| Horizon | {horizon} months |
""")

    st.divider()
    if st.button("📄 Download PDF Report", use_container_width=True):
        with st.spinner("Building PDF…"):
            try:
                from generate_pdf_report import build_pdf
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = tmp.name
                build_pdf(results, tmp_path)
                with open(tmp_path, "rb") as f:
                    pdf_bytes = f.read()
                os.unlink(tmp_path)
                st.download_button("⬇️ Save staffing_report.pdf", data=pdf_bytes,
                                   file_name="staffing_report.pdf", mime="application/pdf",
                                   use_container_width=True)
            except Exception as e:
                st.error(f"PDF generation failed: {e}")


# ── TAB 7: Assumptions ─────────────────────────────────────────────────────────
with tab7:
    st.markdown('<div class="sec-label">Model Assumptions</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">Every recommendation comes from a statistical model with explicit assumptions. Review these to know when to trust the output — and when to apply extra judgment.</div>', unsafe_allow_html=True)

    assumptions = [
        ("💰 Gross margin formula",
         f"Gross margin = {gm_pct:.0%} × Sales − ₹{salary} salary/person − {comm_pct:.0%} × Sales. Net margin rate = {gm_pct - comm_pct:.0%} of sales, minus fixed salaries."),
        ("📈 New hires ramp up via log-curve",
         "Performance follows a log-curve — most learning in months 1–3, stabilising thereafter. Calibrated from your own historical hire data."),
        ("🎲 Future hire quality = historical average",
         f"Baseline assumes skill matching the mean of all {n_total} salespeople ever employed."),
        ("🔁 Seasonal patterns repeat",
         "The detected cycle is assumed to continue. Major format changes may reduce accuracy."),
        ("👣 Footfall is the short-term constraint",
         "Capacity ceiling analysis assumes mall footfall — not headcount — limits monthly sales."),
        ("🏬 Stable store format across data period",
         "Major changes (renovation, rebranding, pricing) could make older data less representative."),
        ("📊 Individual sales are accurately recorded",
         "The model attributes each sale to a specific salesperson. Requires correct per-person columns in CSV."),
    ]
    for t, b in assumptions:
        with st.expander(t, expanded=True):
            st.write(b)

    if show_raw:
        st.markdown('<div class="sec-label">Raw Data</div>', unsafe_allow_html=True)
        st.dataframe(df_raw, use_container_width=True)


# ── Page footer ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-footer">
  Built by Sreeja Penumudi &nbsp;·&nbsp; RevInsight Staffing Analysis Assignment
</div>
""", unsafe_allow_html=True)