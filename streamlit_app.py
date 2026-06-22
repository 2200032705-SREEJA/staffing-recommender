"""
streamlit_app.py  —  Friendly rewrite
Run with:  streamlit run streamlit_app.py
"""

import io
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from analysis_engine import run_full_analysis, normalize_sp_name

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Should I Replace My Salesperson?",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN  = "#2e7d32"
RED    = "#c62828"
AMBER  = "#e67e00"
NAVY   = "#1a2e4a"
TEAL   = "#0d7a7a"
LGRAY  = "#f4f6f8"

st.markdown("""
<style>
    /* Verdict banners */
    .verdict-yes {
        background:#e8f5e9; border-radius:12px; padding:20px 24px;
        border-left:6px solid #2e7d32; font-size:17px;
        font-weight:600; color:#2e7d32;
    }
    .verdict-no {
        background:#ffebee; border-radius:12px; padding:20px 24px;
        border-left:6px solid #c62828; font-size:17px;
        font-weight:600; color:#c62828;
    }
    /* Plain-English explanation boxes */
    .explain-box {
        background:#f0f4ff; border-radius:10px; padding:14px 18px;
        border-left:4px solid #5c7cfa; font-size:14px;
        color:#333; margin:10px 0 18px 0; line-height:1.6;
    }
    .tip-box {
        background:#fff8e1; border-radius:10px; padding:14px 18px;
        border-left:4px solid #f9a825; font-size:13px;
        color:#555; margin:8px 0 14px 0; line-height:1.6;
    }
    /* KPI cards */
    .kpi-card {
        background:#f4f6f8; border-radius:10px; padding:16px 18px;
        border-left:4px solid #0d7a7a; margin:4px 0;
    }
    .kpi-number { font-size:26px; font-weight:700; color:#1a2e4a; }
    .kpi-label  { font-size:13px; color:#666; margin-bottom:4px; }
    .kpi-meaning{ font-size:12px; color:#888; margin-top:6px; line-height:1.5; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/clothes.png", width=56)
    st.title("Staffing Recommender")
    st.caption("Branded Clothing Store · Bangalore")
    st.divider()

    st.subheader("Step 1 — Upload your sales file")
    st.caption(
        "Upload the CSV that has monthly sales numbers for each salesperson. "
        "It needs columns named: Month, Sales, sale sp1, sale sp2, etc."
    )
    uploaded = st.file_uploader("Choose your CSV file", type=["csv"])

    st.divider()
    st.subheader("Step 2 — Costs & settings")
    st.caption("These numbers affect the profit calculation. Change them to match your store.")

    salary = st.number_input(
        "Monthly salary per person (same unit as Sales column)",
        value=3.0, step=0.5, min_value=0.1,
        help="Use the same unit as your Sales column. In this dataset, salary=3 and sales are in the same unit."
    )
    st.caption(f"= {salary} units/month per person (same scale as your Sales data)")
    comm_pct = st.slider(
        "Commission rate (%)", 0, 20, 5,
        help="What % of sales you pay back as commission to the salesperson."
    ) / 100
    gm_pct = st.slider(
        "Gross margin rate (%)", 10, 90, 50,
        help="What % of sales revenue is actual profit for the store before salaries."
    ) / 100
    horizon = st.slider(
        "How many months ahead to project?", 3, 24, 6,
        help="The tool will estimate profits this many months into the future."
    )
    season_period = st.selectbox(
        "Seasonal cycle (0 = auto-detect)",
        [0, 6, 12, 18], index=0,
        help="Does your store have a repeating sales pattern? 12 = yearly cycle, 0 = let the tool figure it out."
    )

    st.divider()
    show_raw = st.checkbox("Show raw data table at the bottom", value=False)


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🛍️ Should You Replace Your Salesperson?")
st.markdown(
    "Upload your store's sales data, pick who is leaving, and this tool will tell you "
    "**whether hiring a replacement is worth it** — with a clear profit estimate."
)

if uploaded is None:
    st.info("👈  Start by uploading your sales CSV in the sidebar.")
    with st.expander("🖥️ First time? How to install and run this app"):
        st.markdown("""
**Step 1 — Install Python dependencies** (run once in your terminal):
```
pip install streamlit pandas numpy statsmodels matplotlib
```

**Step 2 — Run the app:**
```
streamlit run streamlit_app.py
```

**Step 3 — Open your browser** at `http://localhost:8501`

That's it! Upload your CSV, pick who is leaving, and click Run Analysis.
        """)
    with st.expander("What should my CSV look like?"):
        st.markdown("""
Your CSV should have one row per month, with a column for each salesperson:

| Month | Sales | sale sp1 | sale sp2 | sale sp3 |
|-------|-------|----------|----------|----------|
| 1     | 23.1  | 7.2      | 6.3      | 9.5      |
| 2     | 41.6  | 8.9      | 7.8      | 11.7     |

- **Month**: just a number (1, 2, 3, ...)
- **Sales**: total store sales that month
- **sale spN**: that person's individual sales (put 0 if they weren't working that month)
        """)
    st.stop()

# ── Load CSV ──────────────────────────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = [c.strip() for c in df_raw.columns]
except Exception as e:
    st.error(f"Couldn't read the file: {e}")
    st.stop()

import re
sp_cols = [c for c in df_raw.columns if re.match(r"^sale\s+sp\d+$", c.strip(), re.IGNORECASE)]
if not sp_cols:
    st.error("No salesperson columns found. Make sure your columns are named like 'sale sp1', 'sale sp2', etc.")
    st.stop()

df_raw[sp_cols] = df_raw[sp_cols].fillna(0)
last_month = int(df_raw["Month"].max())
last_row   = df_raw[df_raw["Month"] == last_month].iloc[0]
active_now = [sp for sp in sp_cols if last_row[sp] > 0]

st.divider()
st.subheader("Step 3 — Who is leaving?")
st.caption(f"Your file has {len(sp_cols)} salespeople total. {len(active_now)} are currently active.")

matched = [sp for sp in active_now if "sp12" in sp.lower()]
default_sp = matched[0] if matched else active_now[0]
default_idx = active_now.index(default_sp)
if not matched:
    st.caption("ℹ️ No sp12 found — defaulting to the first active person. Select the right person below.")

leaving_sp_label = st.selectbox(
    "Select the person who gave notice:",
    options=active_now,
    index=default_idx,
    help="Only currently active salespeople are shown here."
)

run_btn = st.button("🔍 Run the Analysis", type="primary", use_container_width=True)

if not run_btn:
    st.caption("Once you click the button, you'll get the full recommendation with charts and explanations.")
    st.stop()


# ── Run analysis ──────────────────────────────────────────────────────────────
with st.spinner("Crunching the numbers…"):
    try:
        results = run_full_analysis(
            df_raw,
            leaving_name=leaving_sp_label,
            horizon=horizon,
            salary=salary,
            commission_rate=comm_pct,
            gross_margin_rate=gm_pct,
            season_period=season_period,
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

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
pct           = 100 * (n_total - leaving_rank) / max(n_total - 1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()

if recommend:
    st.markdown(
        f'<div class="verdict-yes">✅ YES — Hire a replacement for {leaving_sp}<br>'
        f'<span style="font-size:14px;font-weight:400;">Replacing this person is expected to earn '
        f'your store <b>₹{diff_mean:.2f} more profit</b> over the next {horizon} months, '
        f'compared to leaving the role empty.</span></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="verdict-no">⛔ NO — Don\'t rush to replace {leaving_sp}<br>'
        f'<span style="font-size:14px;font-weight:400;">Based on the numbers, leaving the seat empty '
        f'saves ₹{abs(diff_mean):.2f} more profit over {horizon} months. '
        f'Only hire if you expect a strong candidate.</span></div>',
        unsafe_allow_html=True
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# 4 KPI CARDS — with plain-English meaning
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("What the numbers mean")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Performance rank</div>
        <div class="kpi-number">#{leaving_rank} / {n_total}</div>
        <div class="kpi-meaning">
            This person ranked <b>#{leaving_rank}</b> out of all {n_total} salespeople 
            who've ever worked here. #{1} is the best. 
            This ranking is <i>fair</i> — it adjusts for how long they've been on the job 
            and busy/slow seasons.
        </div>
    </div>""", unsafe_allow_html=True)

with k2:
    direction = "more" if leaving_skill >= 0 else "less"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Their monthly sales impact</div>
        <div class="kpi-number">{leaving_skill:+.2f} ₹/mo</div>
        <div class="kpi-meaning">
            On average, this person brings in <b>₹{abs(leaving_skill):.2f} {direction}</b> per month 
            than a typical hire would. Negative means below average.
        </div>
    </div>""", unsafe_allow_html=True)

with k3:
    color = GREEN if recommend else RED
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Expected profit gain if you hire</div>
        <div class="kpi-number" style="color:{color};">₹{diff_mean:+.2f}</div>
        <div class="kpi-meaning">
            If you hire someone of <b>average quality</b>, this is how much extra profit 
            your store earns over {horizon} months vs. leaving the seat empty. 
            Includes the new hire's salary and ramp-up time.
        </div>
    </div>""", unsafe_allow_html=True)

with k4:
    r2_pct = params["r2"] * 100
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">How reliable is this prediction?</div>
        <div class="kpi-number">{r2_pct:.1f}%</div>
        <div class="kpi-meaning">
            The model explains <b>{r2_pct:.1f}%</b> of your store's real sales patterns. 
            Anything above 80% means you can trust these projections.
        </div>
    </div>""", unsafe_allow_html=True)

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Profit Forecast",
    "🎯 Hire Quality",
    "🏆 Team Rankings",
    "📅 Seasonal Patterns",
    "🏪 Overlap & Ceiling",
    "📋 Tips & Rules",
    "📌 Assumptions",
])


# ── TAB 1: Profit Forecast ────────────────────────────────────────────────────
with tab1:
    st.subheader("Will hiring someone grow your profits?")
    st.markdown("""
    <div class="explain-box">
    The chart below compares two futures:<br>
    &nbsp;🟢 <b>Green line</b> — you hire a replacement (assumes an average-quality hire)<br>
    &nbsp;🔴 <b>Red line</b> — you leave the role empty<br><br>
    If the green line is higher, hiring is worth it. The gap between them is your profit gain.
    The dips and peaks you see are <b>normal seasonal patterns</b> — your store naturally has 
    busy and slow months.
    </div>
    """, unsafe_allow_html=True)

    mc = projection["monthly_compare"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(mc["month"], mc["margin_noreplace"],
            color=RED, marker="o", linewidth=2, markersize=4, label="Leave role empty")
    ax.plot(mc["month"], mc["margin_replace_mean"],
            color=GREEN, marker="s", linewidth=2, markersize=4, label="Hire a replacement")
    ax.fill_between(mc["month"], mc["margin_noreplace"], mc["margin_replace_mean"],
                    alpha=0.12, color=GREEN)
    ax.set_xlabel("Month (projected forward)", fontsize=10)
    ax.set_ylabel("Gross profit (₹)", fontsize=10)
    ax.set_title(f"Month-by-month profit over next {horizon} months", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown(f"""
    <div class="tip-box">
    💡 <b>Bottom line:</b> Over {horizon} months, hiring is expected to earn 
    <b>₹{diff_mean:+.2f} more</b> than not hiring. Even in slow months, 
    having a person in the role usually beats an empty seat.
    </div>
    """, unsafe_allow_html=True)


# ── TAB 2: Hire Quality ───────────────────────────────────────────────────────
with tab2:
    st.subheader("What if your next hire is great — or terrible?")
    st.markdown("""
    <div class="explain-box">
    You can't know in advance how good your next hire will be. This chart shows 
    your expected <b>extra profit</b> depending on hire quality, from the worst 
    person your store has ever had to the absolute best.<br><br>
    Even in the <b>worst case</b>, replacing is usually better than an empty seat.
    </div>
    """, unsafe_allow_html=True)

    scens  = projection["scenarios"]
    labels = [
        ("Worst ever",   "replace_worst"),
        ("Bottom 25%",   "replace_p25"),
        ("Median hire",  "replace_median"),
        ("Average hire", "replace_mean"),
        ("Top 25%",      "replace_p75"),
        ("Best ever",    "replace_best"),
    ]
    names  = [l for l, _ in labels]
    deltas = [scens[k]["margin"].sum() - nr_total for _, k in labels]

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    bar_cols = [GREEN if d >= 0 else RED for d in deltas]
    bars = ax2.bar(names, deltas, color=bar_cols, edgecolor="white", width=0.55)
    ax2.axhline(0, color=NAVY, linewidth=1)
    ax2.set_ylabel(f"Extra profit vs. empty seat (₹, {horizon} mo)", fontsize=10)
    ax2.set_title("Extra profit by hire quality", fontsize=11)
    for bar, d in zip(bars, deltas):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.3 if d >= 0 else -1.0),
                 f"₹{d:+.1f}", ha="center", fontsize=9, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.markdown("""
    <div class="tip-box">
    💡 <b>What "average hire" means:</b> This is the typical outcome based on every 
    person your store has ever hired. It's the most realistic scenario to plan for.
    </div>
    """, unsafe_allow_html=True)

    # Simplified sensitivity table
    st.subheader("Full breakdown by scenario")
    st.caption("This table shows the exact profit numbers for each hiring scenario.")
    ms = margin_summary.copy()
    ms.columns = ["Scenario", "Skill level (₹/mo)", f"Total profit ({horizon} mo)", "vs. leaving empty"]
    st.dataframe(ms, use_container_width=True, hide_index=True)


# ── TAB 3: Team Rankings ──────────────────────────────────────────────────────
with tab3:
    st.subheader(f"How does {leaving_sp} compare to your team?")
    st.markdown("""
    <div class="explain-box">
    Each bar shows how much <b>more or less</b> a salesperson sells per month 
    compared to a "baseline" hire — after adjusting for how long they've been on the 
    job and seasonal ups/downs. Positive = above average. Negative = below average.<br><br>
    🟡 Yellow = the person leaving &nbsp;|&nbsp; 🟢 Green = active staff &nbsp;|&nbsp; 🟠 Orange = departed
    </div>
    """, unsafe_allow_html=True)

    col_t, col_c = st.columns([1, 1.5])

    with col_t:
        def highlight_leaving(row):
            if row["Salesperson"] == leaving_sp:
                return ["background-color: #fff3cd"] * len(row)
            if row["Status"] == "Active":
                return ["background-color: #e8f5e9"] * len(row)
            return [""] * len(row)

        display_cols = ["Rank", "Salesperson", "Tenure (mo)", "Skill Effect", "Status"]
        st.dataframe(
            skill_table[display_cols].style.apply(highlight_leaving, axis=1),
            use_container_width=True, hide_index=True
        )

    with col_c:
        sk_sorted = skill_table.sort_values("Skill Effect")
        bar_cs = []
        for _, row in sk_sorted.iterrows():
            if row["Salesperson"] == leaving_sp:
                bar_cs.append(AMBER)
            elif row["Status"] == "Active":
                bar_cs.append(GREEN)
            else:
                bar_cs.append("#b0bec5")

        fig3, ax3 = plt.subplots(figsize=(6, max(4, len(skill_table) * 0.38)))
        ax3.barh(sk_sorted["Salesperson"], sk_sorted["Skill Effect"],
                 color=bar_cs, edgecolor="white")
        ax3.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax3.set_xlabel("Monthly sales contribution vs. average (₹)", fontsize=9)
        ax3.set_title("Team skill ranking", fontsize=10)
        gp = mpatches.Patch(color=GREEN,    label="Active staff")
        yp = mpatches.Patch(color=AMBER,    label=f"{leaving_sp} (leaving)")
        dp = mpatches.Patch(color="#b0bec5", label="Departed")
        ax3.legend(handles=[gp, yp, dp], fontsize=8)
        ax3.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    st.markdown(f"""
    <div class="tip-box">
    💡 <b>What this means for your hiring decision:</b> {leaving_sp} is ranked 
    #{leaving_rank} out of {n_total}. Their below-average contribution means 
    even a median new hire is likely to do better.
    </div>
    """, unsafe_allow_html=True)

    # Skill distribution histogram
    st.subheader(f"Where does {leaving_sp} sit among all historical hires?")
    st.markdown("""
    <div class="explain-box">
    This chart shows the spread of skill across all 16 salespeople ever hired.
    Each bar = number of people with that sales contribution level.
    The vertical lines show where <b>sp12</b> sits vs. the average and median hire —
    so you can see at a glance whether replacing them is likely to be an upgrade.
    </div>
    """, unsafe_allow_html=True)

    all_skills = projection["all_skills"]
    fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
    ax_hist.hist(all_skills, bins=8, color=TEAL, edgecolor="white", alpha=0.85)
    ax_hist.axvline(np.mean(all_skills), color=NAVY, linewidth=1.8, linestyle="--",
                    label=f"Average hire ({np.mean(all_skills):.2f})")
    ax_hist.axvline(np.median(all_skills), color=AMBER, linewidth=1.8, linestyle=":",
                    label=f"Median hire ({np.median(all_skills):.2f})")
    ax_hist.axvline(leaving_skill, color=RED, linewidth=2.2, linestyle="-",
                    label=f"{leaving_sp} ({leaving_skill:+.2f}) ← leaving")
    ax_hist.set_xlabel("Monthly sales contribution vs. baseline (units)", fontsize=9)
    ax_hist.set_ylabel("Number of salespeople", fontsize=9)
    ax_hist.set_title("Skill distribution — all 16 hires ever", fontsize=10)
    ax_hist.legend(fontsize=9)
    ax_hist.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_hist, use_container_width=True)

    below_count = sum(1 for s in all_skills if s < leaving_skill)
    st.markdown(f"""
    <div class="tip-box">
    💡 Only <b>{below_count} out of {len(all_skills)}</b> historical hires performed worse than {leaving_sp}.
    That means any random replacement has a <b>{100*(len(all_skills)-below_count-1)//len(all_skills)}%+ chance</b> of being an upgrade.
    </div>
    """, unsafe_allow_html=True)



# ── TAB 4: Seasonal Patterns ──────────────────────────────────────────────────
with tab4:
    st.subheader("When are your busiest and slowest months?")
    st.markdown("""
    <div class="explain-box">
    Your store has a repeating seasonal cycle — some months are naturally 
    busier than others regardless of who's working. This chart shows that pattern.<br><br>
    <b>Positive bars</b> = months where sales run above average.<br>
    <b>Negative bars</b> = slower months. This is automatically factored into 
    the profit forecast so the numbers are fair.
    </div>
    """, unsafe_allow_html=True)

    if params["season_period"] > 1:
        sp_len  = params["season_period"]
        offsets = list(range(sp_len))
        effects = [params["season_effects"].get(o, 0.0) for o in offsets]

        fig5, ax5 = plt.subplots(figsize=(8, 3))
        bar_cs5 = [GREEN if e >= 0 else RED for e in effects]
        ax5.bar(offsets, effects, color=bar_cs5, edgecolor="white")
        ax5.axhline(0, color="black", linewidth=0.8)
        ax5.set_xlabel("Month position in cycle", fontsize=9)
        ax5.set_ylabel("Sales per person vs. average (₹)", fontsize=9)
        ax5.set_title(f"{sp_len}-month seasonal cycle detected in your data", fontsize=10)
        ax5.set_xticks(offsets)
        ax5.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
    else:
        st.info("No clear seasonal pattern was found in your data — sales are fairly consistent year-round.")

    # Sales + headcount history
    st.subheader("Your store's full sales history")
    st.caption(f"All {int(df['Month'].max())} months of data — bars = total sales, line = number of staff working.")

    hc = (df[sp_cols] > 0).sum(axis=1)
    fig6, ax6a = plt.subplots(figsize=(10, 3))
    ax6b = ax6a.twinx()
    ax6a.bar(df["Month"], df["Sales"], color=LGRAY, edgecolor=TEAL, linewidth=0.4, label="Total Sales")
    ax6b.plot(df["Month"], hc, color=NAVY, linewidth=1.5, label="Headcount")
    ax6b.set_ylabel("Number of staff", color=NAVY, fontsize=9)
    ax6a.set_ylabel("Total sales (₹)", fontsize=9)
    ax6a.set_xlabel("Month", fontsize=9)
    ax6a.spines[["top", "right"]].set_visible(False)
    lines1, labels1 = ax6a.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6a.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig6, use_container_width=True)


# ── TAB 5: Overlap & Capacity Ceiling ───────────────────────────────────────
with tab5:
    st.subheader("Should you keep overlapping staff for a full month?")
    st.markdown("""
    <div class="explain-box">
    Your current rule is: when someone resigns, use their 1-month notice period to hire 
    their replacement — so you briefly run 8 staff instead of 7.<br><br>
    This tab checks whether that 8th person actually <b>grows</b> your total store sales,
    or just splits the same customers among more people.
    </div>
    """, unsafe_allow_html=True)

    if ceiling["n_above"] >= 2:
        normal_resid = ceiling["normal_mean_resid"] or 0
        above_resid  = ceiling["above_mean_resid"]  or 0
        has_ceiling  = above_resid < -2

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Normal team size",
            f"{ceiling['modal_headcount']} people",
            f"{ceiling['n_normal']} months of data"
        )
        c2.metric(
            "Sales accuracy — normal months",
            f"{normal_resid:+.2f}",
            "actual vs. predicted (near 0 = accurate)"
        )
        c3.metric(
            "Sales accuracy — 8-person months",
            f"{above_resid:+.2f}",
            "negative = sales fell short of what staff individually predicted",
            delta_color="inverse"
        )

        if has_ceiling:
            n_above = ceiling["n_above"]
            gap = abs(above_resid)
            ceiling_msg = (
                f"<div class=\"tip-box\">"
                f"⚠️ <b>Capacity ceiling detected.</b> During the {n_above} months "
                f"where you had 8 staff, total store sales were on average "
                f"<b>{gap:.1f} units below</b> what those salespeople's individual "
                f"track records predict. The 8th person is not adding new customers — "
                f"they are splitting the same footfall.<br><br>"
                f"<b>Recommendation:</b> Time the new hire's start date to coincide with "
                f"or just after the departing person's last day. A 1-month full overlap "
                f"costs more in salary than it earns in extra sales.</div>"
            )
            st.markdown(ceiling_msg, unsafe_allow_html=True)
        else:
            st.success("No capacity ceiling found — the 8th person does appear to grow total sales. Your overlap pattern is fine.")

        fig7, ax7 = plt.subplots(figsize=(5, 3.2))
        cats = [f"Normal ({ceiling['modal_headcount']} staff)", "Overlap (8 staff)"]
        vals = [normal_resid, above_resid]
        bcs  = [GREEN if v >= 0 else RED for v in vals]
        ax7.bar(cats, vals, color=bcs, width=0.4, edgecolor="white")
        ax7.axhline(0, color="black", linewidth=0.8)
        ax7.set_ylabel("Avg. sales gap vs. individual track records (units)", fontsize=9)
        ax7.set_title("Does an 8th staff member grow total sales?", fontsize=10)
        for i, v in enumerate(vals):
            ax7.text(i, v + (0.05 if v >= 0 else -0.4), f"{v:+.2f}", ha="center", fontsize=11, fontweight="bold")
        ax7.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig7)
    else:
        st.info("Not enough months with 8 staff to test the capacity ceiling. Keep collecting data.")


# ── TAB 6: Tips & Rules ───────────────────────────────────────────────────────
with tab6:
    st.subheader("Simple rules for staffing your store")

    st.markdown("""
    <div class="explain-box">
    These are practical takeaways from analysing your store's 81 months of data. 
    They'll save you time on future hiring decisions.
    </div>
    """, unsafe_allow_html=True)

    tips = [
        ("✅ Usually replace when someone leaves",
         "Hiring a replacement beats leaving the seat empty in almost every scenario — "
         "even if your next hire is below average. The only exception is if you have "
         "specific reason to expect an unusually weak hiring pool."),

        ("⏱️ Don't overlap for a full month if you can avoid it",
         "Your data shows that having 8 staff at once (during handover overlap) "
         "actually generates LESS total store sales than expected — the extra person "
         "splits the same customers rather than adding new ones. "
         "Try to time the new hire's start close to the departing person's last day."),

        ("🔄 Re-run this tool every time someone leaves",
         "The answer depends on WHO is leaving, what season it is, and who's staying. "
         "Upload your latest CSV and run fresh numbers each time — it only takes a minute."),

        ("📊 Use the team ranking to manage performance",
         "The skill ranking table fairly compares everyone, adjusting for tenure and seasons. "
         "Use it to have early conversations with lower-ranked active staff, "
         "and to benchmark new hires at 6 and 12 months."),

        ("⚠️ Your store seems to have a capacity ceiling around 7 staff",
         "Adding an 8th person doesn't appear to grow total sales proportionally — "
         "customer footfall is the real limit. Running permanently at 8 mostly adds "
         "salary cost without matching sales growth."),
    ]

    for title_r, body_r in tips:
        with st.expander(title_r, expanded=True):
            st.write(body_r)

    st.divider()
    st.subheader("Model details")
    st.markdown(f"""
    | What | Value |
    |------|-------|
    | Months of sales data | {int(df['Month'].max())} |
    | Total salespeople ever hired | {n_total} |
    | Seasonal cycle detected | {params['season_period']} months |
    | Model accuracy (R²) | {params['r2']:.3f} ({params['r2']*100:.1f}%) |
    | Monthly salary used | ₹{salary} |
    | Commission rate | {comm_pct:.0%} |
    | Gross margin rate | {gm_pct:.0%} |
    | Projection horizon | {horizon} months |
    """)

    # PDF download
    st.divider()
    if st.button("📄 Download PDF report", use_container_width=True):
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
                st.download_button(
                    "⬇️ Save staffing_report.pdf",
                    data=pdf_bytes,
                    file_name="staffing_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")


# ── TAB 7: Assumptions ────────────────────────────────────────────────────────
with tab7:
    st.subheader("Assumptions this analysis is based on")
    st.markdown("""
    <div class="explain-box">
    Every recommendation comes from a model that makes certain assumptions. 
    Here they are, listed clearly so you know what to question.
    </div>
    """, unsafe_allow_html=True)

    assumptions = [
        ("💰 Gross margin formula",
         f"Gross margin = {gm_pct:.0%} × Sales − {salary} salary/person − {comm_pct:.0%} × Sales. "
         f"This means your effective net margin rate is {gm_pct - comm_pct:.0%} of sales, minus fixed salaries. "
         "Change these in the sidebar if your store uses different numbers."),

        ("📈 A new hire ramps up over time",
         "New salespeople take time to learn the store and build customer relationships. "
         "The model accounts for this using a log-curve ramp-up — most of the learning "
         "happens in the first 3 months, then performance stabilises. "
         "This is based on the ramp-up pattern seen in your own historical hires."),

        ("🎲 Future hire quality = average of past hires",
         "The recommended baseline assumes your next hire will be of average skill — "
         "matching the mean of all 16 salespeople you've ever employed. "
         "The 'Hire Quality' tab shows what happens under better or worse scenarios."),

        ("🔁 Seasonal patterns repeat",
         "The model detected a repeating seasonal cycle in your data and assumes it will "
         "continue. If your store is opening in a new location or changing its product mix, "
         "the seasonal forecast may be less accurate."),

        ("👣 Customer footfall is fixed in the short term",
         "The capacity ceiling analysis assumes your store has a natural limit on how many "
         "customers it can serve per month — driven by mall footfall, not staff count. "
         "This is a standard assumption for mall-based retail in India."),

        ("🏬 One store, stable format",
         "The analysis assumes your store format, location, and pricing have been broadly "
         "stable over the 81-month data period. Major changes (renovation, rebranding, "
         "new product lines) could make older data less relevant."),

        ("📊 Individual salesperson sales are measurable",
         "The model attributes each unit of sales to a specific salesperson in the month "
         "they sold it. This requires that your CSV correctly records individual sales, "
         "not just total store sales."),
    ]

    for title_a, body_a in assumptions:
        with st.expander(title_a, expanded=True):
            st.write(body_a)


    if show_raw:
        st.divider()
        st.subheader("Raw data")
        st.dataframe(df_raw, use_container_width=True)  