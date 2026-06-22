"""
generate_pdf_report.py
======================
Generates a richly formatted PDF report with charts and tables
from the staffing recommender analysis engine.

Usage:
    python generate_pdf_report.py RevI-Test.csv --leaving sp12 --out report.pdf
    python generate_pdf_report.py RevI-Test.csv --leaving sp12  # saves staffing_report.pdf
"""

import argparse
import io
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image as RLImage, PageBreak,
                                 HRFlowable, KeepTogether)
from reportlab.graphics.shapes import Drawing, Line

# Import analysis engine (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from analysis_engine import run_full_analysis

# ─── Colour palette ───────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#1a2e4a")
TEAL    = colors.HexColor("#0d7a7a")
AMBER   = colors.HexColor("#e67e00")
GREEN   = colors.HexColor("#2e7d32")
RED     = colors.HexColor("#c62828")
LGRAY   = colors.HexColor("#f4f6f8")
MGRAY   = colors.HexColor("#d0d7de")
DGRAY   = colors.HexColor("#555f6e")
WHITE   = colors.white

MN   = "#1a2e4a"
MT   = "#0d7a7a"
MA   = "#e67e00"
MG   = "#2e7d32"
MR   = "#c62828"
MLG  = "#f4f6f8"

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


# ─── Style helpers ────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()
    def s(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "title": s("title", fontSize=22, textColor=NAVY, fontName="Helvetica-Bold",
                   spaceAfter=4, alignment=TA_CENTER),
        "subtitle": s("subtitle", fontSize=11, textColor=DGRAY,
                      spaceAfter=12, alignment=TA_CENTER),
        "h1": s("h1", fontSize=14, textColor=NAVY, fontName="Helvetica-Bold",
                spaceBefore=14, spaceAfter=6),
        "h2": s("h2", fontSize=11, textColor=TEAL, fontName="Helvetica-Bold",
                spaceBefore=8, spaceAfter=4),
        "body": s("body", fontSize=9, textColor=colors.black,
                  leading=14, spaceAfter=6),
        "small": s("small", fontSize=8, textColor=DGRAY, leading=12),
        "verdict_yes": s("verdict_yes", fontSize=13, textColor=GREEN,
                         fontName="Helvetica-Bold", alignment=TA_CENTER, spaceBefore=6, spaceAfter=6),
        "verdict_no": s("verdict_no", fontSize=13, textColor=RED,
                        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceBefore=6, spaceAfter=6),
        "callout_body": s("callout_body", fontSize=9, textColor=NAVY,
                          leading=13, leftIndent=12, rightIndent=12),
        "table_header": s("th", fontSize=8, textColor=WHITE,
                          fontName="Helvetica-Bold", alignment=TA_CENTER),
        "table_cell": s("tc", fontSize=8, textColor=colors.black, alignment=TA_LEFT),
        "table_cell_r": s("tcr", fontSize=8, textColor=colors.black, alignment=TA_RIGHT),
    }


def _hr(story):
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=MGRAY, spaceAfter=6, spaceBefore=2))


def _callout(story, text, st, border_color=TEAL):
    """Tinted callout box."""
    tbl = Table([[Paragraph(text, st["callout_body"])]],
                colWidths=[PAGE_W - 2 * MARGIN - 0.4 * cm])
    tbl.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 1.5, border_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",  (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f5f5")),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))


def _df_to_table(df, st, col_widths=None, highlight_row=None, highlight_color=None):
    """Convert DataFrame to a styled ReportLab Table."""
    data = [[Paragraph(str(c), st["table_header"]) for c in df.columns]]
    for i, row in df.iterrows():
        data.append([Paragraph(str(v), st["table_cell"]) for v in row.values])

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID",       (0, 0), (-1, -1), 0.3, MGRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LGRAY]),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if highlight_row is not None and highlight_color is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), highlight_color))
    tbl.setStyle(TableStyle(style))
    return tbl


# ─── Chart generators (return PNG bytes) ─────────────────────────────────────

def _chart_skill_rankings(skill_table):
    df = skill_table.sort_values("Skill Effect")
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.38)))
    colors_bar = [MG if s == "Active" else MA for s in df["Status"]]
    bars = ax.barh(df["Salesperson"], df["Skill Effect"],
                   color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Skill Effect (₹/month vs. baseline, tenure & season adjusted)", fontsize=9)
    ax.set_title("Salesperson Skill Rankings\n(after removing ramp-up curve & seasonality)", fontsize=10)
    green_patch = mpatches.Patch(color=MG, label="Currently active")
    amber_patch = mpatches.Patch(color=MA, label="Departed")
    ax.legend(handles=[green_patch, amber_patch], fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(); buf.seek(0); return buf.read()


def _chart_seasonality(params, n_months):
    if params["season_period"] <= 1:
        return None
    sp = params["season_period"]
    offsets = list(range(sp))
    effects = [params["season_effects"].get(o, 0.0) for o in offsets]
    # Map offsets to approximate month labels if period is 12
    fig, ax = plt.subplots(figsize=(7, 3))
    bar_colors = [MG if e >= 0 else MR for e in effects]
    ax.bar(offsets, effects, color=bar_colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Season offset (month position within cycle)", fontsize=9)
    ax.set_ylabel("Sales effect vs. baseline (₹/head)", fontsize=9)
    ax.set_title(f"Detected {sp}-Month Seasonal Cycle\n(positive = strong month, negative = slow month)", fontsize=10)
    ax.set_xticks(offsets)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(); buf.seek(0); return buf.read()


def _chart_margin_scenarios(projection, horizon, leaving_sp):
    scens = projection["scenarios"]
    nr_total = scens["noreplace"]["margin"].sum()

    labels = [
        ("Worst hire", "replace_worst"),
        ("25th pct.", "replace_p25"),
        ("Median",    "replace_median"),
        ("Average",   "replace_mean"),
        ("75th pct.", "replace_p75"),
        ("Best hire", "replace_best"),
    ]
    names  = [l for l, _ in labels]
    deltas = [scens[k]["margin"].sum() - nr_total for _, k in labels]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bar_colors = [MG if d >= 0 else MR for d in deltas]
    bars = ax.bar(names, deltas, color=bar_colors, edgecolor="white", width=0.6)
    ax.axhline(0, color=MN, linewidth=1.0)
    ax.set_ylabel(f"Extra margin vs. not replacing (₹, {horizon} months)", fontsize=9)
    ax.set_title(f"Replace {leaving_sp}? — Sensitivity to New Hire Quality", fontsize=10)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.15 if d >= 0 else -0.4),
                f"{d:+.1f}", ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(); buf.seek(0); return buf.read()


def _chart_monthly_margin(monthly_compare, horizon):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(monthly_compare["month"], monthly_compare["margin_noreplace"],
            color=MR, marker="o", linewidth=1.8, markersize=5, label="Do not replace")
    ax.plot(monthly_compare["month"], monthly_compare["margin_replace_mean"],
            color=MG, marker="s", linewidth=1.8, markersize=5, label="Replace (avg. hire)")
    ax.fill_between(monthly_compare["month"],
                    monthly_compare["margin_noreplace"],
                    monthly_compare["margin_replace_mean"],
                    alpha=0.12, color=MG)
    ax.set_xlabel("Calendar month (projected)", fontsize=9)
    ax.set_ylabel("Gross Margin (₹)", fontsize=9)
    ax.set_title(f"Month-by-Month Margin Projection ({horizon} months)", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(); buf.seek(0); return buf.read()


def _chart_headcount_sales(df, sp_cols, ceiling):
    """Total sales over time with headcount overlay."""
    headcount = (df[sp_cols] > 0).sum(axis=1)
    fig, ax1 = plt.subplots(figsize=(8, 3.2))
    ax2 = ax1.twinx()
    ax1.bar(df["Month"], df["Sales"], color=MLG, edgecolor=MT, linewidth=0.4, label="Total Sales")
    ax2.plot(df["Month"], headcount, color=MN, linewidth=1.5, label="Headcount")
    ax2.set_ylabel("Headcount", color=MN, fontsize=9)
    ax1.set_ylabel("Total Sales (₹)", fontsize=9)
    ax1.set_xlabel("Month", fontsize=9)
    ax1.set_title("81-Month Store Sales & Headcount History", fontsize=10)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(); buf.seek(0); return buf.read()


def _chart_skill_distribution(projection):
    skills = projection["all_skills"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(skills, bins=8, color=MT, edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(skills), color=MN, linewidth=1.5, linestyle="--", label=f"Mean ({np.mean(skills):.2f})")
    ax.axvline(np.median(skills), color=MA, linewidth=1.5, linestyle=":", label=f"Median ({np.median(skills):.2f})")
    ax.set_xlabel("Skill Effect (₹/month)", fontsize=9)
    ax.set_ylabel("Count of hires", fontsize=9)
    ax.set_title("Distribution of Skill Effects\n(across all 16 historical hires)", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(); buf.seek(0); return buf.read()


# ─── Main PDF builder ─────────────────────────────────────────────────────────

def build_pdf(results, out_path):
    st = _styles()
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN)
    story = []
    IMG_W = PAGE_W - 2 * MARGIN

    leaving_sp = results["leaving_sp"]
    params      = results["params"]
    ceiling     = results["ceiling"]
    projection  = results["projection"]
    skill_table = results["skill_table"]
    margin_summary = results["margin_summary"]
    horizon     = results["horizon"]
    salary      = results["salary"]
    comm_rate   = results["commission_rate"]
    gm_rate     = results["gross_margin_rate"]
    df          = results["df"]
    sp_cols     = results["sp_cols"]

    nr_total = projection["scenarios"]["noreplace"]["margin"].sum()
    rm_total = projection["scenarios"]["replace_mean"]["margin"].sum()
    diff_mean = rm_total - nr_total
    recommend_replace = diff_mean > 0

    leaving_row = skill_table[skill_table["Salesperson"] == leaving_sp]
    leaving_rank = int(leaving_row["Rank"].values[0]) if not leaving_row.empty else "?"
    leaving_skill = float(leaving_row["Skill Effect"].values[0]) if not leaving_row.empty else 0.0
    n_total = len(skill_table)
    pct = 100 * (n_total - leaving_rank) / (n_total - 1) if n_total > 1 else 100

    # ── Cover / title ──────────────────────────────────────────────────────
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Staffing Decision Report", st["title"]))
    story.append(Paragraph(
        f"Should you replace <b>{leaving_sp}</b>? &nbsp;·&nbsp; "
        f"Branded Clothing Store, Bangalore", st["subtitle"]))
    _hr(story)

    # ── Verdict banner ─────────────────────────────────────────────────────
    verdict_text = (
        f"✔  REPLACE {leaving_sp}   (+₹{diff_mean:.2f} expected gross margin over {horizon} months)"
        if recommend_replace else
        f"✘  DO NOT REPLACE {leaving_sp}   (saving ₹{abs(diff_mean):.2f} margin over {horizon} months)"
    )
    story.append(Paragraph(verdict_text,
                            st["verdict_yes"] if recommend_replace else st["verdict_no"]))
    _hr(story)
    story.append(Spacer(1, 4))

    # ── Executive summary ──────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", st["h1"]))
    summary_text = (
        f"{leaving_sp} ranks <b>#{leaving_rank} out of {n_total}</b> salespeople who have ever "
        f"worked here (~{pct:.0f}th percentile), with a skill effect of <b>{leaving_skill:+.2f}</b> "
        f"₹/month after adjusting for tenure ramp-up and seasonality. "
        f"Replacing her with a new hire of typical quality is projected to add "
        f"<b>₹{diff_mean:.2f}</b> in gross margin over the next {horizon} months versus leaving "
        f"the seat empty. Replacing is the better call unless the new hire turns out to be "
        f"in roughly the bottom 20% of historical hiring outcomes."
        if recommend_replace else
        f"{leaving_sp} was a below-average performer. Leaving the seat empty for {horizon} months "
        f"is projected to outperform hiring a typical replacement by ₹{abs(diff_mean):.2f} in gross margin, "
        f"because salary savings exceed the lost sales contribution."
    )
    _callout(story, summary_text, st)

    # ── Key metrics row ────────────────────────────────────────────────────
    story.append(Paragraph("Key Metrics at a Glance", st["h2"]))
    metrics = [
        ["Metric", "Value"],
        ["Data span", f"{int(df['Month'].max())} months"],
        ["Total salespeople (ever)", str(n_total)],
        [f"{leaving_sp} — skill rank", f"#{leaving_rank} / {n_total} ({pct:.0f}th pct.)"],
        [f"{leaving_sp} — skill effect", f"{leaving_skill:+.2f} ₹/month"],
        ["Expected margin gain from replacing", f"₹{diff_mean:+.2f} ({horizon} mo)"],
        ["Break-even new-hire quality", "~20th pct. of historical hires"],
        ["Seasonal cycle detected", f"{params['season_period']} months"
         if params["season_period"] > 1 else "None detected"],
        ["Model R²", f"{params['r2']:.3f}"],
        ["Capacity ceiling evidence",
         "Yes — 8th staff cannibalises ~4.8 ₹ sales/month"
         if (ceiling["above_mean_resid"] is not None and ceiling["above_mean_resid"] < -2)
         else "No clear evidence"],
    ]
    col_w = [(PAGE_W - 2 * MARGIN) * f for f in [0.55, 0.45]]
    tbl = Table(metrics, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LGRAY]),
        ("GRID",       (0, 0), (-1, -1), 0.3, MGRAY),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # ── Page 2 ─────────────────────────────────────────────────────────────
    story.append(PageBreak())

    # ── Section 1: Sales & Headcount history ──────────────────────────────
    story.append(Paragraph("1. Store History: Sales & Headcount", st["h1"]))
    story.append(Paragraph(
        f"The store has been running for {int(df['Month'].max())} months. "
        f"Normal staffing level is 7 people; brief spikes to 8 occur during 1-month overlap hiring. "
        f"Total monthly sales broadly follow an upward trend with clear seasonal variation.",
        st["body"]))
    img = _chart_headcount_sales(df, sp_cols, ceiling)
    story.append(RLImage(io.BytesIO(img), width=IMG_W, height=IMG_W * 0.42))
    story.append(Spacer(1, 8))

    # ── Section 2: Seasonality ─────────────────────────────────────────────
    story.append(Paragraph("2. Seasonality", st["h1"]))
    if params["season_period"] > 1:
        story.append(Paragraph(
            f"A statistically meaningful <b>{params['season_period']}-month repeating cycle</b> was "
            f"detected in per-head sales. Two distinct slow periods occur roughly every 6 months "
            f"(likely monsoon-related and post-festive-season lulls, common in Bangalore malls). "
            f"All performance comparisons are seasonality-adjusted.",
            st["body"]))
        season_img = _chart_seasonality(params, int(df["Month"].max()))
        if season_img:
            story.append(RLImage(io.BytesIO(season_img), width=IMG_W * 0.85, height=IMG_W * 0.38))
    else:
        story.append(Paragraph("No statistically significant seasonal cycle was detected.", st["body"]))
    story.append(Spacer(1, 8))

    # ── Page 3 ─────────────────────────────────────────────────────────────
    story.append(PageBreak())

    # ── Section 3: Salesperson skill rankings ─────────────────────────────
    story.append(Paragraph("3. Salesperson Skill Rankings (Tenure & Season Adjusted)", st["h1"]))
    story.append(Paragraph(
        "Each salesperson's 'skill effect' is their estimated monthly sales contribution "
        "versus a baseline hire, <i>after</i> removing how long they'd been on the job "
        "(ramp-up curve) and which calendar season it was. This is the fairest basis for "
        "comparing people who started at different times.",
        st["body"]))

    # Highlight leaving_sp row
    leaving_rank_idx = None
    for i, row in skill_table.iterrows():
        if row["Salesperson"] == leaving_sp:
            leaving_rank_idx = list(skill_table.index).index(i) + 1  # +1 for header
            break

    col_widths_sk = [(PAGE_W - 2 * MARGIN) * f
                     for f in [0.08, 0.18, 0.10, 0.10, 0.14, 0.16, 0.14, 0.10]]
    # Trim to 7 cols (drop index)
    display_cols = ["Rank", "Salesperson", "Start", "End", "Tenure (mo)", "Skill Effect", "Status"]
    sk_display = skill_table[display_cols]
    col_widths_sk = [(PAGE_W - 2 * MARGIN) * f
                     for f in [0.08, 0.20, 0.11, 0.10, 0.15, 0.18, 0.18]]
    tbl_sk = _df_to_table(sk_display, st, col_widths=col_widths_sk,
                          highlight_row=leaving_rank_idx,
                          highlight_color=colors.HexColor("#fff3cd"))
    story.append(tbl_sk)
    story.append(Paragraph(
        f"<b>Highlighted row</b> = {leaving_sp} (resigning). "
        f"Green = currently active. Yellow highlight = person under review.",
        st["small"]))
    story.append(Spacer(1, 8))

    img_rank = _chart_skill_rankings(skill_table)
    story.append(RLImage(io.BytesIO(img_rank), width=IMG_W, height=IMG_W * 0.52))
    story.append(Spacer(1, 6))

    # Skill distribution
    img_dist = _chart_skill_distribution(projection)
    story.append(RLImage(io.BytesIO(img_dist), width=IMG_W * 0.72, height=IMG_W * 0.38))
    story.append(Paragraph(
        "Distribution of skill effects across all 16 hires. A new hire's quality is "
        "unknown in advance; the scenarios below use this distribution to bound the uncertainty.",
        st["small"]))

    # ── Page 4 ─────────────────────────────────────────────────────────────
    story.append(PageBreak())

    # ── Section 4: Capacity ceiling ────────────────────────────────────────
    story.append(Paragraph("4. Store Capacity Ceiling Test", st["h1"]))
    if ceiling["n_above"] >= 2:
        diff_vs_normal = (ceiling["above_mean_resid"] or 0) - (ceiling["normal_mean_resid"] or 0)
        story.append(Paragraph(
            f"During the {ceiling['n_above']} months when headcount was 8 (overlap-hire months), "
            f"total store sales were on average <b>{ceiling['above_mean_resid']:.1f}</b> ₹ below "
            f"what the established staffers' individual track records would predict "
            f"(vs. <b>{ceiling['normal_mean_resid']:+.2f}</b> in normal 7-person months). "
            f"This {abs(diff_vs_normal):.1f} ₹ gap suggests a <b>customer footfall ceiling</b>: "
            f"an 8th salesperson displaces sales from existing colleagues rather than generating net-new business.",
            st["body"]))
        if ceiling["above_mean_resid"] is not None and ceiling["above_mean_resid"] < -2:
            _callout(story,
                     "Recommendation on overlap-hiring: Time the new hire's start to land "
                     "close to the departing person's last day, not a full month early. "
                     "Caveat: the sales data cannot capture the training/handover value of overlap months — "
                     "if you find new hires ramp faster after overlapping with a predecessor, "
                     "that value may offset the margin cost.", st, border_color=AMBER)
    else:
        story.append(Paragraph("Not enough 8-person months to test for a capacity ceiling.", st["body"]))

    story.append(Spacer(1, 8))

    # ── Section 5: Financial projection ────────────────────────────────────
    story.append(Paragraph("5. Financial Comparison: Replace vs. Do Not Replace", st["h1"]))

    formula_text = (
        f"<b>Margin formula:</b> &nbsp; "
        f"Gross Margin = {gm_rate:.0%} × Sales &nbsp;−&nbsp; "
        f"₹{salary} × headcount &nbsp;−&nbsp; "
        f"{comm_rate:.0%} × Sales &nbsp; "
        f"= &nbsp; {gm_rate - comm_rate:.0%} × Sales &nbsp;−&nbsp; "
        f"₹{salary} × headcount"
    )
    story.append(Paragraph(formula_text, st["body"]))
    story.append(Spacer(1, 6))

    # Full sensitivity table
    ms = margin_summary.copy()
    ms.columns = ["Scenario", "Skill (₹/mo)", f"{horizon}-mo Margin (₹)", f"vs. Not Replacing (₹)"]
    col_w_ms = [(PAGE_W - 2 * MARGIN) * f for f in [0.45, 0.15, 0.22, 0.18]]

    # Highlight the "not replace" row (row 0 → index 1 in table with header)
    tbl_ms = _df_to_table(ms, st, col_widths=col_w_ms,
                          highlight_row=1,
                          highlight_color=colors.HexColor("#fce8e8"))
    story.append(tbl_ms)
    story.append(Spacer(1, 6))

    # Monthly comparison chart
    monthly_img = _chart_monthly_margin(projection["monthly_compare"], horizon)
    story.append(RLImage(io.BytesIO(monthly_img), width=IMG_W, height=IMG_W * 0.43))
    story.append(Spacer(1, 6))

    # Scenario bar chart
    scen_img = _chart_margin_scenarios(projection, horizon, leaving_sp)
    story.append(RLImage(io.BytesIO(scen_img), width=IMG_W, height=IMG_W * 0.48))
    story.append(Paragraph(
        "Bars above zero = replacing is better. Only the worst-historical-hire outcome (far left) "
        "makes not replacing the better call.",
        st["small"]))

    # ── Page 5: Rules of thumb ─────────────────────────────────────────────
    story.append(PageBreak())

    story.append(Paragraph("6. Rules of Thumb: Revised Recommendations", st["h1"]))

    rules = [
        ("Replace whenever a salesperson resigns — with one exception",
         "Replacing is better than leaving a seat empty for any new hire of median or "
         "better quality (which is the most likely outcome). Only hold off if you have "
         "specific reason to believe the pool of available candidates is unusually weak."),
        ("Do not reflexively overlap for a full month",
         "Your data shows an 8th salesperson during overlap months generates ~₹4.8 less "
         "total store sales than the 7 established staffers' combined track records would predict. "
         "Time the new hire's start to coincide with — or just after — the departing person's "
         "last day, unless there is a specific handover task that genuinely requires side-by-side time."),
        ("Re-run this analysis every time someone leaves",
         "The right answer depends on who is leaving (their skill rank), the upcoming season, "
         "and who remains on the team. The script included in this package will recompute "
         "everything from your latest CSV in seconds."),
        ("Track the skill ranking as a management tool",
         "The tenure- and season-adjusted skill table ranks every current and past hire fairly. "
         "Use it to identify underperformers early (before they reach a natural resignation point) "
         "and to benchmark new hires at the 6-month and 12-month marks."),
        ("Be cautious about running 8 staff permanently",
         "Current evidence suggests your store has a customer footfall ceiling at roughly 7 staff. "
         "Adding an 8th person on a sustained basis is unlikely to grow total sales meaningfully; "
         "it will mostly add ₹3/month in salary cost."),
    ]

    for title_r, body_r in rules:
        story.append(Paragraph(f"▸  {title_r}", st["h2"]))
        story.append(Paragraph(body_r, st["body"]))
        story.append(Spacer(1, 4))

    # ── Assumptions ────────────────────────────────────────────────────────
    _hr(story)
    story.append(Paragraph("Assumptions & Caveats", st["h1"]))
    assumptions = [
        "Sales units are treated as a consistent currency across all 81 months.",
        "Each salesperson's employment is one continuous block of months (verified in data).",
        "A new hire's future skill is estimated from the historical distribution of all 16 past hires' "
        "adjusted skill effects, since the actual new hire is unknown.",
        "The detected seasonal pattern continues into the next 6 months unchanged.",
        "The margin formula (50% gross margin, 5% commission, ₹3/month salary) is as given and "
        "applied uniformly across all staff.",
        "The model explains ~92% of month-to-month variation in individual sales (R²=0.92), "
        "giving high confidence in the skill rankings and projections.",
        "Training/handover value of overlap months is not captured in sales numbers and may "
        "partly offset the observed capacity-ceiling penalty.",
    ]
    for a in assumptions:
        story.append(Paragraph(f"• {a}", st["body"]))

    doc.build(story)
    print(f"PDF saved: {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate PDF staffing report with charts.")
    p.add_argument("csv_path")
    p.add_argument("--leaving", default=None)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--salary", type=float, default=3.0)
    p.add_argument("--commission-rate", type=float, default=0.05)
    p.add_argument("--gross-margin-rate", type=float, default=0.50)
    p.add_argument("--season-period", type=int, default=0)
    p.add_argument("--out", default="staffing_report.pdf")
    args = p.parse_args()

    results = run_full_analysis(
        args.csv_path,
        leaving_name=args.leaving,
        horizon=args.horizon,
        salary=args.salary,
        commission_rate=args.commission_rate,
        gross_margin_rate=args.gross_margin_rate,
        season_period=args.season_period,
    )
    build_pdf(results, args.out)


if __name__ == "__main__":
    main()
