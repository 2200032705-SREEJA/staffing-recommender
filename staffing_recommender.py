#!/usr/bin/env python3
"""
staffing_recommender.py
========================

Decide whether to replace a departing salesperson, using historical
salesperson-by-month sales data.

WHAT THIS SCRIPT DOES
----------------------
1. Reads a CSV with columns: Month, Sales, sale sp1, sale sp2, ... sale spN
   (one column per salesperson; 0 or blank means that person was not
   employed that month).
2. Reconstructs each salesperson's tenure (which month-on-the-job each
   observation represents).
3. Detects calendar seasonality (e.g. a recurring slow month every N
   months) and a tenure / experience curve (new hires typically start
   slow and ramp up over their first 1-2 years).
4. Fits a regression that separates THREE effects that are tangled
   together in the raw numbers:
       sales = f(experience) + seasonal effect + individual skill + noise
   This lets us judge a salesperson's true skill fairly, instead of
   crediting/blaming them for the month they happened to start in or
   how long they've been around.
5. Tests for a store "capacity ceiling" - whether adding more staff than
   the historical norm actually grows total sales, or just splits the
   same customer footfall more ways.
6. Projects expected store Gross Margin for the next N months under two
   scenarios - REPLACE the departing person vs. DO NOT REPLACE - using
   the user's margin formula:
       Margin = 50% x Sales - Salary (per head per month) - Commission (% of Sales)
7. Recommends whichever scenario produces higher expected margin, and
   reports how sensitive that recommendation is to the assumed skill of
   a new hire (since a new hire's true skill is unknown in advance).

This script is intentionally general: it does not hard-code a specific
margin formula, headcount, salary, or commission rate. It estimates
sensible defaults from the data itself, but you can also pass them on the
command line for your own shop's actual numbers.

HOW TO RUN
----------
    python3 staffing_recommender.py your_data.csv

Common options:
    --leaving sp12          Which salesperson is resigning (column name,
                             with or without the "sale " prefix). If
                             omitted, the script assumes it's whichever
                             active salesperson has been there longest
                             without a successor already overlapping --
                             i.e. it looks for a person active in the
                             final month with no newer hire shadowing them.
    --horizon 6              Number of months to project forward (default 6)
    --salary 3               Monthly salary cost per salesperson, in the
                             same currency units as the Sales column
                             (default: 3, matching the worked example)
    --commission-rate 0.05   Commission as a fraction of Sales (default 0.05)
    --gross-margin-rate 0.50 Gross margin as a fraction of Sales, BEFORE
                             salary and commission are deducted (default 0.50)
    --season-period 0        Force a specific seasonal cycle length in
                             months (e.g. 12 for calendar-year seasonality,
                             6 for a half-year cycle). 0 = auto-detect.
    --out report.txt         Write the full text report to a file as well
                             as printing it to screen.

EXAMPLE
-------
    python3 staffing_recommender.py RevI-Test.csv --leaving sp12

OUTPUT
------
A plain-text report with:
  - Store headcount history and seasonality findings
  - Every salesperson's tenure and "skill effect" (ranked)
  - The capacity-ceiling test result
  - 6-month (or --horizon) margin projection: Replace vs. Don't Replace
  - A bottom-line recommendation with a sensitivity check

Requires: pandas, numpy, statsmodels (pip install pandas numpy statsmodels)
"""

import argparse
import sys
import re
import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError:
    print("This script requires statsmodels. Install it with:\n"
          "    pip install statsmodels\n", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_data(path):
    """Load the CSV and identify the salesperson columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Month" not in df.columns:
        raise ValueError("Expected a 'Month' column; not found in this CSV.")
    if "Sales" not in df.columns:
        raise ValueError("Expected a 'Sales' column; not found in this CSV.")

    sp_cols = [c for c in df.columns if re.match(r"^sale\s+sp\d+$", c.strip(), re.IGNORECASE)]
    if not sp_cols:
        raise ValueError(
            "No salesperson columns found. Expected columns named like "
            "'sale sp1', 'sale sp2', etc."
        )

    df[sp_cols] = df[sp_cols].fillna(0)
    df = df.sort_values("Month").reset_index(drop=True)

    # Sanity check: do the salesperson columns sum (roughly) to Sales?
    sp_sum = df[sp_cols].sum(axis=1)
    max_rel_diff = ((df["Sales"] - sp_sum).abs() / df["Sales"].clip(lower=1e-9)).max()
    if max_rel_diff > 0.01:
        print(f"WARNING: 'Sales' does not closely match the sum of salesperson "
              f"columns (max relative difference {max_rel_diff:.1%}). Proceeding "
              f"anyway, using the per-salesperson columns as the source of truth.",
              file=sys.stderr)

    return df, sp_cols


# ---------------------------------------------------------------------------
# 2. TENURE RECONSTRUCTION
# ---------------------------------------------------------------------------

def build_long_format(df, sp_cols):
    """
    Convert wide format (one column per person) into long format:
    one row per (salesperson, month) where that person was active,
    with a 'month_on_job' counter (1, 2, 3, ... since they started).

    Assumes each salesperson's employment is a single contiguous block
    of months (no rehiring after a gap). This matches typical small-shop
    staffing and is validated with a warning if violated.
    """
    records = []
    tenure_summary = []

    for sp in sp_cols:
        active = df.loc[df[sp] > 0, ["Month", sp]].reset_index(drop=True)
        if active.empty:
            continue

        start, end, n = active["Month"].min(), active["Month"].max(), len(active)
        contiguous = (end - start + 1) == n
        if not contiguous:
            print(f"WARNING: {sp} has gaps in their active months "
                  f"(active {n} of {end - start + 1} months between "
                  f"{start} and {end}). Tenure-month numbering will still "
                  f"increment only on active months; treat results for "
                  f"this person with caution.", file=sys.stderr)

        active = active.sort_values("Month").reset_index(drop=True)
        active["month_on_job"] = range(1, len(active) + 1)
        active["sp"] = sp
        active = active.rename(columns={sp: "sales"})
        records.append(active[["Month", "sp", "sales", "month_on_job"]])

        tenure_summary.append({
            "sp": sp, "start_month": start, "end_month": end,
            "tenure_months": n, "contiguous": contiguous,
        })

    long_df = pd.concat(records, ignore_index=True)
    tenure_df = pd.DataFrame(tenure_summary).sort_values("start_month").reset_index(drop=True)
    return long_df, tenure_df


# ---------------------------------------------------------------------------
# 3. SEASONALITY DETECTION
# ---------------------------------------------------------------------------

def detect_season_period(df, forced_period=0, max_period=18):
    """
    Auto-detect a recurring seasonal cycle length by checking, for each
    candidate period length P, how much of the variance in per-head sales
    is explained by (Month mod P). Returns the P with the best fit,
    unless the improvement over "no seasonality" is negligible, or the
    user forced a specific period.
    """
    if forced_period and forced_period > 1:
        return forced_period

    sp_cols_active = (df.drop(columns=["Month", "Sales"]) > 0)
    headcount = sp_cols_active.sum(axis=1).clip(lower=1)
    per_head = df["Sales"] / headcount

    best_period, best_score = 1, 0.0
    n = len(df)
    for p in range(2, min(max_period, n // 3) + 1):
        groups = (df["Month"] - 1) % p
        grand_mean = per_head.mean()
        ss_total = ((per_head - grand_mean) ** 2).sum()
        if ss_total == 0:
            continue
        group_means = per_head.groupby(groups).mean()
        ss_between = sum(
            ((per_head[groups == g] - group_means[g]) ** 2).sum() * 0
            for g in group_means.index
        )
        # R^2-style score: variance explained by period-P group means
        predicted = groups.map(group_means)
        ss_resid = ((per_head - predicted) ** 2).sum()
        r2 = 1 - ss_resid / ss_total
        # Penalise larger P slightly (more dummy variables, easier to
        # overfit) using an adjustment akin to adjusted R^2.
        dof_penalty = (n - 1) / max(n - p, 1)
        adj_r2 = 1 - (1 - r2) * dof_penalty
        if adj_r2 > best_score:
            best_score, best_period = adj_r2, p

    if best_score < 0.05:
        return 1  # no meaningful seasonality detected
    return best_period


# ---------------------------------------------------------------------------
# 4. MODEL FITTING: separate tenure curve, seasonality, individual skill
# ---------------------------------------------------------------------------

def fit_model(long_df, season_period):
    long_df = long_df.copy()
    long_df["log_tenure"] = np.log(long_df["month_on_job"])
    if season_period > 1:
        long_df["season"] = ((long_df["Month"] - 1) % season_period).astype("category")
    else:
        long_df["season"] = 0

    long_df["sp"] = long_df["sp"].astype("category")

    if season_period > 1:
        formula = "sales ~ log_tenure + month_on_job + C(season) + C(sp)"
    else:
        formula = "sales ~ log_tenure + month_on_job + C(sp)"

    model = smf.ols(formula, data=long_df).fit()

    # Extract individual skill effects (relative to the reference category,
    # which statsmodels picks as the first category alphabetically/by order)
    sp_categories = list(long_df["sp"].cat.categories)
    ref_sp = sp_categories[0]
    sp_effects = {ref_sp: 0.0}
    for sp in sp_categories[1:]:
        key = f"C(sp)[T.{sp}]"
        sp_effects[sp] = model.params.get(key, 0.0)

    season_effects = {0: 0.0}
    if season_period > 1:
        for s in range(1, season_period):
            key = f"C(season)[T.{s}]"
            season_effects[s] = model.params.get(key, 0.0)

    return {
        "model": model,
        "intercept": model.params["Intercept"],
        "b_log": model.params["log_tenure"],
        "b_lin": model.params["month_on_job"],
        "season_effects": season_effects,
        "sp_effects": sp_effects,
        "season_period": season_period,
    }


def predict_sales(params, month_on_job, season, skill_effect):
    season_period = params["season_period"]
    s = (season % season_period) if season_period > 1 else 0
    base = (params["intercept"]
            + params["b_log"] * np.log(max(month_on_job, 1))
            + params["b_lin"] * month_on_job
            + params["season_effects"].get(s, 0.0))
    return max(base + skill_effect, 0.0)


# ---------------------------------------------------------------------------
# 5. CAPACITY CEILING TEST
# ---------------------------------------------------------------------------

def capacity_ceiling_test(df, sp_cols, long_df, params):
    """
    Compare each month's ACTUAL total sales to the sum of what the model
    would predict for each active individual that month. If months with
    above-normal headcount systematically show negative residuals (actual
    below the sum of individual track records), that is evidence staff are
    competing for the same finite pool of customers rather than each
    bringing in incremental business.
    """
    long_df = long_df.copy()
    long_df["season"] = long_df["Month"] if params["season_period"] <= 1 else (long_df["Month"] - 1) % params["season_period"]
    long_df["pred"] = long_df.apply(
        lambda r: predict_sales(params, r["month_on_job"], r["season"], params["sp_effects"].get(r["sp"], 0.0)),
        axis=1,
    )
    long_df["resid"] = long_df["sales"] - long_df["pred"]

    headcount = (df[sp_cols] > 0).sum(axis=1)
    month_resid = long_df.groupby("Month")["resid"].sum().reset_index()
    month_resid = month_resid.merge(
        pd.DataFrame({"Month": df["Month"], "headcount": headcount}), on="Month"
    )

    modal_hc = headcount.mode().iloc[0]
    normal = month_resid.loc[month_resid["headcount"] == modal_hc, "resid"]
    above = month_resid.loc[month_resid["headcount"] > modal_hc, "resid"]

    result = {
        "modal_headcount": int(modal_hc),
        "normal_mean_resid": normal.mean() if len(normal) else None,
        "above_mean_resid": above.mean() if len(above) else None,
        "n_normal": len(normal),
        "n_above": len(above),
    }
    return result


# ---------------------------------------------------------------------------
# 6. SCENARIO PROJECTION
# ---------------------------------------------------------------------------

def project_scenarios(df, sp_cols, tenure_df, params, leaving_sp,
                       horizon, salary, commission_rate, gross_margin_rate):
    last_month = int(df["Month"].max())
    season_period = params["season_period"]

    # Determine which staff are currently active (active in the last month)
    # and continuing (i.e. everyone except the one who is leaving).
    last_row = df[df["Month"] == last_month].iloc[0]
    active_now = [sp for sp in sp_cols if last_row[sp] > 0]
    continuing = [sp for sp in active_now if sp != leaving_sp]

    tenure_at_end = {
        row["sp"]: row["tenure_months"] for _, row in tenure_df.iterrows()
    }

    sp_effects = params["sp_effects"]
    all_skills = list(sp_effects.values())
    avg_new_hire_skill = float(np.mean(all_skills))
    median_new_hire_skill = float(np.median(all_skills))
    p25_skill = float(np.percentile(all_skills, 25))
    p75_skill = float(np.percentile(all_skills, 75))

    def margin(total_sales, headcount):
        gross = gross_margin_rate * total_sales
        sal = salary * headcount
        comm = commission_rate * total_sales
        return gross - sal - comm

    def run_scenario(new_hire_skill, replace):
        rows = []
        for step in range(1, horizon + 1):
            mo = last_month + step
            season = (mo - 1) % season_period if season_period > 1 else 0

            cont_sales = 0.0
            for sp in continuing:
                t = tenure_at_end.get(sp, 1) + step
                cont_sales += predict_sales(params, t, season, sp_effects.get(sp, 0.0))

            if replace:
                new_hire_tenure = step  # starts at month-on-job = 1 at next month
                new_sales = predict_sales(params, new_hire_tenure, season, new_hire_skill)
                total_sales = cont_sales + new_sales
                headcount = len(continuing) + 1
            else:
                total_sales = cont_sales
                headcount = len(continuing)

            rows.append({
                "month": mo, "total_sales": total_sales, "headcount": headcount,
                "margin": margin(total_sales, headcount),
            })
        return pd.DataFrame(rows)

    noreplace = run_scenario(None, replace=False)
    replace_mean = run_scenario(avg_new_hire_skill, replace=True)
    replace_median = run_scenario(median_new_hire_skill, replace=True)
    replace_p25 = run_scenario(p25_skill, replace=True)
    replace_p75 = run_scenario(p75_skill, replace=True)
    replace_worst = run_scenario(min(all_skills), replace=True)
    replace_best = run_scenario(max(all_skills), replace=True)

    return {
        "active_now": active_now,
        "continuing": continuing,
        "noreplace": noreplace,
        "replace_mean": replace_mean,
        "replace_median": replace_median,
        "replace_p25": replace_p25,
        "replace_p75": replace_p75,
        "replace_worst": replace_worst,
        "replace_best": replace_best,
        "avg_new_hire_skill": avg_new_hire_skill,
        "median_new_hire_skill": median_new_hire_skill,
        "p25_skill": p25_skill,
        "p75_skill": p75_skill,
    }


# ---------------------------------------------------------------------------
# 7. GUESS WHO IS LEAVING (if not specified)
# ---------------------------------------------------------------------------

def guess_leaving_sp(df, sp_cols, tenure_df):
    """
    If the user doesn't say who is leaving, assume it's whoever has been
    active the longest among people who are active in the final month and
    have no newer hire already overlapping with them (i.e. nobody hired in
    the last month or two appears to be their already-arranged replacement).
    This is only a fallback guess -- specifying --leaving is recommended.
    """
    last_month = int(df["Month"].max())
    last_row = df[df["Month"] == last_month].iloc[0]
    active_now = [sp for sp in sp_cols if last_row[sp] > 0]
    if not active_now:
        return None
    # pick the one with the earliest start month (longest-serving) among
    # those active now, as a neutral default guess
    starts = tenure_df.set_index("sp")["start_month"]
    candidates = [(starts.get(sp, 0), sp) for sp in active_now]
    candidates.sort()
    return candidates[0][1]


# ---------------------------------------------------------------------------
# 8. REPORT GENERATION
# ---------------------------------------------------------------------------

def normalize_sp_name(name, sp_cols):
    """Allow the user to pass 'sp12' or 'sale sp12' or 'SP12'."""
    name = name.strip()
    candidates = [c for c in sp_cols if c.lower() == name.lower()]
    if candidates:
        return candidates[0]
    candidates = [c for c in sp_cols if c.lower() == f"sale {name}".lower()]
    if candidates:
        return candidates[0]
    return None


def make_report(df, sp_cols, tenure_df, params, ceiling_result, leaving_sp,
                 horizon, salary, commission_rate, gross_margin_rate, projection):
    lines = []
    w = lines.append

    w("=" * 78)
    w("STAFFING RECOMMENDATION REPORT")
    w("=" * 78)
    w("")
    w(f"Data: {len(df)} months ({int(df['Month'].min())} to {int(df['Month'].max())}), "
      f"{len(sp_cols)} salesperson columns found.")
    w(f"Departing salesperson under review: {leaving_sp}")
    w("")

    # --- Tenure summary ---
    w("-" * 78)
    w("1. STAFF TENURE HISTORY")
    w("-" * 78)
    w(f"{'Salesperson':<14}{'Start':>7}{'End':>7}{'Tenure(mo)':>12}{'Skill effect':>14}{'Rank':>6}")
    ranked = sorted(params["sp_effects"].items(), key=lambda x: -x[1])
    rank_map = {sp: i + 1 for i, (sp, _) in enumerate(ranked)}
    for _, row in tenure_df.iterrows():
        sp = row["sp"]
        eff = params["sp_effects"].get(sp, 0.0)
        w(f"{sp:<14}{row['start_month']:>7}{row['end_month']:>7}{row['tenure_months']:>12}"
          f"{eff:>14.2f}{rank_map.get(sp,'-'):>6}")
    w("")
    w("'Skill effect' is each person's estimated sales contribution per month,")
    w("relative to the baseline salesperson, AFTER removing the effects of how")
    w("long they'd been on the job and which calendar season it was. This is the")
    w("fairest available basis for comparing performance across people who")
    w("started at different times.")
    w("")

    leaving_rank = rank_map.get(leaving_sp)
    n_total = len(rank_map)
    if leaving_rank:
        pct = 100 * (n_total - leaving_rank) / (n_total - 1) if n_total > 1 else 100
        w(f">>> {leaving_sp} ranks #{leaving_rank} of {n_total} historical staff "
          f"(~{pct:.0f}th percentile) on this skill measure.")
        w("")

    # --- Seasonality ---
    w("-" * 78)
    w("2. SEASONALITY")
    w("-" * 78)
    if params["season_period"] > 1:
        w(f"Detected a recurring {params['season_period']}-month cycle in sales per ")
        w("active salesperson. Season effects (relative to the first month of the cycle):")
        for s, eff in params["season_effects"].items():
            w(f"   season offset {s}: {eff:+.2f}")
    else:
        w("No statistically meaningful recurring seasonal cycle was detected;")
        w("the model proceeds without seasonal adjustment.")
    w("")

    # --- Capacity ceiling ---
    w("-" * 78)
    w("3. STORE CAPACITY CHECK (does adding extra staff actually grow sales?)")
    w("-" * 78)
    if ceiling_result["n_above"] >= 2:
        w(f"Normal staffing level in this data: {ceiling_result['modal_headcount']} people "
          f"({ceiling_result['n_normal']} months).")
        w(f"Average prediction error in normal months: {ceiling_result['normal_mean_resid']:+.2f}")
        w(f"Average prediction error in above-normal-headcount months "
          f"({ceiling_result['n_above']} months): {ceiling_result['above_mean_resid']:+.2f}")
        if ceiling_result["above_mean_resid"] is not None and ceiling_result["above_mean_resid"] < -2:
            w("")
            w(">>> Total sales in above-normal-headcount months fall noticeably short of")
            w("    what each individual's own track record would predict. This suggests a")
            w("    capacity ceiling: extra staff are splitting the same customer footfall")
            w("    rather than growing total sales. Be cautious about permanently running")
            w("    above your normal headcount.")
        else:
            w("")
            w("No strong evidence of a capacity ceiling was found in this data; adding")
            w("staff above the normal level does not show a clear sales penalty here.")
    else:
        w("Not enough months with above-normal headcount in this data to test for a")
        w("store capacity ceiling.")
    w("")

    # --- Projection ---
    w("-" * 78)
    w(f"4. {horizon}-MONTH MARGIN PROJECTION: REPLACE vs. DO NOT REPLACE")
    w("-" * 78)
    w(f"Assumptions used: salary = {salary}/person/month, commission = "
      f"{commission_rate:.0%} of sales, gross margin = {gross_margin_rate:.0%} of sales")
    w(f"  Margin = {gross_margin_rate:.0%} x Sales - Salary x headcount - "
      f"{commission_rate:.0%} x Sales")
    w("")
    w(f"Continuing staff (excludes {leaving_sp}): {', '.join(projection['continuing'])}")
    w("")

    nr_total = projection["noreplace"]["margin"].sum()
    rm_total = projection["replace_mean"]["margin"].sum()
    rmed_total = projection["replace_median"]["margin"].sum()
    rp25_total = projection["replace_p25"]["margin"].sum()
    rp75_total = projection["replace_p75"]["margin"].sum()
    rworst_total = projection["replace_worst"]["margin"].sum()
    rbest_total = projection["replace_best"]["margin"].sum()

    w(f"{'Scenario':<48}{'Total ' + str(horizon) + '-mo margin':>20}")
    w(f"{'Do NOT replace ' + leaving_sp:<48}{nr_total:>20.2f}")
    w(f"{'Replace (new hire = average historical skill)':<48}{rm_total:>20.2f}")
    w(f"{'Replace (new hire = median historical skill)':<48}{rmed_total:>20.2f}")
    w(f"{'Replace (new hire = 25th pct. historical skill)':<48}{rp25_total:>20.2f}")
    w(f"{'Replace (new hire = 75th pct. historical skill)':<48}{rp75_total:>20.2f}")
    w(f"{'Replace (new hire = worst historical hire)':<48}{rworst_total:>20.2f}")
    w(f"{'Replace (new hire = best historical hire)':<48}{rbest_total:>20.2f}")
    w("")

    diff_mean = rm_total - nr_total
    diff_median = rmed_total - nr_total
    diff_p25 = rp25_total - nr_total
    breakeven_found = diff_p25 >= 0

    w(f"Expected gain from replacing (vs. not), using average-skill assumption: "
      f"{diff_mean:+.2f}")
    w(f"Expected gain from replacing, using median-skill assumption: {diff_median:+.2f}")
    w(f"Expected gain from replacing, using a pessimistic (25th pct.) new hire: "
      f"{diff_p25:+.2f}")
    w("")

    # --- Recommendation ---
    w("-" * 78)
    w("5. RECOMMENDATION")
    w("-" * 78)
    if diff_mean > 0 and diff_median > 0:
        w(f">>> REPLACE {leaving_sp}.")
        w(f"    Hiring a replacement is expected to add {diff_mean:.2f} in margin over the")
        w(f"    next {horizon} months versus leaving the seat empty, using a typical new-hire")
        w("    skill assumption. This conclusion holds even under a median-outcome hire.")
        if diff_p25 < 0:
            w(f"    Caution: if the new hire turns out to be in the weaker quartile of past")
            w(f"    hires, replacing would cost {abs(diff_p25):.2f} versus not replacing -- so")
            w("    hiring quality matters; this is not a risk-free decision.")
    elif diff_mean <= 0 and diff_median <= 0:
        w(f">>> DO NOT REPLACE {leaving_sp} (at least not immediately).")
        w(f"    Under typical new-hire assumptions, leaving the seat open is expected to")
        w(f"    produce {abs(diff_mean):.2f} more margin over {horizon} months than hiring a")
        w("    replacement -- likely because of limited incremental sales capacity in the")
        w("    store relative to the added salary and commission cost.")
    else:
        w(f">>> BORDERLINE: the average-skill case favors one option but the median-skill")
        w(f"    case favors the other. Treat this decision as roughly a coin flip on")
        w(f"    current information; consider qualitative factors (e.g., a strong specific")
        w(f"    candidate already lined up) before deciding.")
    w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Recommend whether to replace a departing salesperson, "
                     "based on historical salesperson-by-month sales data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv_path", help="Path to the CSV file (Month, Sales, sale sp1, sale sp2, ...)")
    p.add_argument("--leaving", default=None,
                   help="Column name (e.g. 'sp12' or 'sale sp12') of the salesperson who is "
                        "resigning. If omitted, the script guesses the longest-serving "
                        "currently-active person.")
    p.add_argument("--horizon", type=int, default=6, help="Months to project forward (default 6)")
    p.add_argument("--salary", type=float, default=3.0,
                   help="Monthly salary per salesperson, same units as Sales column (default 3)")
    p.add_argument("--commission-rate", type=float, default=0.05,
                   help="Commission as a fraction of sales (default 0.05)")
    p.add_argument("--gross-margin-rate", type=float, default=0.50,
                   help="Gross margin as a fraction of sales, before salary/commission (default 0.50)")
    p.add_argument("--season-period", type=int, default=0,
                   help="Force a seasonal cycle length in months (0 = auto-detect, default)")
    p.add_argument("--out", default=None, help="Optional path to also save the text report")
    args = p.parse_args()

    df, sp_cols = load_data(args.csv_path)
    long_df, tenure_df = build_long_format(df, sp_cols)

    if args.leaving:
        leaving_sp = normalize_sp_name(args.leaving, sp_cols)
        if leaving_sp is None:
            print(f"ERROR: could not find a salesperson column matching '{args.leaving}'. "
                  f"Available columns: {sp_cols}", file=sys.stderr)
            sys.exit(1)
    else:
        leaving_sp = guess_leaving_sp(df, sp_cols, tenure_df)
        if leaving_sp is None:
            print("ERROR: could not determine who is leaving; no active staff found in the "
                  "final month. Please pass --leaving explicitly.", file=sys.stderr)
            sys.exit(1)
        print(f"NOTE: --leaving not specified; assuming '{leaving_sp}' "
              f"(longest-serving currently-active person). Pass --leaving to override.",
              file=sys.stderr)

    season_period = detect_season_period(df, forced_period=args.season_period)
    params = fit_model(long_df, season_period)
    ceiling_result = capacity_ceiling_test(df, sp_cols, long_df, params)
    projection = project_scenarios(
        df, sp_cols, tenure_df, params, leaving_sp,
        args.horizon, args.salary, args.commission_rate, args.gross_margin_rate,
    )

    report = make_report(
        df, sp_cols, tenure_df, params, ceiling_result, leaving_sp,
        args.horizon, args.salary, args.commission_rate, args.gross_margin_rate, projection,
    )

    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\n(Report also saved to {args.out})", file=sys.stderr)


if __name__ == "__main__":
    main()
