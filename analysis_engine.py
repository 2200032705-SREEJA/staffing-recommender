"""
analysis_engine.py
==================
Core analysis logic for the staffing recommender.
Used by both the CLI/PDF report and the Streamlit app.
"""

import re
import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError:
    raise ImportError("Install statsmodels: pip install statsmodels")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data(path_or_df):
    """Load CSV (path or DataFrame) and identify salesperson columns."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)
    df.columns = [c.strip() for c in df.columns]

    if "Month" not in df.columns:
        raise ValueError("Expected a 'Month' column.")
    if "Sales" not in df.columns:
        raise ValueError("Expected a 'Sales' column.")

    sp_cols = [c for c in df.columns if re.match(r"^sale\s+sp\d+$", c.strip(), re.IGNORECASE)]
    if not sp_cols:
        raise ValueError("No 'sale spN' columns found.")

    df[sp_cols] = df[sp_cols].fillna(0)
    df = df.sort_values("Month").reset_index(drop=True)
    return df, sp_cols


def normalize_sp_name(name, sp_cols):
    name = name.strip()
    for c in sp_cols:
        if c.lower() == name.lower():
            return c
    for c in sp_cols:
        if c.lower() == f"sale {name}".lower():
            return c
    return None


# ---------------------------------------------------------------------------
# TENURE
# ---------------------------------------------------------------------------

def build_long_format(df, sp_cols):
    records, summary = [], []
    for sp in sp_cols:
        active = df.loc[df[sp] > 0, ["Month", sp]].reset_index(drop=True)
        if active.empty:
            continue
        start, end, n = active["Month"].min(), active["Month"].max(), len(active)
        active = active.sort_values("Month").reset_index(drop=True)
        active["month_on_job"] = range(1, n + 1)
        active["sp"] = sp
        active = active.rename(columns={sp: "sales"})
        records.append(active[["Month", "sp", "sales", "month_on_job"]])
        summary.append({"sp": sp, "start_month": int(start),
                        "end_month": int(end), "tenure_months": n})
    long_df = pd.concat(records, ignore_index=True)
    tenure_df = pd.DataFrame(summary).sort_values("start_month").reset_index(drop=True)
    return long_df, tenure_df


# ---------------------------------------------------------------------------
# SEASONALITY
# ---------------------------------------------------------------------------

def detect_season_period(df, sp_cols, forced_period=0, max_period=18):
    if forced_period and forced_period > 1:
        return forced_period
    headcount = (df[[c for c in sp_cols]] > 0).sum(axis=1).clip(lower=1)
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
        predicted = groups.map(group_means)
        ss_resid = ((per_head - predicted) ** 2).sum()
        r2 = 1 - ss_resid / ss_total
        dof_penalty = (n - 1) / max(n - p, 1)
        adj_r2 = 1 - (1 - r2) * dof_penalty
        if adj_r2 > best_score:
            best_score, best_period = adj_r2, p
    return best_period if best_score >= 0.05 else 1


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def fit_model(long_df, season_period):
    long_df = long_df.copy()
    long_df["log_tenure"] = np.log(long_df["month_on_job"])
    long_df["season"] = (((long_df["Month"] - 1) % season_period).astype("category")
                         if season_period > 1 else 0)
    long_df["sp"] = long_df["sp"].astype("category")

    formula = ("sales ~ log_tenure + month_on_job + C(season) + C(sp)"
               if season_period > 1 else
               "sales ~ log_tenure + month_on_job + C(sp)")
    model = smf.ols(formula, data=long_df).fit()

    sp_cats = list(long_df["sp"].cat.categories)
    ref_sp = sp_cats[0]
    sp_effects = {ref_sp: 0.0}
    for sp in sp_cats[1:]:
        sp_effects[sp] = model.params.get(f"C(sp)[T.{sp}]", 0.0)

    season_effects = {0: 0.0}
    if season_period > 1:
        for s in range(1, season_period):
            season_effects[s] = model.params.get(f"C(season)[T.{s}]", 0.0)

    return {
        "model": model,
        "r2": model.rsquared,
        "intercept": model.params["Intercept"],
        "b_log": model.params["log_tenure"],
        "b_lin": model.params["month_on_job"],
        "season_effects": season_effects,
        "sp_effects": sp_effects,
        "season_period": season_period,
    }


def predict_sales(params, month_on_job, calendar_month, skill_effect):
    sp = params["season_period"]
    s = (calendar_month % sp) if sp > 1 else 0
    base = (params["intercept"]
            + params["b_log"] * np.log(max(month_on_job, 1))
            + params["b_lin"] * month_on_job
            + params["season_effects"].get(s, 0.0))
    return max(base + skill_effect, 0.0)


# ---------------------------------------------------------------------------
# CAPACITY CEILING
# ---------------------------------------------------------------------------

def capacity_ceiling_test(df, sp_cols, long_df, params):
    ld = long_df.copy()
    ld["season"] = (ld["Month"] - 1) % params["season_period"] if params["season_period"] > 1 else 0
    ld["pred"] = ld.apply(
        lambda r: predict_sales(params, r["month_on_job"], r["Month"],
                                params["sp_effects"].get(r["sp"], 0.0)), axis=1)
    ld["resid"] = ld["sales"] - ld["pred"]

    headcount = (df[sp_cols] > 0).sum(axis=1)
    month_resid = ld.groupby("Month")["resid"].sum().reset_index()
    month_resid = month_resid.merge(
        pd.DataFrame({"Month": df["Month"], "headcount": headcount}), on="Month")

    modal_hc = int(headcount.mode().iloc[0])
    normal = month_resid.loc[month_resid["headcount"] == modal_hc, "resid"]
    above = month_resid.loc[month_resid["headcount"] > modal_hc, "resid"]

    return {
        "modal_headcount": modal_hc,
        "normal_mean_resid": float(normal.mean()) if len(normal) else None,
        "above_mean_resid": float(above.mean()) if len(above) else None,
        "n_normal": len(normal),
        "n_above": len(above),
        "month_resid": month_resid,
        "headcount_series": headcount,
    }


# ---------------------------------------------------------------------------
# PROJECTION
# ---------------------------------------------------------------------------

def project_scenarios(df, sp_cols, tenure_df, params, leaving_sp,
                      horizon, salary, commission_rate, gross_margin_rate):
    last_month = int(df["Month"].max())
    last_row = df[df["Month"] == last_month].iloc[0]
    active_now = [sp for sp in sp_cols if last_row[sp] > 0]
    continuing = [sp for sp in active_now if sp != leaving_sp]

    tenure_at_end = {row["sp"]: row["tenure_months"] for _, row in tenure_df.iterrows()}
    sp_effects = params["sp_effects"]
    all_skills = list(sp_effects.values())

    def margin(total_sales, headcount):
        return gross_margin_rate * total_sales - salary * headcount - commission_rate * total_sales

    def run_scenario(new_hire_skill, replace):
        rows = []
        for step in range(1, horizon + 1):
            mo = last_month + step
            cont_sales = sum(
                predict_sales(params, tenure_at_end.get(sp, 1) + step, mo,
                              sp_effects.get(sp, 0.0))
                for sp in continuing)
            if replace:
                new_sales = predict_sales(params, step, mo, new_hire_skill)
                total_sales = cont_sales + new_sales
                hc = len(continuing) + 1
            else:
                total_sales = cont_sales
                hc = len(continuing)
            rows.append({"month": mo, "total_sales": total_sales,
                         "headcount": hc, "margin": margin(total_sales, hc)})
        return pd.DataFrame(rows)

    skill_percentiles = {
        "worst":  float(np.min(all_skills)),
        "p10":    float(np.percentile(all_skills, 10)),
        "p25":    float(np.percentile(all_skills, 25)),
        "median": float(np.median(all_skills)),
        "mean":   float(np.mean(all_skills)),
        "p75":    float(np.percentile(all_skills, 75)),
        "p90":    float(np.percentile(all_skills, 90)),
        "best":   float(np.max(all_skills)),
    }

    scenarios = {}
    scenarios["noreplace"] = run_scenario(None, replace=False)
    for label, skill in skill_percentiles.items():
        scenarios[f"replace_{label}"] = run_scenario(skill, replace=True)

    # Monthly breakdown comparing noreplace vs replace_mean
    monthly_compare = scenarios["noreplace"][["month", "margin"]].copy()
    monthly_compare.columns = ["month", "margin_noreplace"]
    monthly_compare["margin_replace_mean"] = scenarios["replace_mean"]["margin"].values
    monthly_compare["delta"] = monthly_compare["margin_replace_mean"] - monthly_compare["margin_noreplace"]

    return {
        "active_now": active_now,
        "continuing": continuing,
        "leaving_sp": leaving_sp,
        "scenarios": scenarios,
        "skill_percentiles": skill_percentiles,
        "monthly_compare": monthly_compare,
        "all_skills": all_skills,
    }


# ---------------------------------------------------------------------------
# RANKED SKILL TABLE
# ---------------------------------------------------------------------------

def build_skill_table(tenure_df, params, sp_cols, df):
    sp_effects = params["sp_effects"]
    ranked = sorted(sp_effects.items(), key=lambda x: -x[1])
    rank_map = {sp: i + 1 for i, (sp, _) in enumerate(ranked)}

    last_month = int(df["Month"].max())
    last_row = df[df["Month"] == last_month].iloc[0]
    active_now = set(sp for sp in sp_cols if last_row[sp] > 0)

    rows = []
    for _, t in tenure_df.iterrows():
        sp = t["sp"]
        rows.append({
            "Rank": rank_map.get(sp, "-"),
            "Salesperson": sp,
            "Start": int(t["start_month"]),
            "End": int(t["end_month"]),
            "Tenure (mo)": int(t["tenure_months"]),
            "Skill Effect": round(sp_effects.get(sp, 0.0), 2),
            "Status": "Active" if sp in active_now else "Departed",
        })
    rows.sort(key=lambda r: r["Rank"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FULL ANALYSIS RUNNER
# ---------------------------------------------------------------------------

def run_full_analysis(path_or_df, leaving_name=None, horizon=6,
                      salary=3.0, commission_rate=0.05, gross_margin_rate=0.50,
                      season_period=0):
    df, sp_cols = load_data(path_or_df)
    long_df, tenure_df = build_long_format(df, sp_cols)

    if leaving_name:
        leaving_sp = normalize_sp_name(leaving_name, sp_cols)
        if leaving_sp is None:
            raise ValueError(f"Could not find salesperson '{leaving_name}'")
    else:
        last_month = int(df["Month"].max())
        last_row = df[df["Month"] == last_month].iloc[0]
        active_now = [sp for sp in sp_cols if last_row[sp] > 0]
        starts = tenure_df.set_index("sp")["start_month"]
        leaving_sp = min(active_now, key=lambda sp: starts.get(sp, 0))

    sp = detect_season_period(df, sp_cols, forced_period=season_period)
    params = fit_model(long_df, sp)
    ceiling = capacity_ceiling_test(df, sp_cols, long_df, params)
    projection = project_scenarios(df, sp_cols, tenure_df, params, leaving_sp,
                                   horizon, salary, commission_rate, gross_margin_rate)
    skill_table = build_skill_table(tenure_df, params, sp_cols, df)

    # Summary margin table
    nr = projection["scenarios"]["noreplace"]["margin"].sum()
    margin_summary = []
    labels = {
        "noreplace":      ("Do NOT replace", None),
        "replace_worst":  ("Replace — worst historical hire (bottom)", "worst"),
        "replace_p10":    ("Replace — 10th percentile hire", "p10"),
        "replace_p25":    ("Replace — pessimistic hire (25th pct.)", "p25"),
        "replace_median": ("Replace — median hire", "median"),
        "replace_mean":   ("Replace — average hire (recommended baseline)", "mean"),
        "replace_p75":    ("Replace — optimistic hire (75th pct.)", "p75"),
        "replace_p90":    ("Replace — 90th percentile hire", "p90"),
        "replace_best":   ("Replace — best historical hire", "best"),
    }
    for key, (label, pct_key) in labels.items():
        total = projection["scenarios"][key]["margin"].sum()
        skill = projection["skill_percentiles"].get(pct_key, None)
        margin_summary.append({
            "Scenario": label,
            "Skill Assumption": f"{skill:+.2f}" if skill is not None else "—",
            f"Total {horizon}-mo Margin": round(total, 2),
            "vs. Not Replacing": f"{total - nr:+.2f}" if key != "noreplace" else "—",
        })

    return {
        "df": df,
        "sp_cols": sp_cols,
        "long_df": long_df,
        "tenure_df": tenure_df,
        "leaving_sp": leaving_sp,
        "params": params,
        "ceiling": ceiling,
        "projection": projection,
        "skill_table": skill_table,
        "margin_summary": pd.DataFrame(margin_summary),
        "horizon": horizon,
        "salary": salary,
        "commission_rate": commission_rate,
        "gross_margin_rate": gross_margin_rate,
    }
