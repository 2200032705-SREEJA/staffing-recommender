# 🛍️ Staffing Recommender

> 👩‍💼 **Should you replace your resigning salesperson?**  
> Upload your monthly sales CSV and get an instant, data-driven hire/don't-hire recommendation — with profit forecasts, team rankings, capacity analysis, and plain-English explanations.

🏬 Built for branded retail stores. Designed to be reused every time someone resigns.

---

## 🎮 Live demo

Upload `RevI-Test.csv` (included) to try it instantly with 81 months of real store data.

---

## 🚀 Quickstart

```bash
# 1️⃣ Install dependencies
pip install streamlit pandas numpy statsmodels matplotlib reportlab

# 2️⃣ Run the web app
streamlit run streamlit_app.py

# 3️⃣ Open in browser
# http://localhost:8501
```

---

## 🗂️ Three ways to use this

| Mode | File | Best for |
|------|------|----------|
| Web app (recommended) | `streamlit_app.py` | Monthly use, non-technical users |
| PDF report | `generate_pdf_report.py` | Shareable executive report |
| CLI terminal | `staffing_recommender.py` | Quick analysis in terminal |

---

## ✨ Web app features

Upload your CSV, pick who is resigning, and instantly get:

- 🟢 **Verdict banner** — YES or NO in plain English with expected profit figure
- 📊 **4 KPI cards** — skill rank, monthly impact, profit gain, model accuracy (each explained in plain English)
- 📈 **Profit Forecast tab** — month-by-month profit: hire vs. leave empty
- 🎯 **Hire Quality tab** — what if your next hire is great, average, or terrible?
- 🏆 **Team Rankings tab** — skill ranking of all staff + distribution histogram showing where the leaving person sits
- 📅 **Seasonal Patterns tab** — your store's busy/slow cycle + full sales history
- 🏪 **Overlap & Ceiling tab** — does your 1-month overlap pattern actually grow sales?
- 💡 **Tips & Rules tab** — 5 data-backed rules for smarter staffing
- 📌 **Assumptions tab** — every assumption the model makes, listed clearly

---

## 💻 CLI usage

```bash
python staffing_recommender.py RevI-Test.csv --leaving sp12
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--leaving` | (required) | Who is resigning, e.g. `sp12` |
| `--horizon` | `6` | Months to project forward |
| `--salary` | `3` | Monthly salary per person (same unit as Sales column) |
| `--commission-rate` | `0.05` | Commission as fraction of sales |
| `--gross-margin-rate` | `0.50` | Gross margin as fraction of sales |
| `--season-period` | `0` | Seasonal cycle length (0 = auto-detect) |
| `--out` | — | Save report to a text file |

---

## 📄 PDF report

```bash
python generate_pdf_report.py RevI-Test.csv --leaving sp12 --out staffing_report.pdf
```

Produces a multi-page A4 PDF with all charts, tables, and the full recommendation.

---

## 📊 CSV format

```
Month, Sales, sale sp1, sale sp2, sale sp3, ...
1, 23.1, 7.2, 6.3, 9.5, 0, ...
2, 41.6, 8.9, 7.8, 11.7, 7.5, ...
```

| Column | Description |
|--------|-------------|
| `Month` | Sequential integers (1, 2, 3, …) |
| `Sales` | Total store sales that month |
| `sale spN` | Individual salesperson's sales (0 or blank = not employed that month) |

Each person's employed months should be one contiguous block. All values in the same unit (the salary parameter uses the same unit).

---

## 🧠 How it works

1. 📉 **Ramp-up curve** — fits a log+linear tenure model. New hires start slow and improve over ~12 months. Universal across all staff.

2. 🌊 **Seasonality detection** — auto-detects a repeating cycle (12 months in this dataset). All skill comparisons are seasonality-adjusted so slow-season hires aren't unfairly penalised.

3. 🏅 **Skill ranking** — estimates each person's true monthly sales contribution vs. a baseline hire, after removing ramp-up and seasonal noise. Model R² = 0.92 on this dataset.

4. 🏪 **Capacity ceiling test** — checks whether 8-person months (during overlap) generate proportionally higher total sales, or whether staff split the same customer footfall.

5. 💰 **Margin projection** — projects gross margin under Replace vs. Don't Replace for 8 hire-quality scenarios drawn from the historical skill distribution.

6. ✅ **Recommendation** — replace if the expected gain under an average-quality hire is positive. Includes break-even analysis and full sensitivity table.

---

## 📁 Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Interactive web UI |
| `analysis_engine.py` | Core analysis logic (shared by all modes) |
| `staffing_recommender.py` | CLI plain-text report |
| `generate_pdf_report.py` | PDF report generator |
| `RevI-Test.csv` | Sample data — 81 months, 16 salespeople |

---

## 📌 Assumptions

- Gross margin = 50% × Sales − salary × headcount − 5% × Sales (adjustable)
- New hires follow a log-curve ramp-up based on historical patterns in your data
- Future hire quality = average of all past hires (baseline scenario)
- Seasonal patterns detected in your data will repeat
- Customer footfall is the binding constraint, not headcount
- Store format and location are broadly stable over the data period

Full assumption details are in the **Assumptions tab** of the web app.

---

## 📦 Requirements

```
streamlit
pandas
numpy
statsmodels
matplotlib
reportlab
```
