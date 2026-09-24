"""
Impact of AI & Social Media on Students — Pure Analysis Script
==============================================================
Author  : Sneha Mittra
Dataset : AI_SocialMedia_Student_Dataset.csv
Output  : Console report + saved PNG charts in ./charts/
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# ─────────────────────────────────────────────
# 0.  OUTPUT DIRECTORY
# ─────────────────────────────────────────────
os.makedirs("charts", exist_ok=True)

EDU_ORDER    = ["High School", "College", "University"]
DATA_PATH    = "AI_SocialMedia_Student_Dataset.csv"
DIVIDER      = "=" * 72
THIN_DIVIDER = "-" * 72

def banner(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

def save(fig: plt.Figure, name: str) -> None:
    path = f"charts/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


# ══════════════════════════════════════════════
# 1.  COLLECT & LOAD DATASET
# ══════════════════════════════════════════════
banner("1. COLLECT & LOAD DATASET")

raw = pd.read_csv(DATA_PATH)
print(f"  Rows loaded  : {len(raw):,}")
print(f"  Columns      : {list(raw.columns)}")
print(f"\n  First 5 rows :\n")
print(raw.head().to_string(index=False))


# ══════════════════════════════════════════════
# 2.  DATA CLEANING & QUALITY CHECK
# ══════════════════════════════════════════════
banner("2. DATA CLEANING & QUALITY CHECK")

# --- 2a. Missing values ---
null_counts = raw.isnull().sum()
print("\n  Missing values per column:")
print(null_counts.to_string())

# --- 2b. Duplicates ---
n_dupes = raw.duplicated().sum()
print(f"\n  Duplicate rows : {n_dupes}")

# --- 2c. Range checks ---
range_checks = {
    "Daily_Social_Media_Hours"  : {"min": 0,  "max": None},
    "Daily_AI_Tool_Usage_Hours" : {"min": 0,  "max": None},
    "Sleep_Hours"               : {"min": 0,  "max": 24},
    "Physical_Activity_Hours"   : {"min": 0,  "max": None},
    "Mental_Health_Score"       : {"min": 0,  "max": 100},
    "Physical_Health_Score"     : {"min": 0,  "max": 100},
}

print("\n  Out-of-range value counts:")
total_range_issues = 0
for col, bounds in range_checks.items():
    lo = bounds["min"]; hi = bounds["max"]
    mask = pd.Series([False] * len(raw))
    if lo is not None:
        mask |= raw[col] < lo
    if hi is not None:
        mask |= raw[col] > hi
    n = mask.sum()
    total_range_issues += n
    status = "✔  OK" if n == 0 else f"⚠  {n} issue(s)"
    print(f"    {col:<35} {status}")

print(f"\n  Total quality issues found : {null_counts.sum() + n_dupes + total_range_issues}")

# --- Clean ---
df = raw.copy()
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
for col, bounds in range_checks.items():
    df[col] = df[col].clip(lower=bounds["min"], upper=bounds["max"])

print(f"  Rows after cleaning         : {len(df):,}")


# ══════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════
banner("3. FEATURE ENGINEERING")

# Total Screen Time
df["Total_Screen_Time"] = df["Daily_Social_Media_Hours"] + df["Daily_AI_Tool_Usage_Hours"]

# Digital Balance Ratio  (0 = all social-media, 1 = all AI tools)
df["Digital_Balance_Ratio"] = np.where(
    df["Total_Screen_Time"] > 0,
    df["Daily_AI_Tool_Usage_Hours"] / df["Total_Screen_Time"],
    0.0,
)

# Wellness Index  (equal-weight composite, normalised to 0–100)
sleep_norm    = (df["Sleep_Hours"] / 10).clip(0, 1) * 100
activity_norm = (df["Physical_Activity_Hours"] / 5).clip(0, 1) * 100
df["Wellness_Index"] = (
    sleep_norm * 0.25
    + activity_norm * 0.25
    + df["Mental_Health_Score"] * 0.25
    + df["Physical_Health_Score"] * 0.25
)

df["Education_Level"] = pd.Categorical(df["Education_Level"], categories=EDU_ORDER, ordered=True)

print("\n  New features added:")
for feat in ["Total_Screen_Time", "Digital_Balance_Ratio", "Wellness_Index"]:
    print(f"    {feat:<30}  mean={df[feat].mean():.3f}  min={df[feat].min():.3f}  max={df[feat].max():.3f}")

print(f"\n  Engineered dataset sample:\n")
print(df[["Student_ID", "Total_Screen_Time", "Digital_Balance_Ratio", "Wellness_Index"]].head(10).to_string(index=False))


# ══════════════════════════════════════════════
# 4.  GROUP & SUMMARISE DATA
# ══════════════════════════════════════════════
banner("4. GROUP & SUMMARISE DATA")

# ── By Education Level ────────────────────────
print("\n  ── By Education Level ──\n")
edu_group = (
    df.groupby("Education_Level", observed=True)
    .agg(
        Students              = ("Student_ID",               "count"),
        Mean_Screen_Time      = ("Total_Screen_Time",         "mean"),
        Median_Screen_Time    = ("Total_Screen_Time",         "median"),
        Mean_AI_Usage         = ("Daily_AI_Tool_Usage_Hours", "mean"),
        AI_Adoption_Rate_pct  = ("Daily_AI_Tool_Usage_Hours", lambda x: (x > 0).mean() * 100),
        Mean_Sleep            = ("Sleep_Hours",               "mean"),
        Median_Sleep          = ("Sleep_Hours",               "median"),
        Mean_Mental_Health    = ("Mental_Health_Score",       "mean"),
        Mean_Physical_Health  = ("Physical_Health_Score",     "mean"),
        Mean_Wellness         = ("Wellness_Index",            "mean"),
    )
    .round(2)
    .reset_index()
)
print(edu_group.to_string(index=False))

# ── By Gender ─────────────────────────────────
print("\n  ── By Gender ──\n")
gender_group = (
    df.groupby("Gender")
    .agg(
        Students             = ("Student_ID",               "count"),
        Mean_Screen_Time     = ("Total_Screen_Time",         "mean"),
        Mean_AI_Usage        = ("Daily_AI_Tool_Usage_Hours", "mean"),
        Mean_Sleep           = ("Sleep_Hours",               "mean"),
        Mean_Mental_Health   = ("Mental_Health_Score",       "mean"),
        Mean_Physical_Health = ("Physical_Health_Score",     "mean"),
        Mean_Wellness        = ("Wellness_Index",            "mean"),
    )
    .round(2)
    .reset_index()
)
print(gender_group.to_string(index=False))

# ── Cross-tab ─────────────────────────────────
print("\n  ── Education Level × Gender ──\n")
cross = (
    df.groupby(["Education_Level", "Gender"], observed=True)
    .agg(
        Students         = ("Student_ID",         "count"),
        Mean_Screen_Time = ("Total_Screen_Time",   "mean"),
        Mean_Wellness    = ("Wellness_Index",      "mean"),
        Mean_Sleep       = ("Sleep_Hours",         "mean"),
    )
    .round(2)
    .reset_index()
)
print(cross.to_string(index=False))


# ══════════════════════════════════════════════
# 5.  DATA VISUALISATION
# ══════════════════════════════════════════════
banner("5. DATA VISUALISATION")

# ── 5a. Correlation Heatmap ───────────────────
print("\n  Generating correlation heatmap …")
corr_cols = [
    "Daily_Social_Media_Hours", "Daily_AI_Tool_Usage_Hours",
    "Total_Screen_Time", "Sleep_Hours", "Physical_Activity_Hours",
    "Mental_Health_Score", "Physical_Health_Score",
    "Digital_Balance_Ratio", "Wellness_Index",
]
corr = df[corr_cols].corr().round(2)

fig, ax = plt.subplots(figsize=(11, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, linewidths=0.5, ax=ax,
    annot_kws={"size": 8},
)
ax.set_title("Pearson Correlation Matrix — Digital Usage & Health Metrics", pad=14, fontweight="bold")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
fig.tight_layout()
save(fig, "01_correlation_heatmap.png")

# ── 5b. Scatter – Screen Time vs Health Scores ─
print("  Generating scatter plots …")
sample = df.sample(min(2000, len(df)), random_state=42)
edu_palette = {"High School": "#3b82f6", "College": "#22c55e", "University": "#f97316"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, health_col, title in zip(
    axes,
    ["Mental_Health_Score", "Physical_Health_Score"],
    ["Mental Health Score", "Physical Health Score"],
):
    for edu in EDU_ORDER:
        sub = sample[sample["Education_Level"] == edu]
        ax.scatter(sub["Total_Screen_Time"], sub[health_col],
                   alpha=0.35, s=18, color=edu_palette[edu], label=edu)
        # trendline
        m, b = np.polyfit(sub["Total_Screen_Time"], sub[health_col], 1)
        xs = np.linspace(sub["Total_Screen_Time"].min(), sub["Total_Screen_Time"].max(), 100)
        ax.plot(xs, m * xs + b, color=edu_palette[edu], linewidth=1.8)
    ax.set_xlabel("Total Screen Time (hrs/day)")
    ax.set_ylabel(title)
    ax.set_title(f"Screen Time vs {title}", fontweight="bold")
    ax.legend(title="Education Level", fontsize=8)
fig.suptitle("Total Screen Time vs Health Scores", fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "02_scatter_screentime_vs_health.png")

# ── 5c. Scatter – Social Media & AI vs Mental Health ─
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gender_palette = {"Male": "#3b82f6", "Female": "#ec4899", "Non-binary": "#8b5cf6"}

for ax, x_col, xlabel in zip(
    axes,
    ["Daily_Social_Media_Hours", "Daily_AI_Tool_Usage_Hours"],
    ["Social Media (hrs/day)", "AI Tools (hrs/day)"],
):
    for gender, color in gender_palette.items():
        sub = sample[sample["Gender"] == gender]
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub["Mental_Health_Score"],
                   alpha=0.4, s=18, color=color, label=gender)
        m, b = np.polyfit(sub[x_col], sub["Mental_Health_Score"], 1)
        xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
        ax.plot(xs, m * xs + b, color=color, linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mental Health Score")
    ax.set_title(f"{xlabel} vs Mental Health Score", fontweight="bold")
    ax.legend(title="Gender", fontsize=8)
fig.suptitle("Digital Usage vs Mental Health by Gender", fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "03_scatter_digital_vs_mental_health.png")

# ── 5d. Bar – AI Usage & Sleep by Education Level ─
print("  Generating bar charts …")
bar_data = edu_group.set_index("Education_Level")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ["#3b82f6", "#22c55e", "#f97316"]

for ax, col, ylabel, title in zip(
    axes,
    ["Mean_AI_Usage", "Mean_Sleep"],
    ["Avg AI Tool Usage (hrs/day)", "Avg Sleep (hrs/day)"],
    ["Average AI Tool Usage by Education Level", "Average Sleep Hours by Education Level"],
):
    bars = ax.bar(EDU_ORDER, bar_data.loc[EDU_ORDER, col], color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, bar_data.loc[EDU_ORDER, col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, bar_data[col].max() * 1.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
fig.tight_layout()
save(fig, "04_bar_ai_sleep_by_education.png")

# ── 5e. Grouped bar – Health scores by Gender ─
fig, ax = plt.subplots(figsize=(9, 5))
x      = np.arange(len(gender_group))
width  = 0.35
bars1  = ax.bar(x - width / 2, gender_group["Mean_Mental_Health"],  width, label="Mental Health",  color="#6366f1")
bars2  = ax.bar(x + width / 2, gender_group["Mean_Physical_Health"], width, label="Physical Health", color="#22c55e")
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(gender_group["Gender"])
ax.set_ylabel("Score (0–100)")
ax.set_ylim(0, 115)
ax.set_title("Average Mental & Physical Health Scores by Gender", fontweight="bold")
ax.legend()
fig.tight_layout()
save(fig, "05_bar_health_by_gender.png")

# ── 5f. Box – Wellness Index by Education Level ─
fig, ax = plt.subplots(figsize=(9, 5))
groups = [df.loc[df["Education_Level"] == edu, "Wellness_Index"].values for edu in EDU_ORDER]
bp = ax.boxplot(groups, labels=EDU_ORDER, patch_artist=True, notch=False,
                flierprops=dict(marker="o", markersize=3, alpha=0.4))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Wellness Index (0–100)")
ax.set_title("Wellness Index Distribution by Education Level", fontweight="bold")
fig.tight_layout()
save(fig, "06_boxplot_wellness_by_education.png")

# ── 5g. Histogram – Feature Distributions ────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
hist_data = [
    ("Total_Screen_Time",    "Total Screen Time (hrs/day)", "#3b82f6"),
    ("Digital_Balance_Ratio","Digital Balance Ratio",        "#8b5cf6"),
    ("Wellness_Index",       "Wellness Index (0–100)",       "#22c55e"),
]
for ax, (col, xlabel, color) in zip(axes, hist_data):
    ax.hist(df[col], bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution: {col}", fontweight="bold")
fig.tight_layout()
save(fig, "07_histograms_engineered_features.png")

# ── 5h. Line – Screen Time trend across Age ───────
print("  Generating trend line chart …")
age_trend = (
    df.groupby("Age")
    .agg(
        Mean_Screen_Time   = ("Total_Screen_Time",   "mean"),
        Mean_Mental_Health = ("Mental_Health_Score", "mean"),
        Mean_Wellness      = ("Wellness_Index",      "mean"),
    )
    .reset_index()
    .sort_values("Age")
)
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(age_trend["Age"], age_trend["Mean_Screen_Time"], color="#3b82f6", marker="o", markersize=5, label="Avg Screen Time")
ax2.plot(age_trend["Age"], age_trend["Mean_Mental_Health"], color="#f97316", marker="s", markersize=5, linestyle="--", label="Avg Mental Health")
ax1.set_xlabel("Age")
ax1.set_ylabel("Avg Screen Time (hrs/day)", color="#3b82f6")
ax2.set_ylabel("Avg Mental Health Score",   color="#f97316")
ax1.tick_params(axis="y", colors="#3b82f6")
ax2.tick_params(axis="y", colors="#f97316")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax1.set_title("Screen Time & Mental Health Score Trend Across Age Groups", fontweight="bold")
fig.tight_layout()
save(fig, "08_line_screentime_mentalhealth_by_age.png")


# ══════════════════════════════════════════════
# 6.  INSTITUTIONAL & ACTIONABLE DECISIONS
# ══════════════════════════════════════════════
banner("6. INSTITUTIONAL & ACTIONABLE DECISION-MAKING")

# Compute key correlations for evidence
corr_sm_mh = df["Daily_Social_Media_Hours"].corr(df["Mental_Health_Score"])
corr_ai_mh = df["Daily_AI_Tool_Usage_Hours"].corr(df["Mental_Health_Score"])
corr_sl_wi = df["Sleep_Hours"].corr(df["Wellness_Index"])
corr_sc_wi = df["Total_Screen_Time"].corr(df["Wellness_Index"])

print(f"""
  ── Key Correlation Findings ──

  Social Media Hours  ↔  Mental Health Score : {corr_sm_mh:+.4f}
  AI Tool Usage       ↔  Mental Health Score : {corr_ai_mh:+.4f}
  Sleep Hours         ↔  Wellness Index      : {corr_sl_wi:+.4f}
  Total Screen Time   ↔  Wellness Index      : {corr_sc_wi:+.4f}

  AI Adoption Rate    : {(df["Daily_AI_Tool_Usage_Hours"] > 0).mean() * 100:.1f}% of students use AI tools
""")

print(f"""
  ── Avg Wellness Index by Education Level ──
""")
for edu in EDU_ORDER:
    wi = df.loc[df["Education_Level"] == edu, "Wellness_Index"].mean()
    print(f"    {edu:<15} : {wi:.2f}")

print(f"""
  ── Institutional Recommendations ──

  1. CAMPUS WELLNESS PROGRAMS
     • Schedule regular mental-health screenings, especially for University
       students who show the lowest average wellness scores.
     • Run weekly mindfulness and stress-management workshops aligned with
       the academic calendar (exam weeks = highest risk).
     • Flag students with combined screen time > 8 hrs/day for counselling
       check-ins.

  2. DIGITAL HYGIENE INTERVENTIONS
     • Promote the 20-20-20 rule: every 20 min, look 20 ft away for 20 s.
     • Designate phone-free study blocks and quiet zones on campus.
     • Run digital-literacy campaigns using built-in app-usage timers.

  3. ACADEMIC AI INTEGRATION
     • Provide structured AI-literacy training (purposeful, not habitual).
     • Embed AI productivity tools into coursework to shift the Digital
       Balance Ratio toward productive AI use rather than passive social media.
     • Target High School students with introductory AI programmes early.

  4. HEALTHY SLEEP & ACTIVITY TARGETS
     • Evidence-based sleep targets: ≥ 8 hrs for High Schoolers,
       ≥ 7 hrs for College / University students.
     • Gamify physical-activity tracking (campus apps, weekly challenges).
     • During high-screen-time periods, extend evening sports-facility hours
       to encourage de-screen breaks through movement.
""")


# ══════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════
banner("ANALYSIS COMPLETE")
print(f"""
  Dataset rows analysed   : {len(df):,}
  Charts saved to         : ./charts/
    01_correlation_heatmap.png
    02_scatter_screentime_vs_health.png
    03_scatter_digital_vs_mental_health.png
    04_bar_ai_sleep_by_education.png
    05_bar_health_by_gender.png
    06_boxplot_wellness_by_education.png
    07_histograms_engineered_features.png
    08_line_screentime_mentalhealth_by_age.png
""")
