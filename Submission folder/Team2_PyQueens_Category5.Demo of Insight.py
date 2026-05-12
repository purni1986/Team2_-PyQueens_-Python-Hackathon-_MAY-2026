import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

# Plotly for sunburst chart
try:
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

st.set_page_config(
    page_title="CGM Diabetes Dashboard",
    page_icon="💉",
    layout="wide",
)
sns.set_style("whitegrid")

# Custom CSS for a polished look
st.markdown("""
<style>
    h1 {
        color: #1f4e79;
        font-weight: 700;
        padding-bottom: 8px;
        border-bottom: 3px solid #4a90e2;
        margin-bottom: 20px;
    }
    h2, h3 {color: #2c5282;}

  
    .page-title {
        color: #1f4e79;
        font-weight: 700;
        font-size: 30px;
        padding-bottom: 8px;
        border-bottom: 2px solid #4a90e2;
        margin: 0 0 22px 0;
        line-height: 1.3;
    }

    /* Conclusion box */
    .conclusion-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 6px solid #ff9800;
        padding: 18px 22px;
        border-radius: 8px;
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 15px;
        line-height: 1.6;
        color: #333;
    }
    .conclusion-box h4 {
        color: #e65100;
        margin: 0 0 10px 0;
        font-size: 18px;
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3f2fd 0%, #ffffff 100%);
    }
    [data-testid="stSidebar"] label p {font-size: 15px; font-weight: 500;}
    [data-testid="stSidebar"] .stRadio label {font-size: 15px;}

    /* Team box */
    .team-box {
        background: #ffffff;
        border-radius: 8px;
        padding: 18px 20px;
        margin-top: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        font-size: 18px !important;
        line-height: 1.75;
    }
    .team-box h4 {
        margin: 0 0 14px 0 !important;
        color: #1f4e79 !important;
        font-size: 20px !important;
        font-weight: 700 !important;
    }
    .team-box ul {
        margin: 0 !important;
        padding-left: 24px !important;
        color: #222 !important;
        list-style-type: disc;
    }
    .team-box li {
        margin-bottom: 8px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #222 !important;
    }
</style>
""", unsafe_allow_html=True)


def page_title(text):
    """Render a smaller, sleek page title."""
    st.markdown(f'<div class="page-title">{text}</div>',
                unsafe_allow_html=True)


def conclusion(text):
    """Render a polished conclusion box."""
    st.markdown(
        f'<div class="conclusion-box"><h4>📌 Conclusion</h4>{text}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# DATA PATHS
# ============================================================
BASE = "/Users/tejaswinimode/Desktop/NUMPY/PyHackathon/Final cleaned csv files"
MERGED_PATH  = os.path.join(BASE, "Team2_PyQueens_Merged_Final.csv")
CLEANED_PATH = os.path.join(BASE, "Team2_PyQueens_Final_Cleaned_File.csv")
SLEEP_PATH   = os.path.join(BASE, "T1DM_patient_sleep_demographics_with_race.csv")

for p_name, p_value in [("MERGED_PATH", MERGED_PATH),
                        ("CLEANED_PATH", CLEANED_PATH),
                        ("SLEEP_PATH", SLEEP_PATH)]:
    if not os.path.exists(p_value):
        local = os.path.basename(p_value)
        if os.path.exists(local):
            globals()[p_name] = local

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_main():
    df = pd.read_csv(MERGED_PATH)
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    df["Steps"] = df["Steps"].astype("int32")
    df["Hour"] = df["Time"].dt.hour
    df["Date"] = df["Time"].dt.date
    df["Week"] = df["Time"].dt.isocalendar().week.astype(int)
    def cls(g):
        if g < 70: return "Hypoglycemia"
        if g <= 180: return "Normal"
        return "Hyperglycemia"
    df["Glucose_Zone"] = df["Glucose"].apply(cls)
    return df

@st.cache_data
def load_sleep():
    return pd.read_csv(SLEEP_PATH)

try:
    df = load_main()
    sleep_df = load_sleep()
except FileNotFoundError as e:
    st.error(f"Data file not found: {e}\n\nUpdate the BASE path at the top of cgm_dashboard.py.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("CGM Dashboard")

# Team Members — placed right under title
st.sidebar.markdown("""
<div class="team-box">
<h4>👩‍💻 Team PyQueens</h4>
<ul>
<li>Abirami Ramasubramanian</li>
<li>Harshali Gunjal</li>
<li>Mahalakshmi Vinoth Kumar</li>
<li>Purnima Chandrasekaran</li>
<li>Tejaswini Mode</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Section",
    ["Overview", "Descriptive", "Diagnostic", "Prescriptive", "Predictive"]
)

# ── Sidebar labels (full original questions) ──
Q_DESC_1 = "Q1: How much time do patients spend in Hypoglycemia, Normal, and Hyperglycemia zones?"
Q_DESC_2 = "Q2: What are the hourly, daily, and weekly glucose trends in the dataset?"
Q_DESC_3 = "Q3: What is the distribution of physical activity across patients and how does it vary?"
Q_DESC_4 = "Q4: Sleep Quality by Gender & Race"
Q_DESC_5 = "Q5: What are the minimum, maximum, mean, and median heart rate values?"

Q_DIAG_1 = "Q1: Does poor sleep quality cause an increase in glucose levels?"
Q_DIAG_2 = "Q2: How do carbs, insulin, and physical activity together influence glucose spikes?"
Q_DIAG_3 = "Q3: Correlation between previous-day steps and glucose variability (SD & CV). How many steps should a patient aim for?"

Q_PRES_1 = "Q1: Which patients need immediate dietary intervention based on their carb-glucose response?"
Q_PRES_2 = "Q2: What should be done when heart rate and glucose rise together?"

Q_PRED_1 = "Q1: Can we predict whether a T1DM patient will enter a Hypoglycemia or Hyperglycemia zone in the next 30 minutes based on their current glucose, steps, heart rate, carb intake and insulin dose?"

# ── Display titles (descriptive, not in question form) ──
TITLES = {
    Q_DESC_1: "Glucose Level Distribution — Time Spent in Hypoglycemia, Normal & Hyperglycemia Zones",
    Q_DESC_2: "Hourly, Daily and Weekly Glucose Trends",
    Q_DESC_3: "Distribution of Physical Activity (Steps & Calories) Across Patients",
    Q_DESC_4: "Sleep Quality by Gender & Race",
    Q_DESC_5: "Heart Rate Statistics — Minimum, Maximum, Mean & Median",
    Q_DIAG_1: "Effect of Poor Sleep Quality on Glucose Levels",
    Q_DIAG_2: "Combined Influence of Carbs, Insulin & Physical Activity on Glucose Spikes",
    Q_DIAG_3: "Previous-Day Steps and Glucose Variability — Target Daily Step Count",
    Q_PRES_1: "Patients Needing Immediate Dietary Intervention Based on Carb-Glucose Response",
    Q_PRES_2: "Recommended Actions When Heart Rate & Glucose Rise Together",
    Q_PRED_1: "Predicting Hypoglycemia / Hyperglycemia Zones 30 Minutes Ahead",
}

subpages = {
    "Descriptive": [
        "KPIs & Gender Distribution",
        Q_DESC_1, Q_DESC_2, Q_DESC_3, Q_DESC_4, Q_DESC_5,
    ],
    "Diagnostic":  [Q_DIAG_1, Q_DIAG_2, Q_DIAG_3],
    "Prescriptive":[Q_PRES_1, Q_PRES_2],
    "Predictive":  [Q_PRED_1],
}

if section in subpages:
    page = st.sidebar.radio("Question", subpages[section])
else:
    page = section

# ============================================================
# OVERVIEW
# ============================================================
if section == "Overview":
    st.title("CGM Diabetes Analytics Dashboard")
    st.markdown(
        "End-to-end analytics on continuous glucose monitoring (CGM) data from "
        "**25 Type-1 Diabetes patients** — covering Descriptive, Diagnostic, "
        "Prescriptive, and Predictive analytics."
    )

    c1, c2 = st.columns(2)
    c1.metric("Patients", df["Patient_Id"].nunique())
    c2.metric("Total Readings", f"{len(df):,}")

    st.markdown("### Sections in this Dashboard")
    st.markdown("""
- **Descriptive** — KPIs, patient demographics, and the core patterns in the data
- **Diagnostic** — Investigating *why* glucose behaves the way it does
- **Prescriptive** — Recommended actions for each patient
- **Predictive** — Forecasting dangerous glucose events 30 minutes in advance
    """)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20))

# ============================================================
# DESCRIPTIVE — KPIs & Gender + Age Distribution
# ============================================================
elif page == "KPIs & Gender Distribution":
    page_title("KPIs — Gender & Age Distribution")

    st.subheader("Gender Distribution")
    gender_counts = sleep_df["Gender"].value_counts()
    total_g = gender_counts.sum()
    g1, g2, g3 = st.columns(3)
    g1.metric("Male Patients", int(gender_counts.get("Male", 0)),
              f"{gender_counts.get('Male', 0)/total_g*100:.1f}%")
    g2.metric("Female Patients", int(gender_counts.get("Female", 0)),
              f"{gender_counts.get('Female', 0)/total_g*100:.1f}%")
    g3.metric("Male : Female Ratio",
              f"{gender_counts.get('Male', 0)} : {gender_counts.get('Female', 0)}")

    st.subheader("Age Distribution")
    age_per_patient = (df.groupby("Patient_Id")["Age"].first()
                       if "Age" in df.columns
                       else sleep_df.set_index("Patient_Id")["Age"]
                            if "Age" in sleep_df.columns else None)

    if age_per_patient is None:
        st.warning("Age column not found in either dataset.")
    else:
        def age_band(a):
            if a < 30: return "Young (<30)"
            if a < 50: return "Middle (30-49)"
            return "Older (50+)"
        age_groups = age_per_patient.apply(age_band).value_counts()
        total_a = age_groups.sum()

        a1, a2, a3 = st.columns(3)
        a1.metric("Young (<30)", int(age_groups.get("Young (<30)", 0)),
                  f"{age_groups.get('Young (<30)', 0)/total_a*100:.1f}%")
        a2.metric("Middle (30-49)", int(age_groups.get("Middle (30-49)", 0)),
                  f"{age_groups.get('Middle (30-49)', 0)/total_a*100:.1f}%")
        a3.metric("Older (50+)", int(age_groups.get("Older (50+)", 0)),
                  f"{age_groups.get('Older (50+)', 0)/total_a*100:.1f}%")

        a4, a5, a6 = st.columns(3)
        a4.metric("Youngest Patient", f"{age_per_patient.min()} yrs")
        a5.metric("Oldest Patient",   f"{age_per_patient.max()} yrs")
        a6.metric("Mean Age",         f"{age_per_patient.mean():.1f} yrs")

# ============================================================
# DESCRIPTIVE — Q1 Glucose Zones
# ============================================================
elif page == Q_DESC_1:
    page_title(TITLES[Q_DESC_1])

    zone_colors = {"Hypoglycemia": "#FF4C4C",
                   "Normal":        "#2ECC71",
                   "Hyperglycemia": "#FF9F1C"}

    zone_counts = df["Glucose_Zone"].value_counts()
    zone_pct = (zone_counts / len(df) * 100).round(2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Hypoglycemia %", f"{zone_pct.get('Hypoglycemia', 0):.1f}%")
    c2.metric("Normal %",       f"{zone_pct.get('Normal', 0):.1f}%")
    c3.metric("Hyperglycemia %",f"{zone_pct.get('Hyperglycemia', 0):.1f}%")

    patient_zone = (df.groupby(["Patient_Id", "Glucose_Zone"]).size()
                      .unstack(fill_value=0))
    patient_zone_pct = patient_zone.div(patient_zone.sum(axis=1), axis=0) * 100
    for col in ["Hypoglycemia", "Normal", "Hyperglycemia"]:
        if col not in patient_zone_pct.columns:
            patient_zone_pct[col] = 0
    patient_zone_pct = patient_zone_pct[["Hypoglycemia", "Normal", "Hyperglycemia"]]
    patient_zone_pct = patient_zone_pct.sort_values("Normal", ascending=False)

    fig, ax = plt.subplots(figsize=(18, 10), dpi=120)
    patient_zone_pct.plot(kind="barh", stacked=True, ax=ax,
                          color=[zone_colors[c] for c in patient_zone_pct.columns])
    ax.set_title("Per-Patient Time in Each Glucose Zone (%)",
                 fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("% of Time", fontsize=13)
    ax.set_ylabel("Patient ID", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    conclusion(
        "Most patients maintain safe glucose levels most of the time, but a few "
        "outliers spend significant time in dangerous zones — these charts let "
        "you immediately identify <b>who needs prioritised intervention</b>."
    )

# ============================================================
# DESCRIPTIVE — Q2 Trends
# ============================================================
elif page == Q_DESC_2:
    page_title(TITLES[Q_DESC_2])

    granularity = st.selectbox("Select trend granularity",
                               ["Hourly", "Daily", "Weekly"])

    if granularity == "Hourly":
        s = df.groupby("Hour")["Glucose"].mean(); xlab = "Hour"
    elif granularity == "Daily":
        s = df.groupby("Date")["Glucose"].mean(); xlab = "Date"
    else:
        s = df.groupby("Week")["Glucose"].mean(); xlab = "Week"

    fig, ax = plt.subplots(figsize=(18, 7), dpi=120)
    sns.lineplot(x=s.index, y=s.values, marker="o", color="steelblue",
                 linewidth=2.2, markersize=8, ax=ax)
    ax.axhline(180, color="red", linestyle="--", alpha=0.6,
               label="Hyperglycemia (180)")
    ax.axhline(70, color="orange", linestyle="--", alpha=0.6,
               label="Hypoglycemia (70)")
    ax.set_title(f"{granularity} Average Glucose",
                 fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel(xlab, fontsize=13)
    ax.set_ylabel("Avg Glucose (mg/dL)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)
    if granularity == "Daily":
        plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    conclusion(
        "Glucose follows a clear daily rhythm — <b>postprandial spikes after meals</b> "
        "and <b>stable overnight readings</b>. Weekly trends reveal lifestyle-driven "
        "variability, underlining the value of consistent meal and insulin routines."
    )

# ============================================================
# DESCRIPTIVE — Q3 Physical Activity
# ============================================================
elif page == Q_DESC_3:
    page_title(TITLES[Q_DESC_3])

    activity = (df.groupby("Patient_Id")
                  .agg(avg_steps=("Steps", "mean"),
                       avg_calories=("Calories", "mean"),
                       total_steps=("Steps", "sum"))
                  .sort_values("avg_steps", ascending=False))
    activity.columns = ["Avg Steps", "Avg Calories", "Total Steps"]

    def classify(s):
        q33, q66 = s.quantile(0.33), s.quantile(0.66)
        return s.apply(lambda x: 2 if x >= q66 else (1 if x >= q33 else 0))

    activity_T = activity.T
    classified = activity_T.apply(classify, axis=1)
    cmap = ListedColormap(["#FF6B6B", "#FFA726", "#4CAF50"])

    fig, ax = plt.subplots(figsize=(26, 11), dpi=130)
    ax.imshow(classified.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(classified.columns)))
    ax.set_xticklabels(classified.columns, rotation=45, ha="right",
                       fontsize=15, fontweight="bold")
    ax.set_yticks(range(len(classified.index)))
    ax.set_yticklabels(classified.index, fontsize=18, fontweight="bold")
    for i in range(classified.shape[0]):
        for j in range(classified.shape[1]):
            val = activity_T.iloc[i, j]
            ax.text(j, i, f"{val:,.0f}", ha="center", va="center",
                    color="white", fontsize=14, fontweight="bold")
    ax.set_title("Activity Classification (Red = Low, Orange = Mid, Green = High)",
                 fontsize=18, fontweight="bold", pad=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Activity Summary Table")
    st.dataframe(activity.round(1))

    conclusion(
        "<b>Green</b> cells highlight consistently-active patients; <b>red</b> cells "
        "show low-activity patients. The gap between groups is substantial — likely "
        "a key driver of glucose variability."
    )

# ============================================================
# DESCRIPTIVE — Q4 Sleep by Gender & Race
# ============================================================
elif page == Q_DESC_4:
    page_title(TITLES[Q_DESC_4])

    gender_avg = sleep_df.groupby("Gender")["Sleep Quality (1-10)"].mean()
    race_avg = sleep_df.groupby("Race")["Sleep Quality (1-10)"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=120)
    sns.barplot(x=gender_avg.index, y=gender_avg.values,
                hue=gender_avg.index, palette="Set2",
                legend=False, ax=axes[0])
    axes[0].set_title("Average Sleep Quality by Gender",
                      fontsize=16, fontweight="bold", pad=12)
    axes[0].set_ylabel("Avg Sleep Quality", fontsize=13)
    axes[0].tick_params(axis="both", labelsize=12)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}",
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=13, fontweight="bold")

    # Race — pie chart
    pie_colors = sns.color_palette("Set3", n_colors=len(race_avg))
    axes[1].pie(race_avg.values, labels=race_avg.index,
                autopct="%1.1f%%", startangle=90,
                colors=pie_colors,
                textprops={"fontsize": 13, "fontweight": "bold"})
    axes[1].set_title("Sleep Quality Distribution by Race",
                      fontsize=16, fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig)

    conclusion(
        "Sleep quality is fairly evenly distributed across races. No group dominates "
        "or lags dramatically — differences are <b>small but visible</b>."
    )

# ============================================================
# DESCRIPTIVE — Q5 Heart Rate
# ============================================================
elif page == Q_DESC_5:
    page_title(TITLES[Q_DESC_5])

    hr_stats = df["Heart_Rate"].agg(["min", "max", "mean", "median"]).round(2)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min HR",    f"{hr_stats['min']} bpm")
    c2.metric("Max HR",    f"{hr_stats['max']} bpm")
    c3.metric("Mean HR",   f"{hr_stats['mean']} bpm")
    c4.metric("Median HR", f"{hr_stats['median']} bpm")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df["Heart_Rate"], kde=True, bins=40,
                 color="royalblue", ax=axes[0])
    axes[0].set_title("Heart Rate Distribution", fontsize=12)
    axes[0].set_xlabel("Heart Rate")

    sns.boxplot(x=df["Heart_Rate"], color="tomato", ax=axes[1])
    axes[1].set_title("Heart Rate Boxplot", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    conclusion(
        "Min, max, mean, and median HR values reveal <b>recovery quality</b>, "
        "<b>peak strain</b>, and <b>overall cardiovascular load</b> — the foundation "
        "for deeper HR analysis."
    )

# ============================================================
# DIAGNOSTIC — Q1 Sleep vs Glucose
# ============================================================
elif page == Q_DIAG_1:
    page_title(TITLES[Q_DIAG_1])

    p = (df.groupby("Patient_Id")
           .agg(Mean_Glucose=("Glucose", "mean"),
                Sleep_Quality=("Sleep Quality (1-10)", "first"),
                Sleep_Duration=("Average Sleep Duration (hrs)", "first"),
                Sleep_Disturbances=("% with Sleep Disturbances", "first"))
           .reset_index())

    def sleep_cat(s):
        return "Poor" if s <= 4 else ("Average" if s <= 7 else "Good")
    p["Sleep_Group"] = p["Sleep_Quality"].apply(sleep_cat)

    pearson_r, pearson_p = stats.pearsonr(p["Sleep_Quality"], p["Mean_Glucose"])
    spearman_r, spearman_p = stats.spearmanr(p["Sleep_Quality"], p["Mean_Glucose"])

    c1, c2 = st.columns(2)
    c1.metric("Pearson r", f"{pearson_r:+.3f}", f"p = {pearson_p:.4f}")
    c2.metric("Spearman ρ", f"{spearman_r:+.3f}", f"p = {spearman_p:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=p, x="Sleep_Group", y="Mean_Glucose",
                order=["Poor", "Average", "Good"],
                hue="Sleep_Group", palette="coolwarm_r", legend=False, ax=axes[0])
    axes[0].set_title("Mean Glucose by Sleep Quality Group", fontsize=12)
    sns.regplot(data=p, x="Sleep_Quality", y="Mean_Glucose",
                scatter_kws={"s": 60}, line_kws={"color": "red"}, ax=axes[1])
    axes[1].set_title("Sleep Quality vs Mean Glucose", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    conclusion(
        "Poor sleep does <b>NOT cause an increase in glucose</b> in this dataset "
        "(r ≈ 0.05, p > 0.8). No patient falls into the Poor sleep category, "
        "limiting the test."
    )

# ============================================================
# DIAGNOSTIC — Q2 Carbs/Insulin/Activity
# ============================================================
elif page == Q_DIAG_2:
    page_title(TITLES[Q_DIAG_2])

    features = ["Glucose", "Carb_Input", "Basal_Rate",
                "Bolus_Volume_Delivered", "Steps"]
    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Glucose, Carbs, Insulin, and Steps",
                 fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    conclusion("""
    <ul>
    <li><b>Carbs and Bolus insulin</b> are tightly linked — insulin is dosed for meals</li>
    <li><b>Steps</b> show a weak negative relationship with glucose</li>
    <li><b>Basal rate</b> has limited correlation with momentary glucose readings</li>
    <li><b>Carb_Input</b> is the strongest driver of glucose excursions</li>
    </ul>
    """)

# ============================================================
# DIAGNOSTIC — Q3 Steps vs Glucose Variability
# ============================================================
elif page == Q_DIAG_3:
    page_title(TITLES[Q_DIAG_3])

    daily = (df.groupby(["Patient_Id", "Date"])
               .agg(Daily_Steps=("Steps", "sum"),
                    Daily_Glucose_Mean=("Glucose", "mean"),
                    Daily_Glucose_SD=("Glucose", "std"))
               .reset_index())
    daily["Daily_Glucose_CV"] = (
        daily["Daily_Glucose_SD"] / daily["Daily_Glucose_Mean"] * 100
    )
    daily["Prev_Day_Steps"] = daily.groupby("Patient_Id")["Daily_Steps"].shift(1)
    daily = daily.dropna()

    pearson_sd = daily["Prev_Day_Steps"].corr(daily["Daily_Glucose_SD"])
    pearson_cv = daily["Prev_Day_Steps"].corr(daily["Daily_Glucose_CV"])

    c1, c2 = st.columns(2)
    c1.metric("Prev-Day Steps vs SD",  f"{pearson_sd:+.3f}")
    c2.metric("Prev-Day Steps vs CV",  f"{pearson_cv:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.regplot(data=daily, x="Prev_Day_Steps", y="Daily_Glucose_SD",
                scatter_kws={"s": 20, "alpha": 0.4},
                line_kws={"color": "red"}, ax=axes[0])
    axes[0].set_title("Prev Day Steps vs Glucose Standard Deviation", fontsize=12)
    sns.regplot(data=daily, x="Prev_Day_Steps", y="Daily_Glucose_CV",
                scatter_kws={"s": 20, "alpha": 0.4, "color": "green"},
                line_kws={"color": "red"}, ax=axes[1])
    axes[1].set_title("Prev Day Steps vs Glucose CV (%)", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    target = daily.loc[daily["Daily_Glucose_CV"] < 36, "Prev_Day_Steps"].mean()
    conclusion(
        f"Patients with stable glucose (CV < 36%) averaged about "
        f"<b>{target:,.0f} steps</b> the previous day. Aim for this range to "
        "support next-day glucose stability."
    )

# ============================================================
# PRESCRIPTIVE — Q1 Risk Classification
# ============================================================
elif page == Q_PRES_1:
    page_title(TITLES[Q_PRES_1])

    risk = (df.groupby("Patient_Id")
              .agg(Mean_Carb=("Carb_Input", "mean"),
                   Mean_Glucose=("Glucose", "mean"),
                   Hyper_pct=("Glucose", lambda g: (g > 180).mean() * 100))
              .reset_index())

    def risk_class(r):
        if r["Mean_Glucose"] > 180 or r["Hyper_pct"] > 40:
            return "HIGH RISK"
        elif r["Mean_Glucose"] > 154 or r["Hyper_pct"] > 25:
            return "MODERATE"
        else:
            return "LOW RISK"

    risk["Risk_Level"] = risk.apply(risk_class, axis=1)
    summary = risk["Risk_Level"].value_counts()

    c1, c2, c3 = st.columns(3)
    c1.metric("High-Risk Patients",     summary.get("HIGH RISK", 0))
    c2.metric("Moderate-Risk Patients", summary.get("MODERATE", 0))
    c3.metric("Low-Risk Patients",      summary.get("LOW RISK", 0))

    def color_risk(val):
        if val == "HIGH RISK":
            return "background-color: #FF6B6B; color: white"
        if val == "MODERATE":
            return "background-color: #FFA726"
        return "background-color: #4CAF50; color: white"

    st.subheader("Patient Risk Table")
    st.dataframe(risk.sort_values("Hyper_pct", ascending=False)
                     .style.map(color_risk, subset=["Risk_Level"])
                     .format({"Mean_Carb": "{:.1f}",
                              "Mean_Glucose": "{:.1f}",
                              "Hyper_pct": "{:.1f}"}))

    # ── Sunburst chart ──────────────────────────────────────
    st.subheader("Risk Distribution — Sunburst View")
    if PLOTLY_OK:
        sunburst_df = risk.copy()
        sunburst_df["All Patients"] = "All Patients"
        color_map = {"HIGH RISK": "#FF6B6B",
                     "MODERATE":  "#FFA726",
                     "LOW RISK":  "#4CAF50"}
        fig_sb = px.sunburst(
            sunburst_df,
            path=["All Patients", "Risk_Level", "Patient_Id"],
            values="Hyper_pct",
            color="Risk_Level",
            color_discrete_map=color_map,
            title="Patients Grouped by Risk Level (size = Hyperglycemia %)",
        )
        fig_sb.update_layout(height=600, margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig_sb, use_container_width=True)
    else:
        st.warning("Install plotly to see the sunburst chart: `pip install plotly`")

    conclusion("""
    <ul>
    <li><b>HIGH RISK</b> → immediate dietary intervention, carb-reduction plan,
    refined bolus dosing, and post-meal activity</li>
    <li><b>MODERATE</b> → meal timing review and step-count increase</li>
    <li><b>LOW RISK</b> → maintain current routines as best-practice reference</li>
    </ul>
    """)

# ============================================================
# PRESCRIPTIVE — Q2 HR & Glucose Rising Together
# ============================================================
elif page == Q_PRES_2:
    page_title(TITLES[Q_PRES_2])

    df_q2 = df.copy().sort_values(["Patient_Id", "Time"])
    df_q2["HR_change"] = df_q2.groupby("Patient_Id")["Heart_Rate"].diff()
    df_q2["Glu_change"] = df_q2.groupby("Patient_Id")["Glucose"].diff()
    df_q2["Both_Rising"] = ((df_q2["HR_change"] > 0) &
                            (df_q2["Glu_change"] > 0)).astype(int)

    pct_both = df_q2["Both_Rising"].mean() * 100
    st.metric("% of readings where HR & Glucose rise together",
              f"{pct_both:.1f}%")

    sample = df_q2.sample(min(2000, len(df_q2)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=sample, x="HR_change", y="Glu_change",
                    hue="Both_Rising",
                    palette={0: "lightgray", 1: "red"},
                    alpha=0.6, ax=ax)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("Δ Heart Rate vs Δ Glucose (red = both rising = stress event)",
                 fontsize=12)
    ax.set_xlabel("Δ Heart Rate")
    ax.set_ylabel("Δ Glucose")
    st.pyplot(fig)

    conclusion("""
    When HR and glucose rise together it indicates a <b>dual-system stress event</b>
    (autonomic + metabolic stress). Recommended actions:
    <ul>
    <li>Pause and rest for 10–15 minutes — check for stress, illness, or pain</li>
    <li>Hydrate immediately — dehydration amplifies both signals</li>
    <li>Avoid additional bolus correction until the cause is identified</li>
    <li>Monitor for another 30 minutes; if both continue to rise, contact clinician</li>
    </ul>
    """)

# ============================================================
# PREDICTIVE — Q1
# ============================================================
elif page == Q_PRED_1:
    page_title(TITLES[Q_PRED_1])

    with st.expander("Why this model?"):
        st.markdown("""
        - Provides a **30-minute advance warning** before a glucose crisis
        - Uses 5-minute CGM data and lagged features
        - Random Forest handles class imbalance and multi-class output
        - Aligned with what next-gen CGM devices aim to deliver
        """)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, classification_report,
                                     confusion_matrix)

        df_p = df.sort_values(["Patient_Id", "Time"]).copy()
        df_p["Future_Glucose"] = df_p.groupby("Patient_Id")["Glucose"].shift(-6)
        df_p["Future_Zone"] = df_p["Future_Glucose"].apply(
            lambda g: "Hypo" if g < 70 else ("Hyper" if g > 180 else "Normal")
        )
        df_p = df_p.dropna(subset=["Future_Glucose"])

        features = ["Glucose", "Steps", "Heart_Rate", "Carb_Input",
                    "Bolus_Volume_Delivered", "Basal_Rate"]
        X = df_p[features]
        y = df_p["Future_Zone"]

        sample_n = min(50000, len(X))
        X_sample = X.sample(sample_n, random_state=42)
        y_sample = y.loc[X_sample.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.2,
            random_state=42, stratify=y_sample
        )

        with st.spinner("Training Random Forest..."):
            clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                         class_weight="balanced",
                                         random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

        c1, c2 = st.columns(2)
        c1.metric("Test Accuracy", f"{acc*100:.1f}%")
        c2.metric("Baseline (random chance)", "33%")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=["Hypo", "Normal", "Hyper"])
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Hypo", "Normal", "Hyper"],
                    yticklabels=["Hypo", "Normal", "Hyper"], ax=ax)
        ax.set_title("Confusion Matrix", fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Feature Importance")
        fi = pd.DataFrame({"Feature": features,
                           "Importance": clf.feature_importances_}
                         ).sort_values("Importance", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=fi, x="Importance", y="Feature",
                    hue="Feature", palette="viridis",
                    legend=False, ax=ax2)
        ax2.set_title("Which features most influence the prediction?",
                      fontsize=12)
        st.pyplot(fig2)

        conclusion(
            f"The Random Forest model predicts the 30-minute glucose zone with "
            f"<b>{acc*100:.1f}% accuracy</b> — well above the 33% baseline. "
            "The current glucose level is by far the strongest predictor, with "
            "carbs, steps, and heart rate adding meaningful refinement."
        )

    except ImportError:
        st.error("scikit-learn is not installed. Run: `pip install scikit-learn`")
    except Exception as e:
        st.error(f"Could not train model: {e}")

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · Team PyQueens")
