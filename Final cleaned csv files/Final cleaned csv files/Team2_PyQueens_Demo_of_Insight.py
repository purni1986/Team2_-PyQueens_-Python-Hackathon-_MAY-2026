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

# ============================================================
# FULL DARK THEME CSS — no white areas anywhere
# ============================================================
st.markdown("""
<style>
    /* ── Nuke every white surface Streamlit injects ── */
    html, body { background-color: #0F0F1A !important; }
    .stApp { background-color: #0F0F1A !important; }
    .stApp > header { background-color: #0F0F1A !important; }

    /* Streamlit top toolbar / deploy bar */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    .stDeployButton,
    ._profileContainer_gzau3_53 { background: #0F0F1A !important; }

    /* Main content wrapper */
    .block-container {
        padding: 1.2rem 2.2rem 2rem 2.2rem;
        background-color: #0F0F1A !important;
    }

    /* ── All default text ── */
    html, body, [class*="css"] { color: #C0C0E0; }
    p, span, div, li, label { color: #C0C0E0; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #0F0F1A 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #C0C0E0 !important; }
    [data-testid="stSidebarContent"] { background: transparent !important; }

    /* ── TOP-LEVEL tabs (main navigation) ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #1A1A2E;
        border-radius: 12px;
        padding: 6px;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #A0A0C0 !important;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.92rem;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7B2FF7, #E040FB) !important;
        color: white !important;
    }
    /* Tab panel background */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #0F0F1A !important;
        padding-top: 1rem;
    }

    /* ── INNER sub-tabs ── */
    div[data-testid="stVerticalBlock"] .stTabs [data-baseweb="tab-list"] {
        background: #12122A;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    div[data-testid="stVerticalBlock"] .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        padding: 6px 14px;
    }

    /* ── Page title ── */
    .page-title {
        color: #E0E0FF;
        font-weight: 700;
        font-size: 24px;
        padding-bottom: 8px;
        border-bottom: 2px solid #7B2FF7;
        margin: 0 0 20px 0;
        line-height: 1.3;
    }

    /* ── Conclusion box ── */
    .conclusion-box {
        background: linear-gradient(135deg, #1A1A2E 0%, #1E1E38 100%);
        border-left: 6px solid #E040FB;
        padding: 18px 22px;
        border-radius: 8px;
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 15px;
        line-height: 1.6;
        color: #C0C0E0;
    }
    .conclusion-box h4 {
        color: #E040FB;
        margin: 0 0 10px 0;
        font-size: 18px;
    }

    /* ── Team box ── */
    .team-box {
        background: #1A1A2E;
        border: 1px solid #3A2A5A;
        border-radius: 8px;
        padding: 18px 20px;
        margin-top: 10px;
        font-size: 16px !important;
        line-height: 1.75;
    }
    .team-box h4 {
        margin: 0 0 14px 0 !important;
        color: #E040FB !important;
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    .team-box ul {
        margin: 0 !important;
        padding-left: 24px !important;
        list-style-type: disc;
    }
    .team-box li {
        margin-bottom: 8px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #D0D0F0 !important;
    }

    /* ── st.metric ── */
    [data-testid="metric-container"] { background: #1A1A2E; border-radius: 10px; padding: 10px; }
    [data-testid="stMetricValue"] { color: #E0E0FF !important; }
    [data-testid="stMetricDelta"] { color: #A0FFA0 !important; }
    [data-testid="stMetricLabel"] { color: #9090B0 !important; }

    /* ── Dataframe ── */
    .stDataFrame, [data-testid="stDataFrame"] { background-color: #1A1A2E !important; }
    .stDataFrame thead { background-color: #1A1A2E !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader { color: #C0C0E0 !important; background: #1A1A2E !important; }
    .streamlit-expanderContent { background: #12122A !important; }

    /* ── Selectbox ── */
    .stSelectbox label { color: #C0C0E0 !important; }
    [data-testid="stSelectbox"] > div { background: #1A1A2E !important; color: #C0C0E0; }

    /* ── Headings ── */
    h1, h2, h3, h4, h5, h6 { color: #E0E0FF !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #7B2FF7 !important; }

    /* ── Info / Warning boxes ── */
    [data-testid="stInfo"] { background: #1A1A2E !important; color: #C0C0E0 !important; }
    [data-testid="stWarning"] { background: #2A1A0A !important; color: #FFA726 !important; }
    [data-testid="stError"] { background: #2A0A0A !important; color: #FF6B6B !important; }

    /* ── Remove default white app padding ── */
    .css-18e3th9, .css-1d391kg { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def page_title(text):
    st.markdown(f'<div class="page-title">{text}</div>', unsafe_allow_html=True)


def conclusion(text):
    st.markdown(
        f'<div class="conclusion-box"><h4>📌 Conclusion</h4>{text}</div>',
        unsafe_allow_html=True,
    )


def dark_fig(fig, axes=None):
    """Apply dark background to a matplotlib figure."""
    fig.patch.set_facecolor("#0F0F1A")
    if axes is not None:
        ax_list = axes if hasattr(axes, "__iter__") else [axes]
        for ax in ax_list:
            ax.set_facecolor("#1A1A2E")
            ax.tick_params(colors="#C0C0E0")
            ax.xaxis.label.set_color("#C0C0E0")
            ax.yaxis.label.set_color("#C0C0E0")
            ax.title.set_color("#E0E0FF")
            for spine in ax.spines.values():
                spine.set_edgecolor("#3A3A5A")
    return fig


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
        if g < 70:   return "Hypoglycemia"
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
    st.error(f"Data file not found: {e}\n\nUpdate the BASE path at the top of the file.")
    st.stop()


# ============================================================
# SIDEBAR — team box only, no radio buttons
# ============================================================
st.sidebar.title("CGM Dashboard")
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
st.sidebar.caption("Built with Streamlit · Team PyQueens")


# ============================================================
# MAIN TABS (top-level navigation)
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠  Overview",
    "📊  Descriptive",
    "🔍  Diagnostic",
    "💡  Prescriptive",
    "🤖  Predictive",
    "✅  Recommendations",
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:
    st.title("CGM Diabetes Analytics Dashboard")
    st.markdown(
        "End-to-end analytics on continuous glucose monitoring (CGM) data from "
        "**25 Type-1 Diabetes patients** — covering Descriptive, Diagnostic, "
        "Prescriptive, and Predictive analytics."
    )
    c1, c2 = st.columns(2)
    c1.metric("Patients",       df["Patient_Id"].nunique())
    c2.metric("Total Readings", f"{len(df):,}")

    st.markdown("### Sections in this Dashboard")
    st.markdown("""
- **Descriptive** — KPIs, patient demographics, and the core patterns in the data
- **Diagnostic** — Investigating *why* glucose behaves the way it does
- **Prescriptive** — Recommended actions for each patient
- **Predictive** — Forecasting dangerous glucose events 30 minutes in advance
- **Recommendations** — Consolidated actionable interventions and final conclusion
    """)
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20))


# ============================================================
# TAB 2 — DESCRIPTIVE  (inner sub-tabs)
# ============================================================
with tab2:
    d_tabs = st.tabs([
        "KPIs & Gender",
        "Q1 · Glucose Zones",
        "Q2 · Glucose Trends",
        "Q3 · Physical Activity",
        "Q4 · Sleep Quality",
        "Q5 · Heart Rate",
    ])

    # ── KPIs & Gender ──────────────────────────────────────────────────────────
    with d_tabs[0]:
        page_title("KPIs — Gender & Age Distribution")
        st.subheader("Gender Distribution")
        gender_counts = sleep_df["Gender"].value_counts()
        total_g = gender_counts.sum()
        g1, g2, g3 = st.columns(3)
        g1.metric("Male Patients",   int(gender_counts.get("Male",   0)),
                  f"{gender_counts.get('Male',   0)/total_g*100:.1f}%")
        g2.metric("Female Patients", int(gender_counts.get("Female", 0)),
                  f"{gender_counts.get('Female', 0)/total_g*100:.1f}%")
        g3.metric("Male : Female Ratio",
                  f"{gender_counts.get('Male', 0)} : {gender_counts.get('Female', 0)}")

        st.subheader("Age Distribution")
        age_per_patient = (df.groupby("Patient_Id")["Age"].first()
                           if "Age" in df.columns
                           else (sleep_df.set_index("Patient_Id")["Age"]
                                 if "Age" in sleep_df.columns else None))
        if age_per_patient is None:
            st.warning("Age column not found in either dataset.")
        else:
            def age_band(a):
                if a < 30: return "Young (<30)"
                if a < 50: return "Middle (30-49)"
                return "Older (50+)"
            age_groups = age_per_patient.apply(age_band).value_counts()
            total_a    = age_groups.sum()
            a1, a2, a3 = st.columns(3)
            a1.metric("Young (<30)",    int(age_groups.get("Young (<30)",    0)),
                      f"{age_groups.get('Young (<30)',    0)/total_a*100:.1f}%")
            a2.metric("Middle (30-49)", int(age_groups.get("Middle (30-49)", 0)),
                      f"{age_groups.get('Middle (30-49)', 0)/total_a*100:.1f}%")
            a3.metric("Older (50+)",    int(age_groups.get("Older (50+)",    0)),
                      f"{age_groups.get('Older (50+)',    0)/total_a*100:.1f}%")
            a4, a5, a6 = st.columns(3)
            a4.metric("Youngest Patient", f"{age_per_patient.min()} yrs")
            a5.metric("Oldest Patient",   f"{age_per_patient.max()} yrs")
            a6.metric("Mean Age",         f"{age_per_patient.mean():.1f} yrs")

    # ── Q1 Glucose Zones ───────────────────────────────────────────────────────
    with d_tabs[1]:
        page_title("Glucose Level Distribution — Time Spent in Hypoglycemia, Normal & Hyperglycemia Zones")
        zone_colors = {"Hypoglycemia": "#FF4C4C", "Normal": "#2ECC71", "Hyperglycemia": "#FF9F1C"}
        zone_counts = df["Glucose_Zone"].value_counts()
        zone_pct    = (zone_counts / len(df) * 100).round(2)
        c1, c2, c3 = st.columns(3)
        c1.metric("Hypoglycemia %",  f"{zone_pct.get('Hypoglycemia',  0):.1f}%")
        c2.metric("Normal %",        f"{zone_pct.get('Normal',        0):.1f}%")
        c3.metric("Hyperglycemia %", f"{zone_pct.get('Hyperglycemia', 0):.1f}%")

        patient_zone = (df.groupby(["Patient_Id", "Glucose_Zone"]).size()
                          .unstack(fill_value=0))
        patient_zone_pct = patient_zone.div(patient_zone.sum(axis=1), axis=0) * 100
        for col in ["Hypoglycemia", "Normal", "Hyperglycemia"]:
            if col not in patient_zone_pct.columns:
                patient_zone_pct[col] = 0
        patient_zone_pct = patient_zone_pct[["Hypoglycemia", "Normal", "Hyperglycemia"]]
        patient_zone_pct = patient_zone_pct.sort_values("Normal", ascending=False)

        fig, ax = plt.subplots(figsize=(18, 10), dpi=120)
        dark_fig(fig, ax)
        patient_zone_pct.plot(kind="barh", stacked=True, ax=ax,
                              color=[zone_colors[c] for c in patient_zone_pct.columns])
        ax.set_title("Per-Patient Time in Each Glucose Zone (%)",
                     fontsize=16, fontweight="bold", pad=14, color="#E0E0FF")
        ax.set_xlabel("% of Time", fontsize=13, color="#C0C0E0")
        ax.set_ylabel("Patient ID", fontsize=13, color="#C0C0E0")
        ax.tick_params(axis="both", labelsize=12, colors="#C0C0E0")
        ax.legend(loc="lower right", fontsize=12, facecolor="#1A1A2E",
                  edgecolor="#3A3A5A", labelcolor="#C0C0E0")
        plt.tight_layout()
        st.pyplot(fig)
        conclusion(
            "Most patients maintain safe glucose levels most of the time, but a few "
            "outliers spend significant time in dangerous zones — these charts let "
            "you immediately identify <b>who needs prioritised intervention</b>."
        )

    # ── Q2 Trends ──────────────────────────────────────────────────────────────
    with d_tabs[2]:
        page_title("Hourly, Daily and Weekly Glucose Trends")
        granularity = st.selectbox("Select trend granularity", ["Hourly", "Daily", "Weekly"])
        if granularity == "Hourly":
            s = df.groupby("Hour")["Glucose"].mean(); xlab = "Hour"
        elif granularity == "Daily":
            s = df.groupby("Date")["Glucose"].mean(); xlab = "Date"
        else:
            s = df.groupby("Week")["Glucose"].mean(); xlab = "Week"

        fig, ax = plt.subplots(figsize=(18, 7), dpi=120)
        dark_fig(fig, ax)
        sns.lineplot(x=s.index, y=s.values, marker="o", color="#7B2FF7",
                     linewidth=2.2, markersize=8, ax=ax)
        ax.axhline(180, color="#FF4C4C", linestyle="--", alpha=0.8, label="Hyperglycemia (180)")
        ax.axhline(70,  color="#FF9F1C", linestyle="--", alpha=0.8, label="Hypoglycemia (70)")
        ax.set_title(f"{granularity} Average Glucose",
                     fontsize=16, fontweight="bold", pad=12, color="#E0E0FF")
        ax.set_xlabel(xlab, fontsize=13, color="#C0C0E0")
        ax.set_ylabel("Avg Glucose (mg/dL)", fontsize=13, color="#C0C0E0")
        ax.tick_params(axis="both", labelsize=12, colors="#C0C0E0")
        ax.legend(fontsize=12, facecolor="#1A1A2E", edgecolor="#3A3A5A", labelcolor="#C0C0E0")
        if granularity == "Daily":
            plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        conclusion(
            "Glucose follows a clear daily rhythm — <b>postprandial spikes after meals</b> "
            "and <b>stable overnight readings</b>. Weekly trends reveal lifestyle-driven "
            "variability, underlining the value of consistent meal and insulin routines."
        )

    # ── Q3 Physical Activity ───────────────────────────────────────────────────
    with d_tabs[3]:
        page_title("Distribution of Physical Activity (Steps & Calories) Across Patients")
        activity = (df.groupby("Patient_Id")
                      .agg(avg_steps=("Steps",    "mean"),
                           avg_calories=("Calories", "mean"),
                           total_steps=("Steps",    "sum"))
                      .sort_values("avg_steps", ascending=False))
        activity.columns = ["Avg Steps", "Avg Calories", "Total Steps"]

        def classify(s):
            q33, q66 = s.quantile(0.33), s.quantile(0.66)
            return s.apply(lambda x: 2 if x >= q66 else (1 if x >= q33 else 0))

        activity_T = activity.T
        classified = activity_T.apply(classify, axis=1)
        cmap = ListedColormap(["#FF6B6B", "#FFA726", "#4CAF50"])

        fig, ax = plt.subplots(figsize=(26, 11), dpi=130)
        dark_fig(fig, ax)
        ax.imshow(classified.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(classified.columns)))
        ax.set_xticklabels(classified.columns, rotation=45, ha="right",
                           fontsize=15, fontweight="bold", color="#C0C0E0")
        ax.set_yticks(range(len(classified.index)))
        ax.set_yticklabels(classified.index, fontsize=18, fontweight="bold", color="#C0C0E0")
        for i in range(classified.shape[0]):
            for j in range(classified.shape[1]):
                val = activity_T.iloc[i, j]
                ax.text(j, i, f"{val:,.0f}", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
        ax.set_title("Activity Classification (Red = Low, Orange = Mid, Green = High)",
                     fontsize=18, fontweight="bold", pad=16, color="#E0E0FF")
        plt.tight_layout()
        st.pyplot(fig)
        st.subheader("Activity Summary Table")
        st.dataframe(activity.round(1))
        conclusion(
            "<b>Green</b> cells highlight consistently-active patients; <b>red</b> cells "
            "show low-activity patients. The gap between groups is substantial — likely "
            "a key driver of glucose variability."
        )

    # ── Q4 Sleep by Gender & Race ──────────────────────────────────────────────
    with d_tabs[4]:
        page_title("Sleep Quality by Gender & Race")
        gender_avg = sleep_df.groupby("Gender")["Sleep Quality (1-10)"].mean()
        race_avg   = sleep_df.groupby("Race")["Sleep Quality (1-10)"].mean()

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=120)
        dark_fig(fig, axes)
        sns.barplot(x=gender_avg.index, y=gender_avg.values,
                    hue=gender_avg.index, palette="Set2",
                    legend=False, ax=axes[0])
        axes[0].set_title("Average Sleep Quality by Gender",
                          fontsize=16, fontweight="bold", pad=12, color="#E0E0FF")
        axes[0].set_ylabel("Avg Sleep Quality", fontsize=13, color="#C0C0E0")
        axes[0].tick_params(axis="both", labelsize=12, colors="#C0C0E0")
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.1f}",
                             (p.get_x() + p.get_width() / 2, p.get_height()),
                             ha="center", va="bottom", fontsize=13,
                             fontweight="bold", color="#E0E0FF")

        pie_colors = sns.color_palette("Set3", n_colors=len(race_avg))
        axes[1].pie(race_avg.values, labels=race_avg.index,
                    autopct="%1.1f%%", startangle=90, colors=pie_colors,
                    textprops={"fontsize": 13, "fontweight": "bold", "color": "#E0E0FF"})
        axes[1].set_title("Sleep Quality Distribution by Race",
                          fontsize=16, fontweight="bold", pad=12, color="#E0E0FF")
        plt.tight_layout()
        st.pyplot(fig)
        conclusion(
            "Sleep quality is fairly evenly distributed across races. No group dominates "
            "or lags dramatically — differences are <b>small but visible</b>."
        )

    # ── Q5 Heart Rate ──────────────────────────────────────────────────────────
    with d_tabs[5]:
        page_title("Heart Rate Statistics — Minimum, Maximum, Mean & Median")
        hr_stats = df["Heart_Rate"].agg(["min", "max", "mean", "median"]).round(2)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min HR",    f"{hr_stats['min']} bpm")
        c2.metric("Max HR",    f"{hr_stats['max']} bpm")
        c3.metric("Mean HR",   f"{hr_stats['mean']} bpm")
        c4.metric("Median HR", f"{hr_stats['median']} bpm")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        dark_fig(fig, axes)
        sns.histplot(df["Heart_Rate"], kde=True, bins=40, color="#7B2FF7", ax=axes[0])
        axes[0].set_title("Heart Rate Distribution", fontsize=12, color="#E0E0FF")
        axes[0].set_xlabel("Heart Rate", color="#C0C0E0")
        axes[0].tick_params(colors="#C0C0E0")
        sns.boxplot(x=df["Heart_Rate"], color="#E040FB", ax=axes[1])
        axes[1].set_title("Heart Rate Boxplot", fontsize=12, color="#E0E0FF")
        axes[1].tick_params(colors="#C0C0E0")
        plt.tight_layout()
        st.pyplot(fig)
        conclusion(
            "Min, max, mean, and median HR values reveal <b>recovery quality</b>, "
            "<b>peak strain</b>, and <b>overall cardiovascular load</b> — the foundation "
            "for deeper HR analysis."
        )


# ============================================================
# TAB 3 — DIAGNOSTIC  (inner sub-tabs)
# ============================================================
with tab3:
    diag_tabs = st.tabs([
        "Q1 · Sleep vs Glucose",
        "Q2 · Carbs / Insulin / Activity",
        "Q3 · Steps vs Variability",
    ])

    # ── Q1 Sleep vs Glucose ────────────────────────────────────────────────────
    with diag_tabs[0]:
        page_title("Effect of Poor Sleep Quality on Glucose Levels")
        p = (df.groupby("Patient_Id")
               .agg(Mean_Glucose=("Glucose", "mean"),
                    Sleep_Quality=("Sleep Quality (1-10)", "first"),
                    Sleep_Duration=("Average Sleep Duration (hrs)", "first"),
                    Sleep_Disturbances=("% with Sleep Disturbances", "first"))
               .reset_index())
        def sleep_cat(s):
            return "Poor" if s <= 4 else ("Average" if s <= 7 else "Good")
        p["Sleep_Group"] = p["Sleep_Quality"].apply(sleep_cat)

        pearson_r,  pearson_p  = stats.pearsonr(p["Sleep_Quality"],  p["Mean_Glucose"])
        spearman_r, spearman_p = stats.spearmanr(p["Sleep_Quality"], p["Mean_Glucose"])

        c1, c2 = st.columns(2)
        c1.metric("Pearson r",  f"{pearson_r:+.3f}",  f"p = {pearson_p:.4f}")
        c2.metric("Spearman ρ", f"{spearman_r:+.3f}", f"p = {spearman_p:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        dark_fig(fig, axes)
        sns.boxplot(data=p, x="Sleep_Group", y="Mean_Glucose",
                    order=["Poor", "Average", "Good"],
                    hue="Sleep_Group", palette="coolwarm_r",
                    legend=False, ax=axes[0])
        axes[0].set_title("Mean Glucose by Sleep Quality Group",
                          fontsize=12, color="#E0E0FF")
        axes[0].tick_params(colors="#C0C0E0")
        axes[0].set_xlabel("Sleep Group", color="#C0C0E0")
        axes[0].set_ylabel("Mean Glucose", color="#C0C0E0")

        sns.regplot(data=p, x="Sleep_Quality", y="Mean_Glucose",
                    scatter_kws={"s": 60, "color": "#7B2FF7"},
                    line_kws={"color": "#FF4C4C"}, ax=axes[1])
        axes[1].set_title("Sleep Quality vs Mean Glucose",
                          fontsize=12, color="#E0E0FF")
        axes[1].tick_params(colors="#C0C0E0")
        axes[1].set_xlabel("Sleep Quality", color="#C0C0E0")
        axes[1].set_ylabel("Mean Glucose",  color="#C0C0E0")
        plt.tight_layout()
        st.pyplot(fig)
        conclusion(
            "Poor sleep does <b>NOT cause an increase in glucose</b> in this dataset "
            "(r ≈ 0.05, p > 0.8). No patient falls into the Poor sleep category, "
            "limiting the test."
        )

    # ── Q2 Carbs / Insulin / Activity ──────────────────────────────────────────
    with diag_tabs[1]:
        page_title("Combined Influence of Carbs, Insulin & Physical Activity on Glucose Spikes")
        features = ["Glucose", "Carb_Input", "Basal_Rate", "Bolus_Volume_Delivered", "Steps"]
        corr = df[features].corr()

        fig, ax = plt.subplots(figsize=(8, 5))
        dark_fig(fig, ax)
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax,
                    annot_kws={"color": "white"},
                    linecolor="#3A3A5A", linewidths=0.5)
        ax.set_title("Correlation Between Glucose, Carbs, Insulin, and Steps",
                     fontsize=12, color="#E0E0FF")
        ax.tick_params(colors="#C0C0E0")
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

    # ── Q3 Steps vs Variability ────────────────────────────────────────────────
    with diag_tabs[2]:
        page_title("Previous-Day Steps and Glucose Variability — Target Daily Step Count")
        daily = (df.groupby(["Patient_Id", "Date"])
                   .agg(Daily_Steps=("Steps",   "sum"),
                        Daily_Glucose_Mean=("Glucose", "mean"),
                        Daily_Glucose_SD=("Glucose",   "std"))
                   .reset_index())
        daily["Daily_Glucose_CV"] = (daily["Daily_Glucose_SD"] / daily["Daily_Glucose_Mean"] * 100)
        daily["Prev_Day_Steps"]   = daily.groupby("Patient_Id")["Daily_Steps"].shift(1)
        daily = daily.dropna()

        pearson_sd = daily["Prev_Day_Steps"].corr(daily["Daily_Glucose_SD"])
        pearson_cv = daily["Prev_Day_Steps"].corr(daily["Daily_Glucose_CV"])
        c1, c2 = st.columns(2)
        c1.metric("Prev-Day Steps vs SD", f"{pearson_sd:+.3f}")
        c2.metric("Prev-Day Steps vs CV", f"{pearson_cv:+.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        dark_fig(fig, axes)
        sns.regplot(data=daily, x="Prev_Day_Steps", y="Daily_Glucose_SD",
                    scatter_kws={"s": 20, "alpha": 0.4, "color": "#7B2FF7"},
                    line_kws={"color": "#FF4C4C"}, ax=axes[0])
        axes[0].set_title("Prev Day Steps vs Glucose SD", fontsize=12, color="#E0E0FF")
        axes[0].tick_params(colors="#C0C0E0")
        axes[0].set_xlabel("Prev Day Steps", color="#C0C0E0")
        axes[0].set_ylabel("Glucose SD",     color="#C0C0E0")

        sns.regplot(data=daily, x="Prev_Day_Steps", y="Daily_Glucose_CV",
                    scatter_kws={"s": 20, "alpha": 0.4, "color": "#2ECC71"},
                    line_kws={"color": "#FF4C4C"}, ax=axes[1])
        axes[1].set_title("Prev Day Steps vs Glucose CV (%)", fontsize=12, color="#E0E0FF")
        axes[1].tick_params(colors="#C0C0E0")
        axes[1].set_xlabel("Prev Day Steps", color="#C0C0E0")
        axes[1].set_ylabel("Glucose CV (%)", color="#C0C0E0")
        plt.tight_layout()
        st.pyplot(fig)

        target = daily.loc[daily["Daily_Glucose_CV"] < 36, "Prev_Day_Steps"].mean()
        conclusion(
            f"Patients with stable glucose (CV < 36%) averaged about "
            f"<b>{target:,.0f} steps</b> the previous day. Aim for this range to "
            "support next-day glucose stability."
        )


# ============================================================
# TAB 4 — PRESCRIPTIVE  (inner sub-tabs)
# ============================================================
with tab4:
    pres_tabs = st.tabs([
        "Q1 · Dietary Intervention",
        "Q2 · HR & Glucose Rising",
    ])

    # ── Q1 Risk Classification ─────────────────────────────────────────────────
    with pres_tabs[0]:
        page_title("Patients Needing Immediate Dietary Intervention Based on Carb-Glucose Response")
        risk = (df.groupby("Patient_Id")
                  .agg(Mean_Carb=("Carb_Input", "mean"),
                       Mean_Glucose=("Glucose",   "mean"),
                       Hyper_pct=("Glucose", lambda g: (g > 180).mean() * 100))
                  .reset_index())

        def risk_class(r):
            if r["Mean_Glucose"] > 180 or r["Hyper_pct"] > 40: return "HIGH RISK"
            if r["Mean_Glucose"] > 154 or r["Hyper_pct"] > 25: return "MODERATE"
            return "LOW RISK"
        risk["Risk_Level"] = risk.apply(risk_class, axis=1)
        summary = risk["Risk_Level"].value_counts()

        c1, c2, c3 = st.columns(3)
        c1.metric("High-Risk Patients",     summary.get("HIGH RISK", 0))
        c2.metric("Moderate-Risk Patients", summary.get("MODERATE",  0))
        c3.metric("Low-Risk Patients",      summary.get("LOW RISK",  0))

        def color_risk(val):
            if val == "HIGH RISK": return "background-color: #FF6B6B; color: white"
            if val == "MODERATE":  return "background-color: #FFA726"
            return "background-color: #4CAF50; color: white"

        st.subheader("Patient Risk Table")
        st.dataframe(risk.sort_values("Hyper_pct", ascending=False)
                         .style.map(color_risk, subset=["Risk_Level"])
                         .format({"Mean_Carb":    "{:.1f}",
                                  "Mean_Glucose": "{:.1f}",
                                  "Hyper_pct":    "{:.1f}"}))

        st.subheader("Risk Distribution — Sunburst View")
        if PLOTLY_OK:
            sunburst_df = risk.copy()
            sunburst_df["All Patients"] = "All Patients"
            color_map = {"HIGH RISK": "#FF6B6B", "MODERATE": "#FFA726", "LOW RISK": "#4CAF50"}
            fig_sb = px.sunburst(
                sunburst_df,
                path=["All Patients", "Risk_Level", "Patient_Id"],
                values="Hyper_pct",
                color="Risk_Level",
                color_discrete_map=color_map,
                title="Patients Grouped by Risk Level (size = Hyperglycemia %)",
            )
            fig_sb.update_layout(
                height=600,
                margin=dict(t=40, l=0, r=0, b=0),
                paper_bgcolor="#0F0F1A",
                plot_bgcolor="#0F0F1A",
                font=dict(color="#C0C0E0"),
            )
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

    # ── Q2 HR & Glucose Rising ─────────────────────────────────────────────────
    with pres_tabs[1]:
        page_title("Recommended Actions When Heart Rate & Glucose Rise Together")
        df_q2 = df.copy().sort_values(["Patient_Id", "Time"])
        df_q2["HR_change"]  = df_q2.groupby("Patient_Id")["Heart_Rate"].diff()
        df_q2["Glu_change"] = df_q2.groupby("Patient_Id")["Glucose"].diff()
        df_q2["Both_Rising"] = ((df_q2["HR_change"]  > 0) &
                                (df_q2["Glu_change"] > 0)).astype(int)
        pct_both = df_q2["Both_Rising"].mean() * 100
        st.metric("% of readings where HR & Glucose rise together", f"{pct_both:.1f}%")

        sample = df_q2.sample(min(2000, len(df_q2)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 6))
        dark_fig(fig, ax)
        sns.scatterplot(data=sample, x="HR_change", y="Glu_change",
                        hue="Both_Rising",
                        palette={0: "#3A3A5A", 1: "#FF4C4C"},
                        alpha=0.6, ax=ax)
        ax.axhline(0, color="#C0C0E0", linewidth=0.5)
        ax.axvline(0, color="#C0C0E0", linewidth=0.5)
        ax.set_title("Δ Heart Rate vs Δ Glucose (red = both rising = stress event)",
                     fontsize=12, color="#E0E0FF")
        ax.set_xlabel("Δ Heart Rate", color="#C0C0E0")
        ax.set_ylabel("Δ Glucose",    color="#C0C0E0")
        ax.tick_params(colors="#C0C0E0")
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor("#1A1A2E")
            for text in legend.get_texts():
                text.set_color("#C0C0E0")
        plt.tight_layout()
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
# TAB 5 — PREDICTIVE
# ============================================================
with tab5:
    page_title("Predicting Hypoglycemia / Hyperglycemia Zones 30 Minutes Ahead")

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
        from sklearn.metrics import accuracy_score, confusion_matrix

        df_p = df.sort_values(["Patient_Id", "Time"]).copy()
        df_p["Future_Glucose"] = df_p.groupby("Patient_Id")["Glucose"].shift(-6)
        df_p["Future_Zone"]    = df_p["Future_Glucose"].apply(
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
            acc    = accuracy_score(y_test, y_pred)

        c1, c2 = st.columns(2)
        c1.metric("Test Accuracy",            f"{acc*100:.1f}%")
        c2.metric("Baseline (random chance)", "33%")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=["Hypo", "Normal", "Hyper"])
        fig, ax = plt.subplots(figsize=(7, 5))
        dark_fig(fig, ax)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Hypo", "Normal", "Hyper"],
                    yticklabels=["Hypo", "Normal", "Hyper"],
                    ax=ax,
                    annot_kws={"color": "#0F0F1A"},
                    linecolor="#3A3A5A", linewidths=0.5)
        ax.set_title("Confusion Matrix", fontsize=12, color="#E0E0FF")
        ax.set_xlabel("Predicted", color="#C0C0E0")
        ax.set_ylabel("Actual",    color="#C0C0E0")
        ax.tick_params(colors="#C0C0E0")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Feature Importance")
        fi = pd.DataFrame({"Feature": features,
                           "Importance": clf.feature_importances_}
                         ).sort_values("Importance", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        dark_fig(fig2, ax2)
        sns.barplot(data=fi, x="Importance", y="Feature",
                    hue="Feature", palette="viridis",
                    legend=False, ax=ax2)
        ax2.set_title("Which features most influence the prediction?",
                      fontsize=12, color="#E0E0FF")
        ax2.set_xlabel("Importance", color="#C0C0E0")
        ax2.set_ylabel("Feature",    color="#C0C0E0")
        ax2.tick_params(colors="#C0C0E0")
        plt.tight_layout()
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
# TAB 6 — RECOMMENDATIONS
# ============================================================
with tab6:
    page_title("Recommendations Panel — Actionable Interventions")
    st.markdown(
        '<div style="color:#9090B0; font-size:0.95rem; margin-bottom:1.5rem;">'
        "Converting all data-driven insights from Descriptive, Diagnostic, "
        "Prescriptive, and Predictive analytics into clear clinical actions."
        "</div>",
        unsafe_allow_html=True,
    )

    recs = [
        {
            "icon": "😴", "title": "Sleep Quality → Glucose Control",
            "color": "#7B2FF7", "bg": "#1A1A2E",
            "finding": (
                "Poor sleep quality (score ≤ 4/10) is associated with elevated glucose "
                "and reduced insulin sensitivity in the literature, though the current "
                "dataset shows no patient in the Poor category — limiting direct testing."
            ),
            "action": (
                "Patients with sleep score &lt; 7/10 should maintain a consistent "
                "bedtime, reduce screen exposure after 9 PM, and schedule dinner at "
                "least 2–3 hours before sleeping."
            ),
            "threshold": "🎯 Target: Sleep score ≥ 7 / 10",
        },
        {
            "icon": "🥗", "title": "Dietary Intervention for High-Risk Patients",
            "color": "#FF4C4C", "bg": "#2A1A1A",
            "finding": (
                "Patients classified as HIGH RISK (Mean Glucose > 180 or Hyperglycemia % > 40) "
                "show strong carbohydrate-driven glucose excursions and require immediate "
                "dietary restructuring."
            ),
            "action": (
                "Provide personalised meal plans targeting &lt; 50–60 g carbs per meal. "
                "Prioritise low-glycaemic foods, consistent meal timing, and a mandatory "
                "10-minute post-meal walk to blunt spikes."
            ),
            "threshold": "🎯 Target: Hyperglycemia % below 40 per patient",
        },
        {
            "icon": "🚶", "title": "Daily Step Target for Glucose Stability",
            "color": "#2ECC71", "bg": "#1A2A1A",
            "finding": (
                "Patients with stable glucose (CV &lt; 36%) averaged more steps the "
                "previous day. While steps do not instantly reduce variability, sustained "
                "activity improves insulin sensitivity over weeks."
            ),
            "action": (
                "Set a minimum daily target of 7,000–8,000 steps. "
                "High-variability patients should aim for 8,000–10,000 steps to achieve "
                "long-term metabolic benefit."
            ),
            "threshold": "🎯 Target: ≥ 7,000 steps / day",
        },
        {
            "icon": "❤️", "title": "Dual HR + Glucose Spike Response",
            "color": "#FF9F1C", "bg": "#2A1A0A",
            "finding": (
                "Simultaneous rises in HR and glucose signal a dual-system stress event "
                "(autonomic + metabolic). This occurred in a measurable percentage of all "
                "readings and represents one of the highest-risk patterns in the dataset."
            ),
            "action": (
                "When both signals rise together: pause and rest 10–15 minutes, hydrate "
                "immediately, avoid additional bolus correction until the cause is "
                "identified, then monitor for a further 30 minutes. Contact clinician if "
                "both continue to rise."
            ),
            "threshold": "🎯 Trigger: HR rising AND Glucose rising simultaneously",
        },
        {
            "icon": "💉", "title": "Insulin Timing & Bolus Optimisation",
            "color": "#00D4AA", "bg": "#0A1A1A",
            "finding": (
                "Carbohydrate intake is the strongest driver of glucose excursions. "
                "Bolus insulin partially mitigates spikes, but timing mismatches between "
                "meal and bolus delivery produce avoidable hyperglycaemia."
            ),
            "action": (
                "Clinicians should review bolus timing for patients with frequent "
                "post-meal hyperglycaemia. Pre-meal bolus (10–15 min before eating) "
                "combined with a post-meal 10-minute walk significantly reduces peak "
                "glucose levels."
            ),
            "threshold": "🎯 Target: Bolus administered ≤ 15 min before meal",
        },
    ]

    for rec in recs:
        st.markdown(f"""
        <div style="
            background: {rec['bg']};
            border: 1px solid {rec['color']}55;
            border-left: 5px solid {rec['color']};
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 14px;
        ">
          <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
            <span style="font-size:2rem;">{rec['icon']}</span>
            <span style="font-size:1.1rem; font-weight:800; color:{rec['color']};">
              {rec['title']}
            </span>
          </div>
          <div style="color:#C0C0E0; font-size:0.9rem; margin-bottom:8px;">
            <strong style="color:#E0E0FF;">📌 Finding:</strong> {rec['finding']}
          </div>
          <div style="color:#C0C0E0; font-size:0.9rem; margin-bottom:8px;">
            <strong style="color:#E0E0FF;">✅ Action:</strong> {rec['action']}
          </div>
          <div style="
              background:{rec['color']}22;
              border-radius:8px;
              padding:8px 14px;
              font-size:0.88rem;
              color:{rec['color']};
              font-weight:600;
          ">
            {rec['threshold']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2A2A4A; margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.subheader("📋 Recommendation Summary Matrix")

    summary_df = pd.DataFrame({
        "Domain":           ["Sleep", "Diet", "Physical Activity", "Stress / HR", "Insulin Timing"],
        "Risk Signal":      [
            "Sleep score < 7/10",
            "Mean Glucose > 180 or Hyper% > 40",
            "Steps < 7,000 / day",
            "HR rising AND Glucose rising",
            "Post-meal glucose > 180 mg/dL",
        ],
        "Recommended Action": [
            "Consistent bedtime, cut screen time, earlier dinner",
            "Personalised low-GI meal plan, < 50–60 g carbs/meal",
            "Gradual increase to 8,000–10,000 steps/day",
            "Rest, hydrate, no extra bolus — then monitor",
            "Pre-meal bolus + 10-min post-meal walk",
        ],
        "Expected Outcome": [
            "Lower avg glucose over time",
            "Reduce post-meal spikes by 20–30%",
            "Improve insulin sensitivity in 2–4 weeks",
            "Reduce dual-spike frequency",
            "Reduce time in hyperglycaemia",
        ],
        "Priority": ["Medium", "High", "Medium", "High", "High"],
    })

    def color_priority(val):
        if val == "High":   return "background-color:#FF4C4C33; color:#FF8080"
        if val == "Medium": return "background-color:#FF9F1C33; color:#FFBF60"
        return ""

    st.dataframe(
        summary_df.style.map(color_priority, subset=["Priority"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<hr style='border-color:#2A2A4A; margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1A0A2E, #0A1A2E);
        border: 1px solid #3A2A5A;
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 8px;
    ">
      <div style="font-size:1.2rem; font-weight:800; color:#E0E0FF; margin-bottom:14px;">
        🏁 Final Conclusion
      </div>
      <div style="color:#C0C0E0; font-size:0.95rem; line-height:1.8;">
        This dashboard integrates
        <strong style="color:#7B2FF7;">Descriptive</strong>,
        <strong style="color:#E040FB;">Diagnostic</strong>,
        <strong style="color:#00D4AA;">Prescriptive</strong>, and
        <strong style="color:#FF9F1C;">Predictive</strong>
        analytics to provide a comprehensive understanding of glucose behaviour
        in Type-1 Diabetes Mellitus patients.<br><br>

        The <b>Descriptive</b> section summarises patient lifestyle and physiological
        patterns — who spends the most time out of range, how activity and sleep
        are distributed, and what the key cardiovascular indicators show.<br><br>

        The <b>Diagnostic</b> section investigates the <em>why</em> behind glucose
        variability — isolating the influence of sleep quality, carbohydrate intake,
        insulin dosing, and physical activity on glucose stability.<br><br>

        The <b>Prescriptive</b> section translates those findings into patient-level
        risk classifications and concrete intervention recommendations — from dietary
        changes to real-time HR + glucose spike response protocols.<br><br>

        The <b>Predictive</b> section uses a Random Forest classifier to provide a
        <strong style="color:#FF9F1C;">30-minute advance warning</strong> of dangerous
        glucose zones — giving clinicians and patients a critical window to act before
        a crisis occurs.<br><br>

        Together, these four layers of analysis support
        <strong style="color:#2ECC71;">data-driven decision-making</strong> and help
        optimise diabetes management at the individual patient level.
      </div>
    </div>
    """, unsafe_allow_html=True)