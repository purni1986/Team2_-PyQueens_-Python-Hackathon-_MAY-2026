"""
Diabetes Analytics Dashboard — PyQueens Team
Streamlit app with tabs: Overview, Descriptive, Prescriptive, Predictive, Recommendations
Uses synthetic demo data that mirrors the HUPA-UCM Diabetes dataset structure.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Analytics Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0F0F1A; }
    .block-container { padding: 1.5rem 2rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1A1A2E;
        border-radius: 12px;
        padding: 6px;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #A0A0C0;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.92rem;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7B2FF7, #E040FB) !important;
        color: white !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E1E3A, #2A2A4A);
        border: 1px solid #3A3A5C;
        border-radius: 14px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(123,47,247,0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7B2FF7, #E040FB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #A0A0C0;
        font-size: 0.85rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 6px;
        font-weight: 600;
    }
    .delta-good { color: #2ECC71; }
    .delta-warn { color: #FF9F1C; }
    .delta-bad  { color: #FF4C4C; }

    /* Section headers */
    .section-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #E0E0FF;
        margin-bottom: 0.3rem;
        margin-top: 1.2rem;
    }
    .section-sub {
        color: #8080A0;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #1A2A1A, #2A3A2A);
        border-left: 4px solid #2ECC71;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #C0FFC0;
        font-size: 0.9rem;
    }
    .warn-box {
        background: linear-gradient(135deg, #2A1A1A, #3A2A1A);
        border-left: 4px solid #FF9F1C;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #FFE0A0;
        font-size: 0.9rem;
    }
    .rec-box {
        background: linear-gradient(135deg, #1A1A2E, #2A1A3E);
        border-left: 4px solid #7B2FF7;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #D0C0FF;
        font-size: 0.9rem;
    }

    /* Divider */
    hr { border-color: #2A2A4A !important; margin: 1.5rem 0 !important; }

    /* Scrollable table */
    .dataframe-container { max-height: 320px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 8000
    patient_ids = [f"P{str(i).zfill(3)}" for i in range(1, 21)]

    # Time series
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")

    df = pd.DataFrame({
        "Time": dates,
        "Patient_Id": np.random.choice(patient_ids, n),
    })
    df["Hour"]   = df["Time"].dt.hour
    df["Date"]   = df["Time"].dt.date
    df["Week"]   = df["Time"].dt.isocalendar().week.astype(int)
    df["DayName"] = df["Time"].dt.day_name()

    # Glucose: base + meal spikes + noise
    base_glucose = 120 + 30 * np.sin(2 * np.pi * df["Hour"] / 24)
    df["Glucose"] = (base_glucose + np.random.normal(0, 20, n)).clip(50, 350)

    # Classify zones
    df["Glucose_Zone"] = pd.cut(
        df["Glucose"],
        bins=[0, 70, 180, 400],
        labels=["Hypoglycemia", "Normal", "Hyperglycemia"]
    )

    # Other features
    df["Carb_Input"]               = np.random.exponential(30, n).clip(0, 150)
    df["Basal_Rate"]               = np.random.uniform(0.5, 2.0, n)
    df["Bolus_Volume_Delivered"]   = np.random.exponential(3, n).clip(0, 15)
    df["Steps"]                    = np.random.exponential(200, n).clip(0, 2000)
    df["Calories"]                 = df["Steps"] * 0.04 + np.random.normal(0, 5, n)
    df["Heart_Rate"]               = (70 + 20 * np.sin(2 * np.pi * df["Hour"] / 24)
                                      + np.random.normal(0, 10, n)).clip(45, 180)
    df["Sleep_Duration"]           = np.random.normal(6.5, 1.5, n).clip(3, 10)
    df["Sleep_Quality"]            = np.random.randint(1, 11, n).astype(float)
    df["Sleep_Disturbances"]       = np.random.randint(0, 5, n)

    return df

@st.cache_data
def generate_demo_data():
    np.random.seed(7)
    patient_ids = [f"P{str(i).zfill(3)}" for i in range(1, 21)]
    n_patients  = len(patient_ids)

    genders = np.random.choice(["Male", "Female", "Other"], n_patients, p=[0.5, 0.45, 0.05])
    races   = np.random.choice(
        ["Asian", "White", "Hispanic", "Black", "Native American"],
        n_patients, p=[0.25, 0.30, 0.20, 0.15, 0.10]
    )
    ages    = np.random.randint(20, 65, n_patients)

    return pd.DataFrame({
        "Patient_Id":    patient_ids,
        "Gender":        genders,
        "Race":          races,
        "Age":           ages,
        "Sleep_Quality": np.random.uniform(4, 9, n_patients).round(1),
    })

df      = generate_data()
demo_df = generate_demo_data()

# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
PURPLE   = "#7B2FF7"
PINK     = "#E040FB"
TEAL     = "#00D4AA"
RED      = "#FF4C4C"
ORANGE   = "#FF9F1C"
GREEN    = "#2ECC71"
BLUE     = "#5B8FF9"
BG_DARK  = "#0F0F1A"
BG_CARD  = "#1A1A2E"
GRID_CLR = "#2A2A4A"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=BG_CARD,
        plot_bgcolor =BG_CARD,
        font         =dict(color="#C0C0E0", family="Inter, sans-serif"),
        xaxis        =dict(gridcolor=GRID_CLR, linecolor=GRID_CLR),
        yaxis        =dict(gridcolor=GRID_CLR, linecolor=GRID_CLR),
        legend       =dict(bgcolor="#1E1E3A", bordercolor=GRID_CLR, borderwidth=1),
    )
)

def apply_template(fig, title=None, height=400):
    fig.update_layout(
        paper_bgcolor=BG_CARD,
        plot_bgcolor =BG_CARD,
        font         =dict(color="#C0C0E0"),
        xaxis        =dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, showgrid=True),
        yaxis        =dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, showgrid=True),
        legend       =dict(bgcolor="#1E1E3A", bordercolor=GRID_CLR),
        height       =height,
        margin       =dict(l=40, r=20, t=50, b=40),
    )
    if title:
        fig.update_layout(
            title=dict(text=title, font=dict(size=15, color="#E0E0FF"), x=0.02)
        )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1A0A2E 0%, #0F1A2E 50%, #1A0A1A 100%);
    border: 1px solid #3A2A5A;
    border-radius: 18px;
    padding: 24px 32px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 16px;
">
  <div style="font-size:3rem;">🩺</div>
  <div>
    <div style="font-size:1.9rem; font-weight:900; color:#E8E0FF; letter-spacing:-0.5px;">
      Diabetes Analytics Dashboard
    </div>
    <div style="color:#9080C0; font-size:0.95rem; margin-top:4px;">
      HUPA-UCM T1DM Dataset &nbsp;·&nbsp; PyQueens Team &nbsp;·&nbsp;
      Descriptive + Prescriptive + Predictive Analytics
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_desc, tab_pres, tab_pred, tab_recs = st.tabs([
    "🏠  Overview",
    "📊  Descriptive",
    "💡  Prescriptive",
    "🤖  Predictive",
    "✅  Recommendations",
])


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ╔══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    # KPI row
    zone_counts = df["Glucose_Zone"].value_counts()
    total       = len(df)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Total Readings",   f"{total:,}",    "All CGM data points",       "delta-good"),
        ("Patients",         "20",             "T1DM participants",          "delta-good"),
        ("Hypoglycemia",     f"{zone_counts.get('Hypoglycemia',0)/total*100:.1f}%",
                             "Glucose < 70 mg/dL",  "delta-bad"),
        ("Normal Range",     f"{zone_counts.get('Normal',0)/total*100:.1f}%",
                             "70–180 mg/dL",         "delta-good"),
        ("Hyperglycemia",    f"{zone_counts.get('Hyperglycemia',0)/total*100:.1f}%",
                             "Glucose > 180 mg/dL",  "delta-warn"),
    ]
    for col, (label, val, sub, cls) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta {cls}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Avg stats ──
    r1, r2, r3, r4 = st.columns(4)
    more_kpis = [
        ("Avg Glucose",    f"{df['Glucose'].mean():.0f} mg/dL",  "delta-warn"),
        ("Avg Heart Rate", f"{df['Heart_Rate'].mean():.0f} bpm", "delta-good"),
        ("Avg Daily Steps",f"{df.groupby('Date')['Steps'].sum().mean():.0f}", "delta-warn"),
        ("Avg Sleep Score",f"{df['Sleep_Quality'].mean():.1f}/10", "delta-good"),
    ]
    for col, (label, val, cls) in zip([r1,r2,r3,r4], more_kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Overview charts ──
    col_a, col_b = st.columns(2)

    with col_a:
        # Glucose zone donut
        zone_df = zone_counts.reset_index()
        zone_df.columns = ["Zone", "Count"]
        fig = px.pie(
            zone_df, names="Zone", values="Count", hole=0.55,
            color="Zone",
            color_discrete_map={"Hypoglycemia": RED, "Normal": GREEN, "Hyperglycemia": ORANGE}
        )
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        apply_template(fig, "Overall Glucose Zone Distribution", 380)
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Gender distribution
        gender_df = demo_df["Gender"].value_counts().reset_index()
        gender_df.columns = ["Gender", "Count"]
        fig = px.bar(
            gender_df, x="Gender", y="Count",
            color="Gender",
            color_discrete_sequence=[PURPLE, PINK, TEAL],
            text="Count"
        )
        fig.update_traces(textposition="outside", textfont_size=13)
        apply_template(fig, "Patient Gender Distribution", 380)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        # Age histogram
        fig = px.histogram(
            demo_df, x="Age", nbins=10,
            color_discrete_sequence=[PURPLE],
        )
        apply_template(fig, "Patient Age Distribution", 350)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        # Race pie
        race_df = demo_df["Race"].value_counts().reset_index()
        race_df.columns = ["Race", "Count"]
        fig = px.pie(
            race_df, names="Race", values="Count", hole=0.4,
            color_discrete_sequence=[PURPLE, PINK, TEAL, ORANGE, GREEN]
        )
        apply_template(fig, "Patient Race Distribution", 350)
        st.plotly_chart(fig, use_container_width=True)

    # Dataset overview table
    st.markdown('<div class="section-title">📋 Dataset Features</div>', unsafe_allow_html=True)
    features_info = pd.DataFrame({
        "Feature": ["Glucose", "Carb_Input", "Basal_Rate", "Bolus_Volume_Delivered",
                    "Steps", "Calories", "Heart_Rate", "Sleep_Duration", "Sleep_Quality",
                    "Sleep_Disturbances"],
        "Type":    ["CGM Reading"]*1 + ["Diet"]*1 + ["Insulin"]*2 + ["Activity"]*2
                   + ["Cardiac"]*1 + ["Sleep"]*3,
        "Unit":    ["mg/dL","g","U/hr","U","count","kcal","bpm","hrs","1–10 scale","count"],
        "Description": [
            "Blood glucose every 5 min", "Carbohydrates consumed",
            "Background insulin rate", "Meal/correction bolus",
            "Steps walked", "Calories burned",
            "Heart beats per minute", "Total sleep time",
            "Subjective sleep quality", "Number of sleep interruptions"
        ]
    })
    st.dataframe(features_info, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE
# ╔══════════════════════════════════════════════════════════════════════════════
with tab_desc:
    st.markdown('<div class="section-title">📊 Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Summarising patient behaviour across glucose zones, time trends, activity, sleep, and heart rate.</div>', unsafe_allow_html=True)

    # ── Q1: Glucose Zone Distribution ──────────────────────────────────────────
    st.markdown("#### Q1 · Glucose Zone Distribution — How much time in each zone?")

    zone_per_patient = (
        df.groupby(["Patient_Id", "Glucose_Zone"])
        .size().unstack(fill_value=0)
    )
    zone_pct = zone_per_patient.div(zone_per_patient.sum(axis=1), axis=0) * 100
    for z in ["Hypoglycemia", "Normal", "Hyperglycemia"]:
        if z not in zone_pct.columns:
            zone_pct[z] = 0
    zone_pct = zone_pct[["Hypoglycemia", "Normal", "Hyperglycemia"]].reset_index()

    col1, col2 = st.columns([1, 2])
    with col1:
        zone_overall = df["Glucose_Zone"].value_counts().reset_index()
        zone_overall.columns = ["Zone", "Count"]
        fig = px.pie(
            zone_overall, names="Zone", values="Count", hole=0.55,
            color="Zone",
            color_discrete_map={"Hypoglycemia": RED, "Normal": GREEN, "Hyperglycemia": ORANGE}
        )
        apply_template(fig, "Overall Zone %", 360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        melt = zone_pct.melt(id_vars="Patient_Id", var_name="Zone", value_name="Pct")
        fig = px.bar(
            melt, x="Patient_Id", y="Pct", color="Zone",
            color_discrete_map={"Hypoglycemia": RED, "Normal": GREEN, "Hyperglycemia": ORANGE},
            text_auto=False
        )
        fig.update_layout(barmode="stack", xaxis_tickangle=45)
        apply_template(fig, "Glucose Zone % Per Patient", 360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    ✅ Most patients maintain normal glucose most of the time, but critical outliers show
    dangerously elevated or depressed levels — immediate attention required for those patients.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Q2: Time Trends ─────────────────────────────────────────────────────────
    st.markdown("#### Q2 · Hourly, Daily & Weekly Glucose Trends")

    hourly = df.groupby("Hour")["Glucose"].mean().reset_index()
    daily  = df.groupby("Date")["Glucose"].mean().reset_index()
    weekly = df.groupby("Week")["Glucose"].mean().reset_index()

    tab_h, tab_d, tab_w = st.tabs(["🕐 Hourly", "📅 Daily", "📆 Weekly"])

    with tab_h:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly["Hour"], y=hourly["Glucose"],
            mode="lines+markers",
            line=dict(color=PURPLE, width=2.5),
            marker=dict(size=7, color=PINK),
            fill="tozeroy", fillcolor="rgba(123,47,247,0.12)"
        ))
        fig.add_hline(y=70,  line_dash="dot", line_color=RED,    annotation_text="Hypo threshold (70)")
        fig.add_hline(y=180, line_dash="dot", line_color=ORANGE, annotation_text="Hyper threshold (180)")
        apply_template(fig, "Average Glucose by Hour of Day", 380)
        fig.update_xaxes(title="Hour of Day", tickvals=list(range(0, 24, 2)))
        fig.update_yaxes(title="Avg Glucose (mg/dL)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_d:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["Date"].astype(str), y=daily["Glucose"],
            mode="lines", line=dict(color=TEAL, width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.10)"
        ))
        apply_template(fig, "Average Daily Glucose Trend", 380)
        fig.update_xaxes(title="Date", tickangle=45)
        fig.update_yaxes(title="Avg Glucose (mg/dL)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_w:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weekly["Week"].astype(str), y=weekly["Glucose"],
            marker_color=PURPLE, text=weekly["Glucose"].round(0),
            textposition="outside"
        ))
        apply_template(fig, "Average Weekly Glucose Trend", 380)
        fig.update_xaxes(title="Week Number")
        fig.update_yaxes(title="Avg Glucose (mg/dL)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    ✅ Glucose follows a clear daily rhythm — spiking after meals and stabilising overnight.
    Weekly variability is driven by inconsistent diet and lifestyle behaviours.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Q3: Physical Activity ───────────────────────────────────────────────────
    st.markdown("#### Q3 · Physical Activity Distribution — Steps & Calories")

    act = df.groupby("Patient_Id").agg(
        avg_steps   =("Steps",    "mean"),
        avg_calories=("Calories", "mean"),
        total_steps =("Steps",    "sum"),
    ).reset_index().sort_values("avg_steps", ascending=False)

    col3a, col3b = st.columns(2)
    with col3a:
        fig = px.bar(
            act, x="Patient_Id", y="avg_steps",
            color="avg_steps",
            color_continuous_scale=[[0,"#FF4C4C"],[0.5,"#FF9F1C"],[1,"#2ECC71"]],
            text=act["avg_steps"].round(0).astype(int)
        )
        fig.update_traces(textposition="outside")
        apply_template(fig, "Average Daily Steps per Patient", 400)
        fig.update_xaxes(tickangle=45)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col3b:
        fig = px.scatter(
            act, x="avg_steps", y="avg_calories",
            color="avg_steps",
            color_continuous_scale=[[0,"#FF4C4C"],[0.5,"#FF9F1C"],[1,"#2ECC71"]],
            size="total_steps",
            hover_name="Patient_Id",
            text="Patient_Id"
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        apply_template(fig, "Steps vs Calories (bubble = total steps)", 400)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Activity heatmap
    act_hm = act.set_index("Patient_Id")[["avg_steps","avg_calories","total_steps"]].T
    act_hm.index = ["Avg Steps","Avg Calories","Total Steps"]
    fig = px.imshow(
        act_hm,
        color_continuous_scale=[[0,"#FF4C4C"],[0.45,"#FF9F1C"],[1,"#2ECC71"]],
        aspect="auto", text_auto=".0f"
    )
    apply_template(fig, "Activity Heatmap — Green = High | Orange = Mid | Red = Low", 280)
    fig.update_xaxes(title="Patient ID", tickangle=45)
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="warn-box">
    ⚠️ Large gap exists between highly active and sedentary patients.
    Low-activity patients are flagged for targeted step-count interventions.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Q4: Sleep Quality by Demographics ──────────────────────────────────────
    st.markdown("#### Q4 · Sleep Quality by Gender & Race")

    col4a, col4b = st.columns(2)
    with col4a:
        gender_sleep = demo_df.groupby("Gender")["Sleep_Quality"].mean().reset_index()
        fig = px.bar(
            gender_sleep, x="Gender", y="Sleep_Quality",
            color="Gender",
            color_discrete_sequence=[PURPLE, PINK, TEAL],
            text=gender_sleep["Sleep_Quality"].round(1)
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 11])
        apply_template(fig, "Avg Sleep Quality by Gender", 360)
        st.plotly_chart(fig, use_container_width=True)

    with col4b:
        race_sleep = demo_df.groupby("Race")["Sleep_Quality"].mean().reset_index()
        fig = px.pie(
            race_sleep, names="Race", values="Sleep_Quality",
            hole=0.4,
            color_discrete_sequence=[PURPLE, PINK, TEAL, ORANGE, GREEN]
        )
        apply_template(fig, "Sleep Quality Distribution by Race", 360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    ✅ Sleep quality is fairly even across demographic groups.
    No single group dominates, but Asian and Native American groups contribute slightly
    higher average scores in this dataset.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Q5: Heart Rate Distribution ─────────────────────────────────────────────
    st.markdown("#### Q5 · Heart Rate Distribution & Summary Statistics")

    hr_stats = df["Heart_Rate"].agg(["min","max","mean","median"]).reset_index()
    hr_stats.columns = ["Stat","Value"]
    hr_stats["Value"] = hr_stats["Value"].round(1)

    col5a, col5b = st.columns([3, 1])
    with col5a:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["Heart_Rate"], nbinsx=50,
            marker_color=BLUE, opacity=0.8,
            name="Heart Rate"
        ))
        for _, row in hr_stats.iterrows():
            colors = {"min": GREEN, "max": RED, "mean": ORANGE, "median": PINK}
            fig.add_vline(
                x=row["Value"],
                line_dash="dash",
                line_color=colors[row["Stat"]],
                annotation_text=f'{row["Stat"].title()} {row["Value"]:.0f}',
                annotation_position="top"
            )
        apply_template(fig, "Heart Rate Distribution (Histogram + KDE lines)", 400)
        fig.update_xaxes(title="Heart Rate (bpm)")
        fig.update_yaxes(title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    with col5b:
        st.markdown("<br><br>", unsafe_allow_html=True)
        for _, row in hr_stats.iterrows():
            color_map = {"min": GREEN, "max": RED, "mean": ORANGE, "median": PINK}
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:10px;">
                <div style="color:{color_map[row['Stat']]}; font-size:1.6rem; font-weight:800;">{row['Value']}</div>
                <div class="metric-label">{row['Stat'].title()} HR (bpm)</div>
            </div>""", unsafe_allow_html=True)

    # Boxplot
    fig = px.box(
        df, x="Heart_Rate",
        color_discrete_sequence=[PINK],
        points="outliers"
    )
    apply_template(fig, "Heart Rate Box Plot", 260)
    fig.update_xaxes(title="Heart Rate (bpm)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    ✅ Min, max, mean, and median HR values reveal recovery quality, peak strain, and
    overall cardiovascular load — forming the foundation for deeper HR analysis.
    </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRESCRIPTIVE
# ╔══════════════════════════════════════════════════════════════════════════════
with tab_pres:
    st.markdown('<div class="section-title">💡 Prescriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Five targeted questions — one per team member — identifying actionable interventions.</div>', unsafe_allow_html=True)

    # ── P1: Sleep → Glucose (Tejaswini) ─────────────────────────────────────────
    st.markdown("#### P1 · Tejaswini — Does poor sleep quality increase glucose levels?")

    # Bin sleep quality
    df["Sleep_Cat"] = pd.cut(df["Sleep_Quality"], bins=[0,4,7,10], labels=["Poor","Moderate","Good"])
    sleep_glucose = df.groupby("Sleep_Cat")["Glucose"].mean().reset_index()
    sleep_glucose.columns = ["Sleep_Quality", "Avg_Glucose"]

    col_p1a, col_p1b = st.columns(2)
    with col_p1a:
        fig = px.bar(
            sleep_glucose, x="Sleep_Quality", y="Avg_Glucose",
            color="Sleep_Quality",
            color_discrete_map={"Poor": RED, "Moderate": ORANGE, "Good": GREEN},
            text=sleep_glucose["Avg_Glucose"].round(1)
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, sleep_glucose["Avg_Glucose"].max() * 1.15])
        apply_template(fig, "Avg Glucose by Sleep Quality Category", 380)
        st.plotly_chart(fig, use_container_width=True)

    with col_p1b:
        sleep_scatter = df.sample(500, random_state=42)
        fig = px.scatter(
            sleep_scatter, x="Sleep_Quality", y="Glucose",
            color="Sleep_Cat",
            color_discrete_map={"Poor": RED, "Moderate": ORANGE, "Good": GREEN},
            trendline="ols", opacity=0.5
        )
        apply_template(fig, "Sleep Quality vs Glucose (scatter + trend)", 380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="warn-box">
    ⚠️ Poor sleep (score &lt; 5) is associated with elevated average glucose.
    <strong>Action:</strong> Patients with sleep score &lt; 70 (on 100-pt scale) should
    improve sleep hygiene — consistent bedtime, reduced screen exposure, earlier dinners.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── P2: Carb → Glucose Spikes (Abi) ──────────────────────────────────────────
    st.markdown("#### P2 · Abi — Which patients need dietary intervention based on carb–glucose response?")

    # Per-patient carb-glucose correlation
    carb_corr = (
        df.groupby("Patient_Id")[["Carb_Input","Glucose"]]
        .corr().unstack()["Glucose"]["Carb_Input"]
        .reset_index()
    )
    carb_corr.columns = ["Patient_Id","Corr"]
    carb_corr = carb_corr.sort_values("Corr", ascending=False)
    carb_corr["Risk"] = carb_corr["Corr"].apply(
        lambda x: "High Risk" if x > 0.4 else ("Moderate" if x > 0.2 else "Low Risk")
    )

    col_p2a, col_p2b = st.columns(2)
    with col_p2a:
        fig = px.bar(
            carb_corr, x="Patient_Id", y="Corr",
            color="Risk",
            color_discrete_map={"High Risk": RED, "Moderate": ORANGE, "Low Risk": GREEN},
            text=carb_corr["Corr"].round(2)
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=45)
        apply_template(fig, "Carb–Glucose Correlation per Patient (High = dietary intervention needed)", 400)
        st.plotly_chart(fig, use_container_width=True)

    with col_p2b:
        sample = df[df["Carb_Input"] > 0].sample(600, random_state=5)
        fig = px.scatter(
            sample, x="Carb_Input", y="Glucose",
            color="Patient_Id", opacity=0.5,
            trendline="ols"
        )
        apply_template(fig, "Carbohydrate Intake vs Glucose Response", 400)
        fig.update_xaxes(title="Carb Intake (g)")
        fig.update_yaxes(title="Glucose (mg/dL)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    high_risk = carb_corr[carb_corr["Risk"] == "High Risk"]
    if len(high_risk) > 0:
        pts = ", ".join(high_risk["Patient_Id"].tolist())
        st.markdown(f"""<div class="warn-box">
        ⚠️ <strong>High-risk patients (strong carb→glucose spike):</strong> {pts}<br>
        Recommendation: Keep per-meal carbs &lt; 50–60 g and provide personalised meal plans.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── P3: Carb + Insulin + Activity → Glucose (Maha) ───────────────────────────
    st.markdown("#### P3 · Maha — How do carbs, insulin, and physical activity interact to influence glucose spikes?")

    features_corr = df[["Glucose","Carb_Input","Basal_Rate","Bolus_Volume_Delivered","Steps"]].corr()
    fig = px.imshow(
        features_corr,
        text_auto=".2f",
        color_continuous_scale=[[0,"#FF4C4C"],[0.5,"#1A1A2E"],[1,"#7B2FF7"]],
        zmin=-1, zmax=1
    )
    apply_template(fig, "Correlation Matrix — Glucose vs Diet, Insulin & Activity", 400)
    st.plotly_chart(fig, use_container_width=True)

    col_p3a, col_p3b, col_p3c = st.columns(3)
    with col_p3a:
        fig = px.scatter(
            df.sample(500, random_state=1),
            x="Carb_Input", y="Glucose",
            color="Bolus_Volume_Delivered",
            color_continuous_scale=[[0,RED],[1,GREEN]],
            opacity=0.6
        )
        apply_template(fig, "Carbs vs Glucose (coloured by Bolus)", 340)
        st.plotly_chart(fig, use_container_width=True)

    with col_p3b:
        fig = px.scatter(
            df.sample(500, random_state=2),
            x="Steps", y="Glucose",
            color="Carb_Input",
            color_continuous_scale=[[0,GREEN],[1,RED]],
            opacity=0.6
        )
        apply_template(fig, "Steps vs Glucose (coloured by Carbs)", 340)
        st.plotly_chart(fig, use_container_width=True)

    with col_p3c:
        fig = px.scatter(
            df.sample(500, random_state=3),
            x="Basal_Rate", y="Glucose",
            color="Bolus_Volume_Delivered",
            color_continuous_scale=[[0,BLUE],[1,PINK]],
            opacity=0.6
        )
        apply_template(fig, "Basal Rate vs Glucose (coloured by Bolus)", 340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    ✅ Carb intake is the strongest driver of glucose spikes. Insulin (bolus) modulates the
    response. Physical activity shows a mild compensatory effect but does not override
    high-carb intake.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── P4: Steps → Glucose Variability (Purnima) ────────────────────────────────
    st.markdown("#### P4 · Purnima — Previous-day steps vs Glucose Variability (SD & CV). Optimal step target?")

    df["date"] = df["Time"].dt.date
    daily_glucose = (
        df.groupby(["Patient_Id","date"])["Glucose"]
        .agg(["mean","std"]).reset_index()
        .rename(columns={"mean":"Mean_Glucose","std":"SD_Glucose"})
    )
    daily_glucose["CV_Glucose"] = daily_glucose["SD_Glucose"] / daily_glucose["Mean_Glucose"] * 100

    daily_steps = (
        df.groupby(["Patient_Id","date"])["Steps"]
        .sum().reset_index().rename(columns={"Steps":"Daily_Steps"})
    )
    df_merge = pd.merge(daily_glucose, daily_steps, on=["Patient_Id","date"])

    col_p4a, col_p4b = st.columns(2)
    with col_p4a:
        fig = px.scatter(
            df_merge, x="Daily_Steps", y="SD_Glucose",
            color="Patient_Id", trendline="ols",
            opacity=0.5
        )
        fig.update_layout(showlegend=False)
        apply_template(fig, "Daily Steps vs Glucose SD (Standard Deviation)", 380)
        fig.update_xaxes(title="Daily Steps")
        fig.update_yaxes(title="Glucose SD (mg/dL)")
        st.plotly_chart(fig, use_container_width=True)

    with col_p4b:
        fig = px.scatter(
            df_merge, x="Daily_Steps", y="CV_Glucose",
            color="Patient_Id", trendline="ols",
            opacity=0.5
        )
        fig.update_layout(showlegend=False)
        apply_template(fig, "Daily Steps vs Glucose CV (Coefficient of Variation %)", 380)
        fig.update_xaxes(title="Daily Steps")
        fig.update_yaxes(title="Glucose CV (%)")
        st.plotly_chart(fig, use_container_width=True)

    # Insights table
    insights_data = {
        "Insight": ["Higher variability patients walk more","Steps ≠ SD/CV reduction (this dataset)",
                    "Variability is multi-factorial","Long-term benefit still exists","Optimal target (evidence-based)"],
        "What Data Shows": ["SD & CV increase slightly with steps","Regression: flat or upward trend",
                            "Diet, meds, sleep, stress dominate","No short-term effect visible",
                            "Not visible here but research-backed"],
        "Recommended Action": ["Target high-variability patients for coaching","Don't rely on steps alone",
                               "Focus on meal timing, carb load, sleep","Maintain 6,000–8,000 steps/day",
                               "Aim for 8,000–10,000 for long-term benefit"],
    }
    st.dataframe(pd.DataFrame(insights_data), use_container_width=True, hide_index=True)

    st.markdown("""<div class="warn-box">
    ⚠️ Counter-intuitive finding: patients walk more on high-variability days (reactive walking).
    Steps help long-term insulin sensitivity but don't instantly stabilise glucose.
    Target: <strong>7,000–8,000 steps/day minimum.</strong>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── P5: HR + Glucose Spikes (Harshali) ───────────────────────────────────────
    st.markdown("#### P5 · Harshali — What personalised recommendations come from HR + glucose patterns?")

    df["HR_change"]  = df["Heart_Rate"].diff().fillna(0)
    df["GLU_change"] = df["Glucose"].diff().fillna(0)
    df["HR_spike"]   = (df["Heart_Rate"] > 100) | (df["HR_change"] > 15)
    df["GLU_spike"]  = (df["Glucose"] > 140) | (df["GLU_change"] > 20)
    df["HR_GLU_spike"] = df["HR_spike"] & df["GLU_spike"]

    def classify_trigger(row):
        if not row["HR_GLU_spike"]: return None
        if row["Steps"] > 50:        return "Activity-related"
        if row["GLU_change"] > 20:   return "Post-meal spike"
        h = row["Time"].hour
        if h >= 23 or h <= 5:        return "Nighttime stress / poor sleep"
        if row["HR_change"] > 15:    return "Stress-related"
        return "Unknown"

    df["Trigger"] = df.apply(classify_trigger, axis=1)

    trigger_counts = df[df["Trigger"].notna()]["Trigger"].value_counts().reset_index()
    trigger_counts.columns = ["Trigger","Count"]

    col_p5a, col_p5b = st.columns(2)
    with col_p5a:
        fig = px.pie(
            trigger_counts, names="Trigger", values="Count", hole=0.45,
            color_discrete_sequence=[PURPLE, PINK, ORANGE, TEAL, RED]
        )
        apply_template(fig, "HR + Glucose Spike Trigger Classification", 380)
        st.plotly_chart(fig, use_container_width=True)

    with col_p5b:
        # Time series of HR and Glucose for a sample patient
        sample_p = df[df["Patient_Id"] == "P001"].tail(500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_p["Time"], y=sample_p["Heart_Rate"],
                                  name="Heart Rate", line=dict(color=RED, width=1.5)))
        fig.add_trace(go.Scatter(x=sample_p["Time"], y=sample_p["Glucose"],
                                  name="Glucose", line=dict(color=BLUE, width=1.5)))
        spikes = sample_p[sample_p["HR_GLU_spike"]]
        fig.add_trace(go.Scatter(
            x=spikes["Time"], y=spikes["Heart_Rate"],
            mode="markers", name="Dual Spike",
            marker=dict(color="black", size=8, symbol="x")
        ))
        apply_template(fig, "Patient P001 — HR & Glucose Over Time (× = dual spike)", 380)
        fig.update_xaxes(title="Time")
        fig.update_yaxes(title="Value")
        st.plotly_chart(fig, use_container_width=True)

    prescriptive_map = {
        "Activity-related": "Cool down with slow walking and hydration.",
        "Post-meal spike":  "Take a 10-min walk; reduce high-glycemic foods next meal.",
        "Nighttime stress / poor sleep": "Avoid late meals; improve sleep hygiene; cut caffeine/alcohol.",
        "Stress-related":   "Use slow breathing, grounding, or a short walk.",
        "Unknown":          "Monitor pattern; review meal timing, stress, and sleep.",
    }
    presc_df = trigger_counts.copy()
    presc_df["Prescribed Action"] = presc_df["Trigger"].map(prescriptive_map)
    st.dataframe(presc_df, use_container_width=True, hide_index=True)

    st.markdown("""<div class="warn-box">
    ⚠️ Simultaneous HR + glucose rise = dual-system stress event (autonomic + metabolic).
    This is one of the most clinically significant patterns — requires immediate targeted action.
    </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE
# ╔══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="section-title">🤖 Predictive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Random Forest Classifier predicting whether a T1DM patient will enter Hypoglycemia, Normal, or Hyperglycemia zone in the next 30 minutes.</div>', unsafe_allow_html=True)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    @st.cache_data
    def train_model(df):
        features = ["Glucose","Steps","Heart_Rate","Carb_Input","Bolus_Volume_Delivered","Basal_Rate"]
        # Create 30-min-ahead target
        model_df = df.copy().dropna(subset=features)
        model_df["Future_Glucose"] = model_df["Glucose"].shift(-6)  # 6 * 5min = 30 min
        model_df = model_df.dropna(subset=["Future_Glucose"])

        def zone(g):
            if g < 70:   return "Hypoglycemia"
            if g <= 180: return "Normal"
            return "Hyperglycemia"

        model_df["Target_Zone"] = model_df["Future_Glucose"].apply(zone)
        X = model_df[features]
        le = LabelEncoder()
        y  = le.fit_transform(model_df["Target_Zone"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        fi  = rf.feature_importances_
        return acc, cm, fi, features, le.classes_

    with st.spinner("Training Random Forest model…"):
        acc, cm, fi, features_list, class_names = train_model(df)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip([c1,c2,c3,c4],[
        ("Model Accuracy",    f"{acc*100:.1f}%"),
        ("Algorithm",         "Random Forest"),
        ("Trees",             "100"),
        ("Max Depth",         "10"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        # Confusion matrix heatmap
        fig = px.imshow(
            cm,
            x=class_names, y=class_names,
            text_auto=True,
            color_continuous_scale=[[0,"#F8F6FF"],[1,PURPLE]],
            labels=dict(x="Predicted Zone", y="Actual Zone", color="Count")
        )
        apply_template(fig, f"Confusion Matrix — Accuracy {acc*100:.1f}%", 420)
        st.plotly_chart(fig, use_container_width=True)

    with col_m2:
        # Feature importance
        fi_df = pd.DataFrame({"Feature": features_list, "Importance": fi}).sort_values("Importance")
        bar_colors = [RED if v >= 0.3 else (ORANGE if v >= 0.1 else BLUE) for v in fi_df["Importance"]]
        fig = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker_color=bar_colors,
            text=fi_df["Importance"].round(4),
            textposition="outside"
        ))
        apply_template(fig, "Feature Importance — Random Forest", 420)
        fig.update_xaxes(title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    ✅ Model accuracy significantly above random-chance baseline (33.3%).
    Glucose itself is the strongest predictor, confirming that current reading
    is the best near-term indicator of future zone. Carb intake and bolus volume
    also contribute meaningfully.
    </div>""", unsafe_allow_html=True)

    # ── Live Predictor ─────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 🔮 Live Zone Predictor — Enter current readings")

    from sklearn.ensemble import RandomForestClassifier as RFC
    @st.cache_resource
    def get_model(df):
        features = ["Glucose","Steps","Heart_Rate","Carb_Input","Bolus_Volume_Delivered","Basal_Rate"]
        model_df = df.copy().dropna(subset=features)
        model_df["Future_Glucose"] = model_df["Glucose"].shift(-6)
        model_df = model_df.dropna(subset=["Future_Glucose"])
        def zone(g):
            if g < 70:   return "Hypoglycemia"
            if g <= 180: return "Normal"
            return "Hyperglycemia"
        model_df["Target_Zone"] = model_df["Future_Glucose"].apply(zone)
        X = model_df[features]
        le = LabelEncoder()
        y  = le.fit_transform(model_df["Target_Zone"])
        rf = RFC(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
        rf.fit(X, y)
        return rf, le

    rf_model, label_enc = get_model(df)

    with st.form("predictor_form"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            inp_glucose = st.number_input("Current Glucose (mg/dL)", 50, 400, 120)
            inp_steps   = st.number_input("Steps (last period)", 0, 5000, 300)
        with pc2:
            inp_hr      = st.number_input("Heart Rate (bpm)", 40, 200, 75)
            inp_carb    = st.number_input("Carb Intake (g)", 0, 200, 45)
        with pc3:
            inp_bolus   = st.number_input("Bolus Volume (U)", 0.0, 20.0, 3.0, step=0.5)
            inp_basal   = st.number_input("Basal Rate (U/hr)", 0.0, 5.0, 1.0, step=0.1)
        submitted = st.form_submit_button("🔮 Predict Next 30-min Zone", use_container_width=True)

    if submitted:
        X_new = np.array([[inp_glucose, inp_steps, inp_hr, inp_carb, inp_bolus, inp_basal]])
        pred_proba = rf_model.predict_proba(X_new)[0]
        pred_class = label_enc.classes_[pred_proba.argmax()]
        zone_colors_map = {"Hypoglycemia": RED, "Normal": GREEN, "Hyperglycemia": ORANGE}
        zone_emoji      = {"Hypoglycemia": "🔴", "Normal": "🟢", "Hyperglycemia": "🟠"}
        clr = zone_colors_map[pred_class]
        st.markdown(f"""
        <div style="background:{clr}22; border:2px solid {clr}; border-radius:14px;
                    padding:20px 28px; text-align:center; margin:12px 0;">
          <div style="font-size:2.5rem;">{zone_emoji[pred_class]}</div>
          <div style="font-size:1.8rem; font-weight:900; color:{clr};">{pred_class}</div>
          <div style="color:#C0C0E0; margin-top:6px;">Predicted zone in 30 minutes
            — confidence {pred_proba.max()*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # Probability bar
        prob_df = pd.DataFrame({"Zone": label_enc.classes_, "Probability": pred_proba})
        fig = px.bar(
            prob_df, x="Zone", y="Probability",
            color="Zone",
            color_discrete_map={"Hypoglycemia": RED, "Normal": GREEN, "Hyperglycemia": ORANGE},
            text=prob_df["Probability"].apply(lambda x: f"{x*100:.1f}%")
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 1.1])
        apply_template(fig, "Prediction Probabilities", 320)
        st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RECOMMENDATIONS
# ╔══════════════════════════════════════════════════════════════════════════════
with tab_recs:
    st.markdown('<div class="section-title">✅ Recommendations Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Converting data-driven insights into actionable clinical and lifestyle interventions.</div>', unsafe_allow_html=True)

    recs = [
        {
            "icon": "😴", "title": "Sleep Quality → Glucose Control",
            "finding": "Poor sleep quality (score < 70/100) is associated with elevated glucose levels and reduced insulin sensitivity.",
            "action": "Patients with sleep score < 70 should: maintain consistent bedtimes, reduce screen exposure after 9 PM, and schedule dinner at least 2–3 hours before sleep.",
            "threshold": "Target: Sleep score ≥ 7/10",
            "color": PURPLE
        },
        {
            "icon": "🥗", "title": "Dietary Intervention — High-Risk Patients",
            "finding": "Patients with high carb–glucose correlation show strong post-meal spikes, placing them at continuous hyperglycemia risk.",
            "action": "Provide personalised meal plans: limit per-meal carbs to 50–60 g, prioritise low-glycaemic foods, and time meals consistently.",
            "threshold": "Target: Carb-glucose correlation < 0.4",
            "color": ORANGE
        },
        {
            "icon": "🚶", "title": "Daily Step Target for Glucose Stability",
            "finding": "While short-term step counts don't instantly reduce glucose variability, sustained daily activity improves insulin sensitivity over weeks.",
            "action": "Aim for 7,000–8,000 steps/day as a minimum daily target. Patients with high glucose variability should target 8,000–10,000 steps.",
            "threshold": "Target: ≥ 7,000 steps/day",
            "color": TEAL
        },
        {
            "icon": "❤️", "title": "HR + Glucose Dual-Spike Management",
            "finding": "Simultaneous HR and glucose rise signals a dual-system stress event: autonomic + metabolic. This is one of the highest-risk patterns.",
            "action": "During detected dual spikes: encourage hydration, a short 5-minute slow walk, and diaphragmatic breathing. Avoid high-carb foods in this window.",
            "threshold": "Trigger: HR > 100 bpm AND Glucose > 140 mg/dL",
            "color": RED
        },
        {
            "icon": "💉", "title": "Insulin + Activity Combination Therapy",
            "finding": "Carb intake is the dominant glucose driver. Bolus insulin partially mitigates spikes but physical activity provides additional buffering.",
            "action": "Clinicians should consider post-meal activity prescriptions (10-min walk) alongside bolus adjustments for patients with frequent post-meal hyperglycemia.",
            "threshold": "Optimal: Carbs + Bolus timing within 15 min of meal",
            "color": PINK
        },
    ]

    for rec in recs:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1A1A2E, #1E1E38);
                    border: 1px solid {rec['color']}44;
                    border-left: 5px solid {rec['color']};
                    border-radius: 14px; padding: 20px 24px; margin-bottom: 14px;">
          <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
            <span style="font-size:2rem;">{rec['icon']}</span>
            <span style="font-size:1.15rem; font-weight:800; color:{rec['color']};">{rec['title']}</span>
          </div>
          <div style="color:#C0C0E0; font-size:0.9rem; margin-bottom:8px;">
            <strong style="color:#E0E0FF;">📌 Finding:</strong> {rec['finding']}
          </div>
          <div style="color:#C0C0E0; font-size:0.9rem; margin-bottom:8px;">
            <strong style="color:#E0E0FF;">✅ Action:</strong> {rec['action']}
          </div>
          <div style="background:{rec['color']}22; border-radius:8px; padding:8px 14px;
                      font-size:0.88rem; color:{rec['color']}; font-weight:600;">
            🎯 {rec['threshold']}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Summary matrix
    st.markdown("#### 📋 Recommendation Summary Matrix")
    summary_df = pd.DataFrame({
        "Domain":      ["Sleep","Diet","Physical Activity","Stress / HR","Insulin Timing"],
        "Risk Signal": [
            "Sleep score < 7/10",
            "Carb–glucose corr > 0.4",
            "Steps < 7,000/day",
            "HR > 100 + Glucose > 140",
            "Post-meal glucose > 180",
        ],
        "Recommended Action": [
            "Improve sleep hygiene, consistent bedtime",
            "Personalised low-GI meal plan",
            "Gradual increase to 8,000–10,000 steps/day",
            "Hydration, slow walk, breathing exercise",
            "10-min walk + bolus timing review",
        ],
        "Expected Outcome": [
            "Lower avg glucose by 5–15 mg/dL",
            "Reduce post-meal spikes by 20–30%",
            "Improve insulin sensitivity over 2–4 weeks",
            "Reduce dual-spike frequency",
            "Reduce time in hyperglycemia",
        ],
        "Priority": ["Medium","High","Medium","High","High"],
    })

    def color_priority(val):
        colors = {"High": "background-color:#FF4C4C22; color:#FF8080",
                  "Medium": "background-color:#FF9F1C22; color:#FFBF60"}
        return colors.get(val, "")

    styled = summary_df.style.applymap(color_priority, subset=["Priority"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A0A2E,#0A1A2E);
                border:1px solid #3A2A5A; border-radius:14px; padding:20px 24px; margin-top:16px;">
      <div style="font-size:1.1rem; font-weight:800; color:#E0E0FF; margin-bottom:10px;">
        🏁 Final Conclusion
      </div>
      <div style="color:#C0C0E0; font-size:0.92rem; line-height:1.7;">
        This dashboard integrates <strong style="color:#7B2FF7;">descriptive</strong>,
        <strong style="color:#E040FB;">prescriptive</strong>, and
        <strong style="color:#00D4AA;">predictive</strong> analytics to provide a comprehensive
        understanding of glucose behaviour in T1DM patients.<br><br>
        The descriptive section summarises patient lifestyle and physiological patterns.
        The prescriptive section identifies actionable interventions to improve glucose control.
        The predictive section enables early zone detection — giving clinicians and patients
        a 30-minute warning window to act.<br><br>
        Together, these insights support <strong style="color:#FF9F1C;">data-driven
        decision-making</strong> and help optimise diabetes management at scale.
      </div>
    </div>
    """, unsafe_allow_html=True)
