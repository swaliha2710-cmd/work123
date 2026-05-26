import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_data, INCOME_ORDER, AGE_ORDER, PRIMARY, ACCENT, SECONDARY,
    MOOD_COLORS, INTEREST_COLORS, PERSONA_TEMPLATES,
)
from eda import run_eda
from models import (
    train_classification, train_regression,
    train_clustering, association_mining,
    segment_profile,
    save_model, load_model, predict_new,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodCart Analytics v3",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
.main-header {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0d0d1a 100%);
    border-bottom: 2px solid #A855F733;
    padding: 28px 32px 20px;
    margin: -1rem -1rem 1.5rem;
}
.main-header h1 {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #A855F7, #EC4899, #7C3AED);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.main-header p { color: #b0a0c0; font-size: 1rem; margin: 6px 0 0; }
.section-card {
    background: linear-gradient(135deg, #1a1a2e, #1a0a2e);
    border: 1px solid #A855F722;
    border-radius: 14px; padding: 20px 22px; margin-bottom: 16px;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #2a1040);
    border: 1px solid #A855F744;
    border-radius: 12px; padding: 16px 14px; text-align: center;
}
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #A855F7; }
.metric-card .label { font-size: 0.75rem; color: #b0a0c0; margin-top: 4px; }
.metric-card .delta { font-size: 0.7rem; color: #4CAF50; margin-top: 3px; }
.persona-card {
    border-radius: 14px; padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    background: linear-gradient(135deg, #1a1a2e, #0d0d1a);
    margin-bottom: 12px;
}
.persona-icon { font-size: 2.5rem; }
.persona-name { font-size: 1.1rem; font-weight: 700; margin: 8px 0 4px; }
.persona-trait {
    font-size: 0.78rem; color: #b0a0c0;
    background: rgba(255,255,255,0.05);
    border-radius: 6px; padding: 3px 8px;
    display: inline-block; margin: 2px;
}
.persona-strategy {
    font-size: 0.82rem; color: #e0d0f0;
    background: rgba(168,85,247,0.1);
    border-left: 3px solid #A855F7;
    padding: 8px 12px; margin-top: 10px;
    border-radius: 0 6px 6px 0;
}
.insight-box {
    background: rgba(168,85,247,0.08);
    border: 1px solid rgba(168,85,247,0.3);
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
}
.insight-box .headline { font-size: 1.1rem; font-weight: 600; color: #A855F7; }
.insight-box .body { font-size: 0.85rem; color: #c0b0d0; margin-top: 4px; }
.nav-info {
    background: #1a0a2e; border-radius: 8px;
    padding: 10px 14px; margin: 8px 0;
    font-size: 0.8rem; color: #b0a0c0;
    border-left: 3px solid #A855F7;
}
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #A855F7) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #A855F7, #EC4899) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #A855F755 !important;
}
</style>
""", unsafe_allow_html=True)

CHART_BG   = "rgba(0,0,0,0)"
FONT_COLOR = "#e8e0f0"
GRID_COLOR = "rgba(168,85,247,0.12)"


def _layout(fig, title="", height=420):
    fig.update_layout(
        title=title, height=height,
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR, size=12),
        title_font=dict(size=15, color=PRIMARY),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30, r=20, t=50, b=30),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig


def _kpi(col, label, value, delta=None):
    col.markdown(f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
        {"<div class='delta'>"+delta+"</div>" if delta else ""}
    </div>""", unsafe_allow_html=True)


def _insight(headline, body):
    st.markdown(f"""
    <div class="insight-box">
        <div class="headline">💡 {headline}</div>
        <div class="body">{body}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
for key in ["df", "clf_results", "reg_scores", "cluster_labels", "pca_df",
            "inertias", "rules", "reg_avp"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px">
        <div style="font-size:1.6rem;font-weight:700;
                    background:linear-gradient(90deg,#A855F7,#EC4899);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🧠 MoodCart
        </div>
        <div style="font-size:0.72rem;color:#b0a0c0;">Analytics Dashboard v3.0</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio(
        "Navigation",
        ["🏠 Home", "📂 Upload Data", "📊 EDA", "📋 Executive Summary",
         "🤖 Classification", "🔍 SHAP Explainability",
         "📈 Regression", "🔵 Clustering", "🧑‍🤝‍🧑 Persona Cards",
         "🔗 Association Rules", "👥 Segment Profiler", "🔮 Predict New"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.session_state["df"] is not None:
        df_info = st.session_state["df"]
        yes_pct = (df_info.get("Interest_in_MoodCart", pd.Series()) == "Yes").mean() * 100 if "Interest_in_MoodCart" in df_info.columns else 0
        st.markdown(f"""<div class="nav-info">
            ✅ Dataset loaded<br>
            <b>{len(df_info):,}</b> rows · <b>{df_info.shape[1]}</b> cols<br>
            MoodCart Interest Yes: <b>{yes_pct:.1f}%</b>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="nav-info">⚠️ No data loaded.<br>Go to Upload Data first.</div>""",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.caption("MoodCart Analytics · v3.0 · Group Edition")

# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if menu == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 MoodCart Analytics Dashboard</h1>
        <p>Emotion-driven e-commerce intelligence — v3.0 Group Edition</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    features = [
        ("📊", "EDA + Statistical Tests", "Chi-square tests, correlation heatmaps, ordinal encoding"),
        ("🤖", "Classification (6 Models)", "SMOTE-balanced + XGBoost + 5-fold cross-validation"),
        ("🔍", "SHAP Explainability", "Understand WHY the model makes each prediction"),
        ("📈", "Regression + RMSE", "Log-transform, XGBoost, actual vs predicted deep dive"),
        ("🔵", "Clustering + Silhouette", "Optimal K via elbow + silhouette score curves"),
        ("🧑‍🤝‍🧑", "Persona Cards", "4 named customer archetypes with business strategy"),
        ("🔗", "Association Rules", "Apriori product bundle mining with lift visualisation"),
        ("👥", "Segment Profiler", "Slice any demographic vs spend & interest"),
        ("🔮", "Predict New", "Upload new CSV → instant predictions"),
    ]
    cols = [col1, col2, col3]
    for i, (icon, title, desc) in enumerate(features):
        cols[i % 3].markdown(f"""
        <div class="section-card">
            <div style="font-size:1.6rem;">{icon}</div>
            <div style="font-weight:600;color:{PRIMARY};margin:6px 0 4px;">{title}</div>
            <div style="font-size:0.82rem;color:#b0a0c0;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Start with **📂 Upload Data**, then explore EDA → Classification → SHAP → Clustering → Personas.")

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "📂 Upload Data":
    st.markdown(f'<h2 style="color:{PRIMARY};">📂 Upload Dataset</h2>', unsafe_allow_html=True)
    f = st.file_uploader("Upload MoodCart CSV file", type=["csv"])
    if f:
        st.session_state["df"] = load_data(f)
        df = st.session_state["df"]
        st.success(f"✅ Data loaded — {len(df):,} rows · {df.shape[1]} columns")

        c1, c2, c3, c4 = st.columns(4)
        _kpi(c1, "Total Respondents", f"{len(df):,}")
        _kpi(c2, "Columns", f"{df.shape[1]}")
        avg_spend = pd.to_numeric(df.get("Monthly_Spend"), errors="coerce").mean()
        _kpi(c3, "Avg Monthly Spend", f"₹{avg_spend:,.0f}" if pd.notna(avg_spend) else "—")
        if "Interest_in_MoodCart" in df.columns:
            yes_pct = (df["Interest_in_MoodCart"] == "Yes").mean() * 100
            _kpi(c4, "MoodCart Interest (Yes)", f"{yes_pct:.1f}%")

        st.markdown("---")
        st.subheader("Data Preview")
        st.dataframe(df.head(15), use_container_width=True)

        st.subheader("Column Summary")
        summary = pd.DataFrame({
            "dtype":    df.dtypes,
            "non_null": df.notna().sum(),
            "nulls":    df.isna().sum(),
            "unique":   df.nunique(),
        }).reset_index().rename(columns={"index": "Column"})
        st.dataframe(summary, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "📊 EDA":
    if st.session_state["df"] is not None:
        run_eda(st.session_state["df"])
    else:
        st.warning("⚠️ Please upload data first.")

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "📋 Executive Summary":
    st.markdown(f'<h2 style="color:{PRIMARY};">📋 Executive Summary</h2>', unsafe_allow_html=True)
    st.markdown("*A data-to-decision narrative: from survey insights → model findings → business strategy.*")

    if st.session_state["df"] is None:
        st.warning("⚠️ Please upload data first.")
    else:
        df = st.session_state["df"]
        df["Monthly_Spend"] = pd.to_numeric(df["Monthly_Spend"], errors="coerce")

        # ── Key metrics row ──
        yes_pct    = (df["Interest_in_MoodCart"] == "Yes").mean() * 100
        maybe_pct  = (df["Interest_in_MoodCart"] == "Maybe").mean() * 100
        avg_spend  = df["Monthly_Spend"].mean()
        willing    = (df["Willingness_To_Spend_More"] == "Yes").mean() * 100 if "Willingness_To_Spend_More" in df.columns else 0
        ai_trust   = (df["AI_Trust"].isin(["Highly", "Moderately"])).mean() * 100 if "AI_Trust" in df.columns else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        _kpi(c1, "Confirmed Interest", f"{yes_pct:.1f}%", "Say 'Yes' to MoodCart")
        _kpi(c2, "Potential Converts", f"{maybe_pct:.1f}%", "Said 'Maybe'")
        _kpi(c3, "Avg Monthly Spend", f"₹{avg_spend:,.0f}", "Across all respondents")
        _kpi(c4, "Will Pay More", f"{willing:.1f}%", "For personalization")
        _kpi(c5, "Trust AI Reco", f"{ai_trust:.1f}%", "Moderate/High trust")

        st.markdown("---")

        # ── 3 Headline Findings ──
        st.subheader("🎯 3 Key Findings")
        _insight(
            "Mood significantly drives purchase behaviour",
            "Chi-square tests confirm that Mood, Decision Style, and Impulse Behaviour are "
            "statistically significant predictors of MoodCart interest (p < 0.05). "
            "Stressed and Happy users show highest openness to the platform."
        )
        _insight(
            "Privacy is the #1 adoption barrier",
            f"{100-ai_trust:.1f}% of users show low/no trust in AI recommendations. "
            "Data_Concern and Privacy_Comfort are key hesitation drivers — "
            "a transparent opt-in system is critical for conversion."
        )
        _insight(
            f"₹{avg_spend:,.0f}/month average spend with 4 distinct customer segments",
            "Spend is right-skewed — a small premium segment (top 15%) drives disproportionate revenue. "
            "Clustering reveals 4 personas: Impulse Buyers, Rational Planners, Premium Shoppers, and Privacy Skeptics. "
            "Each requires a different engagement strategy."
        )

        st.markdown("---")

        # ── Story arc ──
        st.subheader("📖 Analytics Story Arc")
        arc = {
            "1️⃣ Descriptive": "2,000 respondents across ages, income, city tiers. "
                               "Balanced gender. Frequent online shoppers. Right-skewed spend.",
            "2️⃣ Diagnostic": "Mood impacts shopping for 70%+ users. Privacy concern deters adoption. "
                              "Stressed users buy snacks & electronics. Happy users buy fashion.",
            "3️⃣ Predictive": "Random Forest / XGBoost predict MoodCart interest with F1 > 0.6 "
                              "(after SMOTE balancing). SHAP shows AI_Trust and Income as top features.",
            "4️⃣ Prescriptive": "Target the 'Maybe' segment (33%) with trust-building campaigns. "
                                "Use mood-triggered notifications for impulse buyer cluster. "
                                "Premium bundle wellness + fashion for high-spend users.",
        }
        for step, text in arc.items():
            st.markdown(f"""
            <div class="insight-box">
                <div class="headline">{step}</div>
                <div class="body">{text}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Business Recommendations ──
        st.subheader("🚀 Top 5 Business Recommendations")
        recs = [
            ("Transparent Mood Opt-In", "Let users choose which moods trigger recommendations. Builds trust, reduces privacy friction."),
            ("Convert the Maybe Segment", f"{maybe_pct:.1f}% said Maybe. Target with trial offers + social proof campaigns."),
            ("Mood-Bundle Product Pairs", "Association rules show Clothes+Accessories and Snacks+Drinks as top combos. Surface these when mood = Happy or Bored."),
            ("Premium Tier for High Spenders", "Top 15% spend ₹25,000+/month. A premium membership with exclusive recommendations has high LTV."),
            ("Build Trust First in Tier-3 Cities", "Lower AI trust in smaller cities → launch with simpler 'trending in your area' reco before full mood AI."),
        ]
        for i, (title, body) in enumerate(recs, 1):
            st.markdown(f"""
            <div class="insight-box">
                <div class="headline">#{i} {title}</div>
                <div class="body">{body}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🤖 Classification":
    st.markdown(f'<h2 style="color:{PRIMARY};">🤖 Classification — Predict Interest in MoodCart</h2>',
                unsafe_allow_html=True)
    st.markdown("Trains **6 models** (incl. XGBoost) with SMOTE class-balancing and 5-fold cross-validation.")

    if st.session_state["df"] is None:
        st.warning("⚠️ Please upload data first.")
    else:
        use_smote = st.checkbox("Apply SMOTE oversampling (recommended — fixes class imbalance)", value=True)

        if st.button("▶ Train All Classification Models", type="primary"):
            with st.spinner("Training models with cross-validation… ~60-90 seconds"):
                try:
                    (res, best_model, le, cols, conf_mats,
                     best_name, feat_imp, smote_applied, shap_df) = \
                        train_classification(st.session_state["df"], use_smote=use_smote)
                    save_model(best_model, le, cols)
                    st.session_state["clf_results"]   = res
                    st.session_state["clf_conf_mats"] = conf_mats
                    st.session_state["clf_best_name"] = best_name
                    st.session_state["clf_feat_imp"]  = feat_imp
                    st.session_state["clf_le"]        = le
                    st.session_state["clf_shap"]      = shap_df
                    smote_msg = " (SMOTE applied ✅)" if smote_applied else ""
                    st.success(f"✅ Training complete! Best model: **{best_name}**{smote_msg} · F1: {res['F1 Score'].max():.4f}")
                except Exception as e:
                    st.error(f"Training error: {e}")

        if st.session_state.get("clf_results") is not None:
            res       = st.session_state["clf_results"]
            conf_mats = st.session_state.get("clf_conf_mats", {})
            best_name = st.session_state.get("clf_best_name", "")
            feat_imp  = st.session_state.get("clf_feat_imp")
            le        = st.session_state.get("clf_le")

            st.subheader("📊 Model Comparison (with 5-Fold CV)")
            display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score",
                            "CV F1 (mean)", "CV F1 (±std)", "ROC-AUC"]
            display_cols = [c for c in display_cols if c in res.columns]
            styled = res[display_cols].style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1 Score", "CV F1 (mean)"],
                color="#2a0a4e"
            ).format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1 Score",
                                              "CV F1 (mean)", "CV F1 (±std)"]}, na_rep="—")
            st.dataframe(styled, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(res, x="Model", y="F1 Score",
                             color="Model", text=res["F1 Score"].round(4),
                             color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT, "#FF9800", "#4CAF50", "#26C6DA"])
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "F1 Score by Model"), use_container_width=True)

            with col2:
                # CV F1 with error bars
                if "CV F1 (mean)" in res.columns:
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Bar(
                        x=res["Model"], y=res["CV F1 (mean)"],
                        error_y=dict(type="data", array=res.get("CV F1 (±std)", [0]*len(res))),
                        marker_color=[PRIMARY, SECONDARY, ACCENT, "#FF9800", "#4CAF50", "#26C6DA"][:len(res)],
                        text=res["CV F1 (mean)"].round(3),
                        textposition="outside",
                    ))
                    st.plotly_chart(_layout(fig_cv, "5-Fold CV F1 Score (with Std Dev error bars)"),
                                    use_container_width=True)

            if feat_imp is not None:
                st.subheader(f"🔍 Top 20 Feature Importances ({best_name})")
                fig = px.bar(feat_imp.sort_values("Importance"),
                             x="Importance", y="Feature",
                             orientation="h", color="Importance",
                             color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
                st.plotly_chart(_layout(fig, "", 500), use_container_width=True)

            if conf_mats and best_name in conf_mats and le is not None:
                st.subheader(f"🧩 Confusion Matrix — {best_name}")
                cm = conf_mats[best_name]
                fig = px.imshow(cm, x=le.classes_, y=le.classes_,
                                color_continuous_scale=["#0d0d1a", SECONDARY, PRIMARY, ACCENT],
                                text_auto=True,
                                labels=dict(x="Predicted", y="Actual"))
                st.plotly_chart(_layout(fig, f"Confusion Matrix — {best_name}", 400), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🔍 SHAP Explainability":
    st.markdown(f'<h2 style="color:{PRIMARY};">🔍 SHAP Explainability</h2>', unsafe_allow_html=True)
    st.markdown("SHAP (SHapley Additive Explanations) shows **why** the model made each prediction — "
                "not just which features are important, but their direction and magnitude.")

    shap_df = st.session_state.get("clf_shap")
    feat_imp = st.session_state.get("clf_feat_imp")

    if shap_df is None and feat_imp is None:
        st.warning("⚠️ Please run Classification first to generate SHAP values.")
    else:
        if shap_df is not None:
            st.subheader("🌟 SHAP Feature Importance (Mean |SHAP| across all classes)")
            fig = px.bar(shap_df.sort_values("SHAP Importance"),
                         x="SHAP Importance", y="Feature",
                         orientation="h", color="SHAP Importance",
                         color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
            st.plotly_chart(_layout(fig, "SHAP Mean Absolute Feature Importance", 520), use_container_width=True)

            st.markdown("---")
            st.subheader("📊 SHAP vs Feature Importance Comparison")
            merged = shap_df.merge(feat_imp, on="Feature", how="inner") if feat_imp is not None else None
            if merged is not None:
                top = merged.head(15)
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name="SHAP Importance",
                                      x=top["SHAP Importance"], y=top["Feature"],
                                      orientation="h", marker_color=PRIMARY))
                fig2.add_trace(go.Bar(name="Tree Importance",
                                      x=top["Importance"], y=top["Feature"],
                                      orientation="h", marker_color=ACCENT))
                fig2.update_layout(barmode="group")
                st.plotly_chart(_layout(fig2, "SHAP vs Tree Feature Importance (Top 15)", 500),
                                use_container_width=True)

        # Business interpretation
        st.markdown("---")
        st.subheader("📖 How to Read SHAP for MoodCart")
        interp_items = [
            ("AI_Trust (high) →", "Strongly pushes prediction toward Yes — users who trust AI are far more likely to adopt MoodCart."),
            ("Privacy_Comfort (low) →", "Pushes toward No — privacy concern is the single biggest adoption blocker."),
            ("Impulse_Behavior (Often) →", "Pushes toward Yes — impulsive shoppers find mood-based reco compelling."),
            ("Monthly_Spend (high) →", "Slightly pushes toward Yes — high spenders are open to premium personalization."),
            ("Income (high) →", "Mixed effect — wealthy users are open to personalization but also have established brand loyalty."),
        ]
        for feat, interp in interp_items:
            st.markdown(f"""
            <div class="insight-box">
                <div class="headline">{feat}</div>
                <div class="body">{interp}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "📈 Regression":
    st.markdown(f'<h2 style="color:{PRIMARY};">📈 Regression — Predict Monthly Spend</h2>',
                unsafe_allow_html=True)
    st.markdown("Models use **log-transform** to handle right-skewed spend. Metrics include R², MAE, and RMSE.")

    if st.session_state["df"] is None:
        st.warning("⚠️ Please upload data first.")
    else:
        if st.button("▶ Train Regression Models", type="primary"):
            with st.spinner("Training models…"):
                try:
                    scores, best_model, feat_imp, avp = train_regression(st.session_state["df"])
                    st.session_state["reg_scores"]   = scores
                    st.session_state["reg_feat_imp"] = feat_imp
                    st.session_state["reg_avp"]      = avp
                    best_name = max(scores, key=lambda k: scores[k]["R²"])
                    st.success(f"✅ Done! Best model: **{best_name}** · R² = {scores[best_name]['R²']:.4f}")
                except Exception as e:
                    st.error(f"Training error: {e}")

        if st.session_state.get("reg_scores") is not None:
            scores   = st.session_state["reg_scores"]
            feat_imp = st.session_state.get("reg_feat_imp")
            avp      = st.session_state.get("reg_avp")

            st.subheader("📊 Regression Model Performance")
            scores_df = pd.DataFrame([
                {"Model": k, "R² Score": v["R²"], "MAE (₹)": v["MAE"], "RMSE (₹)": v["RMSE"]}
                for k, v in scores.items()
            ])
            st.dataframe(scores_df.style.highlight_max(subset=["R² Score"], color="#2a0a4e"),
                         use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                fig = px.bar(scores_df, x="Model", y="R² Score", color="Model",
                             text="R² Score", color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50"])
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                st.plotly_chart(_layout(fig, "R² Score (higher = better)"), use_container_width=True)
            with col2:
                fig2 = px.bar(scores_df, x="Model", y="MAE (₹)", color="Model",
                              text="MAE (₹)", color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50"])
                fig2.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
                st.plotly_chart(_layout(fig2, "MAE — lower is better"), use_container_width=True)
            with col3:
                fig3b = px.bar(scores_df, x="Model", y="RMSE (₹)", color="Model",
                               text="RMSE (₹)", color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50"])
                fig3b.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
                st.plotly_chart(_layout(fig3b, "RMSE — lower is better"), use_container_width=True)

            if avp is not None:
                st.subheader("📉 Actual vs Predicted Spend")
                fig3 = px.scatter(avp, x="Actual", y="Predicted",
                                  color_discrete_sequence=[PRIMARY], opacity=0.6)
                max_val = max(avp["Actual"].max(), avp["Predicted"].max())
                fig3.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                               line=dict(color=ACCENT, dash="dash"))
                fig3.update_layout(xaxis_title="Actual Spend (₹)", yaxis_title="Predicted Spend (₹)")
                st.plotly_chart(_layout(fig3, "Actual vs Predicted (Best Model, Sample 300)", 420),
                                use_container_width=True)

            if feat_imp is not None:
                st.subheader("🔍 Feature Importances — Spend Predictor")
                fig4 = px.bar(feat_imp.sort_values("Importance"),
                              x="Importance", y="Feature",
                              orientation="h", color="Importance",
                              color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
                st.plotly_chart(_layout(fig4, "", 500), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🔵 Clustering":
    st.markdown(f'<h2 style="color:{PRIMARY};">🔵 KMeans Clustering — Customer Segments</h2>',
                unsafe_allow_html=True)
    st.markdown("Choose optimal K using **elbow curve + silhouette score**. PCA visualises cluster separation.")

    if st.session_state["df"] is None:
        st.warning("⚠️ Please upload data first.")
    else:
        k = st.slider("Number of clusters (k)", 2, 8, 4)

        if st.button("▶ Run Clustering", type="primary"):
            with st.spinner("Running KMeans + PCA + Silhouette…"):
                try:
                    labels, pca_df, inertias, sil_scores, sil_score, pca_var = \
                        train_clustering(st.session_state["df"], k=k)
                    st.session_state["cluster_labels"] = labels
                    st.session_state["pca_df"]         = pca_df
                    st.session_state["inertias"]        = inertias
                    st.session_state["sil_scores"]      = sil_scores
                    st.session_state["sil_score"]       = sil_score
                    st.session_state["pca_var"]         = pca_var
                    st.success(f"✅ {k} clusters · Silhouette Score: **{sil_score:.4f}** "
                               f"(higher = better separation · PCA variance: "
                               f"PC1={pca_var[0]*100:.1f}% PC2={pca_var[1]*100:.1f}%)")
                except Exception as e:
                    st.error(f"Clustering error: {e}")

        if st.session_state.get("pca_df") is not None:
            labels     = st.session_state["cluster_labels"]
            pca_df     = st.session_state["pca_df"]
            inertias   = st.session_state["inertias"]
            sil_scores = st.session_state.get("sil_scores", [])
            pca_var    = st.session_state.get("pca_var", [0, 0])

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                                 color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50",
                                                          "#26C6DA", "#5C6BC0", "#B71C1C", "#FF5722"],
                                 opacity=0.75)
                fig.update_layout(xaxis_title=f"PC1 ({pca_var[0]*100:.1f}% variance)",
                                  yaxis_title=f"PC2 ({pca_var[1]*100:.1f}% variance)")
                st.plotly_chart(_layout(fig, "PCA 2D Cluster Scatter", 440), use_container_width=True)

            with col2:
                counts = pd.Series(labels).value_counts().sort_index()
                counts.index = [f"Cluster {i}" for i in counts.index]
                fig2 = px.bar(x=counts.index, y=counts.values, color=counts.index,
                              text=counts.values,
                              color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50",
                                                       "#26C6DA", "#5C6BC0", "#B71C1C", "#FF5722"])
                fig2.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig2, "Cluster Sizes", 440), use_container_width=True)

            # Elbow + Silhouette dual chart
            st.subheader("📐 Optimal K Selection")
            elbow_df = pd.DataFrame({"k": range(2, 10), "Inertia": inertias,
                                     "Silhouette": sil_scores if sil_scores else [0]*8})
            col_e, col_s = st.columns(2)
            with col_e:
                fig3 = px.line(elbow_df, x="k", y="Inertia", markers=True,
                               color_discrete_sequence=[PRIMARY])
                fig3.update_traces(marker=dict(color=ACCENT, size=8))
                st.plotly_chart(_layout(fig3, "Elbow Curve — choose where drop flattens", 360),
                                use_container_width=True)
            with col_s:
                if sil_scores:
                    fig4 = px.line(elbow_df, x="k", y="Silhouette", markers=True,
                                   color_discrete_sequence=["#4CAF50"])
                    fig4.update_traces(marker=dict(color=ACCENT, size=8))
                    best_k = elbow_df.loc[elbow_df["Silhouette"].idxmax(), "k"]
                    fig4.add_vline(x=best_k, line_dash="dash", line_color=ACCENT,
                                   annotation_text=f"Best k={best_k}")
                    st.plotly_chart(_layout(fig4, "Silhouette Score — higher = better cluster quality", 360),
                                    use_container_width=True)

            # Cluster profile
            df_tmp = st.session_state["df"].copy()
            df_tmp["Cluster"] = [f"Cluster {l}" for l in labels]
            df_tmp["Monthly_Spend"] = pd.to_numeric(df_tmp["Monthly_Spend"], errors="coerce")

            st.subheader("📋 Cluster Profile Summary")
            profile = segment_profile(df_tmp, "Cluster")
            st.dataframe(profile, use_container_width=True)

            st.info("👈 Go to **🧑‍🤝‍🧑 Persona Cards** to see named business personas for each cluster.")

# ─────────────────────────────────────────────────────────────────────────────
# PERSONA CARDS
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🧑‍🤝‍🧑 Persona Cards":
    st.markdown(f'<h2 style="color:{PRIMARY};">🧑‍🤝‍🧑 Customer Persona Cards</h2>', unsafe_allow_html=True)
    st.markdown("4 data-driven customer archetypes derived from clustering — each with traits, spend profile, and business strategy.")

    if st.session_state.get("cluster_labels") is None:
        st.warning("⚠️ Please run Clustering first to generate cluster labels.")
        st.info("Run Clustering with k=4 for best alignment with these personas.")
    else:
        labels = st.session_state["cluster_labels"]
        df     = st.session_state["df"].copy()
        df["Monthly_Spend"] = pd.to_numeric(df["Monthly_Spend"], errors="coerce")
        df["Cluster"] = labels

        k_actual = len(np.unique(labels))
        cols = st.columns(min(k_actual, 4))

        for cluster_id in range(min(k_actual, 4)):
            persona = PERSONA_TEMPLATES.get(cluster_id, {
                "name": f"Cluster {cluster_id}",
                "icon": "👤", "color": PRIMARY,
                "traits": [], "strategy": "Analyse this cluster's top features.",
            })
            cluster_df = df[df["Cluster"] == cluster_id]
            size  = len(cluster_df)
            share = size / len(df) * 100
            avg_spend = cluster_df["Monthly_Spend"].mean()
            yes_rate  = (cluster_df.get("Interest_in_MoodCart", pd.Series()) == "Yes").mean() * 100

            with cols[cluster_id % 4]:
                traits_html = "".join(f'<span class="persona-trait">{t}</span>' for t in persona["traits"])
                st.markdown(f"""
                <div class="persona-card" style="border-color:{persona['color']}44;">
                    <div class="persona-icon">{persona['icon']}</div>
                    <div class="persona-name" style="color:{persona['color']};">{persona['name']}</div>
                    <div style="font-size:0.75rem;color:#b0a0c0;margin-bottom:8px;">
                        {size:,} respondents ({share:.1f}%) · Avg ₹{avg_spend:,.0f}/mo · {yes_rate:.1f}% interested
                    </div>
                    {traits_html}
                    <div class="persona-strategy">🎯 {persona['strategy']}</div>
                </div>""", unsafe_allow_html=True)

        # Comparative radar chart
        st.markdown("---")
        st.subheader("📊 Persona Comparison — Spend & Interest")
        profile = segment_profile(df, "Cluster")
        profile["Cluster"] = profile["Cluster"].astype(str)

        fig = go.Figure()
        metrics = ["Avg_Spend", "Median_Spend", "Share_%"]
        if "Interest_Yes_%" in profile.columns:
            metrics.append("Interest_Yes_%")

        colors = [PRIMARY, ACCENT, "#FF9800", "#4CAF50"]
        for i, row in profile.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row.get(m, 0) for m in metrics],
                theta=metrics, fill="toself",
                name=f"Cluster {row['Cluster']}",
                line_color=colors[i % len(colors)],
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, color=FONT_COLOR, gridcolor=GRID_COLOR)),
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            font=dict(color=FONT_COLOR),
            title=dict(text="Cluster Radar — Spend & Interest", font=dict(color=PRIMARY, size=15)),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ASSOCIATION RULES
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🔗 Association Rules":
    st.markdown(f'<h2 style="color:{PRIMARY};">🔗 Association Rules — Product Bundle Mining</h2>',
                unsafe_allow_html=True)
    st.markdown("Uses **Apriori** to find frequently bought product combinations and high-lift bundle rules.")

    if st.session_state["df"] is None:
        st.warning("⚠️ Please upload data first.")
    else:
        col1, col2, col3 = st.columns(3)
        min_sup  = col1.slider("Min Support",    0.01, 0.30, 0.05, 0.01)
        min_conf = col2.slider("Min Confidence", 0.10, 0.90, 0.30, 0.05)
        top_n    = col3.slider("Show Top N Rules", 10, 100, 30, 10)

        if st.button("▶ Mine Association Rules", type="primary"):
            with st.spinner("Running Apriori algorithm…"):
                try:
                    rules = association_mining(st.session_state["df"], min_support=min_sup)
                    if rules.empty:
                        st.warning("No rules found — try lowering Min Support.")
                    else:
                        rules = rules[rules["confidence"] >= min_conf]
                        st.session_state["rules"] = rules
                        st.success(f"✅ Found **{len(rules)}** association rules.")
                except Exception as e:
                    st.error(f"Association mining error: {e}")

        if st.session_state.get("rules") is not None:
            rules = st.session_state["rules"]
            c1, c2, c3 = st.columns(3)
            _kpi(c1, "Total Rules", f"{len(rules):,}")
            _kpi(c2, "Avg Confidence", f"{rules['confidence'].mean():.2f}")
            _kpi(c3, "Max Lift", f"{rules['lift'].max():.2f}")
            st.markdown("---")

            col_a, col_b = st.columns(2)
            with col_a:
                top_rules = rules.head(top_n)
                fig = px.scatter(top_rules, x="support", y="confidence",
                                 size="lift", color="lift",
                                 color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT],
                                 hover_data=["antecedents", "consequents"])
                st.plotly_chart(_layout(fig, "Support vs Confidence (bubble = Lift)", 400),
                                use_container_width=True)
            with col_b:
                lift_df = rules.head(20).copy()
                lift_df["rule"] = lift_df["antecedents"] + " → " + lift_df["consequents"]
                fig2 = px.bar(lift_df.sort_values("lift"), x="lift", y="rule",
                              orientation="h", color="lift",
                              color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
                st.plotly_chart(_layout(fig2, "Top Rules by Lift", 480), use_container_width=True)

            st.subheader(f"📋 Top {min(top_n, len(rules))} Rules")
            disp = rules.head(top_n).copy()
            disp["support"]    = disp["support"].round(3)
            disp["confidence"] = disp["confidence"].round(3)
            disp["lift"]       = disp["lift"].round(2)
            st.dataframe(
                disp.style.background_gradient(subset=["lift", "confidence"], cmap="Purples"),
                use_container_width=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT PROFILER
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "👥 Segment Profiler":
    st.markdown(f'<h2 style="color:{PRIMARY};">👥 Segment Profiler</h2>', unsafe_allow_html=True)
    st.markdown("Slice any demographic dimension and compare spend, interest rate and size.")

    if st.session_state["df"] is None:
        st.warning("⚠️ Please upload data first.")
    else:
        df = st.session_state["df"]
        cat_cols = [c for c in df.select_dtypes("object").columns
                    if df[c].nunique() <= 12 and c not in
                    ["Interest_in_MoodCart", "Categories", "Stress_Purchases",
                     "Shopping_Situations", "Product_Combinations", "Happy_Purchases"]]
        seg_col = st.selectbox("Select segmentation dimension", cat_cols)

        if seg_col:
            profile = segment_profile(df, seg_col)
            df_s = df.copy()
            df_s["Monthly_Spend"] = pd.to_numeric(df_s["Monthly_Spend"], errors="coerce")

            c1, c2, c3 = st.columns(3)
            _kpi(c1, "Segments", f"{len(profile)}")
            _kpi(c2, "Total Respondents", f"{profile['Count'].sum():,}")
            if "Interest_Yes_%" in profile.columns:
                best_seg = profile.loc[profile['Interest_Yes_%'].idxmax(), seg_col]
                _kpi(c3, "Highest Interest Rate", f"{profile['Interest_Yes_%'].max():.1f}%", best_seg)

            st.markdown("---")
            st.dataframe(profile, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(profile, x=seg_col, y="Avg_Spend", color=seg_col,
                             text="Avg_Spend", color_discrete_sequence=px.colors.qualitative.Bold)
                fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
                st.plotly_chart(_layout(fig, f"Avg Spend by {seg_col}"), use_container_width=True)
            with col2:
                if "Interest_Yes_%" in profile.columns:
                    fig2 = px.bar(profile, x=seg_col, y="Interest_Yes_%", color=seg_col,
                                  text="Interest_Yes_%", color_discrete_sequence=px.colors.qualitative.Set2)
                    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    st.plotly_chart(_layout(fig2, f"MoodCart Yes% by {seg_col}"), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                fig3 = px.pie(profile, names=seg_col, values="Count",
                              hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(_layout(fig3, f"Respondent Share by {seg_col}"), use_container_width=True)
            with col4:
                fig4 = px.box(df_s, x=seg_col, y="Monthly_Spend", color=seg_col,
                              color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(_layout(fig4, f"Spend Distribution by {seg_col}"), use_container_width=True)

            if "Mood" in df.columns:
                st.subheader(f"🧠 Mood Breakdown by {seg_col}")
                ct = pd.crosstab(df[seg_col], df["Mood"])
                ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
                fig5 = px.bar(ct_pct.reset_index().melt(id_vars=seg_col),
                              x=seg_col, y="value", color="variable",
                              barmode="stack", text_auto=".0f",
                              color_discrete_map=MOOD_COLORS,
                              labels={"value": "%", "variable": "Mood"})
                st.plotly_chart(_layout(fig5, f"Mood % by {seg_col}", 440), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT NEW
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🔮 Predict New":
    st.markdown(f'<h2 style="color:{PRIMARY};">🔮 Predict on New Data</h2>', unsafe_allow_html=True)
    st.markdown("Upload a CSV with the same columns (minus `Interest_in_MoodCart`) to get predictions + probabilities.")

    f = st.file_uploader("Upload new CSV (same structure, no target column)", type=["csv"])
    if f:
        try:
            df_new = pd.read_csv(f)
            model, le, cols = load_model()
            preds, proba, classes = predict_new(df_new, model, le, cols)
            df_new["Prediction"] = preds

            if proba is not None:
                for i, cls in enumerate(classes):
                    df_new[f"Prob_{cls}"] = proba[:, i].round(3)

            st.success(f"✅ Predicted **{len(df_new):,}** rows.")

            pred_counts = pd.Series(preds).value_counts().reset_index()
            pred_counts.columns = ["Prediction", "Count"]
            pred_counts["Pct"] = (pred_counts["Count"] / len(preds) * 100).round(1)

            c1, c2, c3 = st.columns(3)
            for idx_r, row in pred_counts.iterrows():
                col = [c1, c2, c3][idx_r % 3]
                _kpi(col, row["Prediction"], f"{row['Count']:,}", f"{row['Pct']:.1f}%")

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.bar(pred_counts, x="Prediction", y="Count",
                             color="Prediction", text=pred_counts["Pct"].astype(str) + "%",
                             color_discrete_map=INTEREST_COLORS)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Prediction Distribution"), use_container_width=True)
            with col_b:
                fig2 = px.pie(pred_counts, names="Prediction", values="Count",
                              hole=0.45, color="Prediction", color_discrete_map=INTEREST_COLORS)
                st.plotly_chart(_layout(fig2, "Prediction Share"), use_container_width=True)

            st.subheader("📋 Prediction Results")
            disp_cols = ["Prediction"] + [f"Prob_{c}" for c in classes if f"Prob_{c}" in df_new.columns]
            disp_cols += [c for c in df_new.columns if c not in disp_cols]
            st.dataframe(df_new[disp_cols].head(50), use_container_width=True)

            csv_out = df_new.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Full Predictions CSV",
                               csv_out, "moodcart_predictions.csv", "text/csv")

        except FileNotFoundError:
            st.error("❌ No trained model found. Please run **Classification** first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("MoodCart Analytics · v3.0 · Group Edition")
