import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from utils import (
    MOOD_COLORS, INTEREST_COLORS, PRIMARY, SECONDARY, ACCENT,
    INCOME_ORDER, AGE_ORDER, FREQ_ORDER, LAST_BUY_ORDER,
    build_segment_profile, chi_square_test,
)

PASTEL  = px.colors.qualitative.Pastel
BOLD    = px.colors.qualitative.Bold
SET2    = px.colors.qualitative.Set2
PLOTLY  = px.colors.qualitative.Plotly

CHART_BG   = "rgba(0,0,0,0)"
PAPER_BG   = "rgba(0,0,0,0)"
FONT_COLOR = "#e8e0f0"
GRID_COLOR = "rgba(168,85,247,0.12)"


def _layout(fig, title="", height=420):
    fig.update_layout(
        title=title,
        height=height,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=PAPER_BG,
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
    <div style="background:linear-gradient(135deg,#1a1a2e,#2a1a3e);
                border:1px solid {PRIMARY}33; border-radius:12px;
                padding:18px 16px; text-align:center; margin-bottom:8px;">
        <div style="font-size:24px; font-weight:700; color:{PRIMARY};">{value}</div>
        <div style="font-size:12px; color:#b0a0c0; margin-top:4px;">{label}</div>
        {"<div style='font-size:11px;color:#4CAF50;margin-top:2px;'>"+delta+"</div>" if delta else ""}
    </div>""", unsafe_allow_html=True)


def _chi_badge(result: dict):
    sig = result["significant"]
    color = "#4CAF50" if sig else "#FF9800"
    label = "Statistically significant ✅" if sig else "Not significant ⚠️"
    return (
        f"<span style='background:{color}22;color:{color};"
        f"border:1px solid {color}55;border-radius:20px;"
        f"padding:2px 10px;font-size:0.75rem;'>"
        f"χ²={result['chi2']} · p={result['p_value']} · Cramér's V={result['cramers_v']} · {label}"
        f"</span>"
    )


def run_eda(df: pd.DataFrame):
    st.markdown(f"""
    <h2 style="color:{PRIMARY};margin-bottom:4px;">📊 Exploratory Data Analysis</h2>
    <p style="color:#b0a0c0;margin-top:0;">Deep-dive with statistical tests, correlation analysis,
    and mood-behaviour insights.</p>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "🔍 Overview", "📊 Statistical Tests", "👥 Demographics",
        "🛒 Shopping Behaviour", "🧠 Mood & Emotions",
        "💰 Spend Analysis", "📦 Products & Bundles", "🚧 Barriers & Trust",
    ])

    # ── 0 OVERVIEW ────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Dataset Snapshot")
        df_num = df.copy()
        df_num["Monthly_Spend"] = pd.to_numeric(df_num["Monthly_Spend"], errors="coerce")

        c1, c2, c3, c4, c5 = st.columns(5)
        _kpi(c1, "Total Respondents", f"{len(df):,}")
        _kpi(c2, "Columns", f"{df.shape[1]}")
        _kpi(c3, "Avg Monthly Spend", f"₹{df_num['Monthly_Spend'].mean():,.0f}")
        _kpi(c4, "Interested (Yes)",
             f"{(df['Interest_in_MoodCart']=='Yes').sum():,}",
             f"{(df['Interest_in_MoodCart']=='Yes').mean()*100:.1f}% of total")
        _kpi(c5, "Mood Diversity",
             f"{df['Mood'].nunique()} moods" if 'Mood' in df.columns else "—")

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            counts = df["Interest_in_MoodCart"].value_counts().reset_index()
            counts.columns = ["Interest", "Count"]
            counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
            present = counts["Interest"].tolist()
            cmap = {k: v for k, v in INTEREST_COLORS.items() if k in present}
            fig = px.bar(counts, x="Interest", y="Count",
                         color="Interest", text=counts["Pct"].astype(str) + "%",
                         color_discrete_map=cmap)
            fig.update_traces(textposition="outside")
            st.plotly_chart(_layout(fig, "Interest in MoodCart"), use_container_width=True)

        with col_b:
            fig2 = px.pie(counts, names="Interest", values="Count",
                          hole=0.5, color="Interest", color_discrete_map=cmap)
            fig2.update_traces(textinfo="percent+label")
            st.plotly_chart(_layout(fig2, "Interest Share"), use_container_width=True)

        st.subheader("🔎 Data Quality Check")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("✅ No missing values — dataset is complete.")
        else:
            m_df = missing.reset_index()
            m_df.columns = ["Column", "Missing"]
            m_df["Pct"] = (m_df["Missing"] / len(df) * 100).round(2)
            st.dataframe(m_df, use_container_width=True)

        # Class balance info
        st.subheader("⚖️ Class Balance")
        cb = df["Interest_in_MoodCart"].value_counts()
        st.info(
            f"**Yes:** {cb.get('Yes',0)} ({cb.get('Yes',0)/len(df)*100:.1f}%)  |  "
            f"**Maybe:** {cb.get('Maybe',0)} ({cb.get('Maybe',0)/len(df)*100:.1f}%)  |  "
            f"**No:** {cb.get('No',0)} ({cb.get('No',0)/len(df)*100:.1f}%)  \n"
            "ℹ️ Mild imbalance — SMOTE oversampling is applied in the ML pipeline."
        )

        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # ── 1 STATISTICAL TESTS ───────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("📊 Chi-Square Tests — What Drives MoodCart Interest?")
        st.markdown("Testing statistical significance of each variable's relationship with `Interest_in_MoodCart`.")

        test_cols = ["Mood", "Age", "Income", "Gender", "City_Tier",
                     "Decision_Style", "AI_Trust", "Privacy_Comfort",
                     "Impulse_Behavior", "Habit_Type", "Shopping_Frequency"]
        test_cols = [c for c in test_cols if c in df.columns]

        results = []
        for col in test_cols:
            try:
                r = chi_square_test(df, col, "Interest_in_MoodCart")
                results.append({"Variable": col, **r})
            except Exception:
                pass

        if results:
            res_df = pd.DataFrame(results).sort_values("cramers_v", ascending=False)
            res_df["Significant"] = res_df["significant"].map({True: "✅ Yes", False: "⚠️ No"})

            st.dataframe(
                res_df[["Variable", "chi2", "p_value", "cramers_v", "Significant"]]
                .style.background_gradient(subset=["cramers_v"], cmap="Purples"),
                use_container_width=True
            )

            fig = px.bar(res_df.sort_values("cramers_v"),
                         x="cramers_v", y="Variable", orientation="h",
                         color="cramers_v",
                         color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT],
                         labels={"cramers_v": "Cramér's V (association strength)"})
            fig.add_vline(x=0.1, line_dash="dash", line_color=ACCENT,
                          annotation_text="Weak (0.1)", annotation_position="top right")
            fig.add_vline(x=0.3, line_dash="dash", line_color="#4CAF50",
                          annotation_text="Moderate (0.3)", annotation_position="top right")
            st.plotly_chart(_layout(fig, "Variable Association Strength with MoodCart Interest", 440),
                            use_container_width=True)

            st.markdown("**Interpretation:** Cramér's V > 0.3 = moderate association. "
                        "Variables with p < 0.05 are statistically significant.")

        # Numeric correlation
        st.subheader("🔗 Numeric Feature Correlation")
        df_enc = df.copy()
        for col in ["Age", "Income", "Shopping_Frequency"]:
            if col in df_enc.columns:
                from utils import ORDINAL_MAPS
                if col in ORDINAL_MAPS:
                    df_enc[col] = df_enc[col].map(ORDINAL_MAPS[col])
        df_enc["Monthly_Spend"] = pd.to_numeric(df_enc["Monthly_Spend"], errors="coerce")
        numeric_df = df_enc.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr.round(2),
                                 color_continuous_scale=["#0d0d1a", SECONDARY, PRIMARY, ACCENT],
                                 text_auto=True, aspect="auto")
            st.plotly_chart(_layout(fig_corr, "Pearson Correlation Matrix", 450), use_container_width=True)

    # ── 2 DEMOGRAPHICS ────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("👥 Respondent Demographics")
        col1, col2 = st.columns(2)

        with col1:
            if "Age" in df.columns:
                age_counts = df["Age"].value_counts().reindex(AGE_ORDER).dropna().reset_index()
                age_counts.columns = ["Age", "Count"]
                fig = px.bar(age_counts, x="Age", y="Count", color="Age",
                             color_discrete_sequence=BOLD, text="Count")
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Age Group Distribution"), use_container_width=True)

            if "Occupation" in df.columns:
                occ = df["Occupation"].value_counts().reset_index()
                occ.columns = ["Occupation", "Count"]
                fig = px.pie(occ, names="Occupation", values="Count",
                             hole=0.4, color_discrete_sequence=PASTEL)
                st.plotly_chart(_layout(fig, "Occupation Split"), use_container_width=True)

        with col2:
            if "Gender" in df.columns:
                gen = df["Gender"].value_counts().reset_index()
                gen.columns = ["Gender", "Count"]
                fig = px.pie(gen, names="Gender", values="Count",
                             hole=0.45, color_discrete_sequence=[PRIMARY, ACCENT, "#26C6DA", "#FF9800"])
                st.plotly_chart(_layout(fig, "Gender Distribution"), use_container_width=True)

            if "City_Tier" in df.columns:
                tier = df["City_Tier"].value_counts().reset_index()
                tier.columns = ["City_Tier", "Count"]
                fig = px.bar(tier, x="City_Tier", y="Count", color="City_Tier",
                             color_discrete_sequence=SET2, text="Count")
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "City Tier Distribution"), use_container_width=True)

        if "Income" in df.columns:
            inc = df["Income"].value_counts().reindex(INCOME_ORDER).dropna().reset_index()
            inc.columns = ["Income", "Count"]
            fig = px.bar(inc, x="Income", y="Count", color="Income",
                         color_discrete_sequence=BOLD, text="Count")
            fig.update_traces(textposition="outside")
            st.plotly_chart(_layout(fig, "Income Bracket Distribution"), use_container_width=True)

        # Cross-tabs with chi-square badges
        if "Age" in df.columns and "Interest_in_MoodCart" in df.columns:
            st.subheader("📊 Age × Interest in MoodCart")
            chi_r = chi_square_test(df, "Age", "Interest_in_MoodCart")
            st.markdown(_chi_badge(chi_r), unsafe_allow_html=True)
            ct = pd.crosstab(df["Age"], df["Interest_in_MoodCart"])
            ct = ct.reindex([a for a in AGE_ORDER if a in ct.index]).fillna(0)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            melted = ct_pct.reset_index().melt(id_vars="Age", var_name="Interest", value_name="%")
            cmap = {k: v for k, v in INTEREST_COLORS.items() if k in melted["Interest"].unique()}
            fig = px.bar(melted, x="Age", y="%", color="Interest",
                         barmode="stack", text="%", color_discrete_map=cmap)
            fig.update_traces(texttemplate="%{text:.1f}%")
            st.plotly_chart(_layout(fig, "Interest % by Age Group"), use_container_width=True)

        if "Income" in df.columns and "Interest_in_MoodCart" in df.columns:
            st.subheader("📊 Income × Interest in MoodCart")
            chi_r2 = chi_square_test(df, "Income", "Interest_in_MoodCart")
            st.markdown(_chi_badge(chi_r2), unsafe_allow_html=True)
            ct2 = pd.crosstab(df["Income"], df["Interest_in_MoodCart"])
            ct2 = ct2.reindex([i for i in INCOME_ORDER if i in ct2.index]).fillna(0)
            ct2_pct = ct2.div(ct2.sum(axis=1), axis=0) * 100
            melted2 = ct2_pct.reset_index().melt(id_vars="Income", var_name="Interest", value_name="%")
            cmap2 = {k: v for k, v in INTEREST_COLORS.items() if k in melted2["Interest"].unique()}
            fig = px.bar(melted2, x="Income", y="%", color="Interest",
                         barmode="stack", text="%", color_discrete_map=cmap2)
            fig.update_traces(texttemplate="%{text:.1f}%")
            st.plotly_chart(_layout(fig, "Interest % by Income Bracket"), use_container_width=True)

    # ── 3 SHOPPING BEHAVIOUR ──────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("🛒 Shopping Behaviour Patterns")
        col1, col2 = st.columns(2)

        with col1:
            if "Shopping_Frequency" in df.columns:
                freq = df["Shopping_Frequency"].value_counts().reindex(FREQ_ORDER).dropna().reset_index()
                freq.columns = ["Frequency", "Count"]
                fig = px.bar(freq, x="Frequency", y="Count", color="Frequency",
                             text="Count", color_discrete_sequence=BOLD)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Shopping Frequency"), use_container_width=True)

            if "Habit_Type" in df.columns:
                habit = df["Habit_Type"].value_counts().reset_index()
                habit.columns = ["Habit", "Count"]
                fig = px.pie(habit, names="Habit", values="Count",
                             hole=0.4, color_discrete_sequence=SET2)
                st.plotly_chart(_layout(fig, "Habit Type"), use_container_width=True)

        with col2:
            if "Last_Purchase" in df.columns:
                lp = df["Last_Purchase"].value_counts().reindex(LAST_BUY_ORDER).dropna().reset_index()
                lp.columns = ["Last_Purchase", "Count"]
                fig = px.bar(lp, x="Last_Purchase", y="Count", color="Last_Purchase",
                             text="Count", color_discrete_sequence=PASTEL)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Last Purchase Recency"), use_container_width=True)

            if "Cart_Abandonment" in df.columns:
                ca = df["Cart_Abandonment"].value_counts().reset_index()
                ca.columns = ["Cart_Abandonment", "Count"]
                fig = px.pie(ca, names="Cart_Abandonment", values="Count",
                             hole=0.45, color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50"])
                st.plotly_chart(_layout(fig, "Cart Abandonment Frequency"), use_container_width=True)

        if "Purchase_Influence" in df.columns:
            st.subheader("📣 What Influences Purchase Decisions?")
            infl = df["Purchase_Influence"].value_counts().reset_index()
            infl.columns = ["Influence", "Count"]
            fig = px.bar(infl.sort_values("Count"), x="Count", y="Influence",
                         orientation="h", color="Count",
                         color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
            st.plotly_chart(_layout(fig, "Purchase Influence Factors", 360), use_container_width=True)

        if "Decision_Style" in df.columns:
            st.subheader("🧩 Decision Style × Interest")
            chi_ds = chi_square_test(df, "Decision_Style", "Interest_in_MoodCart")
            st.markdown(_chi_badge(chi_ds), unsafe_allow_html=True)
            ct_ds = pd.crosstab(df["Decision_Style"], df["Interest_in_MoodCart"])
            ct_pct_ds = ct_ds.div(ct_ds.sum(axis=1), axis=0) * 100
            melted_ds = ct_pct_ds.reset_index().melt(id_vars="Decision_Style", var_name="Interest", value_name="%")
            cmap_ds = {k: v for k, v in INTEREST_COLORS.items() if k in melted_ds["Interest"].unique()}
            fig = px.bar(melted_ds, x="Decision_Style", y="%", color="Interest",
                         barmode="stack", color_discrete_map=cmap_ds)
            st.plotly_chart(_layout(fig, "Decision Style vs MoodCart Interest"), use_container_width=True)

    # ── 4 MOOD & EMOTIONS ─────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("🧠 Mood & Emotional Patterns")

        if "Mood" in df.columns:
            col1, col2 = st.columns(2)
            mood_counts = df["Mood"].value_counts().reset_index()
            mood_counts.columns = ["Mood", "Count"]
            with col1:
                fig = px.bar(mood_counts, x="Mood", y="Count",
                             color="Mood", color_discrete_map=MOOD_COLORS, text="Count")
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Mood Distribution"), use_container_width=True)
            with col2:
                fig2 = px.pie(mood_counts, names="Mood", values="Count",
                              hole=0.45, color="Mood", color_discrete_map=MOOD_COLORS)
                st.plotly_chart(_layout(fig2, "Mood Share"), use_container_width=True)

        if "Mood" in df.columns and "Interest_in_MoodCart" in df.columns:
            st.subheader("🧩 Mood × Interest (Chi-Square Tested)")
            chi_mood = chi_square_test(df, "Mood", "Interest_in_MoodCart")
            st.markdown(_chi_badge(chi_mood), unsafe_allow_html=True)
            ct = pd.crosstab(df["Mood"], df["Interest_in_MoodCart"])
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            fig = px.imshow(ct_pct.round(1),
                            color_continuous_scale=["#0d0d1a", SECONDARY, PRIMARY, ACCENT],
                            text_auto=True)
            st.plotly_chart(_layout(fig, "Mood vs Interest Heatmap (% row)", 420), use_container_width=True)

        if "Mood" in df.columns and "Monthly_Spend" in df.columns:
            st.subheader("💸 Avg Monthly Spend by Mood")
            df3 = df.copy()
            df3["Monthly_Spend"] = pd.to_numeric(df3["Monthly_Spend"], errors="coerce")
            ms = df3.groupby("Mood")["Monthly_Spend"].agg(["mean", "std"]).reset_index()
            ms.columns = ["Mood", "Avg_Spend", "Std"]
            ms = ms.sort_values("Avg_Spend", ascending=False)
            fig = px.bar(ms, x="Mood", y="Avg_Spend",
                         color="Mood", text=ms["Avg_Spend"].round(0),
                         color_discrete_map=MOOD_COLORS,
                         error_y="Std")
            fig.update_traces(texttemplate="₹%{text}", textposition="outside")
            st.plotly_chart(_layout(fig, "Average Spend per Mood (with Std Dev)"), use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            if "Impulse_Behavior" in df.columns:
                ib = df["Impulse_Behavior"].value_counts().reset_index()
                ib.columns = ["Impulse", "Count"]
                chi_ib = chi_square_test(df, "Impulse_Behavior", "Interest_in_MoodCart")
                st.markdown(_chi_badge(chi_ib), unsafe_allow_html=True)
                fig = px.pie(ib, names="Impulse", values="Count", hole=0.4,
                             color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50"])
                st.plotly_chart(_layout(fig, "Impulse Buying Behaviour"), use_container_width=True)

        with col6:
            if "Mood_Impact" in df.columns:
                mi = df["Mood_Impact"].value_counts().reset_index()
                mi.columns = ["Mood_Impact", "Count"]
                fig = px.bar(mi, x="Mood_Impact", y="Count", color="Mood_Impact",
                             text="Count", color_discrete_sequence=SET2)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Does Mood Impact Shopping?"), use_container_width=True)

        if "Post_Purchase_Feeling" in df.columns:
            ppf = df["Post_Purchase_Feeling"].value_counts().reset_index()
            ppf.columns = ["Feeling", "Count"]
            fig = px.bar(ppf.sort_values("Count", ascending=True),
                         x="Count", y="Feeling", orientation="h",
                         color="Count",
                         color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
            st.plotly_chart(_layout(fig, "Post-Purchase Emotion Distribution", 360), use_container_width=True)

    # ── 5 SPEND ANALYSIS ──────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("💰 Spend Analysis")
        df_s = df.copy()
        df_s["Monthly_Spend"] = pd.to_numeric(df_s["Monthly_Spend"], errors="coerce")

        c1, c2, c3, c4 = st.columns(4)
        _kpi(c1, "Mean Spend",   f"₹{df_s['Monthly_Spend'].mean():,.0f}")
        _kpi(c2, "Median Spend", f"₹{df_s['Monthly_Spend'].median():,.0f}")
        _kpi(c3, "Max Spend",    f"₹{df_s['Monthly_Spend'].max():,.0f}")
        _kpi(c4, "Std Dev",      f"₹{df_s['Monthly_Spend'].std():,.0f}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_s, x="Monthly_Spend", nbins=50,
                               color_discrete_sequence=[PRIMARY])
            fig.add_vline(x=df_s["Monthly_Spend"].mean(), line_dash="dash",
                          line_color=ACCENT, annotation_text=f"Mean ₹{df_s['Monthly_Spend'].mean():,.0f}")
            fig.add_vline(x=df_s["Monthly_Spend"].median(), line_dash="dot",
                          line_color="#4CAF50", annotation_text=f"Median ₹{df_s['Monthly_Spend'].median():,.0f}")
            st.plotly_chart(_layout(fig, "Monthly Spend Distribution (right-skewed → log transform in ML)"), use_container_width=True)

        with col2:
            if "Income" in df.columns:
                fig = px.box(df_s, x="Income", y="Monthly_Spend", color="Income",
                             category_orders={"Income": INCOME_ORDER},
                             color_discrete_sequence=BOLD)
                st.plotly_chart(_layout(fig, "Spend by Income Bracket"), use_container_width=True)

        if "Age" in df.columns:
            fig = px.box(df_s, x="Age", y="Monthly_Spend", color="Age",
                         category_orders={"Age": AGE_ORDER},
                         color_discrete_sequence=BOLD)
            st.plotly_chart(_layout(fig, "Spend by Age Group"), use_container_width=True)

        if "Willingness_To_Spend_More" in df.columns:
            st.subheader("📈 Willingness to Spend More")
            wts = df["Willingness_To_Spend_More"].value_counts().reset_index()
            wts.columns = ["Willingness", "Count"]
            wts["Pct"] = (wts["Count"] / wts["Count"].sum() * 100).round(1)
            col3, col4 = st.columns(2)
            with col3:
                fig = px.bar(wts, x="Willingness", y="Count", color="Willingness",
                             text=wts["Pct"].astype(str) + "%",
                             color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800"])
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Willingness to Pay More for Personalized Reco"), use_container_width=True)
            with col4:
                # Spend vs willingness box
                fig2 = px.box(df_s, x="Willingness_To_Spend_More", y="Monthly_Spend",
                              color="Willingness_To_Spend_More",
                              color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800"])
                st.plotly_chart(_layout(fig2, "Spend Distribution by Willingness"), use_container_width=True)

        if "Mood" in df.columns and "Willingness_To_Spend_More" in df.columns:
            st.subheader("🧠 Willingness to Spend More × Mood")
            ct = pd.crosstab(df["Mood"], df["Willingness_To_Spend_More"])
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            fig = px.imshow(ct_pct.round(1),
                            color_continuous_scale=["#0d0d1a", SECONDARY, PRIMARY, ACCENT],
                            text_auto=True)
            st.plotly_chart(_layout(fig, "Mood vs WTP Heatmap (% row)", 380), use_container_width=True)

    # ── 6 PRODUCTS & BUNDLES ──────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("📦 Category & Product Preferences")

        if "Categories" in df.columns:
            cats_raw = df["Categories"].fillna("")
            all_cats: dict = {}
            for row in cats_raw:
                for c in str(row).split("|"):
                    c = c.strip()
                    if c:
                        all_cats[c] = all_cats.get(c, 0) + 1
            cats_df = pd.DataFrame(list(all_cats.items()), columns=["Category", "Count"])
            cats_df = cats_df.sort_values("Count", ascending=True)
            fig = px.bar(cats_df, x="Count", y="Category", orientation="h",
                         color="Count", color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
            st.plotly_chart(_layout(fig, "Category Preferences (multi-select)", 380), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "Stress_Purchases" in df.columns:
                sp_raw = df["Stress_Purchases"].fillna("")
                sp_dict: dict = {}
                for row in sp_raw:
                    for c in str(row).split("|"):
                        c = c.strip()
                        if c:
                            sp_dict[c] = sp_dict.get(c, 0) + 1
                sp_df = pd.DataFrame(list(sp_dict.items()), columns=["Category", "Count"])
                sp_df = sp_df.sort_values("Count", ascending=False)
                fig = px.bar(sp_df, x="Category", y="Count", color="Category",
                             text="Count", color_discrete_sequence=BOLD)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Stress Purchase Categories"), use_container_width=True)

        with col2:
            if "Shopping_Situations" in df.columns:
                sit_raw = df["Shopping_Situations"].fillna("")
                sit_dict: dict = {}
                for row in sit_raw:
                    for c in str(row).split("|"):
                        c = c.strip()
                        if c:
                            sit_dict[c] = sit_dict.get(c, 0) + 1
                sit_df = pd.DataFrame(list(sit_dict.items()), columns=["Situation", "Count"])
                fig = px.bar(sit_df.sort_values("Count", ascending=False),
                             x="Situation", y="Count", color="Situation",
                             text="Count", color_discrete_sequence=SET2)
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "Shopping Trigger Situations"), use_container_width=True)

        if "Product_Combinations" in df.columns:
            st.subheader("🧺 Popular Product Bundle Combinations")
            pc_raw = df["Product_Combinations"].fillna("")
            pc_dict: dict = {}
            for row in pc_raw:
                for c in str(row).split("|"):
                    c = c.strip()
                    if c:
                        pc_dict[c] = pc_dict.get(c, 0) + 1
            pc_df = pd.DataFrame(list(pc_dict.items()), columns=["Bundle", "Count"])
            pc_df = pc_df.sort_values("Count", ascending=True)
            fig = px.bar(pc_df, x="Count", y="Bundle", orientation="h",
                         color="Count", color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
            st.plotly_chart(_layout(fig, "Product Bundle Frequency", 380), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "Value_Perception" in df.columns:
                vp = df["Value_Perception"].value_counts().reset_index()
                vp.columns = ["Perception", "Count"]
                fig = px.pie(vp, names="Perception", values="Count",
                             hole=0.4, color_discrete_sequence=PASTEL)
                st.plotly_chart(_layout(fig, "Value Perception"), use_container_width=True)
        with col4:
            if "Priority" in df.columns:
                pr = df["Priority"].value_counts().reset_index()
                pr.columns = ["Priority", "Count"]
                fig = px.bar(pr.sort_values("Count"), x="Count", y="Priority",
                             orientation="h", color="Count",
                             color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
                st.plotly_chart(_layout(fig, "Top Purchase Priority Factors", 360), use_container_width=True)

    # ── 7 BARRIERS & TRUST ────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("🚧 Barriers, Hesitations & AI Trust")

        col1, col2 = st.columns(2)
        with col1:
            if "AI_Trust" in df.columns:
                chi_at = chi_square_test(df, "AI_Trust", "Interest_in_MoodCart")
                st.markdown(_chi_badge(chi_at), unsafe_allow_html=True)
                at = df["AI_Trust"].value_counts().reset_index()
                at.columns = ["AI_Trust", "Count"]
                fig = px.bar(at, x="AI_Trust", y="Count", color="AI_Trust",
                             text="Count", color_discrete_sequence=[PRIMARY, ACCENT, "#FF9800", "#4CAF50"])
                fig.update_traces(textposition="outside")
                st.plotly_chart(_layout(fig, "AI Trust Level"), use_container_width=True)

        with col2:
            if "Privacy_Comfort" in df.columns:
                chi_pc = chi_square_test(df, "Privacy_Comfort", "Interest_in_MoodCart")
                st.markdown(_chi_badge(chi_pc), unsafe_allow_html=True)
                pc = df["Privacy_Comfort"].value_counts().reset_index()
                pc.columns = ["Privacy_Comfort", "Count"]
                fig = px.pie(pc, names="Privacy_Comfort", values="Count",
                             hole=0.4, color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT, "#FF9800"])
                st.plotly_chart(_layout(fig, "Privacy Comfort Level"), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "Hesitation" in df.columns:
                hes = df["Hesitation"].value_counts().reset_index()
                hes.columns = ["Hesitation", "Count"]
                fig = px.bar(hes.sort_values("Count"), x="Count", y="Hesitation",
                             orientation="h", color="Count",
                             color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
                st.plotly_chart(_layout(fig, "Main Hesitation Factors", 360), use_container_width=True)

        with col4:
            if "Data_Concern" in df.columns:
                dc = df["Data_Concern"].value_counts().reset_index()
                dc.columns = ["Data_Concern", "Count"]
                fig = px.bar(dc.sort_values("Count"), x="Count", y="Data_Concern",
                             orientation="h", color="Count",
                             color_continuous_scale=["#2a1a3e", PRIMARY, ACCENT])
                st.plotly_chart(_layout(fig, "Data Concern Reasons", 350), use_container_width=True)

        # AI trust vs interest cross-tab
        if "AI_Trust" in df.columns and "Interest_in_MoodCart" in df.columns:
            st.subheader("🔗 AI Trust × Interest in MoodCart")
            ct_at = pd.crosstab(df["AI_Trust"], df["Interest_in_MoodCart"])
            ct_pct_at = ct_at.div(ct_at.sum(axis=1), axis=0) * 100
            fig = px.imshow(ct_pct_at.round(1),
                            color_continuous_scale=["#0d0d1a", SECONDARY, PRIMARY, ACCENT],
                            text_auto=True)
            st.plotly_chart(_layout(fig, "AI Trust vs Interest Heatmap (% row)", 380), use_container_width=True)
