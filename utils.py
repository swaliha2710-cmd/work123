import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# ─────────────────────────────────────────────────────────────────────────────
# Column categories
# ─────────────────────────────────────────────────────────────────────────────
PIPE_COLS = ["Categories", "Stress_Purchases", "Shopping_Situations", "Product_Combinations"]
DROP_FOR_MODEL = ["Happy_Purchases"]

# Ordered encodings (ordinal, not nominal)
INCOME_ORDER    = ["<20k", "20k-50k", "50k-1L", ">1L"]
AGE_ORDER       = ["Under 18", "18-24", "25-34", "35-44", "45+"]
FREQ_ORDER      = ["Rarely", "Monthly", "Weekly", "Daily"]
LAST_BUY_ORDER  = [">1 Month", "This Month", "This Week", "Today"]
BROWSING_ORDER  = ["<15 min", "15-30 min", "30-60 min", ">1 hour"]
ABANDON_ORDER   = ["Never", "Rarely", "Sometimes", "Often", "Always"]
EMOTIONAL_FREQ_ORDER = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
AI_TRUST_ORDER  = ["Not at all", "Slightly", "Moderately", "Highly"]
PRIVACY_ORDER   = ["Not comfortable", "Slightly comfortable", "Comfortable", "Very comfortable"]

ORDINAL_MAPS = {
    "Income":            {v: i for i, v in enumerate(INCOME_ORDER)},
    "Age":               {v: i for i, v in enumerate(AGE_ORDER)},
    "Shopping_Frequency":{v: i for i, v in enumerate(FREQ_ORDER)},
    "Last_Purchase":     {v: i for i, v in enumerate(LAST_BUY_ORDER)},
    "Browsing_Time":     {v: i for i, v in enumerate(BROWSING_ORDER)},
    "Cart_Abandonment":  {v: i for i, v in enumerate(ABANDON_ORDER)},
    "Emotional_Frequency":{v: i for i, v in enumerate(EMOTIONAL_FREQ_ORDER)},
    "AI_Trust":          {v: i for i, v in enumerate(AI_TRUST_ORDER)},
    "Privacy_Comfort":   {v: i for i, v in enumerate(PRIVACY_ORDER)},
}

MOOD_COLORS = {
    "Happy":   "#4CAF50",
    "Sad":     "#5C6BC0",
    "Bored":   "#FF9800",
    "Anxious": "#F44336",
    "Excited": "#E91E63",
    "Neutral": "#90A4AE",
    "Angry":   "#B71C1C",
    "Calm":    "#26C6DA",
    "Stressed":"#FF5722",
}

INTEREST_COLORS = {
    "Yes":   "#A855F7",
    "No":    "#F44336",
    "Maybe": "#FF9800",
}

PRIMARY   = "#A855F7"
SECONDARY = "#7C3AED"
ACCENT    = "#EC4899"

# ─────────────────────────────────────────────────────────────────────────────
# Persona definitions (named clusters for business storytelling)
# ─────────────────────────────────────────────────────────────────────────────
PERSONA_TEMPLATES = {
    0: {
        "name": "The Emotional Impulse Buyer",
        "icon": "⚡",
        "color": "#EC4899",
        "traits": ["Shops when stressed or bored", "High impulse behaviour", "Responds to mood-based deals"],
        "strategy": "Send real-time mood-triggered notifications. Bundle comfort + snack products.",
    },
    1: {
        "name": "The Rational Planner",
        "icon": "🧮",
        "color": "#A855F7",
        "traits": ["Planned purchases only", "Price-sensitive", "Reads reviews before buying"],
        "strategy": "Offer comparison tools, price-match guarantees, and detailed review summaries.",
    },
    2: {
        "name": "The Premium Lifestyle Shopper",
        "icon": "💎",
        "color": "#26C6DA",
        "traits": ["High monthly spend", "Values quality over price", "Open to personalization"],
        "strategy": "Premium membership tier. Curated recommendations. Early access to new products.",
    },
    3: {
        "name": "The Privacy-First Skeptic",
        "icon": "🔒",
        "color": "#FF9800",
        "traits": ["High data concern", "Low AI trust", "Mostly browses, rarely converts"],
        "strategy": "Transparent data policy, opt-in mood features, trust badges on all pages.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["Monthly_Spend"] = pd.to_numeric(df["Monthly_Spend"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# One-hot encode pipe-separated multi-select columns
# ─────────────────────────────────────────────────────────────────────────────
def one_hot_encode_multiselect(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    s = df[column].fillna("")
    items: set = set()
    for x in s:
        for it in str(x).split("|"):
            it = it.strip()
            if it:
                items.add(it)
    for it in sorted(items):
        df[f"{column}__{it}"] = s.apply(
            lambda x: 1 if it in [t.strip() for t in str(x).split("|")] else 0
        )
    return df.drop(columns=[column])


# ─────────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline — with ordinal encoding
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Apply ordinal encoding where order matters
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    # One-hot multi-selects
    for col in PIPE_COLS:
        if col in df.columns:
            df = one_hot_encode_multiselect(df, col)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Encode for ML  (returns X, y)
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_feature_names(X: "pd.DataFrame") -> "pd.DataFrame":
    """Remove/replace characters that XGBoost rejects: [ ] < >"""
    X = X.copy()
    X.columns = (
        X.columns
        .str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "lt", regex=False)
        .str.replace(">", "gt", regex=False)
    )
    return X


def encode_for_model(df: "pd.DataFrame", target_col=None):
    df = df.copy()
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])
    df = df.drop(columns=[c for c in DROP_FOR_MODEL if c in df.columns])
    X = pd.get_dummies(df, drop_first=True)
    X = sanitize_feature_names(X)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Chi-square test between two categorical columns
# ─────────────────────────────────────────────────────────────────────────────
def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(ct)
    cramers_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "dof": dof,
        "cramers_v": round(cramers_v, 4),
        "significant": p < 0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Segment helper
# ─────────────────────────────────────────────────────────────────────────────
def build_segment_profile(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    df2 = df.copy()
    df2["Monthly_Spend"] = pd.to_numeric(df2["Monthly_Spend"], errors="coerce")
    agg = (
        df2.groupby(segment_col)
        .agg(
            Count=("Monthly_Spend", "count"),
            Avg_Spend=("Monthly_Spend", "mean"),
            Median_Spend=("Monthly_Spend", "median"),
        )
        .round(0)
        .reset_index()
    )
    agg["Share_%"] = (agg["Count"] / agg["Count"].sum() * 100).round(1)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# PSM helpers (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
PSM_MIDPOINTS = {
    "<₹200":       175,
    "₹200-500":    350,
    "₹500-1000":   750,
    "₹1000-2000": 1500,
    "₹2000-3500": 2750,
    ">₹3500":     4000,
}

def psm_midpoint(val):
    val = str(val).strip()
    return PSM_MIDPOINTS.get(val, np.nan)
