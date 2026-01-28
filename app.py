import pandas as pd
import joblib
import streamlit as st
LABEL_STYLES = {
    "worth_it": {"bg": "#EEDFCF", "fg": "#5C3A21", "border": "#DDBFA5"},
    "underrated": {"bg": "#F7EEDF", "fg": "#4B3B2A", "border": "#D9C7AE"},
    "overrated": {"bg": "#EAD7C5", "fg": "#5A3E2B", "border": "#D9BFA9"},
}
MODEL_PATH = "models/rf_hype_classifier.joblib"
DATA_PATH = "data/labeled_products.csv"
TARGET = "hype_label"

st.set_page_config(page_title="Sephora Worth-the-Hype", layout="wide")
st.markdown("""
<style>
/* Progress bar background */
div[data-testid="stProgress"] > div > div {
    background-color: #F5EFE6;  /* creamy background */
}

/* Progress bar fill */
div[data-testid="stProgress"] > div > div > div {
    background-color: #C8B6A6;  /* soft taupe / beige */
}
</style>
""", unsafe_allow_html=True)
st.title("🧴 Sephora: Worth the Hype? (ML Classifier)")
st.caption("Pick a product and see the model's prediction based on product metadata (brand, category, price, flags).")
st.divider()

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

st.sidebar.header("Filters")

# Filters
brands = ["All"] + sorted(df["brand_name"].dropna().unique().tolist())
cats = ["All"] + sorted(df["primary_category"].dropna().unique().tolist())

brand = st.sidebar.selectbox("Brand", brands)
cat = st.sidebar.selectbox("Primary category", cats)

filtered = df.copy()
if brand != "All":
    filtered = filtered[filtered["brand_name"] == brand]
if cat != "All":
    filtered = filtered[filtered["primary_category"] == cat]

st.subheader("Select a product")

options = filtered["product_name"].fillna("Unknown").astype(str).unique().tolist()
if not options:
    st.warning("No products match the filters.")
    st.stop()

product_name = st.selectbox("Product", options)

row = filtered[filtered["product_name"].astype(str) == product_name].iloc[0]

left, right = st.columns(2)

with left:
    st.markdown("### 🧾 Product card")

    card_cols = [
        "product_name", "brand_name",
        "primary_category", "secondary_category", "tertiary_category",
        "price_usd", "sale_price_usd",
        "sephora_exclusive", "limited_edition", "new", "online_only", "out_of_stock"
    ]
    card_cols = [c for c in card_cols if c in row.index]

    # Show as nice key-value list
    show = []
    for c in card_cols:
        val = row[c]
        if pd.isna(val):
            continue
        # prettier booleans
        if isinstance(val, (bool,)) or c in ["sephora_exclusive", "limited_edition", "new", "online_only", "out_of_stock"]:
            val = "Yes" if str(val).lower() in ["1", "true", "yes"] else "No"
        show.append((c, val))

    card_df = pd.DataFrame(show, columns=["field", "value"])
    card_df["value"] = card_df["value"].astype(str)

    st.dataframe(card_df, use_container_width=True, hide_index=True)
with right:
    st.markdown("### Prediction")

    X = pd.DataFrame([row]).drop(columns=[TARGET], errors="ignore")
    pred = model.predict(X)[0]
    proba = float(model.predict_proba(X).max())

    style = LABEL_STYLES.get(pred, {"bg": "#F2F2F2", "fg": "#333", "border": "#DDD"})

    # extra info (safe)
    price = row["price_usd"] if "price_usd" in row.index else None
    cat = row["primary_category"] if "primary_category" in row.index else None

    price_txt = f"${float(price):.2f}" if price is not None and str(price) != "nan" else "N/A"
    cat_txt = str(cat) if cat is not None and str(cat) != "nan" else "N/A"
    conf_txt = f"{int(proba*100)}%"

    card_html = f"""
    <div style="
        background:#FBF7F2;
        border:1px solid #EFE6D8;
        border-radius:18px;
        padding:18px 18px;
    ">
      <div style="font-size:13px; letter-spacing:0.6px; color:#7A6A5A; text-transform:uppercase;">
        Model prediction
      </div>

      <div style="margin-top:12px;">
        <span style="
          display:inline-block;
          padding:10px 14px;
          border-radius:14px;
          background:{style['bg']};
          color:{style['fg']};
          border:1px solid {style['border']};
          font-weight:700;
          font-size:20px;
        ">
          {pred.replace('_',' ').title()}
        </span>
      </div>

      <div style="margin-top:16px; display:flex; gap:14px; flex-wrap:wrap;">
        <div style="flex:1; min-width:120px; background:#FFFFFF; border:1px solid #EFE6D8; border-radius:14px; padding:12px;">
          <div style="font-size:12px; color:#7A6A5A;">Confidence</div>
          <div style="font-size:22px; font-weight:700; color:#4B3B2A;">{conf_txt}</div>
        </div>
        <div style="flex:1; min-width:120px; background:#FFFFFF; border:1px solid #EFE6D8; border-radius:14px; padding:12px;">
          <div style="font-size:12px; color:#7A6A5A;">Price</div>
          <div style="font-size:22px; font-weight:700; color:#4B3B2A;">{price_txt}</div>
        </div>
        <div style="flex:1; min-width:160px; background:#FFFFFF; border:1px solid #EFE6D8; border-radius:14px; padding:12px;">
          <div style="font-size:12px; color:#7A6A5A;">Category</div>
          <div style="font-size:16px; font-weight:700; color:#4B3B2A;">{cat_txt}</div>
        </div>
      </div>

      <div style="margin-top:14px; font-size:12.5px; color:#7A6A5A;">
        This prediction uses <b>metadata-only</b> features (brand, category, price, flags). No rating/review fields are used.
      </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)