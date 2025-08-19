import streamlit as st
import requests
import pandas as pd
import math
import random
import json
import re
from urllib.parse import urlencode

# ========================= 
# USER CONFIG (secrets)
# =========================

CLIENT_ID = st.secrets["ebay"]["CLIENT_ID"]
CLIENT_SECRET = st.secrets["ebay"]["CLIENT_SECRET"]


# =========================
# CONSTANTS
# =========================
CATEGORY_IDS = {
    "CDs": "176984",
    "Cassettes": "176983",
    "DVDs/Blu-ray": "617",
}

KEY_ASPECTS = {
    "CDs": ["Artist", "Genre", "Record Label", "Release Year", "Format"],
    "Cassettes": ["Artist", "Genre", "Record Label", "Release Year", "Format"],
    "DVDs/Blu-ray": ["Director", "Genre", "Studio", "Release Year", "Format"],
}

# Trend targets we actually rank
TREND_TARGETS = {
    "CDs": ["Decade", "Genre", "Artist", "Record Label"],
    "Cassettes": ["Decade", "Genre", "Artist", "Record Label"],
    "DVDs/Blu-ray": ["Decade", "Genre", "Studio", "Director"],
}

SORT_OPTIONS = {
    "Best Match": "best_match",
    "Price: High to Low": "-price",
    "Price: Low to High": "price",
    "Newly Listed": "-startTime",
    "Ending Soonest": "endTime",
}

MAX_PER_CALL = 1000
CATEGORY_TREE_ID = "0"

# =========================
# AUTH
# =========================
@st.cache_data(ttl=3500)
def get_access_token():
    url = "https://api.ebay.com/identity/v1/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    auth = (CLIENT_ID, CLIENT_SECRET)
    try:
        r = requests.post(url, headers=headers, data=data, auth=auth, timeout=30)
        if r.status_code == 200:
            return r.json().get("access_token")
        st.error(f"OAuth token request failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        st.error(f"OAuth request error: {str(e)}")
        return None

# =========================
# HELPERS (parsing & splitting)
# =========================
_SPLIT_PAT = re.compile(r"\s*(,|/|;|\s&\s|\sand\s)\s*", flags=re.IGNORECASE)

def _normalize_multival(val: str):
    """Extract multiple values from a string, but return only the first one for CSV storage"""
    if not isinstance(val, str) or not val.strip():
        return []
    s = _SPLIT_PAT.sub("|", val)
    parts = [p.strip() for p in s.split("|") if p.strip()]
    return parts

def _get_first_value(val: str):
    """Get only the first value from a multi-value string for CSV storage"""
    if not isinstance(val, str) or not val.strip():
        return None
    parts = _normalize_multival(val)
    return parts[0] if parts else None

def parse_year(value: str):
    if not isinstance(value, str):
        return None
    s = value.strip()

    # 1990s / 2000s
    m_decade = re.search(r"\b(19|20)\d0s\b", s)
    if m_decade:
        return int(m_decade.group(0)[:4])

    # explicit 4-digit years (take earliest)
    years = re.findall(r"(19|20)\d{2}", s)
    if years:
        return int(min(years))

    # '90s style
    m_2 = re.search(r"\b'?\d0s\b", s.lower())
    if m_2:
        n = re.search(r"\d", m_2.group(0)).group(0)
        decade = int(n) * 10
        prefix = "20" if decade in (0, 10) else "19"
        return int(f"{prefix}{decade:02d}")

    return None

def year_to_decade(year: int):
    if not isinstance(year, int):
        return None
    return f"{(year // 10) * 10}s"

# =========================
# ASPECT EXTRACTION
# =========================
def extract_item_aspects(item: dict, key_aspects: list):
    """
    Build two dicts:
      aspects_mapped -> only requested key aspects (best-effort) - SINGLE VALUES ONLY
      all_raw_aspects -> everything found (human readable)
    Works with detail response; also handles summary fallbacks.
    """
    all_raw = {}

    # Detail: localizedAspects
    if "localizedAspects" in item:
        for a in item["localizedAspects"]:
            name = a.get("name", "").strip()
            val = a.get("value", "")
            if name:
                if isinstance(val, list):
                    all_raw[name] = ", ".join(str(v) for v in val if v)
                else:
                    all_raw[name] = str(val)

    # Fallback: itemSpecifics (if present)
    if "itemSpecifics" in item:
        for spec in item["itemSpecifics"]:
            name = spec.get("name", "").strip()
            vals = spec.get("values", [])
            if name and vals and name not in all_raw:
                all_raw[name] = ", ".join(str(v) for v in vals if v)

    # Fallback: product.aspects
    if "product" in item and "aspects" in item["product"]:
        for name, vals in item["product"]["aspects"].items():
            if name not in all_raw and vals:
                all_raw[name] = ", ".join(str(v) for v in vals if v)

    # Normalize: treat Style as Genre if Genre missing
    if "Genre" not in all_raw and "Style" in all_raw:
        all_raw["Genre"] = all_raw["Style"]

    # Map with tolerant synonyms - BUT RETURN ONLY FIRST VALUE
    def match_value(target_name: str):
        if target_name in all_raw:
            return _get_first_value(all_raw[target_name])
        tn = target_name.lower()
        for raw_name, raw_val in all_raw.items():
            rn = raw_name.lower()
            if rn == tn:
                return _get_first_value(raw_val)
            if tn == "artist" and any(x in rn for x in ["artist", "performer", "musician"]):
                return _get_first_value(raw_val)
            if tn == "director" and "director" in rn:
                return _get_first_value(raw_val)
            if tn == "genre" and any(x in rn for x in ["genre", "style"]):
                return _get_first_value(raw_val)
            if tn == "record label" and any(x in rn for x in ["label", "record label", "publisher", "record company"]):
                return _get_first_value(raw_val)
            if tn == "studio" and any(x in rn for x in ["studio", "publisher", "label"]):
                return _get_first_value(raw_val)
            if tn == "release year" and any(x in rn for x in ["release year", "year", "release date", "published"]):
                return _get_first_value(raw_val)
            if tn == "brand" and "brand" in rn:
                return _get_first_value(raw_val)
            if tn == "model" and "model" in rn:
                return _get_first_value(raw_val)
            if tn == "type" and "type" in rn:
                return _get_first_value(raw_val)
            if tn == "color" and "color" in rn:
                return _get_first_value(raw_val)
            if tn == "connectivity" and "connectivity" in rn:
                return _get_first_value(raw_val)
        return None

    aspects_mapped = {k: match_value(k) for k in key_aspects}
    return aspects_mapped, all_raw

# =========================
# API CALLS
# =========================
def browse_search(token: str, category_id: str, min_price: float, max_price: float,
                  limit: int, offset: int, sort_order: str):
    params = {
        "category_ids": category_id,
        "filter": f"price:[{min_price:.2f}..{max_price:.2f}],priceCurrency:USD,buyingOptions:{{FIXED_PRICE}},itemLocationCountry:US",
        "delivery_country": "US",
        "limit": min(MAX_PER_CALL, limit),
        "offset": offset,
        "sort": sort_order,
        "fieldgroups": "EXTENDED,PRODUCT",
    }
    url = "https://api.ebay.com/buy/browse/v1/item_summary/search?" + urlencode(params)
    headers = {"Authorization": f"Bearer {token}"}
    return requests.get(url, headers=headers, timeout=45)

def fetch_item_details(token: str, item_id: str) -> dict:
    url = f"https://api.ebay.com/buy/browse/v1/item/{item_id}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code == 200:
        return r.json()
    return {}

@st.cache_data(ttl=1800)
def fetch_category_data(category: str, min_price: float, max_price: float, sort_order_key: str, 
                       listings_to_fetch: int = 150, enrich_target: int = 150):
    """
    1) Pull up to listings_to_fetch summaries (fast)
    2) Randomly pick enrich_target items and call detail endpoint to get aspects
    3) Merge aspects and derive Decade
    4) Store only FIRST value for multi-value fields in CSV columns
    """
    token = get_access_token()
    if not token:
        return pd.DataFrame()

    category_id = CATEGORY_IDS[category]
    key_aspects = KEY_ASPECTS.get(category, [])
    sort_order = SORT_OPTIONS[sort_order_key]

    # ---- summaries
    pages = max(1, math.ceil(listings_to_fetch / MAX_PER_CALL))
    all_items = []
    prog = st.progress(0.0, text="Fetching summaries…")
    for p in range(pages):
        remaining = listings_to_fetch - len(all_items)
        if remaining <= 0:
            break
        r = browse_search(token, category_id, min_price, max_price, remaining, p * MAX_PER_CALL, sort_order)
        if r.status_code != 200:
            st.error(f"Browse API error on page {p+1}: {r.status_code} - {r.text}")
            break
        data = r.json()
        items = data.get("itemSummaries", [])
        for it in items:
            price_info = it.get("price", {})
            try:
                price = float(price_info.get("value"))
            except (TypeError, ValueError):
                continue
            title = it.get("title", "")
            row = {
                "Title": title[:100] + "..." if len(title) > 100 else title,
                "Price": price,
                "Currency": price_info.get("currency", "USD"),
                "Item_URL": it.get("itemWebUrl", ""),
                "Item_ID": it.get("itemId", ""),
                "Condition": it.get("condition", ""),
                "Seller": it.get("seller", {}).get("username", ""),
                "Raw_Aspects_Count": 0,
                "Raw_Aspects_JSON": None,
            }
            all_items.append(row)
        prog.progress((p+1)/pages, text=f"Fetched page {p+1}/{pages} — {len(all_items)} items")
    prog.empty()

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    # ---- aspect enrichment (random sample)
    ids = df["Item_ID"].dropna().unique().tolist()
    sample_n = min(enrich_target, len(ids))
    random.seed(42)  # reproducible
    sample_ids = random.sample(ids, sample_n)

    st.info(f"Enriching aspects for {sample_n} items (detail calls)…")
    eprog = st.progress(0.0, text="Fetching item details…")

    # map back by item_id
    enrich_map = {}
    for i, iid in enumerate(sample_ids, 1):
        detail = fetch_item_details(token, iid)
        if detail:
            mapped, raw = extract_item_aspects(detail, key_aspects)
            enrich_map[iid] = (mapped, raw)
        eprog.progress(i / sample_n, text=f"Fetched {i}/{sample_n} detail records")
    eprog.empty()

    # apply to df
    df.set_index("Item_ID", inplace=True)
    # ensure columns exist
    for k in key_aspects:
        if k not in df.columns:
            df[k] = pd.NA
    for iid, (mapped, raw) in enrich_map.items():
        for k, v in mapped.items():
            df.at[iid, k] = v  # v is already the first value only
        df.at[iid, "Raw_Aspects_JSON"] = json.dumps(raw, ensure_ascii=False)
        df.at[iid, "Raw_Aspects_Count"] = len(raw)
    df.reset_index(inplace=True)

    # ---- derived decade from Release Year (if present)
    if "Release Year" in df.columns:
        years = df["Release Year"].fillna("").astype(str).map(parse_year)
        df["Release_Year_Parsed"] = years
        df["Decade"] = years.map(year_to_decade)
    else:
        df["Decade"] = None

    return df

# =========================
# ANALYSIS
# =========================
def _explode_multi_for_analysis(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """For ANALYSIS ONLY - explode multi-values to see all genres in trend analysis"""
    if col not in df.columns:
        df = df.copy()
        df[col] = pd.NA
        return df
    
    # For analysis, we want to see all values, so we reconstruct them from raw JSON if available
    def extract_all_values(row):
        # First try the single value from the column
        single_val = row[col]
        if pd.notna(single_val) and str(single_val).strip():
            # Also check raw JSON for additional values
            raw_json = row.get("Raw_Aspects_JSON")
            if pd.notna(raw_json):
                try:
                    raw_data = json.loads(raw_json)
                    # Look for Genre in raw data
                    for key, value in raw_data.items():
                        if key.lower() in ["genre", "style"] and value:
                            all_vals = _normalize_multival(value)
                            if all_vals:
                                return all_vals
                except:
                    pass
            # Fallback to single value as list
            return [str(single_val)]
        return []
    
    lists = df.apply(extract_all_values, axis=1)
    out = df.copy()
    out[col] = lists
    out = out.explode(col)
    out[col] = out[col].replace("", pd.NA)
    return out

def analyze_profitable_aspect(df: pd.DataFrame, aspect: str, min_occurrences: int = 3):
    if df.empty or "Price" not in df.columns:
        return pd.DataFrame()
    work = _explode_multi_for_analysis(df, aspect).dropna(subset=[aspect])
    if work.empty:
        return pd.DataFrame()

    global_median = df["Price"].median()
    grouped = (work.groupby(aspect)["Price"]
               .agg(Count="count", Median_Price="median", Mean_Price="mean")
               .round(2)
               .reset_index())
    grouped = grouped[grouped["Count"] >= min_occurrences]
    if grouped.empty:
        return pd.DataFrame()

    grouped["Premium_vs_Global_Median"] = (grouped["Median_Price"] - global_median).round(2)
    grouped["Premium_Percentage"] = ((grouped["Median_Price"] / global_median - 1) * 100).round(1)
    grouped["Value_Score"] = (grouped["Premium_vs_Global_Median"] * grouped["Count"] / 10).round(2)
    grouped = grouped.rename(columns={aspect: "Aspect_Value"})
    grouped.insert(0, "Aspect_Name", aspect)
    return grouped.sort_values(["Value_Score", "Premium_vs_Global_Median", "Count"], ascending=[False, False, False])

# =========================
# UI
# =========================
st.set_page_config(page_title="eBay Reselling Trends (Aspects)", page_icon="📈", layout="wide")
st.title("📈 eBay Reselling Trends - Media")
st.markdown("**Note:** CSV exports contain only the **first genre/value** for clean data analysis, but trend analysis shows all genres.")

st.sidebar.header("⚙️ Parameters")
category = st.sidebar.selectbox("📦 Category", list(CATEGORY_IDS.keys()), index=0)

# Configurable sample sizes
listings_count = st.sidebar.number_input("📊 Listings to Fetch", value=150, min_value=50, max_value=1000, step=25)
detail_calls = st.sidebar.number_input("🔍 Detail Calls (Aspect Enrichment)", value=150, min_value=25, max_value=1000, step=25)

min_price = st.sidebar.number_input("💰 Min Price ($)", value=5.0, min_value=0.0, step=1.0)
max_price = st.sidebar.number_input("💰 Max Price ($)", value=100.0, min_value=1.0, step=1.0)
sort_order_key = st.sidebar.selectbox("🔄 Sort Order", list(SORT_OPTIONS.keys()), index=0)
min_occurrences = st.sidebar.slider("📈 Min Occurrences (keep values appearing at least…)", 2, 20, 3)
top_n = st.sidebar.slider("🏆 Show Top N per Aspect", 5, 100, 25, step=5)

# Usage note
browse_calls = math.ceil(listings_count / MAX_PER_CALL)
st.sidebar.markdown(
    f"**API usage this run:** {browse_calls} browse call + {detail_calls} detail calls = **{browse_calls + detail_calls} total**"
)

if st.sidebar.button("🚀 Analyze Trends", type="primary"):
    key_aspects = KEY_ASPECTS.get(category, [])
    trend_targets = TREND_TARGETS.get(category, [])

    with st.spinner("📥 Fetching and enriching…"):
        df = fetch_category_data(category, float(min_price), float(max_price), sort_order_key, 
                               listings_count, detail_calls)

    if df.empty:
        st.error("No data returned. Try different parameters.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Profitable Trends",
        "📋 Listings (with aspects)",
        "🔍 Raw Aspect JSON",
        "📊 Market Overview",
    ])

    with tab1:
        st.subheader("🎯 Most Profitable Aspect Values (Median premium vs. global median)")
        st.caption("⚠️ Trend analysis shows ALL genres/values found (exploded from raw data)")
        frames = []
        for tgt in trend_targets:
            tdf = analyze_profitable_aspect(df, tgt, min_occurrences=min_occurrences)
            if not tdf.empty:
                frames.append(tdf.groupby("Aspect_Name").head(top_n))
        if frames:
            trends = pd.concat(frames, ignore_index=True)
            st.dataframe(
                trends,
                column_config={
                    'Median_Price': st.column_config.NumberColumn("Median Price", format="$%.2f"),
                    'Premium_vs_Global_Median': st.column_config.NumberColumn("Premium vs Median", format="$%.2f"),
                    'Premium_Percentage': st.column_config.NumberColumn("Premium %", format="%.1f%%"),
                    'Value_Score': st.column_config.NumberColumn("Value Score", format="%.2f"),
                },
                use_container_width=True, hide_index=True
            )
            st.download_button("📥 Download Trends CSV", trends.to_csv(index=False),
                               file_name=f"{category}_trends_{detail_calls}.csv", mime="text/csv")
        else:
            st.info("No significant trends at current thresholds.")

    with tab2:
        st.subheader("📋 Listings with Key Aspects (enriched sample included)")
        st.caption("⚠️ Displayed data shows FIRST genre/value only (clean for CSV export)")
        display_cols = ['Title', 'Price', 'Condition', 'Seller', 'Item_URL'] + key_aspects + ["Decade"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols],
            column_config={
                "Item_URL": st.column_config.LinkColumn("Link", display_text="View Listing"),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            },
            use_container_width=True
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Items", len(df))
        with c2: st.metric("Median Price", f"${df['Price'].median():.2f}")
        with c3: st.metric("Average Price", f"${df['Price'].mean():.2f}")
        with c4: st.metric("Price Range", f"${df['Price'].min():.2f} - ${df['Price'].max():.2f}")

    with tab3:
        st.subheader("🔍 Sample Raw Aspect JSON (from enriched items)")
        st.caption("Raw data contains ALL genres/values found - only first is used in CSV")
        sample = df[df["Raw_Aspects_JSON"].notna()].head(10)
        if sample.empty:
            st.info("No enriched items found (unexpected).")
        else:
            idx = st.selectbox(
                "Pick an item",
                list(sample.index),
                format_func=lambda i: f"{sample.loc[i, 'Title'][:70]}"
            )
            st.json(json.loads(sample.loc[idx, "Raw_Aspects_JSON"]))

    with tab4:
        st.subheader("📊 Price Distribution")
        pmin, pmax = df["Price"].min(), df["Price"].max()
        if pmax > pmin:
            bw = (pmax - pmin) / 10
            ranges, counts = [], []
            for i in range(10):
                s = pmin + i * bw
                e = pmin + (i + 1) * bw
                if i == 9:
                    count = len(df[(df["Price"] >= s) & (df["Price"] <= e)])
                else:
                    count = len(df[(df["Price"] >= s) & (df["Price"] < e)])
                ranges.append(f"${s:.0f}-${e:.0f}")
                counts.append(count)
            st.bar_chart(pd.DataFrame({"Price Range": ranges, "Count": counts}).set_index("Price Range"))
        else:
            st.write("Not enough variance to draw histogram.")

        # quick breakdowns for trend targets (these show all values for analysis)
        st.caption("⚠️ Breakdowns below show ALL values for comprehensive analysis")
        for aspect in TREND_TARGETS.get(category, []):
            if aspect in df.columns and df[aspect].notna().sum() > 0:
                st.subheader(f"Top {aspect} by Median Price (all values)")
                work = _explode_multi_for_analysis(df, aspect).dropna(subset=[aspect])
                stats = (work.groupby(aspect)["Price"]
                         .agg(Count="count", Median_Price="median")
                         .round(2)
                         .sort_values("Median_Price", ascending=False)
                         .head(10))
                st.dataframe(
                    stats,
                    column_config={'Median_Price': st.column_config.NumberColumn("Median Price", format="$%.2f")},
                    use_container_width=True
                )

    st.markdown("---")
    st.subheader("📥 Download Full Dataset")
    st.caption("⚠️ CSV contains FIRST genre/value only for clean data analysis")
    st.download_button(f"Download {category} dataset (CSV)", df.to_csv(index=False),
                       file_name=f"{category}_dataset_{detail_calls}.csv", mime="text/csv")
