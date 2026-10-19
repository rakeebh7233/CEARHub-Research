"""
Chatham County, GA: flood-damaged homeowners WITHOUT flood insurance.
Source: OpenFEMA "Individuals and Households Program - Valid Registrations" (v2).

Version 2 changes:
  - Reuses chatham_raw_records.csv if it exists (no re-download). Delete it to refresh.
  - Checks how detailed censusGeoid is and adds census-tract results.
  - Checks ownRent codes instead of assuming "O".
  - Two flood definitions: STRICT (floodDamage flag) and BROAD (flag OR flood damage
    amount > 0 OR inspected water level > 0 OR a high-water location recorded).
  - Combines all storms and hides areas with too few flooded owners (MIN_COUNT).

Run:   pip install requests pandas
       python chatham_uninsured_flood_owners.py
"""
import os
import requests
import pandas as pd

API = "https://www.fema.gov/api/open/v2/IndividualsAndHouseholdsProgramValidRegistrations"
DISASTERS = {4284: "Matthew (2016)", 4338: "Irma (2017)",
             4821: "Debby (2024)", 4830: "Helene (2024)"}
RAW_FILE = "chatham_raw_records.csv"
PAGE = 10000
MIN_COUNT = 10  # minimum flooded owners before an area's percentage is reported


# ---------- Download (only if no saved copy) ----------
def fetch(filter_str):
    rows, skip = [], 0
    while True:
        r = requests.get(API, params={"$filter": filter_str, "$top": PAGE, "$skip": skip}, timeout=120)
        r.raise_for_status()
        batch = r.json().get("IndividualsAndHouseholdsProgramValidRegistrations", [])
        rows += batch
        if len(batch) < PAGE:
            return rows
        skip += PAGE


def download():
    frames = []
    for num, name in DISASTERS.items():
        recs = fetch(f"disasterNumber eq {num} and county eq 'Chatham (County)'")
        if not recs:
            print(f"{name}: county filter empty, pulling all records for this disaster...")
            recs = fetch(f"disasterNumber eq {num}")
        df = pd.DataFrame(recs)
        if df.empty:
            continue
        df = df[df["county"].astype(str).str.lower().str.contains("chatham")]
        df["storm"] = name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(RAW_FILE, index=False)
    return data


# ---------- Helpers ----------
def truthy(v):
    return str(v).strip().lower() in {"true", "t", "1", "1.0", "yes", "y", "x"}


def positive(series):
    return pd.to_numeric(series, errors="coerce").fillna(0) > 0


def has_text(series):
    s = series.astype(str).str.strip().str.lower()
    return ~s.isin({"", "nan", "none", "null", "0", "false"})


def clean_code(series, width=None):
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    s = s.where(~s.str.lower().isin({"nan", "none", ""}))
    return s.str.zfill(width) if width else s


def summarize(df, group_cols, flood_col):
    sub = df[df["is_owner"] & df[flood_col]]
    out = sub.groupby(group_cols).agg(flooded_owners=("has_flood_ins", "size"),
                                      insured=("has_flood_ins", "sum")).reset_index()
    out["uninsured"] = out["flooded_owners"] - out["insured"]
    out["pct_uninsured"] = (100 * out["uninsured"] / out["flooded_owners"]).round(1)
    return out


# ---------- Main ----------
data = pd.read_csv(RAW_FILE, dtype=str, low_memory=False) if os.path.exists(RAW_FILE) else download()
print(f"Loaded {len(data):,} Chatham applications\n")

# 1. What codes does ownRent use?
print("ownRent values:\n", data["ownRent"].value_counts(dropna=False).to_string(), "\n")
data["is_owner"] = data["ownRent"].astype(str).str.strip().str.lower().str.startswith("o")

# 2. How detailed is censusGeoid?
data["geoid"] = clean_code(data["censusGeoid"])
lengths = data["geoid"].str.len().value_counts(dropna=False).sort_index()
print("censusGeoid lengths (11 = tract, 12 = block group, 15 = block):\n", lengths.to_string(), "\n")
data["tract"] = data["geoid"].where(data["geoid"].str.len() >= 11).str[:11]
data["zip"] = clean_code(data["damagedZipCode"], 5)

# 2b. DIAGNOSTIC: what do the flood-related columns actually contain?
print("=== Flood column check (top values, including blanks) ===")
for c in ["floodDamage", "floodInsurance", "floodDamageAmount", "waterLevel", "highWaterLocation"]:
    if c in data.columns:
        print(f"\n{c}:\n{data[c].value_counts(dropna=False).head(8).to_string()}")
    else:
        print(f"\n{c}: COLUMN NOT FOUND")
print()

# 3. Flood definitions
data["flood_strict"] = data["floodDamage"].map(truthy)
data["flood_broad"] = (data["flood_strict"]
                       | positive(data["floodDamageAmount"])
                       | positive(data["waterLevel"])
                       | has_text(data["highWaterLocation"]))
data["has_flood_ins"] = data["floodInsurance"].map(truthy)


if not (data["is_owner"] & data["flood_strict"]).any():
    print("WARNING: zero flood-damaged owners found. Look at the 'Flood column check' above:")
    print("the flood columns may use different values than this script expects.\n")

# 4. Storm-by-storm totals under both definitions
for col, label in [("flood_strict", "STRICT"), ("flood_broad", "BROAD")]:
    t = summarize(data, ["storm"], col)
    print(f"=== Flood-damaged homeowners by storm ({label}) ===")
    print(t.to_string(index=False), "\n")

# 5. All storms combined, by ZIP and by tract (BROAD definition), small areas hidden
for geo in ["zip", "tract"]:
    s = summarize(data.dropna(subset=[geo]), [geo], "flood_broad")
    s["reliable"] = s["flooded_owners"] >= MIN_COUNT
    s = s.sort_values("uninsured", ascending=False)
    s.to_csv(f"chatham_uninsured_by_{geo}_all_storms.csv", index=False)
    shown = s[s["reliable"]]
    print(f"=== All storms combined, by {geo} (at least {MIN_COUNT} flooded owners) ===")
    print(shown.head(15).to_string(index=False) if len(shown) else "No areas meet the minimum.")
    print(f"({(~s['reliable']).sum()} smaller areas saved to CSV but not shown)\n")

# 6. Storm-by-ZIP detail file (for the per-storm view)
summarize(data, ["storm", "zip"], "flood_broad").to_csv("chatham_uninsured_by_zip_by_storm.csv", index=False)
print("Saved: chatham_uninsured_by_zip_all_storms.csv, chatham_uninsured_by_tract_all_storms.csv,"
      " chatham_uninsured_by_zip_by_storm.csv")