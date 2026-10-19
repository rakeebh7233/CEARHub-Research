"""
Chatham County, GA: flood-damaged homeowners WITHOUT flood insurance, by ZIP code.
Source: OpenFEMA "Individuals and Households Program - Valid Registrations" (v2).

Run:   pip install requests pandas
       python chatham_uninsured_flood_owners.py
Output: chatham_raw_records.csv, chatham_uninsured_by_zip.csv (+ printed summary)
"""
import requests
import pandas as pd

API = "https://www.fema.gov/api/open/v2/IndividualsAndHouseholdsProgramValidRegistrations"
DISASTERS = {
    4284: "Matthew (2016)",   # confirm Chatham appears; script reports 0 if not
    4338: "Irma (2017)",
    4821: "Debby (2024)",
    4830: "Helene (2024)",
}
PAGE = 10000


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


def truthy(v):
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


frames = []
for num, name in DISASTERS.items():
    # Try a county filter first; if the county label format differs, fall back to state-wide + local filter.
    recs = fetch(f"disasterNumber eq {num} and county eq 'Chatham (County)'")
    if not recs:
        print(f"{name}: county filter returned nothing, pulling all GA records (slower)...")
        recs = fetch(f"disasterNumber eq {num}")
    df = pd.DataFrame(recs)
    if df.empty:
        print(f"{name}: no records found")
        continue
    df = df[df["county"].astype(str).str.lower().str.contains("chatham")]
    df["storm"] = name
    print(f"{name}: {len(df):,} Chatham applications")
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
data.to_csv("chatham_raw_records.csv", index=False)
print("\nFields available:", sorted(data.columns))

data["is_owner"] = data["ownRent"].astype(str).str.upper().eq("O")
data["flood_damaged"] = data["floodDamage"].map(truthy)
data["has_flood_ins"] = data["floodInsurance"].map(truthy)

owners_flooded = data[data["is_owner"] & data["flood_damaged"]]
summary = (
    owners_flooded.groupby(["storm", "damagedZipCode"])
    .agg(flooded_owners=("has_flood_ins", "size"),
         insured=("has_flood_ins", "sum"))
    .reset_index()
)
summary["uninsured"] = summary["flooded_owners"] - summary["insured"]
summary["pct_uninsured"] = (100 * summary["uninsured"] / summary["flooded_owners"]).round(1)
summary = summary.sort_values(["storm", "uninsured"], ascending=[True, False])
summary.to_csv("chatham_uninsured_by_zip.csv", index=False)

print("\n=== Flood-damaged homeowners, Chatham County ===")
totals = owners_flooded.groupby("storm").agg(
    flooded_owners=("has_flood_ins", "size"), insured=("has_flood_ins", "sum"))
totals["uninsured"] = totals["flooded_owners"] - totals["insured"]
totals["pct_uninsured"] = (100 * totals["uninsured"] / totals["flooded_owners"]).round(1)
print(totals.to_string())
print("\nTop ZIPs by uninsured flooded owners:")
print(summary.head(15).to_string(index=False))