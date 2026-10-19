"""
What do the highWaterLocation codes mean, and how do the flood fields line up?
Run:   python check_water_fields.py
"""
import requests
import pandas as pd

FILE = "chatham_raw_records_BACKUP_pre_refresh.csv"

# 1. Official code meanings from FEMA's code lookup service (DataSetCodes)
print("=== 1. FEMA's definitions for highWaterLocation codes ===")
r = requests.get("https://www.fema.gov/api/open/v1/DataSetCodes", params={
    "$filter": "datasetName eq 'IndividualsAndHouseholdsProgramValidRegistrations' "
               "and fieldName eq 'highWaterLocation'",
    "$top": 100,
}, timeout=60)
codes = r.json().get("DataSetCodes", []) if r.status_code == 200 else []
if codes:
    for c in sorted(codes, key=lambda x: str(x.get("code"))):
        print(f"  {c.get('code')!s:4} = {c.get('description')}  (version {c.get('datasetVersion')})")
else:
    print(f"  No codes returned (status {r.status_code}). Try without the dataset filter, or check")
    print("  the data dictionary on the dataset's FEMA page.")

d = pd.read_csv(FILE, dtype=str, low_memory=False)
d["loc"] = d["highWaterLocation"].fillna("(blank)")
d["level"] = pd.to_numeric(d["waterLevel"], errors="coerce").fillna(0)
d["flag"] = d["floodDamage"].astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})

# 2. How each location code relates to water level and the flood flag
print("\n=== 2. By location code: rows, water level, and flood flag ===")
t = d.groupby("loc").agg(rows=("level", "size"),
                         level_above_0=("level", lambda s: (s > 0).sum()),
                         median_level=("level", "median"),
                         max_level=("level", "max"),
                         flood_flag_true=("flag", "sum"))
print(t.sort_values("rows", ascending=False).to_string())

# 3. How much the flag and the water measurements overlap
wet = d["level"] > 0
print("\n=== 3. Overlap between flood flag and measured water ===")
print(f"  Flag true AND water measured:  {(d['flag'] & wet).sum():,}")
print(f"  Flag true, NO water measured:  {(d['flag'] & ~wet).sum():,}")
print(f"  Water measured, flag NOT true: {(~d['flag'] & wet).sum():,}")