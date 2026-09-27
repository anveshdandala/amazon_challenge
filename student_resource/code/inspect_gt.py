import polars as pl

gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=5)
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", n_rows=5)
s1_map = {r["entity_id"]: r for r in s1.iter_rows(named=True)}

for row in gt.iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    m = row["matched_entity_ids"]
    if not m:
        continue
    m_ids = m.split(",")[:2]
    s1_rec = s1_map.get(s1_id)
    print(f"=== S1 ID: {s1_id} ===")
    print("  S1 Name:", s1_rec["business_name"] if s1_rec else "None")
    print("  S1 Addr:", s1_rec["business_address"] if s1_rec else "None")
    print("  True Matches:", m_ids)
    for mid in m_ids:
        src = "train_source2.tsv" if mid.startswith("S2-") else "train_source3.tsv"
        target = pl.scan_csv("student_resource/dataset/train/" + src, separator="\t").filter(pl.col("entity_id") == mid).collect().to_dicts()
        if target:
            print(f"  Target {mid}: Name={target[0]['business_name']!r}")
            print(f"               Addr={target[0]['business_address']!r}")
