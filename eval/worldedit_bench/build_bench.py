import os
import json
import pandas as pd
from PIL import Image
from io import BytesIO

pq_path = "WorldEdit/test-00000-of-00001.parquet"
out_dir = ""

img_dir = os.path.join(out_dir, "images")
os.makedirs(img_dir, exist_ok=True)

print("Loading parquet...")
df = pd.read_parquet(pq_path)

print("Sorting by id...")
df = df.sort_values("id").reset_index(drop=True)

meta_list = []

print("Extracting images + json...")
for i, row in df.iterrows():
    idx = int(row["id"])

    
    img_bytes = row["original_image"]
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    img_path = os.path.join(img_dir, f"{idx}.png")
    img.save(img_path)

    
    item = {
        "id": idx,
        "image": f"{idx}.png",
        "implicit_instructions": row["implicit_instructions"],
        "mode": row["mode"],
        "cot": row["cot"],
    }
    meta_list.append(item)

    if i % 50 == 0:
        print(f"processed {i}/{len(df)}")


json_path = os.path.join(out_dir, "test.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(meta_list, f, indent=2, ensure_ascii=False)

print("\nDone.")
print("Images saved to:", img_dir)
print("Json saved to:", json_path)
