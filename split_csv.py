
import pandas as pd
from pathlib import Path

input_file = Path("backend/data/amazon_reviews_clean.csv")
output_dir = Path("backend/data/split_csv")
rows_per_file = 500

output_dir.mkdir(parents=True, exist_ok=True)

for old_file in output_dir.glob("part_*.csv"):
    old_file.unlink()

for part_number, chunk in enumerate(
    pd.read_csv(input_file, chunksize=rows_per_file),
    start=1
):
    output_file = output_dir / f"part_{part_number:03d}.csv"
    chunk.to_csv(output_file, index=False)
    print(f"Created {output_file} ({len(chunk)} rows)")

print("CSV splitting complete!")
