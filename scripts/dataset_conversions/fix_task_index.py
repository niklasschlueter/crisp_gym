"""Fix task_index in parquet files: set all task_index values to 0."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa


def fix_task_index(data_dir: str):
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return

    fixed = 0
    for fpath in parquet_files:
        table = pq.read_table(fpath)
        if "task_index" not in table.column_names:
            continue

        col = table.column("task_index")
        # Check if any value != 0
        if col.to_pylist() == [0] * len(col):
            continue

        # Replace task_index column with all zeros
        idx = table.column_names.index("task_index")
        new_col = pa.array([0] * len(table), type=col.type)
        table = table.set_column(idx, "task_index", new_col)
        pq.write_table(table, fpath)
        fixed += 1
        print(f"Fixed: {fpath.name}")

    print(f"\nDone. Fixed {fixed}/{len(parquet_files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix task_index in parquet files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / ".cache/huggingface/lerobot/continuallearning/real_1_stack_bowls/data/chunk-000"),
        help="Path to the directory containing parquet files",
    )
    args = parser.parse_args()
    fix_task_index(args.data_dir)
