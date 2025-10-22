"""
Compare Arrow schema of two parquet files and highlight differences
"""
import pyarrow.parquet as pq
import argparse

def compare_schemas(path1, path2):
    schema1 = pq.read_schema(path1)
    schema2 = pq.read_schema(path2)

    print(f"Schema of {path1}:\n{schema1}\n")
    print(f"Schema of {path2}:\n{schema2}\n")

    fields1 = {f.name: f for f in schema1}
    fields2 = {f.name: f for f in schema2}

    all_fields = set(fields1.keys()).union(fields2.keys())
    print("Differences between schemas:")
    for f in all_fields:
        if f not in fields1:
            print(f"  Field '{f}' missing in first schema")
        elif f not in fields2:
            print(f"  Field '{f}' missing in second schema")
        elif str(fields1[f].type) != str(fields2[f].type):
            print(f"  Field '{f}' type differs: {fields1[f].type} vs {fields2[f].type}")
    print("Comparison finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Path to original parquet file")
    parser.add_argument("--new", required=True, help="Path to new parquet file")
    args = parser.parse_args()

    compare_schemas(args.original, args.new)
