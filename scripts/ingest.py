"""命令行索引：python -m scripts.ingest ./docs --collection hr"""

import argparse
from pathlib import Path

from app.db import init_db
from app.rag.ingest import SUPPORTED_SUFFIXES, ingest_file

parser = argparse.ArgumentParser()
parser.add_argument("path", help="文件或目录")
parser.add_argument("--collection", default="default")
parser.add_argument("--recursive", action="store_true")
parser.add_argument("--init-db", action="store_true")
args = parser.parse_args()

if args.init_db:
    init_db()

path = Path(args.path)
files = (
    [path]
    if path.is_file()
    else sorted(p for p in (path.rglob if args.recursive else path.glob)("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
)

if not files:
    print("没有可索引的文件，支持：", sorted(SUPPORTED_SUFFIXES))
else:
    ok = 0
    for f in files:
        result = ingest_file(f, collection=args.collection)
        ok += int(result["status"] == "ready")
        print(f"{f.name} -> {result['status']}，分片 {result['chunk_count']} {result.get('error', '')}")
    print(f"完成：{ok}/{len(files)} 个文件索引成功")
