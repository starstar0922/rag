"""端到端自检：建表 -> 索引一个文件 -> 混合检索（不需要大模型 Key）。

用法：DATABASE_URL=... python3 scripts/e2e_check.py
"""

from app.db import init_db
from app.rag.ingest import ingest_file
from app.rag.retrieval import retrieve

COLLECTION = "e2e"


def main() -> None:
    init_db()
    result = ingest_file("/tmp/shouce.md", collection=COLLECTION)
    print("索引结果:", result["status"], "分片数", result["chunk_count"], result.get("error", ""))
    docs = retrieve("差旅报销多久到账", COLLECTION)
    print("召回条数:", len(docs))
    for d in docs[:3]:
        print("  ->", d.metadata.get("document_name"), "|", d.page_content[:60].replace("\n", " "))


if __name__ == "__main__":
    main()
