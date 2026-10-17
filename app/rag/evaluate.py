"""评测：RAGAS（生成质量） + 离线检索指标（召回质量）。

流程：自动生成评测集 -> 跑检索与生成 -> 算指标 -> 结果存成 JSON 文件。
"""

import json
import random
import uuid
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.rag.chain import answer
from app.rag.retrieval import retrieve
from app.rag.store import get_llm, load_collection_documents

GEN_PROMPT = """基于下面这段文档内容，生成 1 个问答对，用于评测检索问答系统。

要求：问题要像真实员工提问，答案必须能由这段内容直接回答，不超过 80 字。
只输出 JSON：{{"question": "...", "answer": "..."}}

文档内容：
{content}
"""


def generate_samples(collection: str, size: int, settings: Settings | None = None) -> list[dict]:
    """从已索引的分片里抽样，用大模型合成「问题 + 标准答案」。"""
    s = settings or get_settings()
    docs = load_collection_documents(collection, s)
    docs = [d for d in docs if len(d.page_content) > 100] or docs
    if not docs:
        raise ValueError(f"知识库 {collection} 为空，请先上传文档")

    random.seed(42)
    picked = random.sample(docs, min(size, len(docs)))
    llm = get_llm(s)

    samples: list[dict] = []
    for d in picked:
        gold_id = d.metadata.get("chunk_id", "")
        if not s.openai_api_key:
            # 没配模型时的兜底：用原文片段当问题，保证流程可跑
            samples.append(
                {"question": d.page_content[:40], "ground_truth": d.page_content[:200], "gold_chunk_id": gold_id}
            )
            continue
        try:
            text = llm.invoke(GEN_PROMPT.format(content=d.page_content[:2000])).content
            data = _parse_json(text)
        except Exception:
            data = {}
        if data.get("question") and data.get("answer"):
            samples.append(
                {"question": data["question"], "ground_truth": data["answer"], "gold_chunk_id": gold_id}
            )
    return samples


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start : end + 1]) if start >= 0 and end > start else {}
    except Exception:
        return {}


def _ragas_scores(rows: list[dict], settings: Settings) -> dict:
    """调用 RAGAS：忠实度 / 答案相关性 / 上下文召回 / 上下文精确率。"""
    if not settings.openai_api_key:
        return {"ragas_error": "未配置 OPENAI_API_KEY，跳过 RAGAS（仍会输出检索指标）"}
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except Exception as exc:
        return {"ragas_error": f"RAGAS 依赖不可用：{exc}"}

    try:
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=r["question"],
                    retrieved_contexts=r["contexts"],
                    response=r["answer"],
                    reference=r["ground_truth"],
                )
                for r in rows
            ]
        )
        judge_llm = LangchainLLMWrapper(
            ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key,
                       base_url=settings.openai_base_url, temperature=0)
        )
        judge_emb = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key,
                             base_url=settings.openai_base_url)
        )
        result = evaluate(
            dataset,
            metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall(), LLMContextPrecisionWithReference()],
            llm=judge_llm,
            embeddings=judge_emb,
        )
    except Exception as exc:
        return {"ragas_error": f"RAGAS 运行失败：{exc}"}

    try:
        df = result.to_pandas()
        return {f"ragas_{col}": round(float(df[col].mean()), 4) for col in df.columns if col != "user_input"}
    except Exception:
        return {"ragas_error": "RAGAS 结果解析失败"}


def run_evaluation(
    collection: str,
    size: int = 20,
    use_ragas: bool = True,
    settings: Settings | None = None,
    run_id: str | None = None,
) -> dict:
    """执行一次评测，结果写入 data/eval/run_<id>.json（后台任务与 CLI 共用）。"""
    s = settings or get_settings()
    run_id = run_id or uuid.uuid4().hex[:12]
    _save(run_id, {"id": run_id, "status": "running", "collection": collection})
    try:
        samples = generate_samples(collection, size, s)
        rows: list[dict] = []
        hits, mrr_sum = 0, 0.0

        for sample in samples:
            docs: list[Document] = retrieve(sample["question"], collection, s)
            ids = [d.metadata.get("chunk_id", "") for d in docs]
            gold = sample["gold_chunk_id"]
            hit = gold in ids
            hits += int(hit)
            mrr_sum += 1.0 / (ids.index(gold) + 1) if hit else 0.0

            gen = answer(sample["question"], collection, settings=s) if s.openai_api_key else {
                "answer": "", "contexts": [d.page_content for d in docs]
            }
            rows.append(
                {
                    "question": sample["question"],
                    "ground_truth": sample["ground_truth"],
                    "answer": gen.get("answer", ""),
                    "contexts": gen.get("contexts", []),
                    "hit": hit,
                    "retrieved": ids[:5],
                }
            )

        n = max(len(samples), 1)
        metrics: dict[str, Any] = {
            "sample_size": len(samples),
            "hit_rate": round(hits / n, 4),      # 正确答案片段是否被召回
            "mrr": round(mrr_sum / n, 4),        # 正确答案片段的平均排名倒数
        }
        if use_ragas:
            metrics.update(_ragas_scores(rows, s))

        result = {"id": run_id, "status": "done", "collection": collection, "metrics": metrics, "rows": rows}
        _save(run_id, result)
        return result
    except Exception as exc:
        result = {"id": run_id, "status": "failed", "collection": collection, "error": str(exc)}
        _save(run_id, result)
        return result


def _save(run_id: str, payload: dict) -> None:
    s = get_settings()
    s.eval_dir.mkdir(parents=True, exist_ok=True)
    (s.eval_dir / f"run_{run_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_run(run_id: str) -> dict | None:
    path = get_settings().eval_dir / f"run_{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(limit: int = 20) -> list[dict]:
    s = get_settings()
    if not s.eval_dir.exists():
        return []
    files = sorted(s.eval_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for f in files[:limit]:
        data = json.loads(f.read_text(encoding="utf-8"))
        runs.append(
            {
                "id": data.get("id"),
                "status": data.get("status"),
                "collection": data.get("collection"),
                "metrics": data.get("metrics", {}),
                "error": data.get("error", ""),
            }
        )
    return runs
