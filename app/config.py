"""全局配置（从环境变量 / .env 读取）。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "企业知识库问答系统"
    api_prefix: str = "/api/v1"

    # ---- 数据库（Postgres + pgvector）----
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/kb"

    # ---- 大模型（OpenAI 兼容协议）----
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ---- 切分与检索 ----
    default_collection: str = "default"
    chunk_size: int = 800          # 每个分片的最大字符数
    chunk_overlap: int = 120       # 相邻分片重叠字符数，避免语义被切断
    top_k: int = 8                 # 混合检索（向量 + BM25）召回条数
    rerank_top_k: int = 4          # 重排后真正送给大模型的条数
    enable_rerank: bool = True     # 是否启用重排

    # ---- 其他 ----
    api_key: str = ""              # 留空则不鉴权
    upload_dir: Path = ROOT / "data" / "uploads"
    eval_dir: Path = ROOT / "data" / "eval"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
