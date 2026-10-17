"""全局配置：**所有参数都从 .env / 环境变量读取**。

规则：
1. 任何可调参数都写在这里，不要在业务代码里硬编码；
2. 字段名大写即 .env 里的键，例如 chunk_size -> CHUNK_SIZE=800；
3. 每个字段都有默认值，所以没有 .env 也能启动（会用假向量跑通流程）。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------------- 服务 ----------------
    app_name: str = "企业知识库问答系统"
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"                 # uvicorn 用：uvicorn app.main:app --host $HOST
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "*"               # 多个用逗号分隔，例如 http://a.com,http://b.com
    api_key: str = ""                     # 留空则不鉴权

    # ---------------- 数据库 ----------------
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/kb"

    # ---------------- 大模型（OpenAI 兼容协议）----------------
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.0
    llm_timeout: int = 60
    embedding_dim: int = 1536             # 未配置 KEY 时使用「假向量」的维度

    # ---------------- 索引（切分）----------------
    default_collection: str = "default"
    chunk_size: int = 800
    chunk_overlap: int = 120
    chunk_separators: list[str] = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    pdf_extract_tables: bool = True       # PDF 表格转成 Markdown
    max_upload_mb: int = 100

    # ---------------- 检索 ----------------
    top_k: int = 8                        # 单路召回条数
    rerank_top_k: int = 4                 # 重排后送进大模型的条数
    enable_rerank: bool = True
    enable_bm25: bool = True              # 关闭则只用向量检索
    vector_weight: float = 0.5            # 混合检索中向量检索的权重
    bm25_weight: float = 0.5              # 混合检索中 BM25 的权重
    bm25_cache_ttl: int = 300             # BM25 索引缓存秒数

    # ---------------- 提示词（可选：用文件覆盖默认提示词）----------------
    system_prompt_file: Path | None = None
    rewrite_prompt_file: Path | None = None

    # ---------------- 评测 ----------------
    eval_size: int = 20
    eval_use_ragas: bool = True

    # ---------------- 存储目录 ----------------
    upload_dir: Path = ROOT / "data" / "uploads"
    eval_dir: Path = ROOT / "data" / "eval"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def upload_max_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


def load_prompt(path: Path | None, default: str) -> str:
    """提示词：配置了 *_PROMPT_FILE 就读文件，否则用代码里的默认值。"""
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return default
