"""应用入口：FastAPI + 静态页面（所有配置来自 .env）。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import router
from app.config import get_settings

settings = get_settings()
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时建表（失败也不阻止启动，方便用 /health 排查）。"""
    try:
        from app.db import init_db

        init_db()
    except Exception as exc:
        print(f"[startup] 数据库初始化失败：{exc}")
    yield


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="LangChain 1.x + pgvector 的企业知识库问答系统：混合检索 + 重排 + RAGAS 评测",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api_prefix)


# ---------- 页面 ----------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
