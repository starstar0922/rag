# 企业知识库问答系统（LangChain 版）

面向新手的 **极简 RAG 系统**：上传 PDF / Word / Markdown → 自动切分、向量化、存入 Postgres(pgvector) → 提问时做「向量 + BM25」混合检索 → 重排 → 大模型带引用作答 → 可用 RAGAS 一键评测。

**全部核心逻辑都用 LangChain 现成组件拼装**，一共只有 13 个 Python 文件。**所有配置都从 `.env` 读取**（没有任何硬编码参数）。

> 📖 **新手请看这里**：[`docs/GUIDE.md`](docs/GUIDE.md) 是完整教程——从「怎么跑起来」到「每行代码讲什么」，再到「怎么改、怎么调、怎么排错」。

### 技术栈（全部为当前最新稳定版）

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| LangChain | `1.4.x` | 主框架（LCEL） |
| langchain-classic | `1.0.x` | **LangChain 1.0 起，`create_retrieval_chain` 等经典链迁入此包** |
| langchain-community | `0.4.x` | Loader / BM25 / Flashrank（官方已宣布 sunset，后续会拆成独立包） |
| langchain-openai | `1.7.x` | 模型（OpenAI 兼容协议） |
| langchain-postgres | `0.0.19+` | PGVector 向量库 |
| FastAPI | `0.143.x` | Web 服务（已用 lifespan 替代废弃的 on_event） |
| RAGAS | `0.4.x` | 评测（指标名改为 `AnswerRelevancy` / `ContextRecall`…） |
| Postgres | `pgvector/pgvector:pg16` | 向量数据库 |
| Python | `3.12` | 镜像内用 `uv` 装依赖（比 pip 快 5~10 倍） |

---

## 一、整体流程（3 段，对应 3 个文件）

```
【索引】  app/rag/ingest.py
  文档 → LangChain Loader(PDF/Word/MD) → RecursiveCharacterTextSplitter 切分
       → PGVector.add_documents(自动向量化并写入 pgvector)

【检索】  app/rag/retrieval.py
  问题 → 向量检索(PGVector) ┐
                            ├─ EnsembleRetriever 融合 → FlashrankRerank 重排 → Top4
         BM25 关键词检索    ┘

【生成】  app/rag/chain.py
  历史改写问题 → 检索资料 → 拼进 Prompt → 大模型流式作答（带 [1][2] 引用）

【评测】  app/rag/evaluate.py
  自动生成问答对 → 算 HitRate/MRR（检索） + RAGAS（忠实度/相关性/召回/精确率）
```

## 二、5 分钟跑起来

```bash
# 1. 准备配置
make init                      # 等价于 cp .env.example .env
#    把 OPENAI_API_KEY 填进去（也可以填 DeepSeek / 通义等 OpenAI 兼容地址）

# 2. 启动（Postgres+pgvector 与 Web 服务一起起）
make build                     # docker compose build
make up                        # docker compose up -d

# 3. 打开页面
浏览器访问 http://localhost:8000
API 文档：http://localhost:8000/docs
```

> 不想用 make 也可以：`cp .env.example .env` + `docker compose up -d --build`。
> 忘记命令了就敲 `make help`。

本地开发（不用 Docker 跑应用）：

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d db                 # 只起数据库
export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/kb
uvicorn app.main:app --reload           # http://localhost:8000
```

命令行索引与评测：

```bash
python -m scripts.ingest ./docs/员工手册.pdf --collection hr --init-db
python -m scripts.evaluate --collection hr --size 20
```

## 三、文件都是干什么的

```
app/
├── config.py        所有配置（模型、切分长度、TopK…），改参数只动这里
├── db.py            documents 表：记录每篇文档的状态与分片数
├── schemas.py       接口的请求/响应格式
├── api.py           HTTP 接口：上传、列表、删除、检索、问答、评测
├── main.py          启动 FastAPI，挂载页面
└── rag/
    ├── store.py     创建 Embedding / 大模型 / PGVector（换模型改这里）
    ├── ingest.py    索引：加载 → 切分 → 入库
    ├── retrieval.py 检索：向量 + BM25 混合 + 重排
    ├── chain.py     问答链：历史改写 + 检索增强生成 + 流式输出
    └── evaluate.py  评测：自动生成评测集 + 检索指标 + RAGAS
web/                 页面（原生 HTML/CSS/JS，不需要 npm）
scripts/             命令行索引、命令行评测
tests/               单元测试
Makefile             常用命令（make up / make ingest / make test）
.github/             PR / Issue 模板（CI 模板见 docs/ci.yml.example）
```

## 四、关键概念（用大白话）

| 概念 | 是什么 | 本项目在哪 |
| --- | --- | --- |
| 分片 Chunk | 长文档切成小段，方便检索与塞进 Prompt | `ingest.py` 的 `split_documents` |
| 向量 Embedding | 把文字变成数字数组，语义相近则距离近 | `store.py` 的 `get_embeddings` |
| 向量库 | 存向量并做相似度搜索（Postgres + pgvector） | `store.py` 的 `get_vectorstore` |
| BM25 | 关键词匹配，擅长精确词（人名、编号、术语） | `retrieval.py` 的 `build_bm25_retriever` |
| 混合检索 | 向量 + 关键词两路结果合并，召回更全 | `EnsembleRetriever` |
| 重排 Rerank | 用更精细的小模型给候选重新排序，压到 4 条 | `retrieval.py` 的 `rerank` |
| 历史改写 | 「那这个呢？」→ 改成可独立检索的完整问题 | `chain.py` 的 `create_history_aware_retriever` |
| RAGAS | 自动给「答案是否忠于资料、是否相关」打分 | `evaluate.py` 的 `_ragas_scores` |

## 五、常用改动

**换模型**（`.env`）：

```bash
# DeepSeek
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
# 注意：向量模型仍需单独选一个 embedding 服务（如 OpenAI / 硅流动的 BGE-M3）
```

**调参数**（`.env`）：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `CHUNK_SIZE` | 800 | 分片字符数，长文档可加到 1200 |
| `CHUNK_OVERLAP` | 120 | 分片重叠，避免语义被切断 |
| `TOP_K` | 8 | 混合检索召回条数 |
| `RERANK_TOP_K` | 4 | 最终送给模型的资料条数 |
| `ENABLE_RERANK` | true | 关掉可提速（精度下降） |
| `ENABLE_BM25` | true | 关掉则只用向量检索 |
| `VECTOR_WEIGHT` / `BM25_WEIGHT` | 0.5 / 0.5 | 混合检索两路权重；术语多→偏 BM25，口语提问多→偏向量 |
| `BM25_CACHE_TTL` | 300 | BM25 索引缓存秒数 |
| `CHUNK_SEPARATORS` | JSON 数组 | 切分分隔符，中文务必保留 `。！？` 等标点 |
| `PDF_EXTRACT_TABLES` | true | PDF 表格转 Markdown |
| `LLM_TEMPERATURE` / `LLM_TIMEOUT` | 0 / 60 | 生成温度、超时秒数 |
| `MAX_UPLOAD_MB` | 100 | 单文件上限 |
| `EVAL_SIZE` / `EVAL_USE_RAGAS` | 20 / true | 评测样本数、是否跑 RAGAS |
| `SYSTEM_PROMPT_FILE` | 空 | 用文件覆盖默认提示词 |

> 完整可复制的清单见 [`.env.example`](.env.example)。改完重启：`docker compose restart app`；
> 想确认有没有生效，访问 `GET /api/v1/config`（密钥已脱敏）。

**新增文档类型**：在 `app/rag/ingest.py` 的 `load_file` 里加一个 `elif suffix == ".xxx": loader = ...`（LangChain 有上百种 Loader）。

**换向量库**：只改 `app/rag/store.py` 的 `get_vectorstore`（如换成 FAISS / Chroma / Qdrant，LangChain 接口一致）。

## 六、接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/config` | 查看当前生效的配置（来自 `.env`，密钥脱敏） |
| POST | `/api/v1/documents/upload` | 上传文档（`collection`、`sync` 参数） |
| GET | `/api/v1/documents` | 文档列表 |
| DELETE | `/api/v1/documents/{id}` | 删除文档及分片 |
| POST | `/api/v1/search` | 检索调试 |
| POST | `/api/v1/chat` | 问答（一次性返回） |
| POST | `/api/v1/chat/stream` | 问答（SSE 流式） |
| POST | `/api/v1/eval/run` | 启动评测（返回 `run_id`） |
| GET | `/api/v1/eval/runs/{run_id}` | 查看评测结果 |

## 七、放到 GitHub

```bash
git init                       # 我已帮你初始化好仓库，可跳过
git add . && git commit -m "feat: 企业知识库问答系统（LangChain 版）"
git branch -M main
git remote add origin git@github.com:<你的账号>/kb-qa.git
git push -u origin main
```

`.gitignore` 已排除 `.env`（密钥）与 `data/`（文档和评测结果），不会误传敏感信息。

### GitHub 相关文件

| 文件 | 作用 |
| --- | --- |
| `docs/ci.yml.example` | CI 模板：复制成 `.github/workflows/ci.yml` 即可启用（每次 push / PR 自动跑 `pytest` + `docker build`）。**注意：推送 workflow 文件需要令牌带 `workflow` 权限** |
| `.github/PULL_REQUEST_TEMPLATE.md` | 提 PR 时的检查清单 |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | 提 bug 的结构化模板 |

如果想让 CI 徽标显示在 README 顶部，把下面这行的用户名/仓库名替换掉：

```
![CI](https://github.com/<你的账号>/kb-qa/actions/workflows/ci.yml/badge.svg)
```

### 推荐的分支习惯

```bash
git checkout -b feat/rerank-tuning   # 新功能分支
git add . && git commit -m "feat: 调整重排参数"
git push -u origin feat/rerank-tuning
# GitHub 上提 PR → CI 通过 → 合并到 main
```

## 八、常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 页面能开但问答没内容 | 还没上传文档，或文档状态不是 `ready`（在「知识库」页刷新看状态） |
| 报 `OPENAI_API_KEY` 相关错误 | `.env` 没填 key；没填时系统用「假向量」跑通流程，检索无意义 |
| 中文检索不准 | 用中文向量模型（BGE-M3 / bge-large-zh），并把 `EMBEDDING_MODEL` 改成对应服务名 |
| Word 解析为空 | `.doc`（老格式）不支持，另存为 `.docx` |
| RAGAS 报错 | RAGAS 需要能调用大模型；只想看检索指标用 `--no-ragas` |
| 换了向量模型后检索异常 | 向量维度变了，需要清空数据：`docker compose down -v` 后重建 |
| 第一次问答很慢 | Flashrank 首次使用会下载 ~100MB 重排模型；不想要就设 `ENABLE_RERANK=false` |
| 启动时看到 `langchain-community is being sunset` 警告 | 官方预告的弃用提醒，不影响运行；后续可迁移到独立集成包 |
