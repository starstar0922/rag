# 企业知识库问答系统 · 从零到彻底掌握

这份文档面向刚接触 RAG / LangChain 的同学：**先带你跑起来，再一行行讲代码，最后教你怎么改**。

阅读建议：

| 你现在的状态 | 直接看 |
| --- | --- |
| 想先看效果 | 第 3、4、5 章 |
| 想搞懂每段代码 | 第 6、7 章 |
| 想调效果 / 二次开发 | 第 8、9 章 |
| 出问题了 | 第 10 章 |

---

## 1. 先搞懂 RAG 是什么（30 秒）

大模型有两个毛病：**不知道你公司的内部资料**，而且**一本正经地编答案**。
RAG（检索增强生成）就是给模型一场「开卷考试」：

```
考前（索引）：把公司文档 → 切成小卡片 → 给每张卡片算一个"语义指纹"（向量）→ 存进卡片柜
考中（检索）：用户提问 → 从卡片柜里挑出最相关的几张卡片
考后（生成）：把卡片和问题一起交给大模型 → 让它只照着卡片回答，并标注引用
```

对应到本项目的三个文件：索引 `app/rag/ingest.py`、检索 `app/rag/retrieval.py`、生成 `app/rag/chain.py`。

---

## 2. 项目全景

```
                       ┌─────────────── 索引（离线，上传时做一次） ───────────────┐
PDF / Word / Markdown ─▶ LangChain Loader ─▶ RecursiveCharacterTextSplitter ─▶ OpenAI Embedding ─▶ PGVector(pgvector)
                       └────────────────────────────────────────────────────────────────────────┘

                       ┌─────────────── 检索 + 生成（在线，每次提问） ──────────────┐
用户问题 ─▶ 多轮改写 ─┬─▶ 向量检索(PGVector, Top8) ─┐
                      └─▶ BM25 关键词检索(Top8)   ─┴─▶ EnsembleRetriever 融合 ─▶ FlashrankRerank 重排(Top4)
                                                                                        │
                                                          ┌─────────────────────────────┘
                                                          ▼
                                           拼进 Prompt ─▶ 大模型流式作答（带 [1][2] 引用）

                       ┌─────────────── 评测 ───────────────┐
自动生成问答对 ─▶ 跑检索算 HitRate/MRR ─▶ RAGAS 算忠实度/相关性/召回/精确率 ─▶ data/eval/run_*.json
```

**每个阶段用的 LangChain 组件（照着搜官方文档即可）**

| 阶段 | 用什么 | 代码位置 |
| --- | --- | --- |
| 加载 | `PyMuPDFLoader` / `Docx2txtLoader` / `TextLoader` | `app/rag/ingest.py:25` |
| 切分 | `RecursiveCharacterTextSplitter` | `app/rag/ingest.py:44` |
| 向量化 + 存储 | `OpenAIEmbeddings` + `PGVector.add_documents` | `app/rag/store.py:16`、`app/rag/ingest.py:98` |
| 稀疏检索 | `BM25Retriever`（jieba 分词） | `app/rag/retrieval.py:35` |
| 融合 | `EnsembleRetriever`（RRF 倒数排名融合） | `app/rag/retrieval.py`（`langchain_classic.retrievers`） |
| 重排 | `FlashrankRerank`（本地 CPU 小模型） | `app/rag/retrieval.py`（`langchain_community.document_compressors`） |
| 多轮改写 | `create_history_aware_retriever` | `app/rag/chain.py`（`langchain_classic.chains`） |
| 生成 | `create_retrieval_chain` + `create_stuff_documents_chain` | `app/rag/chain.py`（`langchain_classic.chains`） |
| 评测 | `ragas.evaluate` | `app/rag/evaluate.py` |

> **LangChain 1.x 迁移要点**：`langchain.chains` / `langchain.retrievers` 已从主包移除，
> 经典链与融合检索器现在从 **`langchain-classic`** 导入（`langchain_classic.chains`、`langchain_classic.retrievers`）；
> Loader、BM25、Flashrank 仍在 `langchain-community`（官方已宣布 sunset，只是警告，不影响使用）。
> 文档里的行号可能随版本微调，以函数名搜索为准。

**目录树**

```
app/
├── config.py        配置中心（模型、切分长度、TopK…）        改参数只动这里
├── db.py            documents 表 + 建库建扩展
├── schemas.py       接口数据结构（Pydantic）
├── api.py           所有 HTTP 接口
├── main.py          启动 FastAPI、挂载页面
└── rag/
    ├── store.py     Embedding / 大模型 / 向量库的创建处        换模型只动这里
    ├── ingest.py    索引：加载 → 切分 → 入库
    ├── retrieval.py 检索：向量 + BM25 混合 + 重排
    ├── chain.py     问答链 + 流式输出
    └── evaluate.py  评测：自动生成评测集 + 检索指标 + RAGAS
web/                 index.html / style.css / app.js（原生，无 npm）
scripts/             ingest.py（命令行索引）、evaluate.py（命令行评测）
tests/               test_ingest.py（4 个单元测试，不需要数据库）
Makefile             命令速查（make up / make ingest / make test）
docker-compose.yml   db(pgvector) + app 两个服务
Dockerfile           镜像构建
```

---

## 3. 运行前的准备

**必需软件**（先确认一下）：

```bash
docker -v && docker compose version   # 推荐方式，需 Docker 20.10+
python3 --version                     # 本地方式，需 3.10+（本项目用 3.11）
make -v                               # 可选，只是命令别名
```

**必需的大模型 Key**（二选一即可）：

- OpenAI：`OPENAI_API_KEY=sk-...`，`OPENAI_BASE_URL=https://api.openai.com/v1`
- 国产 / 自建（DeepSeek、通义、硅流动、vLLM、Xinference）：只要服务是 **OpenAI 兼容协议**，改 `OPENAI_BASE_URL` 和 `LLM_MODEL` 即可，代码一行都不用改。

> 没有 Key 也能跑：系统会自动用 LangChain 自带的「假向量」先把流程跑通（`app/rag/store.py:29`），但检索结果是随机的，仅供联调界面。
> **向量模型**建议单独选：中文场景优先 `BGE-M3` / `bge-large-zh`（硅流动、Xinference 都提供 OpenAI 兼容的 embedding 接口）。

---

## 4. 三种运行方式（详细步骤 + 预期输出）

### 方式 A：Docker 一键（推荐，最省事）

```bash
cd 项目目录
make init                 # ① 生成 .env（等价于 cp .env.example .env）
vim .env                  # ② 填 OPENAI_API_KEY / LLM_MODEL / EMBEDDING_MODEL
make build                # ③ 构建镜像（第一次约 3～10 分钟，装依赖）
make up                   # ④ 启动：数据库 + 服务（-d 后台运行）
docker compose ps         # ⑤ 看状态，db 应为 healthy，app 为 Up
```

`docker compose ps` 预期输出：

```
NAME      SERVICE   STATUS
kb-app    app       Up
kb-db     db        Up (healthy)
```

验证服务活着：

```bash
curl http://localhost:8000/api/v1/health
```

```json
{"status":"ok","llm":"gpt-4o-mini","embedding":"text-embedding-3-small","llm_configured":true,"database":"ok","collection":"default"}
```

> `llm_configured: false` 说明 `.env` 里的 Key 没生效，改完 `.env` 要 `docker compose restart app`。

打开页面：<http://localhost:8000>；接口文档：<http://localhost:8000/docs>

**这几个命令背后做了什么**

- `docker-compose.yml` 定义了两个服务：`db` 用官方 `pgvector/pgvector:pg16` 镜像（Postgres 装好了向量扩展），`app` 用本项目 `Dockerfile` 构建。
- `app` 挂了 `./app:/app/app`，所以**本地改代码后 `make restart` 就生效**，不用重新 build。
- `db` 有 healthcheck，`app` 会等数据库健康后再启动（`depends_on: service_healthy`），避免启动顺序问题。

### 方式 B：本地虚拟环境（改代码最舒服）

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d db                       # 只起数据库
export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/kb
export OPENAI_API_KEY=sk-xxx                  # 或直接写进 .env，程序会自动读
uvicorn app.main:app --reload                 # 代码改动自动重启
```

看到 `Uvicorn running on http://127.0.0.1:8000` 就成功了。

### 方式 C：命令行（不启 Web，适合批量灌数据）

```bash
python -m scripts.ingest ./docs/员工手册.pdf --collection hr --init-db
# 输出：员工手册.pdf -> ready，分片 37

python -m scripts.evaluate --collection hr --size 20
# 输出：{"sample_size": 20, "hit_rate": 0.85, "mrr": 0.7, "ragas_faithfulness": 0.91, ...}
```

`--init-db` 第一次要加（建 documents 表和 vector 扩展），之后可省略。

### 停止 / 清理 / 看日志

```bash
make down      # 停止容器（数据卷保留，下次 make up 数据还在）
make logs      # 实时看服务日志（Ctrl+C 退出）
make restart   # 改完代码重启 app
make clean     # 危险：删容器 + 删数据卷 + 清空 data/（换 embedding 模型后必须做）
```

---

## 5. 第一次完整演练（手把手，全部可复制）

### 第 1 步：造一个测试文档

```bash
mkdir -p docs && cat > docs/员工手册.md <<'EOF'
# 员工手册

## 差旅报销
差旅报销需要在 OA 系统提交《差旅报销单》，并附上车票、住宿发票。
审批流程：员工提交 → 直属主管审批 → 财务复核 → 财务打款，通常在 5 个工作日内到账。
单笔超过 5000 元的差旅费需 VP 额外审批。

## 年假
入职满 1 年可享受 10 天年假，每满 1 年增加 1 天，上限 15 天。
年假需提前 3 个工作日在 OA 提交申请。
EOF
```

### 第 2 步：上传（页面拖拽，或用 curl）

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@docs/员工手册.md" -F "collection=hr"
```

```json
{"status":"processing","name":"员工手册.md","collection":"hr"}
```

上传接口是**异步**的（后台跑索引），想同步等结果加 `-F "sync=true"`。

### 第 3 步：看索引状态

```bash
curl "http://localhost:8000/api/v1/documents?collection=hr"
```

```json
[{"id":"a1b2...","name":"员工手册.md","doc_type":"md","collection":"hr","status":"ready","chunk_count":3,...}]
```

`status` 三种：`pending`（排队中）→ `ready`（成功）/ `failed`（失败，看 `error` 字段）。

### 第 4 步：检索调试（看看召回的东西对不对）

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"差旅报销的审批流程","collection":"hr","top_k":3}'
```

命中不佳时，先查这一步——**检索准了答案才会准**。

### 第 5 步：问答

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"差旅报销多久到账？","collection":"hr"}'
```

```json
{"question":"...","answer":"通常在 5 个工作日内到账 [1]。","sources":[{"index":1,"document_name":"员工手册.md","page":null,"snippet":"..."}],...}
```

多轮对话（带历史，验证「它」这类指代能被改写）：

```bash
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d '{
  "question":"那超过 5000 呢？","collection":"hr",
  "history":[{"role":"user","content":"差旅报销流程是什么？"},
             {"role":"assistant","content":"提交→主管→财务复核→打款"}]}'
```

看流式输出（页面用的就是这个）：

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream -H "Content-Type: application/json" \
  -d '{"question":"年假怎么算？","collection":"hr"}'
# data: {"type":"sources","sources":[...]}
# data: {"type":"token","content":"入职"}
# data: {"type":"token","content":"满 1"}
# ...
# data: {"type":"done"}
```

### 第 6 步：跑评测

```bash
RUN=$(curl -s -X POST http://localhost:8000/api/v1/eval/run -H "Content-Type: application/json" \
  -d '{"collection":"hr","size":10,"use_ragas":true}' | python3 -c "import sys,json;print(json.load(sys.stdin)['run_id'])")
sleep 20
curl http://localhost:8000/api/v1/eval/runs/$RUN
```

结果长这样（含义见 7.7）：

```json
{"id":"...","status":"done","metrics":{"sample_size":10,"hit_rate":0.9,"mrr":0.77,"ragas_faithfulness":0.94,"ragas_answer_relevancy":0.88,...}}
```

### 第 7 步：看看数据库里到底存了什么

```bash
docker compose exec db psql -U postgres -d kb
```

```sql
\dt                                            -- 会看到 documents / langchain_pg_collection / langchain_pg_embedding
SELECT id,name,status,chunk_count FROM documents;
SELECT count(*) FROM langchain_pg_embedding;   -- 全部分片数
SELECT left(document,60) AS 内容, cmetadata->>'document_name' AS 文档名, cmetadata->>'chunk_id' AS 分片ID
FROM langchain_pg_embedding LIMIT 5;
```

**这张表是理解全局的关键**：一行 = 一个分片 = 一条向量 + 原文 + JSON 版元数据。

---

## 6. 每个文件是干什么的（以及什么时候会改它）

| 文件 | 职责 | 你什么时候会改 |
| --- | --- | --- |
| `app/config.py` | 从 `.env`/环境变量读配置，全局单例 | 加一个新配置项（如超时时间） |
| `app/db.py` | `documents` 表 CRUD、建表、建 vector 扩展 | 想给文档加字段（如上传者） |
| `app/schemas.py` | 接口的请求/响应结构 | 加接口参数 |
| `app/rag/store.py` | 创建 Embedding、大模型、PGVector | **换模型 / 换向量库** |
| `app/rag/ingest.py` | 加载 → 切分 → 入库全流程 | **加文档类型 / 改切分策略** |
| `app/rag/retrieval.py` | 混合检索 + 重排 | 调融合权重、加第三种检索源 |
| `app/rag/chain.py` | 问答主链、流式、引用格式化 | 改提示词、改输出格式 |
| `app/rag/evaluate.py` | 生成评测集 + 指标 + RAGAS | 加自己的指标/数据集 |
| `app/api.py` | 9 个 HTTP 接口 | 加接口（如导出、权限） |
| `app/main.py` | 组装 FastAPI、挂静态页 | 加中间件（限流、日志） |
| `web/*` | 页面 | 改 UI |

---

## 7. 代码精讲（一段一段拆）

### 7.1 配置中心 `app/config.py`

```python
11:37:app/config.py
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "企业知识库问答系统"
    api_prefix: str = "/api/v1"
    ...
```

- `BaseSettings`（Pydantic）会自动从**环境变量**或 `.env` 读同名变量，所以 `.env` 里写 `CHUNK_SIZE=1200` 就等于改这个字段。
- **项目里没有任何硬编码参数**：切分长度、TopK、权重、提示词、目录、鉴权……全部在这里声明，全部可由 `.env` 覆盖。
- 想确认线上到底用了哪套配置：访问 `GET /api/v1/config`（密钥自动脱敏）。
- `get_settings()` 上加了 `@lru_cache(maxsize=1)`：整个进程只读一次配置，避免重复解析文件，并保证 `data/` 目录只创建一次。
- 想加配置：在这里加一行 `my_flag: bool = False`，然后 `.env` 里写 `MY_FLAG=true` 即可。

### 7.2 数据库 `app/db.py`

只有一张业务表 `documents`（`app/db.py:17`），向量数据由 LangChain 的 PGVector 自己管，不重复造轮子。

```python
53:64:app/db.py
@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    session: Session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

这是 Python 的**上下文管理器**写法，`with session_scope() as s:` 会自动提交、出错回滚、最后一定关闭连接——新手最容易忘的就是「关闭连接」，这里统一兜住了。

删除文档时要连向量一起删，用的是 Postgres 的 JSONB 查询：

```python
108:118:app/db.py
def delete_document_chunks(document_id: str) -> int:
    engine = get_engine()
    sql = text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = :doc_id")
```

`cmetadata->>'document_id'` 意思是「取 JSON 字段 cmetadata 里的 document_id」——这也是元数据能带来源、页码的原因。

### 7.3 索引 `app/rag/ingest.py`（三件事）

**① 加载**：按后缀选 Loader，PDF 还会尽量把表格抽成 Markdown。

```python
25:41:app/rag/ingest.py
def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            loader = PyMuPDFLoader(str(path), extract_tables="markdown")
        except TypeError:
            loader = PyMuPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in {".md", ".markdown", ".txt"}:
        loader = TextLoader(str(path), encoding="utf-8")
```

返回的 `Document` 就是 LangChain 的通用数据结构：`page_content`（文本）+ `metadata`（来源、页码等字典）。**PDF 每一页是一个 Document**，所以能拿到页码。

**② 切分**：为什么要切？Embedding 模型有长度上限，且长文本稀释语义、塞不进 Prompt。

```python
44:52:app/rag/ingest.py
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_documents(docs)
```

它的逻辑是「**递归**」：先按段落 `\n\n` 切，切完还太长就按句子（这里额外加了中文标点）切，还太长才硬切字符。中文项目一定要像上面这样把 `。！？；，` 加进 `separators`，否则整段会被当成一个长句硬切断。

**③ 打元数据 + 入库**（`app/rag/ingest.py:88`）：

```python
88:99:app/rag/ingest.py
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "document_id": document_id,
                    "document_name": name,
                    "collection": collection,
                    "chunk_id": f"{document_id}_{i}",
                }
            )

        vectorstore = get_vectorstore(collection)
        vectorstore.add_documents(chunks)  # 内部自动调用 Embedding 并写入 pgvector
```

`add_documents` 一行干了三件事：调 Embedding 接口 → 生成向量 → 写进 `langchain_pg_embedding`。`chunk_id` 后面评测算「命中率」和「按文档删除」全靠它。

> **新手常见疑问：图片/流程图怎么办？**
> 本项目只抽取文本（表格会被转成 Markdown 文本）。要理解图片，可在 `load_file` 里换成带 OCR / 视觉理解的方案（例如 `UnstructuredFileLoader` + 图片描述，`PyMuPDFLoader(extract_images=True)` 取图后交给多模态模型生成描述再入库）。这是进阶项，第 9 章给了改法。

### 7.4 模型与向量库 `app/rag/store.py`

```python
16:31:app/rag/store.py
def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if s.openai_api_key:
        return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key, base_url=s.openai_base_url)
    from langchain_community.embeddings import DeterministicFakeEmbedding
    return DeterministicFakeEmbedding(size=1536)
```

- 三个组件（Embedding、Chat 模型、向量库）**都集中在这里创建**，所以换模型只需要改这一个文件。
- `@lru_cache(maxsize=8)` 让同一个知识库复用同一个 `PGVector` 对象（内部持有数据库连接池），避免每次检索都新建连接。
- `create_tables()` / `create_collection()` 外面包了 `try/except`：已存在时抛错直接忽略，达到「幂等」效果（重复调用不会出问题）。

### 7.5 检索 `app/rag/retrieval.py`（本项目的核心）

**为什么要有两路检索？**

- 向量检索：懂语义（「报销要多久」≈「到账时间」），但对精确词（工号、法条编号、专有名词）不敏感。
- BM25：纯关键词匹配，精确词很强，但不懂同义词。
- 两路结果合并 → 召回更全，这就是**混合检索**。

**BM25 为什么要自己造索引？** LangChain 的 `BM25Retriever` 需要内存里的文档列表，而文档存在 Postgres 里，所以先从库里读出来再建索引，并缓存 5 分钟：

```python
35:48:app/rag/retrieval.py
def build_bm25_retriever(collection: str, k: int, settings: Settings | None = None) -> BM25Retriever | None:
    s = settings or get_settings()
    cached = _BM25_CACHE.get(collection)
    if cached and time.time() - cached[0] < _BM25_TTL:
        retriever = cached[1]
    else:
        docs = load_collection_documents(collection, s)
        retriever = BM25Retriever.from_documents(docs, preprocess_func=chinese_tokenizer) if docs else None
        _BM25_CACHE[collection] = (time.time(), retriever)
```

`preprocess_func=chinese_tokenizer` 是**中文关键**：BM25 默认按空格分词，中文整句会被当一个词。这里用 jieba 切词（`app/rag/retrieval.py:27`）。上传/删除文档后调用 `invalidate_bm25()` 让缓存失效重建。

**融合 + 重排**：

```python
80:97:app/rag/retrieval.py
    vector_retriever = get_vectorstore(collection).as_retriever(search_kwargs={"k": s.top_k})
    retrievers = [vector_retriever]
    weights = [0.6]
    bm25 = build_bm25_retriever(collection, s.top_k, s)
    if bm25 is not None:
        retrievers.append(bm25)
        weights = [0.5, 0.5]
    if len(retrievers) == 1:
        base: BaseRetriever = vector_retriever
    else:
        base = EnsembleRetriever(retrievers=retrievers, weights=weights)
    return RerankRetriever(base_retriever=base, top_k=s.rerank_top_k, collection=collection)
```

- `EnsembleRetriever` 用 RRF（倒数排名融合）：一个文档在第 1 路排第 3、第 2 路排第 1，综合得分 = `0.5/3 + 0.5/1`，两路都靠前才最终靠前。
- 重排为什么必要？向量检索是「粗排」（快但粗），重排模型（交叉编码器）会把问题和每个候选**一起**送进模型精算相关性，把 8 条压到 4 条，精度明显提升。`FlashrankRerank` 是 CPU 可跑的轻量模型，无需 GPU。
- `RerankRetriever`（`app/rag/retrieval.py:66`）继承 LangChain 的 `BaseRetriever`，只实现 `_get_relevant_documents`，这样「混合 + 重排」就变成了一个标准 Retriever，能直接塞进后面的链里——**这是本项目唯一一处自定义类，就是为了把两个步骤粘起来**。

### 7.6 生成 `app/rag/chain.py`

三个 LangChain 高阶函数拼出完整问答链：

```python
63:74:app/rag/chain.py
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rewrite_prompt)
    ...
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

1. `create_history_aware_retriever`：先让大模型把「那这个呢？」结合历史改写成「差旅报销超过 5000 元怎么审批」，再去检索。没有这一步，多轮对话会因为指代而检索失败。
2. `create_stuff_documents_chain`：把检索到的文档「塞」（stuff）进 Prompt 的 `{context}` 位置（模板见 `app/rag/chain.py:18` 的 `SYSTEM_PROMPT`，里面有「不许编造、要标 [1] 引用」等规则）。
3. `create_retrieval_chain`：把 1、2 串起来，输入 `{"input": 问题, "chat_history": 历史}`，输出 `{"answer":..., "context":[...]}`。

流式输出的事件协议（`app/rag/chain.py:90`）：先推一次 `sources`（前端立刻显示引用），再逐 token 推 `answer`，最后 `done`。前端 `web/app.js:77` 手动按 `\n\n` 拆包解析 SSE（用原生 `EventSource` 无法发 POST，所以用 `fetch + ReadableStream`）。

### 7.7 评测 `app/rag/evaluate.py`

**① 自动生成评测集**（`app/rag/evaluate.py:29`）：随机抽已入库分片 → 让大模型基于该片段生成「问题 + 标准答案」→ 记住这个分片的 `chunk_id` 作为「金标准」。没有现成数据集时这是最省事的做法；有真实业务 Q&A 时，直接把 `generate_samples` 换成读你的 JSON/CSV 即可。

**② 检索指标**（`app/rag/evaluate.py:141`）：

- `hit_rate`：金标准分片有没有出现在召回结果里（召回率，越高越好）
- `mrr`：金标准分片排第几（排第 1 得 1 分，第 2 得 0.5），衡量「排序质量」

**③ RAGAS 生成指标**（`app/rag/evaluate.py:109`，需要配置 Key）：

| 指标 | 问的问题 | 低了说明什么 |
| --- | --- | --- |
| `faithfulness` 忠实度 | 答案是不是只用了资料里的内容 | 模型在编（换更强的模型 / 收紧 Prompt） |
| `answer_relevancy` 答案相关性 | 答案是否切题 | 答非所问（检查改写或切分） |
| `context_recall` 上下文召回 | 标准答案所需信息是否被检索到 | **检索不好**（加大 `TOP_K`、开启 BM25） |
| `context_precision` 上下文精确率 | 召回回来的有多少是有用的 | 噪声多（开启重排、减小 `TOP_K`） |

> RAGAS 0.4 的指标类名变了：`ResponseRelevancy → AnswerRelevancy`、`LLMContextRecall → ContextRecall`、
> `LLMContextPrecisionWithReference → ContextPrecisionWithReference`，且要从 **`ragas.metrics.collections`** 导入。

**看指标定位问题的口诀**：召回低 → 改检索；精确率低 → 加重排/调切分；忠实度低 → 改 Prompt/换模型。

### 7.8 接口与服务 `app/api.py`、`app/main.py`

- **上传为什么是异步的**：索引要调 Embedding 接口，几十页 PDF 可能要几十秒，HTTP 请求一直挂着会超时。所以先落盘，再用 FastAPI 的 `BackgroundTasks` 后台跑（`app/api.py:84`），前端 3 秒/10 秒后刷新列表看状态。调试时加 `sync=true` 可同步返回。
- **鉴权**：`check_api_key` 挂在 router 级别（`app/api.py:36`），`.env` 里没配 `API_KEY` 就自动放行，配了就必须带 `X-API-Key` 头。
- **静态页面**：`app/main.py:40` 的 `/` 返回 `web/index.html`，`/static` 挂载整个 `web/` 目录，所以 CSS/JS 直接引用 `/static/style.css`。没有前端构建步骤，改完刷新即可。
- **启动即建表**：`app/main.py:29` 的 startup 事件里调 `init_db()`，包了 try/except，数据库没起来时服务仍能启动，方便看 `/health` 里的 `database` 字段排查。

---

## 8. 参数怎么调（在 `.env` 里改，改完 `make restart`）

| 参数 | 默认 | 调大/调小的效果 | 建议 |
| --- | --- | --- | --- |
| `CHUNK_SIZE` | 800 | 大：上下文完整但稀释语义；小：精确但可能断句 | 政策/手册类 600～1000；FAQ 类 300～500 |
| `CHUNK_OVERLAP` | 120 | 防止一句话被切断 | 设为 `CHUNK_SIZE` 的 10%～20% |
| `TOP_K` | 8 | 召回候选数，越大召回越全但噪声越多 | 5～15 |
| `RERANK_TOP_K` | 4 | 真正送进模型的条数，超了会挤爆上下文 | 3～6 |
| `ENABLE_RERANK` | true | 关掉能提速约 30%，精度下降 | 先开，性能瓶颈再关 |
| `ENABLE_BM25` | true | 关掉只剩向量检索 | 纯语义问答可关 |
| `VECTOR_WEIGHT` / `BM25_WEIGHT` | 0.5 / 0.5 | 融合权重（在 `app/rag/retrieval.py` 的 `build_retriever`） | 术语多→偏 BM25；口语提问多→偏向量 |
| `CHUNK_SEPARATORS` | JSON 数组 | 切分分隔符 | 中文务必保留 `。！？；，` |
| `LLM_TEMPERATURE` | 0 | 越大越发散 | 问答场景保持 0 |
| `EVAL_SIZE` / `EVAL_USE_RAGAS` | 20 / true | 评测默认参数 | 只想看检索指标就关 RAGAS |

**改动有没有效，别靠感觉**：改之前跑一次 `python -m scripts.evaluate`，改之后再跑一次，对比 `hit_rate` 和 `mrr`。

---

## 9. 常见二次开发（照抄即可）

**换大模型（DeepSeek 为例）** —— 只改 `.env`：

```bash
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

**换中文 Embedding（BGE-M3，硅流动为例）**：

```bash
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
```

> 换了 Embedding 后向量维度会变（1536 → 1024），旧数据不兼容，必须 `make clean` 后重新索引。

**换向量库（不想要 Postgres，改成 FAISS 本地文件）** —— 只改 `app/rag/store.py` 的 `get_vectorstore`：

```python
from langchain_community.vectorstores import FAISS
def get_vectorstore(collection: str):
    path = get_settings().upload_dir.parent / f"faiss_{collection}"
    if path.exists():
        return FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)
    return FAISS.from_texts(["初始化"], get_embeddings(), metadatas=[{"chunk_id": "init"}])
```
（同时要把 `app/db.py:108` 的删除分片逻辑换成 FAISS 的 `delete`。）

**加文档类型（Excel / PPT）** —— 在 `app/rag/ingest.py:25` 加分支：

```python
elif suffix in {".xlsx", ".pptx"}:
    from langchain_community.document_loaders import UnstructuredFileLoader
    loader = UnstructuredFileLoader(str(path))
```

**多知识库隔离**：所有接口都有 `collection` 参数（页面右上角输入框）。不同部门用不同 collection，检索时天然隔离，向量存在同一张表里靠 `collection_id` 区分。

**给文档加字段（如上传者）**：`app/db.py` 的 `Document` 加一列 → `app/rag/ingest.py` 里传入 → `app/schemas.py` 的 `DocumentOut` 加字段即可。

---

## 10. 排错手册

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| 页面能开但问答答非所问 | 没配 Key，用了假向量 | `/health` 看 `llm_configured`，填 `.env` 后 `make restart` |
| 文档一直是 `pending` | 后台索引在跑（大 PDF 慢）或崩了 | `make logs` 看报错；状态 `failed` 时看列表里的 `error` |
| 上传报「不支持的文件类型」 | 老格式 `.doc` | 另存为 `.docx` |
| PDF 抽出内容为空 | 扫描件（纯图片） | 需要 OCR：`pymupdf` + OCR，或用 `UnstructuredFileLoader` |
| 检索召回不准 | Embedding 不是中文模型 / 切分不合适 | 换 BGE-M3；调 `CHUNK_SIZE`；用 `/search` 逐条看 |
| `database: error` in `/health` | 数据库没起来或地址不对 | `docker compose ps`；容器内地址是 `db:5432`，不是 localhost |
| 换 Embedding 后报维度错误 | 新旧向量维度不一致 | `make clean` 清库后重新索引 |
| RAGAS 报错 / 很慢 | 需要调大模型打分，样本多就慢 | 先用 `--size 5` 试；只想看检索指标用 `--no-ragas` |
| 改了 `.env` 没生效 | 容器启动时才读 | `make restart`（或 `docker compose up -d` 重建） |

---

## 11. 放到 GitHub 与自动 CI

```bash
git init                 # 仓库已初始化，可跳过
git add . && git commit -m "feat: 企业知识库问答系统（LangChain 版）"
git branch -M main
git remote add origin git@github.com:<你的账号>/kb-qa.git
git push -u origin main
```

- `.gitignore` 已排除 `.env`（密钥）和 `data/`（上传文件、评测结果），不会误传。
- `docs/ci.yml.example`：CI 模板。把它复制成 `.github/workflows/ci.yml` 提交即可启用（**推送 workflow 文件需要令牌带 `workflow` 权限**）；启用后每次 push / PR 自动跑 `pytest` 并 `docker build`，失败会在 PR 上标红。
- `.github/PULL_REQUEST_TEMPLATE.md`、`.github/ISSUE_TEMPLATE/bug_report.yml`：协作模板。
- 推荐流程：`git checkout -b feat/xxx` → 提交 → 提 PR → CI 绿了再合并。

---

## 12. 下一步学什么

按这个顺序啃，基本就能独立改这个项目了：

1. LangChain 三个基础概念：`Document` / `Retriever` / `Runnable（LCEL）`
2. 切分策略：递归切分 vs 按标题（MarkdownHeaderTextSplitter）切分
3. 检索进阶：MMR（多样性）、HyDE（用假答案去检索）、查询扩展
4. 分片进阶：父子分片（小片检索、大片生成）、按表格/图片特殊处理
5. 评测进阶：用自己的业务 Q&A 数据集、A/B 对比不同参数
6. Agent：把「检索」变成模型可自主调用的工具

---

## 13. 命令速查表

```bash
# ---- 服务 ----
make init / build / up / down / restart / logs / clean
curl http://localhost:8000/api/v1/health          # 健康检查
docker compose exec db psql -U postgres -d kb     # 进数据库

# ---- 数据 ----
python -m scripts.ingest ./docs --collection hr --recursive --init-db
python -m scripts.evaluate --collection hr --size 20
python -m pytest -q                               # 单元测试

# ---- 接口 ----
POST /api/v1/documents/upload     (form: file, collection, sync)
GET  /api/v1/documents?collection=hr
DELETE /api/v1/documents/{id}
POST /api/v1/search               {query, collection, top_k}
POST /api/v1/chat                 {question, collection, history}
POST /api/v1/chat/stream          同上，SSE 流式
POST /api/v1/eval/run             {collection, size, use_ragas}
GET  /api/v1/eval/runs/{run_id}
```

> 当前 Docker 环境里 `kb-db` 和 `kb-app` 已在运行，可直接访问 <http://localhost:8000> 体验；不用时 `make down`。
