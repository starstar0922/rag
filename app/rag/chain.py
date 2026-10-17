"""问答链（LangChain LCEL）。

LangChain 1.x 说明：create_* 系列经典链已迁入 langchain-classic 包：
  - langchain_classic.chains.create_history_aware_retriever：用历史把「最新问题」改写成可独立检索的查询
  - langchain_classic.chains.combine_documents.create_stuff_documents_chain：把资料塞进 Prompt 生成答案
  - langchain_classic.chains.create_retrieval_chain：把上面两步串起来
提示词默认写在代码里，也可以用 .env 的 SYSTEM_PROMPT_FILE / REWRITE_PROMPT_FILE 指向文件覆盖。
"""

from typing import Any, AsyncIterator

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import Settings, get_settings, load_prompt
from app.rag.retrieval import build_retriever
from app.rag.store import get_llm

DEFAULT_SYSTEM_PROMPT = """你是企业知识库问答助手，只依据【参考资料】回答问题。

规则：
1. 资料里没有的内容不要编造，直接说明「现有资料无法确定」；
2. 回答用与问题相同的语言，条理清晰、要点化；
3. 关键结论后标注来源编号，例如 [1]、[2]；
4. 涉及流程/步骤时按序号列出。

【参考资料】
{context}"""

DEFAULT_REWRITE_PROMPT = """结合对话历史，把用户最新的问题改写成一个可以独立理解、适合检索的完整问题。
如果没有对话历史，或原问题已经完整，则原样返回。

对话历史：
{chat_history}

最新问题：{input}
"""


def to_messages(history: list[dict[str, str]] | None) -> list[Any]:
    """把前端传来的 [{"role": "user", "content": "..."}] 转成 LangChain 消息对象。"""
    result: list[Any] = []
    for turn in history or []:
        if turn.get("role") == "user":
            result.append(HumanMessage(content=turn.get("content", "")))
        elif turn.get("role") == "assistant":
            result.append(AIMessage(content=turn.get("content", "")))
    return result


def build_chain(collection: str, settings: Settings | None = None):
    s = settings or get_settings()
    llm = get_llm(s)
    retriever = build_retriever(collection, s)

    system_prompt = load_prompt(s.system_prompt_file, DEFAULT_SYSTEM_PROMPT)
    rewrite_prompt_text = load_prompt(s.rewrite_prompt_file, DEFAULT_REWRITE_PROMPT)

    # 1) 多轮改写：让「那这个呢？」这类问题也能正确检索
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rewrite_prompt_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rewrite_prompt)

    # 2) 问答主链
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "问题：{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def _sources(docs) -> list[dict]:
    return [
        {
            "index": i,
            "document_name": d.metadata.get("document_name", ""),
            "page": d.metadata.get("page"),
            "source": d.metadata.get("source", ""),
            "snippet": d.page_content[:300],
        }
        for i, d in enumerate(docs, start=1)
    ]


async def astream_answer(
    question: str,
    collection: str,
    history: list[dict[str, str]] | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[dict]:
    """流式回答：先推送来源，再逐 token 推送答案。"""
    chain = build_chain(collection, settings)
    sent_sources = False
    async for chunk in chain.astream({"input": question, "chat_history": to_messages(history)}):
        if "context" in chunk and not sent_sources:
            sent_sources = True
            yield {"type": "sources", "sources": _sources(chunk["context"])}
        if "answer" in chunk and chunk["answer"]:
            yield {"type": "token", "content": chunk["answer"]}
    yield {"type": "done"}


def answer(
    question: str,
    collection: str,
    history: list[dict[str, str]] | None = None,
    settings: Settings | None = None,
) -> dict:
    """一次性回答（非流式）。"""
    chain = build_chain(collection, settings)
    result = chain.invoke({"input": question, "chat_history": to_messages(history)})
    return {
        "question": question,
        "answer": result.get("answer", ""),
        "sources": _sources(result.get("context", [])),
        "contexts": [d.page_content for d in result.get("context", [])],
    }
