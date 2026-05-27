import logging
import threading
import os
import time

import chromadb
import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_core.tools import tool

from config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY

logger = logging.getLogger(__name__)
_client = None
_lock = threading.Lock()

EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 2


class DashScopeEmbedding(EmbeddingFunction):
    """使用 DashScope text-embedding API 作为 ChromaDB 的嵌入函数"""

    def _call_api(self, texts: Documents) -> Embeddings:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
            headers={
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "text-embedding-v2",
                "input": {"texts": texts},
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error("Embedding API returned %s: %s", resp.status_code, resp.text[:200])
            raise RuntimeError(f"Embedding API error: {resp.status_code} {resp.text}")
        data = resp.json()
        return [e["embedding"] for e in data["output"]["embeddings"]]

    def __call__(self, texts: Documents) -> Embeddings:
        if not texts:
            return []
        last_exc = None
        for attempt in range(1, EMBEDDING_RETRIES + 1):
            try:
                return self._call_api(texts)
            except Exception as e:
                last_exc = e
                if attempt < EMBEDDING_RETRIES:
                    wait = EMBEDDING_RETRY_DELAY * attempt
                    logger.warning("Embedding API retry %d/%d after error: %s",
                                   attempt, EMBEDDING_RETRIES, e)
                    time.sleep(wait)
        raise RuntimeError(f"Embedding API 请求失败 (retried {EMBEDDING_RETRIES}x): {last_exc}") from last_exc


def get_chroma_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def search_faq(query: str, category: str = "all", top_k: int = 3) -> list[str]:
    """从 ChromaDB 检索相关 FAQ，支持按 category 过滤"""
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name="faq_knowledge",
            embedding_function=DashScopeEmbedding(),
        )
        if collection.count() == 0:
            return []
        where_filter = None if category == "all" else {"category": category}
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )
        return results["documents"][0] if results and results["documents"] else []
    except Exception:
        logger.exception("search_faq 查询失败")
        return []


@tool
def search_faq_tool(query: str, category: str = "all") -> str:
    """搜索产品知识库。当需要查找产品文档、FAQ、政策条款、技术规格时调用。
    参数: query（搜索问题）、category（technical/aftersales/customer/all，默认all）。
    返回匹配的知识条目，用于回答用户关于产品功能、售后政策、使用指南等事实性问题。"""
    from utils.cache import cached_search_faq
    results_str = cached_search_faq(query, category)
    if not results_str:
        return "未在知识库中找到相关内容。"
    return results_str
