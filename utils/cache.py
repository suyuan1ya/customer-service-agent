"""FAQ 查询内存缓存，避免重复 Embedding + LLM 调用"""
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

FAQ_CACHE_SIZE = 512


@lru_cache(maxsize=FAQ_CACHE_SIZE)
def cached_search_faq(query: str, category: str) -> str:
    """缓存 FAQ 查询结果。参数 hash 作为 key，相同查询直接返回缓存。"""
    from tools.knowledge_base import search_faq
    logger.debug("FAQ cache miss: query='%s', category='%s'", query[:80], category)
    results = search_faq(query, category=category, top_k=3)
    return "\n---\n".join(results) if results else ""


def clear_faq_cache():
    info = cached_search_faq.cache_info()
    cached_search_faq.cache_clear()
    logger.info("FAQ cache cleared (was %d entries, %.1f%% hit rate)",
                info.currsize, info.hits / max(info.hits + info.misses, 1) * 100)
