"""FAQ 知识库加载器 — 将三类 FAQ 文档向量化存入 ChromaDB"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from config import CHROMA_PERSIST_DIR
from tools.knowledge_base import DashScopeEmbedding

FAQ_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faq")


def load_faqs():
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    try:
        client.delete_collection("faq_knowledge")
    except Exception:
        pass

    embedding_fn = DashScopeEmbedding()
    collection = client.get_or_create_collection(
        name="faq_knowledge",
        embedding_function=embedding_fn,
    )

    docs = []
    metadatas = []
    ids = []

    for filename in sorted(os.listdir(FAQ_DIR)):
        if not filename.endswith(".txt"):
            continue
        category = filename.replace(".txt", "")
        filepath = os.path.join(FAQ_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        entries = [e.strip() for e in content.split("##") if e.strip()]
        for i, entry in enumerate(entries):
            docs.append(entry)
            metadatas.append({"category": category, "source": filename})
            ids.append(f"{category}_{i}")

    if docs:
        collection.add(documents=docs, metadatas=metadatas, ids=ids)  # type: ignore
        print(f"已加载 {len(docs)} 条 FAQ 到知识库 ({CHROMA_PERSIST_DIR})")
    else:
        print("未找到 FAQ 文档，知识库为空")
    return len(docs)


if __name__ == "__main__":
    load_faqs()
