"""
CorpChat Search — Shared Utilities
===================================
Helper functions used across the search package: text cleaning, Chinese
word segmentation, and structural relationship computation.
"""

from typing import Dict, List

from .config import _JIEBA_AVAILABLE, _METADATA_MARKER, logger


def _clean_text_from_enriched(text: str) -> str:
    """
    从 enriched text 中提取干净的内容文本 (去掉 title 前缀和 metadata 后缀)。

    返回: 去除了标题行和 Metadata 部分的原始消息内容。
    """
    # 去掉 Metadata 后缀 (兼容旧索引)
    if _METADATA_MARKER in text:
        text = text.split(_METADATA_MARKER)[0]
    # 去掉 title 前缀 (第一行 "---" 之前的内容和 "---" 分隔符)
    parts = text.split("\n---\n", 1)
    if len(parts) > 1:
        return parts[1]
    return text


def _segment(text: str) -> str:
    """
    使用 jieba 对中文文本进行分词, 以空格连接。

    使 txtai 默认的 Unicode 分词器能按 jieba 的词语边界切分中文,
    从而让 BM25 能匹配未加空格的中文短语 (如 投資美國債券跟藍籌股)。
    索引与查询两侧使用同一分词器, 保证一致性。
    """
    if not text:
        return text
    if _JIEBA_AVAILABLE:
        import jieba
        return " ".join(jieba.cut_for_search(text))
    return text


def _format_citations(results, max_sources: int = 3) -> str:
    """从搜索结果构建引用块 (sender · 日期 · label); 空结果返回空串。"""
    lines = []
    for r in results[:max_sources]:
        meta = r.get("metadata", {}) if isinstance(r, dict) else {}
        sender = meta.get("customer_name") or meta.get("external_userid", "?")
        ts = str(meta.get("send_time", ""))[:10]
        label = meta.get("label", "-")
        lines.append(f"- {sender} · {ts} · [{label}]")
    return "\n【來源】\n" + "\n".join(lines) if lines else ""


def _compute_structural_relationships(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    从分块元数据计算五个结构关系, 返回 {chunk_id: [relationships]}.

    关系类型: same_conversation, sender_receiver, same_sender, same_company, same_label.
    """
    metas = {chunk["id"]: chunk.get("metadata", {}) for chunk in chunks}
    relationships: Dict[str, List[Dict]] = {cid: [] for cid in metas}
    ids = list(metas.keys())

    for i, a_id in enumerate(ids):
        a = metas[a_id]
        for b_id in ids[i + 1:]:
            b = metas[b_id]
            rels: set = set()

            if a.get("open_kfid") and a["open_kfid"] == b.get("open_kfid"):
                rels.add("same_conversation")

            if a.get("open_kfid") and a["open_kfid"] == b.get("open_kfid"):
                if a.get("external_userid") == b.get("servicer_userid") or \
                   b.get("external_userid") == a.get("servicer_userid"):
                    rels.add("sender_receiver")

            if a.get("external_userid") and a["external_userid"] == b.get("external_userid"):
                rels.add("same_sender")

            if a.get("company") and a["company"] == b.get("company"):
                rels.add("same_company")

            if a.get("label") and a["label"] == b.get("label"):
                rels.add("same_label")

            for rel in rels:
                relationships[a_id].append({"id": b_id, "relation": rel})
                relationships[b_id].append({"id": a_id, "relation": rel})

    return relationships