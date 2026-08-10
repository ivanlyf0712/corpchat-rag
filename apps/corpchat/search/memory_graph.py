"""
CorpChat Search — Hindsight Memory Graph
==========================================
Entity-level memory graph ("星座视图") built from message metadata and
conversation memory.

Nodes: person / company / label / keyword / message.
Edges: mention (entity↔message), association (person↔company, person↔label,
person↔person same conversation), reference (message↔label).

POC entity extraction is rule-based (structured metadata + jieba keywords);
LLM entity extraction is a future enhancement.
"""

from collections import Counter
from typing import Dict, List, Optional

from .utils import _segment  # noqa: F401  (保持分词依赖可寻)

_STOPWORDS = set(
    "的 了 和 是 在 有 我 你 他 她 它 这 那 也 就 都 而 及 与 或 一个 一下 什么 怎么 我们 你们 他们 好 的 嗎 嗎 啊 吧 呢 请 請"
)


def build_entity_graph(
    messages: List[Dict],
    agent_memory: Optional[List[Dict]] = None,
    sources: Optional[List[str]] = None,
    risk_labels: Optional[set] = None,
    max_keywords: int = 8,
) -> Dict:
    """从消息元数据 + 对话记忆抽取实体关系图。

    Args:
        messages: 消息文档列表 (含 metadata: customer_name/company/label/open_kfid/...)
        agent_memory: agent_memory turn 列表 (user/bot 文本, 贡献关键词)
        sources: 数据源门控, 子集 of {"messages", "contacts"}
        risk_labels: 高怀疑度模式下高亮的风险标签集合
        max_keywords: 关键词节点数量上限

    Returns:
        {"nodes": [{id,label,type,size,highlighted}], "edges": [{source,target,type,label}]}
    """
    sources = sources or ["messages", "contacts"]
    risk_labels = risk_labels or {"old_friend_reconnect", "詐騙", "fraud"}

    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []

    def _add_node(nid, label, ntype):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "label": label, "type": ntype, "size": 0, "highlighted": False}

    def _add_edge(src, dst, etype, label):
        if not src or not dst or src == dst:
            return
        edges.append({"source": src, "target": dst, "type": etype, "label": label})

    # ── 实体节点 + 关系 ──
    for msg in messages:
        meta = msg.get("metadata", {}) or {}
        mid = str(msg.get("id", ""))
        cust = meta.get("customer_name") or meta.get("external_userid")
        company = meta.get("company")
        label = meta.get("label")
        kfid = meta.get("open_kfid")

        if mid:
            _add_node(mid, (str(msg.get("text", ""))[:20]), "message")

        if "contacts" in sources and cust:
            _add_node(f"person:{cust}", str(cust), "person")
            if mid:
                _add_edge(f"person:{cust}", mid, "mention", "提及")
            if company:
                _add_node(f"company:{company}", str(company), "company")
                _add_edge(f"person:{cust}", f"company:{company}", "association", "关联")

        if "messages" in sources and label:
            _add_node(f"label:{label}", str(label), "label")
            if mid:
                _add_edge(f"label:{label}", mid, "reference", "引用")
            if cust:
                _add_edge(f"person:{cust}", f"label:{label}", "association", "关联")

    # ── 同会话 person-person 关联边 ──
    if "contacts" in sources:
        by_kfid: Dict[str, List[str]] = {}
        for msg in messages:
            meta = msg.get("metadata", {}) or {}
            cust = meta.get("customer_name") or meta.get("external_userid")
            kfid = meta.get("open_kfid")
            if cust and kfid:
                by_kfid.setdefault(str(kfid), [])
                if f"person:{cust}" not in by_kfid[str(kfid)]:
                    by_kfid[str(kfid)].append(f"person:{cust}")
        for kfid, persons in by_kfid.items():
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    _add_edge(persons[i], persons[j], "association", "同会话")

    # ── 关键词节点 (jieba, 消息 + agent_memory 文本) ──
    if "messages" in sources:
        kw_counter: Counter = Counter()
        texts = [str(m.get("text", "")) for m in messages]
        if agent_memory:
            texts += [str(t.get("user", "")) for t in agent_memory]
            texts += [str(t.get("bot", "")) for t in agent_memory]
        for text in texts:
            for w in _segment(text).split():
                w_clean = w.strip()
                if len(w_clean) >= 2 and w_clean not in _STOPWORDS:
                    kw_counter[w_clean] += 1
        for w, _c in kw_counter.most_common(max_keywords):
            _add_node(f"kw:{w}", w, "keyword")
            for msg in messages:
                if w in str(msg.get("text", "")):
                    _add_edge(f"kw:{w}", str(msg.get("id", "")), "mention", "提及")

    # ── 节点大小 = 连接密度 ──
    degree: Counter = Counter()
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    for nid, n in nodes.items():
        n["size"] = min(degree[nid], 30) + 5

    # ── 风险标签高亮 (高怀疑度模式) ──
    for n in nodes.values():
        if n["type"] == "label" and n["label"] in risk_labels:
            n["highlighted"] = True

    return {"nodes": list(nodes.values()), "edges": edges}
