"""
CorpChat Search — Index Builder
================================
Builds the txtai hybrid search index with sentence-level chunking,
enrichment, and structural graph relationships.
"""

import json
import os
import time
from typing import Dict, List, Optional

import psycopg2
import txtai

from .config import (
    DB_CONFIG,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_INDEX_PATH,
    _EMBED_MODEL,
    logger,
)
from .utils import _compute_structural_relationships, _segment


class IndexBuilder:
    """构建带分块、丰富化和元数据的混合搜索索引。"""

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── 数据读取 ──────────────────────────────────────────────
    def _fetch_messages(self) -> List[Dict]:
        """从 PostgreSQL 读取消息及关联联系人信息。"""
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT m.msgid, m.content, m.send_time, m.external_userid,
                   m.servicer_userid, m.label, c.full_name AS customer_name,
                   m.open_kfid, m.origin, c.company
            FROM messages m
            LEFT JOIN contacts c ON m.external_userid = c.userid
            WHERE m.content IS NOT NULL AND m.content != ''
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        messages = []
        for row in rows:
            send_time_raw = row[2]
            if hasattr(send_time_raw, 'isoformat'):
                send_time_str = send_time_raw.isoformat()
            else:
                send_time_str = str(send_time_raw) if send_time_raw else None

            messages.append({
                "msgid": row[0],
                "content": row[1],
                "send_time": send_time_raw,
                "send_time_str": send_time_str,
                "external_userid": row[3],
                "servicer_userid": row[4],
                "label": row[5],
                "customer_name": row[6] or str(row[3]),
                "company": row[9] if len(row) > 9 else None,
                "open_kfid": row[7],
                "origin": row[8],
            })
        return messages

    # ── 分块 (§2.2) ────────────────────────────────────────────
    def _chunk_message(self, msg: Dict) -> List[Dict]:
        """将单条消息拆分为句子级块。"""
        content = msg["content"]
        chunks_text = []

        try:
            from chonkie import SentenceChunker

            def _token_counter(text: str) -> int:
                chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
                other_chars = len(text) - chinese_chars
                return int(chinese_chars / 2 + other_chars / 4)

            chunker = SentenceChunker(
                tokenizer_or_token_counter=_token_counter,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                return_type="texts",
            )
            chunks_text = chunker.chunk(content)
        except (ImportError, Exception) as e:
            logger.debug(f"chonkie 不可用 ({e}), 使用 fallback 分块")
            import re
            sentences = re.split(r'(?<=[.!?。！？])\s*', content)
            current = []
            current_len = 0
            for sent in sentences:
                sent_len = len(sent)
                if current_len + sent_len < self.chunk_size * 4:
                    current.append(sent)
                    current_len += sent_len
                else:
                    if current:
                        chunks_text.append(" ".join(current).strip())
                    current = [sent]
                    current_len = sent_len
            if current:
                chunks_text.append(" ".join(current).strip())

        if not chunks_text:
            chunks_text = [content]

        base_id = msg["msgid"]
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_id = f"{base_id}__chunk{i}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "msgid": msg["msgid"],
                    "send_time": msg.get("send_time_str"),
                    "external_userid": msg["external_userid"],
                    "servicer_userid": msg["servicer_userid"],
                    "label": msg["label"],
                    "customer_name": msg["customer_name"],
                    "company": msg.get("company"),
                    "open_kfid": msg["open_kfid"],
                    "origin": msg["origin"],
                    "chunk_index": i,
                },
                "title": (
                    f"{msg['customer_name']} ({msg['label'] or 'general'})"
                ),
            })
        return chunks

    # ── 丰富化 (§2.3) ──────────────────────────────────────────
    def _enrich_chunk(self, chunk: Dict) -> str:
        """
        丰富化: 组合标题 + 内容 → 用于嵌入与匹配的最终文本。

        匹配面 (match surface) 仅包含:
          - 标题: customer_name (label)  — 提供消歧上下文
          - 内容: 消息正文
        元数据 (label, 时间, 发送者, 接收者, msgid 等) 不再拼入文本,
        而是作为结构化 tags 存入 sections.tags 列, 用于过滤/展示/LLM 上下文。

        格式: [title]\n---\n[content]
        """
        title = chunk.get("title", "")
        text = chunk["text"]
        # 中文分词: 让 BM25 能匹配未加空格的中文短语
        return f"{title}\n---\n{_segment(text)}"

    # ── Contacts 索引构建 ─────────────────────────────────────
    @staticmethod
    def _fetch_contacts() -> List[Dict]:
        """从 PostgreSQL 读取联系人数据。"""
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT userid, full_name, email, company, phone, job_title
            FROM contacts
            WHERE full_name IS NOT NULL
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        contacts = []
        for row in rows:
            contacts.append({
                "userid": row[0],
                "full_name": row[1],
                "email": row[2],
                "company": row[3],
                "phone": row[4] or "",
                "job_title": row[5] or "",
            })
        return contacts

    @staticmethod
    def build_contacts_index(contacts_index_path: str) -> txtai.Embeddings:
        """
        为 contacts 表构建独立的 txtai 索引。

        每个联系人为一个文档, 文本内容拼接 full_name + company + job_title,
        元数据包含 userid, full_name, email, company, phone, job_title。
        用于跨表推理时通过语义搜索查找联系人。
        """
        contacts = IndexBuilder._fetch_contacts()
        if not contacts:
            logger.warning("数据库中没有联系人数据")
            # 返回空索引
            config = {"path": _EMBED_MODEL, "content": True, "objects": True, "hybrid": True, "scoring": {"method": "bm25"}}
            embeddings = txtai.Embeddings(config)
            embeddings.save(contacts_index_path)
            return embeddings

        logger.info(f"构建 contacts 索引 ({len(contacts)} 条记录) ...")

        docs = []
        for c in contacts:
            # 索引文本: 姓名 + 公司 + 职位 (便于语义搜索)
            text = f"{c['full_name']}\n---\n{c['company']} {c['job_title']}"
            if c.get("email"):
                text += f"\nEmail: {c['email']}"
            metadata = {
                "userid": c["userid"],
                "full_name": c["full_name"],
                "email": c["email"],
                "company": c["company"],
                "phone": c["phone"],
                "job_title": c["job_title"],
                "source_type": "contact",
            }
            tags_json = json.dumps(metadata, default=str)
            docs.append((c["userid"], text, tags_json))

        config = {
            "path": _EMBED_MODEL,
            "content": True,
            "objects": True,
            "hybrid": True,
            "scoring": {"method": "bm25"},
        }
        embeddings = txtai.Embeddings(config)

        t0 = time.perf_counter()
        embeddings.index(docs)
        logger.info(f"contacts 索引完成, {len(docs)} 条, 耗时 {time.perf_counter()-t0:.2f}s")

        embeddings.save(contacts_index_path)
        logger.info(f"contacts 索引保存至 {contacts_index_path}")
        return embeddings

    # ── 索引构建入口 ──────────────────────────────────────────
    def build(self, force: bool = False, enable_graph: bool = True,
              graph_mode: str = "auto") -> txtai.Embeddings:
        """构建或加载索引。"""
        if os.path.exists(self.index_path) and not force:
            logger.info(f"从 {self.index_path} 加载已有索引 ...")
            embeddings = txtai.Embeddings()
            embeddings.load(self.index_path)
            logger.info(f"已加载 {embeddings.count()} 个块")
            return embeddings

        logger.info("从数据库构建新索引 (含分块+丰富化) ...")
        messages = self._fetch_messages()
        if not messages:
            raise RuntimeError("数据库中没有消息数据")

        all_chunks = []
        for msg in messages:
            chunks = self._chunk_message(msg)
            all_chunks.extend(chunks)
        logger.info(f"分块完成: {len(messages)} 条消息 → {len(all_chunks)} 个块")

        # 计算结构关系 (仅当启用图时)
        relationships: Dict[str, List[Dict]] = {}
        if enable_graph:
            relationships = _compute_structural_relationships(all_chunks)

        docs = []
        for chunk in all_chunks:
            enriched = self._enrich_chunk(chunk)
            tags_json = json.dumps(chunk["metadata"], default=str)
            if enable_graph and relationships:
                docs.append((
                    chunk["id"],
                    {
                        "text": enriched,
                        "relationships": relationships.get(chunk["id"], []),
                    },
                    tags_json,
                ))
            else:
                docs.append((chunk["id"], enriched, tags_json))

        config: Dict = {
            "path": _EMBED_MODEL,
            "content": True,
            "objects": True,
            "hybrid": True,
            "scoring": {"method": "bm25"},
            "columns": {"relationships": "relationships"},
        }
        if enable_graph:
            config["graph"] = True

        logger.info(f"模型: {_EMBED_MODEL}")
        logger.info(f"图功能: {'✅' if enable_graph else '❌'} (模式: structural)")

        embeddings = txtai.Embeddings(config)

        t0 = time.perf_counter()
        logger.info(f"索引 {len(docs)} 个文档 ...")
        embeddings.index(docs)
        logger.info(f"索引完成, 耗时 {time.perf_counter()-t0:.2f}s")

        if enable_graph and embeddings.graph:
            logger.info("图构建完成: 纯结构关系 (same_conversation / sender_receiver / same_sender / same_company / same_label)")

        embeddings.save(self.index_path)
        logger.info(f"索引保存至 {self.index_path}")
        return embeddings
