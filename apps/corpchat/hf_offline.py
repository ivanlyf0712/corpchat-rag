"""HF 离线自动检测 — 必须在任何 huggingface_hub / txtai import 之前调用。

原因: huggingface_hub.constants 在其模块 import 时读取环境变量 HF_HUB_OFFLINE;
一旦 huggingface_hub 已被 import (txtai / sentence_transformers 都会拉入), 之后再设
HF_HUB_OFFLINE=1 不会改变已缓存的常量 → 模型加载仍会联网校验。因此本模块只用
标准库做纯文件系统检查, 由 warmup.py / app.py 在导入搜索栈之前调用。

模型在 HF hub 缓存 (hf-cache 卷) 已就位 → 自动设 HF_HUB_OFFLINE=1 (启动离线提速);
未缓存 → 保持在线, 由 warmup 首次拉取。用户显式设了 HF_HUB_OFFLINE 时以显式值为准。
"""
from __future__ import annotations

import os
from typing import Optional


def _hf_cache_has_model(model_id: str) -> bool:
    """HF hub 缓存是否已含该模型的可用快照 (纯文件系统检查)。

    结构: <cache_root>/hub/models--<repo>/snapshots/<sha>/... 任一非空快照目录
    即视为已缓存。不 import huggingface_hub, 避免在其 constants 计算前提前引入。
    """
    cache_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"
    )
    repo_dir = os.path.join(cache_root, "models--" + model_id.replace("/", "--"))
    snapshots = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return False
    for entry in os.listdir(snapshots):
        snap = os.path.join(snapshots, entry)
        if os.path.isdir(snap) and os.listdir(snap):
            return True
    return False


def apply_auto_offline(embed_model: Optional[str] = None,
                       rerank_model: Optional[str] = None) -> None:
    """嵌入+重排模型都已缓存 → 设 HF_HUB_OFFLINE=1 (未显式设置时)。

    模型 id 默认读环境变量 EMBEDDING_MODEL / RERANKER_MODEL (与 config.py 一致),
    便于用户换模型后仍能正确检测。
    """
    embed_model = embed_model or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    rerank_model = rerank_model or os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
    if os.environ.get("HF_HUB_OFFLINE") is None and (
        _hf_cache_has_model(embed_model) and _hf_cache_has_model(rerank_model)
    ):
        os.environ["HF_HUB_OFFLINE"] = "1"
