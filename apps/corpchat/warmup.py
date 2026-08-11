"""CorpChat 容器启动预热 (fail-fast + 冷启动提速)。

在 `streamlit run` 之前运行, 目标:
  1. 索引缺失/模型损坏 → 容器启动即失败 (清晰日志), 而不是第一次搜索才崩;
  2. 把模型/索引文件读入 OS page cache, 缩短 streamlit 进程首次搜索的冷加载;
  3. 首次联网拉取模型在启动时完成, 不占用用户交互时间。

说明: Python 模块级缓存是进程级的, 本脚本在独立进程运行, 无法填充 streamlit
进程的内存缓存; 但 page cache 预热能显著减少 streamlit 进程首次 load_agent
的磁盘等待, 且把"缺索引/装坏模型"这类问题提前到启动阶段暴露。
"""
from __future__ import annotations

import os
import sys
import time

# 以路径方式运行 (python apps/corpchat/warmup.py) 时, sys.path[0] 是脚本所在目录,
# 仓库根不在其中 → 手动加入, 保证 `apps.corpchat.*` 可导入 (容器内同样适用)。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 在导入搜索栈之前设 HF 离线开关: huggingface_hub.constants 在 import 时读 env,
# 一旦 txtai 提前拉入 huggingface_hub, 自动离线将不生效。
from apps.corpchat.hf_offline import apply_auto_offline
apply_auto_offline()


def main() -> None:
    t0 = time.perf_counter()
    print("[warmup] importing search stack ...", flush=True)

    # 完整 import 链 (transformers / txtai / sentence_transformers / agent)
    from apps.corpchat.search import (
        CONTACTS_INDEX_PATH,
        DEFAULT_INDEX_PATH,
        load_contacts_index,
    )
    from apps.corpchat.agent import load_agent
    from apps.corpchat.search.config import DEFAULT_RERANKER_MODEL
    from apps.corpchat.search.reranker import _load_cross_encoder

    # 1) txtai 消息索引 + bge-m3 嵌入模型 (与首次搜索的 load_agent 同路径)
    agent = load_agent(DEFAULT_INDEX_PATH)
    n_msg = agent.searcher.embeddings.count() if agent.searcher is not None else -1
    print(f"[warmup] messages index ready ({n_msg} chunks) in {time.perf_counter() - t0:.1f}s", flush=True)

    # 2) 联系人索引 (独立 txtai 实例)
    contacts = load_contacts_index(CONTACTS_INDEX_PATH)
    print(f"[warmup] contacts index ready ({contacts.count()} contacts)", flush=True)

    # 3) 交叉编码器重排模型 (bge-reranker-base)
    _load_cross_encoder(DEFAULT_RERANKER_MODEL)
    print(f"[warmup] cross-encoder ready in {time.perf_counter() - t0:.1f}s", flush=True)

    print("[warmup] done — starting streamlit", flush=True)


if __name__ == "__main__":
    main()
