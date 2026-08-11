# CorpChat RAG — 容器化镜像
# 运行: docker build -t corpchat-rag .
#       docker run --rm -p 8501:8501 --network host -e LITELLM_API_KEY=... -e DEEPSEEK_API_KEY=... corpchat-rag
# 说明: postgres 已作为容器运行 (5432), 应用容器通过 host 网络访问 localhost:5432。
# 注意: requirements.txt 的 pin 面向 Python 3.10; 镜像用 3.12 需要放宽 (见下)。

FROM python:3.12-slim

WORKDIR /app

# 系统依赖: psycopg2-binary 需要 libpq; 分词/嵌入无额外系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖: 3.12 下放宽 requirements.txt 的硬 pin (networkx/numpy/pandas 等
# 要求 Python>=3.11 的版本与 3.10 pin 冲突)。关键包保持兼容版本。
COPY requirements.txt .
RUN pip install --no-cache-dir \
    "langchain==1.3.14" "langchain-community==0.4.2" "langchain-core==1.5.3" "langgraph==1.2.10" \
    "txtai[graph]==9.12.0" "chonkie==1.7.0" "sentence-transformers==5.6.1" \
    "streamlit==1.59.2" "jieba==0.42.1" "psycopg2-binary==2.9.12" \
    "pandas" "numpy" "networkx" "requests" "python-dotenv" \
    "Faker" "tabulate" "click" "tenacity" "ollama" "GitPython" \
    "pydantic" "httpx" "altair" "pydeck" "pillow" "pymupdf" "starlette" "uvicorn"

# 应用代码 + 索引 (140 chunks + 30 contacts)
COPY apps/ apps/
COPY core/ core/
COPY lib/ lib/

# 运行环境: Streamlit 监听容器内 8501。
# --server.fileWatcherType=none: 禁用源码热重载监听 —— Streamlit 的模块扫描会
# 触发 transformers 5.15.0 全部 198 个子包的惰性 import, 其中几十个视觉/OCR 模块
# 依赖未安装的 torchvision, 每次首次加载都会卡 ~45s 并刷屏 traceback。
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# 启动前先跑 warmup (加载 txtai 索引 + bge-m3 + 交叉编码器, fail-fast + page cache
# 预热), 再启动 streamlit。
CMD ["sh", "-c", "python apps/corpchat/warmup.py && exec streamlit run apps/corpchat/app.py --server.port=8501 --server.address=0.0.0.0 --server.fileWatcherType=none"]
