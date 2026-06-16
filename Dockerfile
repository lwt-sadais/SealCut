FROM docker.m.daocloud.io/python:3.11-slim
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# onnxruntime 运行时依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY seal_processor.py .
COPY app.py .

# 模型目录通过 volume 挂载，U2NET_HOME 指向该目录
# docker-compose 示例：
#   volumes:
#     - /path/to/models:/app/models
#   environment:
#     - U2NET_HOME=/app/models
# 目录中需包含 u2net.onnx 文件
ENV U2NET_HOME=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
