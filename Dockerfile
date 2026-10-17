FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# 用 uv 安装依赖（比 pip 快 5~10 倍）
RUN pip install --no-cache-dir uv
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

COPY . /app
RUN mkdir -p /app/data/uploads /app/data/eval

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
