# 1️⃣ 选择 Python 基础镜像
FROM python:3.9

# 2️⃣ 设置工作目录
WORKDIR /app

# 3️⃣ 复制项目代码到容器
COPY . .

# 4️⃣ 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ 暴露 API 端口
EXPOSE 8000

# 6️⃣ 运行 FastAPI 服务器
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]