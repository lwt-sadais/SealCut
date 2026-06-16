# SealCut

公章（印章）抠图微服务 API，从文档图片中提取透明背景印章。

基于 AI（U2-Net 模型）自动去除背景，支持红色、蓝色、黑色印章提取，并可自动检测印章颜色，输出透明背景 PNG，可直接叠加到其他文档。

## API 接口

### 健康检查

```
GET /health
```

响应：
```json
{"status": "ok"}
```

### 提取印章

```
POST /api/v1/seal/extract
```

**参数（multipart/form-data）：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | 是 | - | 包含印章的图片文件（PNG/JPG/JPEG），最大 10MB，分辨率不超过 4000x4000 |
| type | int | 否 | 0 | 返回格式：0=直接返回 PNG 二进制，1=JSON + Base64 编码 |
| seal_color | string | 否 | auto | 印章颜色：red / blue / black / auto（自动检测） |

**示例 - 自动检测颜色（默认）：**

```bash
curl -X POST http://localhost:8000/api/v1/seal/extract \
  -F "file=@seal.png" \
  -o result.png
```

**示例 - 指定红色印章：**

```bash
curl -X POST http://localhost:8000/api/v1/seal/extract \
  -F "file=@seal.png" \
  -F "seal_color=red" \
  -o result.png
```

**示例 - JSON + Base64 模式：**

```bash
curl -X POST http://localhost:8000/api/v1/seal/extract \
  -F "file=@seal.png" \
  -F "type=1" \
  -F "seal_color=blue"
```

响应：
```json
{
  "image": "<Base64 编码的 PNG 图片数据>"
}
```

**错误响应：**
```json
{
  "detail": "错误描述信息"
}
```

## Docker 部署

### 前置条件

需要预先下载 U2-Net 模型文件 `u2net.onnx`（约 176MB）：

- 下载地址：`https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx`
- MD5：`60024c5c889badc19c04ad937298a77b`

将下载的文件放到宿主机某个目录下，例如 `/data/sealcut-models/u2net.onnx`。

### 使用 docker-compose（推荐）

1. 修改 `docker-compose.yml` 中的 volume 路径，将 `/path/to/models` 替换为实际模型目录：

```yaml
volumes:
  - /data/sealcut-models:/app/models
```

2. 启动服务：

```bash
docker-compose up -d
```

3. 验证：

```bash
curl http://localhost:8000/health
```

### 手动构建运行

```bash
# 构建镜像
docker build -t sealcut .

# 运行容器
docker run -d -p 8000:8000 \
  --memory=4g \
  -v /data/sealcut-models:/app/models \
  --name sealcut \
  sealcut
```

### 配置说明

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 端口 | 8000 | API 服务端口 |
| 内存 | 4GB | 建议分配，支持 2 并发处理 |
| 模型目录 | `/app/models` | 容器内路径，需通过 volume 挂载 |
| 并发数 | 2 | 超出自动排队 |

## 本地开发

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 下载模型到默认缓存目录（~/.u2net/）
mkdir -p ~/.u2net
cp /path/to/u2net.onnx ~/.u2net/u2net.onnx

# 启动服务
python app.py
```

## 运行测试

```bash
# 单元测试（颜色分类器，无需模型）
pytest tests/test_color_filters.py -v

# 集成测试（需模型文件）
pytest tests/test_api.py -v
```

## 注意事项

- 扫描文档图片效果最佳，照片效果可能不佳
- 首次部署需确保 `u2net.onnx` 模型文件已正确挂载，否则服务启动失败
- 并发限制为 2 个请求，超出自动排队等待
