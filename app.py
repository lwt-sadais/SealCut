import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from seal_processor import SealColor, extract_seal_bytes, init_session

# 配置日志：输出到 stdout，Docker 友好
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sealcut")

# 环境变量控制是否输出请求日志，默认关闭
_REQUEST_LOG_ENABLED = os.getenv("SEALCUT_REQUEST_LOG", "").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预热模型，关闭时清理资源"""
    logger.info("SealCut API 正在启动...")
    init_session()
    logger.info("SealCut API 服务就绪，监听 0.0.0.0:8000")
    yield


app = FastAPI(
    title="SealCut API",
    description="印章提取微服务 API",
    version="1.0.0",
    lifespan=lifespan,
)

# 并发控制：最多 2 个请求同时处理，超出排队等待
_semaphore = asyncio.Semaphore(2)

# 请求限制
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_DIMENSION = 4000


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "ok"}


@app.post("/api/v1/seal/extract")
async def extract_seal(
    file: UploadFile = File(..., description="包含印章的图片文件（PNG/JPG/JPEG）"),
    type: int = Form(0, description="返回格式：0=二进制图片，1=JSON+Base64"),
    seal_color: str = Form("auto", description="印章颜色：red/blue/black/auto（自动检测）"),
):
    """
    提取印章图片

    - **file**: 包含印章的图片文件
    - **type**: 返回格式，0=直接返回 PNG 二进制，1=返回 JSON（Base64 编码）
    - **seal_color**: 印章颜色，支持 red/blue/black，auto 为自动检测
    """
    # 校验 seal_color
    try:
        color_enum = SealColor(seal_color)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"无效的 seal_color '{seal_color}'，可选值：red, blue, black, auto",
        )

    # 校验 type
    if type not in (0, 1):
        raise HTTPException(
            status_code=400,
            detail=f"无效的 type '{type}'，可选值：0（二进制）或 1（JSON+Base64）",
        )

    # 读取文件内容并校验大小
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件大小超过 10MB 限制")

    # 校验图片格式和分辨率
    try:
        img = Image.open(BytesIO(content))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="无效或不支持的图片文件")
    if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"图片分辨率超过 {MAX_DIMENSION}x{MAX_DIMENSION} 限制，当前 {img.width}x{img.height}",
        )

    # 获取信号量，控制并发
    async with _semaphore:
        try:
            if _REQUEST_LOG_ENABLED:
                logger.info(
                    "处理请求: file=%s, seal_color=%s, type=%d",
                    file.filename, seal_color, type,
                )
            start = time.time()
            loop = asyncio.get_event_loop()
            result_bytes = await loop.run_in_executor(
                None, extract_seal_bytes, content, color_enum
            )
            if _REQUEST_LOG_ENABLED:
                logger.info("请求处理完成，耗时: %.2f 秒", time.time() - start)
        except Exception as e:
            logger.exception("印章提取处理失败")
            raise HTTPException(status_code=500, detail="图片处理失败")

    # 根据 type 参数返回不同格式
    if type == 0:
        return Response(content=result_bytes, media_type="image/png")
    else:
        b64 = base64.b64encode(result_bytes).decode("ascii")
        return JSONResponse(content={"image": b64})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
