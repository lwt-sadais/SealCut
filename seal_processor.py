import logging
import os
import time
from enum import Enum
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image
from rembg import remove, new_session
from rembg.sessions.base import BaseSession

logger = logging.getLogger("sealcut")


class SealColor(str, Enum):
    """印章颜色枚举"""
    RED = "red"
    BLUE = "blue"
    BLACK = "black"
    AUTO = "auto"


# 模块级 rembg session，启动时初始化一次，所有请求复用
_session: Optional[BaseSession] = None


def init_session() -> None:
    """预热 rembg 模型，应用启动时调用一次"""
    global _session
    if _session is None:
        model_path = os.path.expanduser(
            os.getenv("U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net"))
        )
        logger.info("正在加载 U2-Net 模型，路径: %s", model_path)
        start = time.time()
        _session = new_session("u2net")
        elapsed = time.time() - start
        logger.info("U2-Net 模型加载完成，耗时: %.2f 秒", elapsed)


def _classify_seal_pixels_red(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """红色印章分类：R 通道主导（1.3 倍，兼容印章边缘混合色）"""
    return (r > 120) & (r > g * 1.3) & (r > b * 1.3)


def _classify_seal_pixels_blue(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """蓝色印章分类：B 通道主导"""
    return (b > 150) & (b > r * 1.5) & (b > g * 1.5)


def _classify_seal_pixels_black(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """黑色印章分类：低亮度且非白色"""
    brightness = (r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32)) / 3.0
    is_dark = brightness < 100
    is_not_white = ~((r > 200) & (g > 200) & (b > 200))
    return is_dark & is_not_white


# 颜色分类器映射（不含 AUTO，AUTO 通过 _auto_detect_color 动态选择）
_CLASSIFIERS = {
    SealColor.RED: _classify_seal_pixels_red,
    SealColor.BLUE: _classify_seal_pixels_blue,
    SealColor.BLACK: _classify_seal_pixels_black,
}

# 自适应颜色检测阈值：某通道均值超过其他通道均值的倍数即判定为主导色
_COLOR_DOMINANCE_RATIO = 1.3


def _auto_detect_color(img_array: np.ndarray) -> SealColor:
    """根据非透明像素的 RGB 分布自动检测印章颜色"""
    mask = img_array[:, :, 3] > 0
    if not mask.any():
        return SealColor.BLACK
    r_mean = img_array[:, :, 0][mask].mean()
    g_mean = img_array[:, :, 1][mask].mean()
    b_mean = img_array[:, :, 2][mask].mean()
    if r_mean > g_mean * _COLOR_DOMINANCE_RATIO and r_mean > b_mean * _COLOR_DOMINANCE_RATIO:
        return SealColor.RED
    if b_mean > r_mean * _COLOR_DOMINANCE_RATIO and b_mean > g_mean * _COLOR_DOMINANCE_RATIO:
        return SealColor.BLUE
    return SealColor.BLACK

# 颜色增强系数
_COLOR_ENHANCE_FACTOR = 1.1

# 像素过滤阈值
_WHITE_THRESHOLD = 200
_GRAY_DIFF_THRESHOLD = 15
_COLOR_DIFF_THRESHOLD = 20
_BRIGHTNESS_THRESHOLD = 210

# 裁剪 padding（像素）
_CROP_PADDING = 8


def _is_background_pixel(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """检测背景像素：白色、灰色、低饱和度或高亮度"""
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    white = (r > _WHITE_THRESHOLD) & (g > _WHITE_THRESHOLD) & (b > _WHITE_THRESHOLD)
    brightness = (r16 + g16 + b16) / 3.0
    gray = (np.abs(r16 - g16) < _GRAY_DIFF_THRESHOLD) & \
           (np.abs(g16 - b16) < _GRAY_DIFF_THRESHOLD) & \
           (np.abs(r16 - b16) < _GRAY_DIFF_THRESHOLD) & \
           (brightness > _BRIGHTNESS_THRESHOLD)
    color_diff = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    low_saturation = (color_diff < _COLOR_DIFF_THRESHOLD) & (brightness > _BRIGHTNESS_THRESHOLD)
    high_brightness = brightness > _BRIGHTNESS_THRESHOLD
    return white | gray | low_saturation | high_brightness


def _has_alpha_channel(image: Image.Image) -> bool:
    """检测图片是否包含有意义的 alpha 通道（非全不透明）"""
    if image.mode != "RGBA":
        return False
    alpha = np.array(image)[:, :, 3]
    return bool((alpha < 255).any())


def _remove_bg(image: Image.Image, session: BaseSession) -> Image.Image:
    """使用 rembg 移除背景，关闭 alpha_matting 避免彩色印章半透明问题"""
    return remove(image, alpha_matting=False, session=session)


def _remove_bg_combined_black(image: Image.Image, session: BaseSession) -> Image.Image:
    """黑色印章专用：正向+反色 rembg 合成，解决 U2-Net 将黑色当背景的问题"""
    img_arr = np.array(image.convert("RGBA"))

    # 正向 rembg：保留印章内部不透明像素
    alpha_fwd = np.array(_remove_bg(image, session))[:, :, 3].astype(np.float32)

    # 反色后 rembg：U2-Net 对"深色背景+浅色物体"识别好
    inverted = Image.fromarray(255 - img_arr[:, :, :3])
    inverted_rgba = inverted.convert("RGBA")
    alpha_inv = np.array(_remove_bg(inverted_rgba, session))[:, :, 3].astype(np.float32)

    # 合并 alpha = max(正向, 反向)
    combined_alpha = np.maximum(alpha_fwd, alpha_inv)
    img_arr[:, :, 3] = combined_alpha.astype(np.uint8)
    return Image.fromarray(img_arr)


def extract_seal_bytes(
    image_bytes: bytes,
    seal_color: SealColor = SealColor.AUTO,
) -> bytes:
    """
    从图片字节中提取印章，返回 PNG 字节流

    Args:
        image_bytes: 原始图片文件字节（PNG/JPEG）
        seal_color: 要提取的印章颜色，auto 为自动检测

    Returns:
        透明背景的 PNG 图片字节
    """
    if _session is None:
        init_session()

    # 从字节流读取图片
    input_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    img_array = np.array(input_image)

    # 阶段一：输入分流 — 有 alpha 通道则跳过 rembg
    if _has_alpha_channel(input_image):
        logger.info("输入图片已有 alpha 通道，跳过 rembg 背景移除")
    else:
        # 阶段二：按印章颜色选择 rembg 策略
        # 先用原图做一次颜色预检测（白底图片的白色像素不影响主导色判定）
        if seal_color == SealColor.AUTO:
            seal_color = _auto_detect_color(img_array)

        if seal_color == SealColor.BLACK:
            logger.info("黑色印章：使用正向+反色 rembg 合成")
            output_image = _remove_bg_combined_black(input_image, _session)
        else:
            logger.info("彩色印章：使用关闭 alpha_matting 的 rembg")
            output_image = _remove_bg(input_image, _session)

        img_array = np.array(output_image)
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    # 自适应检测印章颜色
    if seal_color == SealColor.AUTO:
        seal_color = _auto_detect_color(img_array)

    # 通用非印章像素检测
    is_background = _is_background_pixel(r, g, b)

    # 颜色特定的印章分类
    classifier = _CLASSIFIERS[seal_color]
    is_seal_color = classifier(r, g, b)

    # 移除非印章像素
    pixels_to_remove = is_background & (~is_seal_color)
    img_array[:, :, 3][pixels_to_remove] = 0

    # Alpha 修补：印章颜色像素中 alpha 过低的提升到最低值，防止半透明残缺
    _ALPHA_FLOOR = 200
    low_alpha_seal = is_seal_color & (img_array[:, :, 3] > 0) & (img_array[:, :, 3] < _ALPHA_FLOOR)
    img_array[:, :, 3][low_alpha_seal] = _ALPHA_FLOOR

    # 增强印章颜色
    valid_pixels = img_array[:, :, 3] > 0
    for i in range(3):
        channel = img_array[:, :, i]
        channel[valid_pixels] = np.clip(channel[valid_pixels] * _COLOR_ENHANCE_FACTOR, 0, 255)

    # 裁剪为印章最小矩形区域 + padding
    result_image = Image.fromarray(img_array)
    bbox = result_image.getbbox()
    if bbox is not None:
        left = max(bbox[0] - _CROP_PADDING, 0)
        upper = max(bbox[1] - _CROP_PADDING, 0)
        right = min(bbox[2] + _CROP_PADDING, result_image.width)
        lower = min(bbox[3] + _CROP_PADDING, result_image.height)
        result_image = result_image.crop((left, upper, right, lower))

    # 编码为 PNG 字节
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()
