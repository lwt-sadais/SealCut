from enum import Enum
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image
from rembg import remove, new_session
from rembg.sessions.base import BaseSession


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
        _session = new_session("u2net")


def _classify_seal_pixels_red(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """红色印章分类：R 通道主导"""
    return (r > 150) & (r > g * 1.5) & (r > b * 1.5)


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
_WHITE_THRESHOLD = 220
_GRAY_DIFF_THRESHOLD = 15
_COLOR_DIFF_THRESHOLD = 20


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
    input_image = Image.open(BytesIO(image_bytes))

    # 使用共享 session 进行背景移除
    output_image = remove(
        input_image,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
        session=_session,
    )

    # 转为 numpy 数组进行颜色过滤
    img_array = np.array(output_image)
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    # 自适应检测印章颜色
    if seal_color == SealColor.AUTO:
        seal_color = _auto_detect_color(img_array)

    # 通用非印章像素检测（转 int16 避免 uint8 减法溢出）
    white_pixels = (r > _WHITE_THRESHOLD) & (g > _WHITE_THRESHOLD) & (b > _WHITE_THRESHOLD)
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    gray_pixels = (np.abs(r16 - g16) < _GRAY_DIFF_THRESHOLD) & \
                  (np.abs(g16 - b16) < _GRAY_DIFF_THRESHOLD) & \
                  (np.abs(r16 - b16) < _GRAY_DIFF_THRESHOLD)
    color_diff = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    similar_colors = color_diff < _COLOR_DIFF_THRESHOLD

    # 颜色特定的印章分类
    classifier = _CLASSIFIERS[seal_color]
    is_seal_color = classifier(r, g, b)

    # 移除非印章像素
    pixels_to_remove = (white_pixels | gray_pixels | similar_colors) & (~is_seal_color)
    img_array[:, :, 3][pixels_to_remove] = 0

    # 增强印章颜色
    valid_pixels = img_array[:, :, 3] > 0
    for i in range(3):
        channel = img_array[:, :, i]
        channel[valid_pixels] = np.clip(channel[valid_pixels] * _COLOR_ENHANCE_FACTOR, 0, 255)

    # 编码为 PNG 字节
    result_image = Image.fromarray(img_array)
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()
