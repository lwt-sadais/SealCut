import numpy as np
import pytest

from seal_processor import (
    _classify_seal_pixels_red,
    _classify_seal_pixels_blue,
    _classify_seal_pixels_black,
    _auto_detect_color,
    _is_background_pixel,
    SealColor,
)


class TestRedClassifier:
    """红色印章分类器测试"""

    def test_pure_red_is_seal(self):
        """纯红色像素应被识别为印章"""
        r = np.array([[200]], dtype=np.uint8)
        g = np.array([[50]], dtype=np.uint8)
        b = np.array([[50]], dtype=np.uint8)
        assert _classify_seal_pixels_red(r, g, b)[0, 0] is np.True_

    def test_white_not_seal(self):
        """白色像素不应被识别为红色印章"""
        r = np.array([[230]], dtype=np.uint8)
        g = np.array([[230]], dtype=np.uint8)
        b = np.array([[230]], dtype=np.uint8)
        assert _classify_seal_pixels_red(r, g, b)[0, 0] is np.False_

    def test_green_not_seal(self):
        """绿色像素不应被识别为红色印章"""
        r = np.array([[50]], dtype=np.uint8)
        g = np.array([[200]], dtype=np.uint8)
        b = np.array([[50]], dtype=np.uint8)
        assert _classify_seal_pixels_red(r, g, b)[0, 0] is np.False_

    def test_low_red_not_seal(self):
        """R 通道值低于阈值（120）不应被识别为红色印章"""
        r = np.array([[110]], dtype=np.uint8)
        g = np.array([[50]], dtype=np.uint8)
        b = np.array([[50]], dtype=np.uint8)
        assert _classify_seal_pixels_red(r, g, b)[0, 0] is np.False_

    def test_red_not_dominant_not_seal(self):
        """R 通道不主导（R <= G*1.3）不应被识别为红色印章"""
        r = np.array([[200]], dtype=np.uint8)
        g = np.array([[170]], dtype=np.uint8)
        b = np.array([[50]], dtype=np.uint8)
        assert _classify_seal_pixels_red(r, g, b)[0, 0] is np.False_

    def test_reddish_edge_mixed_is_seal(self):
        """红色印章边缘与背景混合的浅红像素（R>G*1.3 且 R>B*1.3）应被识别为红色印章"""
        r = np.array([[230]], dtype=np.uint8)
        g = np.array([[160]], dtype=np.uint8)
        b = np.array([[170]], dtype=np.uint8)
        assert _classify_seal_pixels_red(r, g, b)[0, 0] is np.True_


class TestBlueClassifier:
    """蓝色印章分类器测试"""

    def test_pure_blue_is_seal(self):
        """纯蓝色像素应被识别为印章"""
        r = np.array([[50]], dtype=np.uint8)
        g = np.array([[50]], dtype=np.uint8)
        b = np.array([[200]], dtype=np.uint8)
        assert _classify_seal_pixels_blue(r, g, b)[0, 0] is np.True_

    def test_red_not_seal(self):
        """红色像素不应被识别为蓝色印章"""
        r = np.array([[200]], dtype=np.uint8)
        g = np.array([[50]], dtype=np.uint8)
        b = np.array([[50]], dtype=np.uint8)
        assert _classify_seal_pixels_blue(r, g, b)[0, 0] is np.False_

    def test_low_blue_not_seal(self):
        """B 通道值低于阈值（150）不应被识别为蓝色印章"""
        r = np.array([[50]], dtype=np.uint8)
        g = np.array([[50]], dtype=np.uint8)
        b = np.array([[140]], dtype=np.uint8)
        assert _classify_seal_pixels_blue(r, g, b)[0, 0] is np.False_

    def test_blue_not_dominant_not_seal(self):
        """B 通道不主导（B <= R*1.5）不应被识别为蓝色印章"""
        r = np.array([[150]], dtype=np.uint8)
        g = np.array([[50]], dtype=np.uint8)
        b = np.array([[200]], dtype=np.uint8)
        assert _classify_seal_pixels_blue(r, g, b)[0, 0] is np.False_


class TestBlackClassifier:
    """黑色印章分类器测试"""

    def test_dark_pixel_is_seal(self):
        """暗色像素应被识别为黑色印章"""
        r = np.array([[30]], dtype=np.uint8)
        g = np.array([[30]], dtype=np.uint8)
        b = np.array([[30]], dtype=np.uint8)
        assert _classify_seal_pixels_black(r, g, b)[0, 0] is np.True_

    def test_bright_pixel_not_seal(self):
        """亮色像素不应被识别为黑色印章"""
        r = np.array([[180]], dtype=np.uint8)
        g = np.array([[180]], dtype=np.uint8)
        b = np.array([[180]], dtype=np.uint8)
        assert _classify_seal_pixels_black(r, g, b)[0, 0] is np.False_

    def test_white_not_seal(self):
        """白色像素不应被识别为黑色印章"""
        r = np.array([[210]], dtype=np.uint8)
        g = np.array([[210]], dtype=np.uint8)
        b = np.array([[210]], dtype=np.uint8)
        assert _classify_seal_pixels_black(r, g, b)[0, 0] is np.False_

    def test_boundary_brightness(self):
        """亮度恰好为 100 的像素不应被识别为黑色印章（阈值 < 100）"""
        r = np.array([[100]], dtype=np.uint8)
        g = np.array([[100]], dtype=np.uint8)
        b = np.array([[100]], dtype=np.uint8)
        assert _classify_seal_pixels_black(r, g, b)[0, 0] is np.False_

    def test_just_below_boundary(self):
        """亮度略低于 100 的像素应被识别为黑色印章"""
        r = np.array([[99]], dtype=np.uint8)
        g = np.array([[99]], dtype=np.uint8)
        b = np.array([[99]], dtype=np.uint8)
        assert _classify_seal_pixels_black(r, g, b)[0, 0] is np.True_


class TestSealColorEnum:
    """SealColor 枚举测试"""

    def test_valid_values(self):
        assert SealColor.RED.value == "red"
        assert SealColor.BLUE.value == "blue"
        assert SealColor.BLACK.value == "black"
        assert SealColor.AUTO.value == "auto"

    def test_from_string(self):
        assert SealColor("red") == SealColor.RED
        assert SealColor("blue") == SealColor.BLUE
        assert SealColor("black") == SealColor.BLACK
        assert SealColor("auto") == SealColor.AUTO

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            SealColor("green")


class TestAutoDetectColor:
    """自适应颜色检测测试"""

    def _make_rgba(self, r, g, b, a=255):
        """构造 1x1 RGBA 数组"""
        return np.array([[[r, g, b, a]]], dtype=np.uint8)

    def test_red_dominant_detected_as_red(self):
        """R 通道主导应检测为红色"""
        img = self._make_rgba(200, 60, 60)
        assert _auto_detect_color(img) == SealColor.RED

    def test_blue_dominant_detected_as_blue(self):
        """B 通道主导应检测为蓝色"""
        img = self._make_rgba(60, 60, 200)
        assert _auto_detect_color(img) == SealColor.BLUE

    def test_dark_neutral_detected_as_black(self):
        """暗色无主导通道应检测为黑色"""
        img = self._make_rgba(30, 30, 30)
        assert _auto_detect_color(img) == SealColor.BLACK

    def test_all_transparent_detected_as_black(self):
        """全透明像素应回退为黑色"""
        img = self._make_rgba(200, 60, 60, a=0)
        assert _auto_detect_color(img) == SealColor.BLACK

    def test_mixed_pixels_red_dominant(self):
        """混合像素中红色占多数应检测为红色"""
        # 3 个红色像素 + 1 个白色像素
        img = np.array([
            [[200, 50, 50, 255], [200, 50, 50, 255], [200, 50, 50, 255], [230, 230, 230, 255]]
        ], dtype=np.uint8)
        assert _auto_detect_color(img) == SealColor.RED


class TestBackgroundPixelDetection:
    """背景像素检测测试（白色、灰色、低饱和度、高亮度）"""

    def test_pure_white_is_background(self):
        """纯白色像素应被检测为背景"""
        r = np.array([[230]], dtype=np.uint8)
        g = np.array([[230]], dtype=np.uint8)
        b = np.array([[230]], dtype=np.uint8)
        assert _is_background_pixel(r, g, b)[0, 0] is np.True_

    def test_light_blue_white_is_background(self):
        """偏蓝浅色像素（文档扫描常见背景）应被检测为背景"""
        r = np.array([[210]], dtype=np.uint8)
        g = np.array([[220]], dtype=np.uint8)
        b = np.array([[240]], dtype=np.uint8)
        assert _is_background_pixel(r, g, b)[0, 0] is np.True_

    def test_high_brightness_non_seal_is_background(self):
        """高亮度非印章像素应被检测为背景"""
        r = np.array([[218]], dtype=np.uint8)
        g = np.array([[239]], dtype=np.uint8)
        b = np.array([[255]], dtype=np.uint8)
        assert _is_background_pixel(r, g, b)[0, 0] is np.True_

    def test_gray_is_background(self):
        """高亮度灰色像素应被检测为背景"""
        r = np.array([[215]], dtype=np.uint8)
        g = np.array([[220]], dtype=np.uint8)
        b = np.array([[218]], dtype=np.uint8)
        assert _is_background_pixel(r, g, b)[0, 0] is np.True_

    def test_low_saturation_is_background(self):
        """高亮度低饱和度像素应被检测为背景"""
        r = np.array([[210]], dtype=np.uint8)
        g = np.array([[220]], dtype=np.uint8)
        b = np.array([[225]], dtype=np.uint8)
        assert _is_background_pixel(r, g, b)[0, 0] is np.True_

    def test_red_seal_not_background(self):
        """红色印章像素不应被检测为背景"""
        r = np.array([[200]], dtype=np.uint8)
        g = np.array([[50]], dtype=np.uint8)
        b = np.array([[50]], dtype=np.uint8)
        assert _is_background_pixel(r, g, b)[0, 0] is np.False_

    def test_dark_pixel_not_background(self):
        """暗色像素不应被检测为背景"""
        r = np.array([[30]], dtype=np.uint8)
        g = np.array([[30]], dtype=np.uint8)
        b = np.array([[30]], dtype=np.uint8)
        assert _is_background_pixel(r, g, b)[0, 0] is np.False_
