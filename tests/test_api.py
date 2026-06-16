import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client():
    """创建 TestClient，整个模块共享，避免重复加载模型"""
    from app import app
    with TestClient(app) as c:
        yield c


def _make_test_image(width=100, height=100, color="red"):
    """创建带彩色矩形的标准测试图片"""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    pixels = img.load()
    cx, cy = width // 2, height // 2
    for x in range(cx - 10, cx + 10):
        for y in range(cy - 10, cy + 10):
            if color == "red":
                pixels[x, y] = (200, 30, 30)
            elif color == "blue":
                pixels[x, y] = (30, 30, 200)
            elif color == "black":
                pixels[x, y] = (20, 20, 20)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestHealthEndpoint:
    """健康检查端点测试"""

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestExtractEndpointValidation:
    """请求参数校验测试"""

    def test_invalid_seal_color_returns_400(self, client):
        img_buf = _make_test_image()
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.png", img_buf, "image/png")},
            data={"seal_color": "green"},
        )
        assert resp.status_code == 400

    def test_invalid_type_returns_400(self, client):
        img_buf = _make_test_image()
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.png", img_buf, "image/png")},
            data={"type": "2"},
        )
        assert resp.status_code == 400

    def test_empty_file_returns_400(self, client):
        empty_buf = io.BytesIO(b"")
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("empty.png", empty_buf, "image/png")},
        )
        assert resp.status_code == 400

    def test_non_image_file_returns_400(self, client):
        text_buf = io.BytesIO(b"this is not an image")
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.txt", text_buf, "text/plain")},
        )
        assert resp.status_code == 400

    def test_resolution_exceeds_limit_returns_400(self, client):
        """构造超过 4000x4000 分辨率的图片（1x4001 以控制文件大小）"""
        img = Image.new("RGB", (1, 4001), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("large.png", buf, "image/png")},
        )
        assert resp.status_code == 400
        assert "分辨率" in resp.json()["detail"]


class TestExtractEndpointProcessing:
    """印章提取处理测试"""

    def test_extract_red_seal_binary_mode(self, client):
        img_buf = _make_test_image(color="red")
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.png", img_buf, "image/png")},
            data={"seal_color": "red", "type": "0"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        # 验证返回的是有效 PNG
        result_img = Image.open(io.BytesIO(resp.content))
        assert result_img.mode == "RGBA"

    def test_extract_red_seal_json_mode(self, client):
        img_buf = _make_test_image(color="red")
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.png", img_buf, "image/png")},
            data={"seal_color": "red", "type": "1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "image" in body
        # 验证 Base64 可解码为有效 PNG
        img_bytes = base64.b64decode(body["image"])
        result_img = Image.open(io.BytesIO(img_bytes))
        assert result_img.mode == "RGBA"

    def test_extract_blue_seal(self, client):
        img_buf = _make_test_image(color="blue")
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.png", img_buf, "image/png")},
            data={"seal_color": "blue"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_extract_black_seal(self, client):
        img_buf = _make_test_image(color="black")
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.png", img_buf, "image/png")},
            data={"seal_color": "black"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_default_params(self, client):
        """不传 type 和 seal_color，使用默认值（type=0, seal_color=auto）"""
        img_buf = _make_test_image(color="red")
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.png", img_buf, "image/png")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_auto_detect_red(self, client):
        """自动检测红色印章"""
        img_buf = _make_test_image(color="red")
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.png", img_buf, "image/png")},
            data={"seal_color": "auto"},
        )
        assert resp.status_code == 200
        result_img = Image.open(io.BytesIO(resp.content))
        assert result_img.mode == "RGBA"

    def test_auto_detect_blue(self, client):
        """自动检测蓝色印章"""
        img_buf = _make_test_image(color="blue")
        resp = client.post(
            "/api/v1/seal/extract",
            files={"file": ("test.png", img_buf, "image/png")},
            data={"seal_color": "auto"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
