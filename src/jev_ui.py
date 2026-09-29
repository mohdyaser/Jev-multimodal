"""Local browser UI for Gemma 4 typed decisions.

Run from this repository with `.venv/bin/python src/jev_ui.py --open`.
The server binds only to 127.0.0.1 and loads model weights on first prediction.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from jev_core import load_model, predict, validate_request

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "ui" / "index.html"
EXPERIMENT = ROOT / "config" / "experiment.json"
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_BODY_BYTES = 72 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class RequestError(ValueError):
    """A request the browser can correct without restarting the server."""


class ModelRegistry:
    """Load each local system once and serialize MLX model access."""

    def __init__(self, config_path: Path = EXPERIMENT):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.base_path = ROOT / self.config["base_model"]["path"]
        self.lock = threading.Lock()
        self.loaded: dict[str, tuple[Any, Any]] = {}

    def public_config(self) -> dict[str, Any]:
        systems = self.config["systems"]
        return {
            "default_model": self.config["default_system"],
            "models": {
                "frozen": {"available": self.base_path.is_dir()},
                "experimental_adapter": {
                    "available": (
                        self.base_path.is_dir()
                        and (
                            ROOT
                            / systems["experimental_adapter"]["adapter"]
                            / "adapters.safetensors"
                        ).is_file()
                    )
                },
            },
            "max_images": MAX_IMAGES,
            "max_options": 16,
        }

    def run(self, choice: str, request: dict[str, Any]) -> dict[str, Any]:
        if choice not in {"frozen", "experimental_adapter"}:
            raise RequestError("Choose frozen or experimental adapter")
        system = self.config["systems"][choice]
        adapter = system["adapter"]
        adapter_path = ROOT / adapter if adapter else None
        if adapter_path is not None:
            weights = adapter_path / "adapters.safetensors"
            if not weights.is_file():
                raise RequestError(f"Adapter weights are missing: {weights}")
            expected = system.get("adapter_sha256")
            if (
                expected
                and hashlib.sha256(weights.read_bytes()).hexdigest() != expected
            ):
                raise RequestError(
                    "Adapter checksum differs from config/experiment.json"
                )
        if not self.base_path.is_dir():
            raise RequestError(f"Base model is missing: {self.base_path}")
        with self.lock:
            if choice not in self.loaded:
                self.loaded[choice] = load_model(
                    str(self.base_path), str(adapter_path) if adapter_path else None
                )
            model, processor = self.loaded[choice]
            result = predict(
                model, processor, request, temperature=float(system["temperature"])
            )
        return {
            **result,
            "model_choice": choice,
            "temperature": float(system["temperature"]),
        }


def _decode_images(images: Any, directory: Path) -> list[str]:
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise RequestError("Add no more than four ordered images")
    paths: list[str] = []
    for index, item in enumerate(images, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("data_url"), str):
            raise RequestError(f"image_{index} must be an uploaded image")
        data_url = item["data_url"]
        if not data_url.startswith("data:image/") or ";base64," not in data_url:
            raise RequestError(f"image_{index} is not a base64 image upload")
        encoded = data_url.split(";base64,", 1)[1]
        if len(encoded) > (MAX_IMAGE_BYTES + 2) * 4 // 3 + 8:
            raise RequestError(f"image_{index} exceeds the 12 MB upload limit")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise RequestError(f"image_{index} has invalid image data") from error
        if len(raw) > MAX_IMAGE_BYTES:
            raise RequestError(f"image_{index} exceeds the 12 MB upload limit")
        try:
            with Image.open(io.BytesIO(raw)) as original:
                if original.format not in {"PNG", "JPEG", "WEBP"}:
                    raise RequestError(f"image_{index} must be PNG, JPEG, or WebP")
                if original.width * original.height > MAX_IMAGE_PIXELS:
                    raise RequestError(f"image_{index} exceeds the 20 megapixel limit")
                oriented = ImageOps.exif_transpose(original)
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                background.alpha_composite(rgba)
                path = directory / f"image_{index}.png"
                background.convert("RGB").save(path, format="PNG")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise RequestError(f"image_{index} could not be read") from error
        paths.append(str(path))
    return paths


def run_payload(payload: Any, registry: ModelRegistry) -> dict[str, Any]:
    """Decode browser uploads, validate the decision, and run one prediction."""
    if not isinstance(payload, dict):
        raise RequestError("Request must be a JSON object")
    with tempfile.TemporaryDirectory(prefix="jev-ui-") as name:
        images = _decode_images(payload.get("images", []), Path(name))
        request = {
            "question": payload.get("question"),
            "context": payload.get("context", ""),
            "images": images,
            "output_type": payload.get("output_type"),
            "options": payload.get("options"),
        }
        if request["output_type"] == "score":
            request["level_values"] = payload.get("level_values")
            if request["level_values"] is None:
                raise RequestError("Score levels need numeric values")
        try:
            request = validate_request(request)
        except (TypeError, ValueError) as error:
            raise RequestError(str(error)) from error
        try:
            return registry.run(payload.get("model_choice", "frozen"), request)
        except (TypeError, ValueError) as error:
            raise RequestError(str(error)) from error


def make_handler(registry: ModelRegistry) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/config":
                self._json(HTTPStatus.OK, registry.public_config())
            elif self.path == "/healthz":
                self._json(HTTPStatus.OK, {"ok": True})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/api/predict":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json(
                    HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length required"}
                )
                return
            if length < 1 or length > MAX_BODY_BYTES:
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "Request is too large"},
                )
                return
            try:
                payload = json.loads(self.rfile.read(length))
                result = run_payload(payload, registry)
            except json.JSONDecodeError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
            except RequestError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # noqa: BLE001 - keep local UI responsive
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            else:
                self._json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            # Requests may contain private prompts/images; do not log them.
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--open", action="store_true", help="open the local UI in a browser"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    registry = ModelRegistry()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(registry))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Jev decision UI: {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
