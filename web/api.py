"""Web API 服务：把多 Agent 编排层暴露给前端。

标准库 http.server 实现，无额外依赖。接口刻意保持薄：
- 只做编排层调用与结果整形，不在 Web 层重复业务逻辑；
- 错误统一返回 {"ok": false, "error": {...}}，前端不解析异常堆栈；
- 每个响应都带 llm_backend，让前端如实展示当前编排后端（是真实大模型还是离线降级）。
"""
from __future__ import annotations

import json
import mimetypes
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config

WEB_DIR = Path(__file__).resolve().parent
MAX_UPLOAD = 8 * 1024 * 1024  # 8MB

# Windows 的 mimetypes 数据库不认 woff/woff2，会回落成 application/octet-stream，
# 浏览器会因 MIME 不符拒绝加载图标字体（页面表现为图标全变方块 tofu）。
# 这里显式补齐字体与样式表类型。
_EXTRA_TYPES = {
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".eot": "application/vnd.ms-fontobject",
    ".svg": "image/svg+xml",
    ".map": "application/json",
    ".webmanifest": "application/manifest+json",
    ".wasm": "application/wasm",
}


def _content_type(path: Path) -> str:
    """推断 Content-Type，字体类型显式优先，避免图标字体加载失败。"""
    ext = path.suffix.lower()
    if ext in _EXTRA_TYPES:
        ctype = _EXTRA_TYPES[ext]
    else:
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
        ctype += "; charset=utf-8"
    return ctype


class Handler(BaseHTTPRequestHandler):
    # ---------------------------------------------------------------- 工具
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, "not found")
            return
        ctype = _content_type(path)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    # ---------------------------------------------------------------- GET
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/api/health":
            return self._json(200, self._health())
        if path == "/api/dashboard":
            return self._json(200, self._dashboard())
        if path == "/api/products":
            import store
            rows = store.query_products(limit=int(qs.get("limit", ["50"])[0]))
            return self._json(200, {"ok": True, "products": [r.model_dump() for r in rows]})
        if path == "/api/comments":
            import store
            rows = store.query_comments(scope=qs.get("scope", ["all"])[0],
                                       limit=int(qs.get("limit", ["50"])[0]))
            return self._json(200, {"ok": True, "count": len(rows), "comments": rows})
        if path == "/api/tags":
            import store
            lib = store.load_tag_library()
            return self._json(200, {"ok": True, "library": lib})
        if path == "/api/history":
            return self._json(200, {"ok": True, "tasks": _history()})
        if path.startswith("/api/task/"):
            tid = path.rsplit("/", 1)[-1]
            p = config.VAR_DIR / f"trace_{tid}.json"
            if not p.exists():
                return self._json(404, {"ok": False, "error": {"code": "NOT_FOUND",
                                                              "message": f"任务不存在: {tid}"}})
            return self._json(200, json.loads(p.read_text(encoding="utf-8")))
        if path == "/api/review":
            import store
            return self._json(200, {"ok": True, "samples": store.load_human_samples()})

        # 静态资源
        rel = path.lstrip("/") or "index.html"
        target = (WEB_DIR / rel).resolve()
        if not str(target).startswith(str(WEB_DIR.resolve())):
            return self.send_error(403, "forbidden")
        return self._file(target)

    # ---------------------------------------------------------------- POST
    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"

        try:
            if path == "/api/chat":
                return self._json(200, self._chat(raw))
            if path == "/api/classify":
                return self._json(200, self._classify(raw))
            if path == "/api/eval":
                return self._json(200, self._eval())
            if path == "/api/upload":
                return self._json(200, self._upload(raw))
            if path == "/api/resolve":
                return self._json(200, self._resolve(raw))
        except Exception as e:
            return self._json(500, {"ok": False, "error": {
                "code": type(e).__name__, "message": str(e)}})
        return self._json(404, {"ok": False, "error": {"code": "NOT_FOUND", "message": path}})

    # ---------------------------------------------------------------- 业务
    def _health(self) -> dict:
        import urllib.request
        out = {"ok": True, "services": {}, "llm_backend": _llm_name()}
        for name, url in (("image", config.IMAGE_SERVICE_URL),
                          ("comment", config.COMMENT_SERVICE_URL)):
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                    data = json.loads(r.read().decode("utf-8"))
                out["services"][name] = {"up": r.status == 200, **data}
            except Exception as e:
                out["services"][name] = {"up": False, "error": str(e)[:80]}
        # 后端可信度注册表：让前端如实展示"哪些后端被采信"
        trust_path = config.VAR_DIR / "backend_trust.json"
        if trust_path.exists():
            try:
                out["backend_trust"] = json.loads(trust_path.read_text(encoding="utf-8"))
            except Exception:
                out["backend_trust"] = {}
        else:
            out["backend_trust"] = {}
        return out

    def _dashboard(self) -> dict:
        import store
        from llm.client import get_llm
        products = store.query_products(limit=200)
        comments = store.all_comments()
        by_cat: dict[str, dict[str, int]] = {}
        for c in comments:
            b = by_cat.setdefault(c.get("category", "未分类"),
                                  {"positive": 0, "negative": 0, "neutral": 0})
            b[c.get("gold_sentiment", "neutral")] = b.get(c.get("gold_sentiment", "neutral"), 0) + 1
        low_stock = [p.model_dump() for p in products if p.stock <= 12]
        return {"ok": True, "llm_backend": get_llm().name,
                "stats": {"products": len(products), "comments": len(comments),
                          "categories": len(by_cat), "low_stock": len(low_stock)},
                "sentiment_by_category": by_cat,
                "low_stock": low_stock[:10],
                "tasks": _history()[:10]}

    def _chat(self, raw: bytes) -> dict:
        body = json.loads(raw or b"{}")
        query = (body.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": {"code": "EMPTY_QUERY", "message": "请输入需求"}}
        from orchestration import MultiAgentWorkflow
        wf = MultiAgentWorkflow()
        paths = body.get("image_paths") or []
        t0 = time.perf_counter()
        st = wf.run(query, image_paths=paths,
                    comment_scope=body.get("scope") or "all")
        cost = round((time.perf_counter() - t0) * 1000, 2)
        payload = st.model_dump(mode="json")
        (config.VAR_DIR / f"trace_{st.task_id}.json").write_text(
            st.model_dump_json(indent=2), encoding="utf-8")
        payload["_meta"] = {"cost_ms": cost, "llm_backend": wf.llm.name}
        return payload

    def _classify(self, raw: bytes) -> dict:
        body = json.loads(raw or b"{}")
        paths = body.get("image_paths") or []
        if not paths:
            import run as runner
            paths = runner._demo_images()
        from orchestration import single_agent_toolcall
        return single_agent_toolcall(body.get("query") or "把这些服装图片归类",
                                     {"image_paths": paths})

    def _eval(self) -> dict:
        from eval.regression import run_eval
        res = run_eval()
        (config.VAR_DIR / "eval_report.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return res

    def _upload(self, raw: bytes) -> dict:
        """接收前端上传的图片（body 即图片二进制，文件名走 X-Filename 头）。"""
        if len(raw) > MAX_UPLOAD:
            return {"ok": False, "error": {"code": "TOO_LARGE",
                                            "message": f"超过 {MAX_UPLOAD // 1024 // 1024}MB"}}
        header_name = self.headers.get("X-Filename", "")
        safe = Path(urllib.parse.unquote(header_name)).name or "upload.png"
        d = config.BASE_DIR / "data" / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        target = d / f"{int(time.time() * 1000)}_{safe}"
        target.write_bytes(raw)
        return {"ok": True, "path": str(target), "name": safe,
                "size": len(raw)}

    def _resolve(self, raw: bytes) -> dict:
        """人工确认回流：前端点确认/修正后写回样本池。"""
        body = json.loads(raw or b"{}")
        kind = body.get("kind") or "tag"
        decision = body.get("decision") or "accept"
        corrected = body.get("corrected_value")
        if not corrected:
            return {"ok": False, "error": {"code": "NO_VALUE", "message": "缺少 corrected_value"}}
        import store
        store.append_human_sample({
            "kind": kind, "task_id": body.get("task_id", ""),
            "original": body.get("original", {}), "decision": decision,
            "corrected_value": corrected,
            "reason": body.get("reason", "前端人工确认"), "by": body.get("reviewer", "owner-web"),
        })
        return {"ok": True, "samples": len(store.load_human_samples())}


def _llm_name() -> str:
    try:
        from llm.client import get_llm
        return get_llm().name
    except Exception:
        return "unknown"


def _history() -> list[dict]:
    out = []
    for p in sorted(config.VAR_DIR.glob("trace_*.json"), key=lambda x: x.stat().st_mtime,
                    reverse=True)[:20]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({"task_id": d.get("task_id"), "intent": d.get("intent"),
                        "status": d.get("status"), "query": d.get("user_query"),
                        "advice_count": len(d.get("advice") or []),
                        "theme_count": len(d.get("themes") or []),
                        "tool_calls": sum(1 for e in d.get("trace", []) if e["step"] == "tool_call"),
                        "mtime": p.stat().st_mtime})
        except Exception:
            continue
    return out


def serve(port: int = 8080, host: str = "127.0.0.1") -> None:
    print(f"[web] http://{host}:{port}  (Ctrl+C 停止)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="服装店多Agent Web 服务")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    serve(a.port, a.host)