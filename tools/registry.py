"""工具注册表（V1：Function Calling 工具化）。

把两个确定性模型服务 + 业务数据查询包装成 OpenAI Function Calling 规范的工具定义。
LLM 只能通过这里暴露的能力获取"判定结论"，从而保证：
1. 结论有出处（每次调用都记 ToolCallRecord，可回归测试参数稳定性）；
2. 参数可校验（pydantic 校验 + 白名单过滤，阻断提示注入造成的越权参数）。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import config
from schemas import CommentAnalyzeResult, ImageClassifyResult, Product, ToolCallRecord

# ---------------------------------------------------------------- 工具定义
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "classify_image",
            "description": "对一张服装图片做品类/款式分类（调用 ResNet50 微调服务，返回 Top-K 标签与置信度）。只能用它拿分类结论，不要自己猜品类。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "本地图片绝对路径"},
                    "image_base64": {"type": "string", "description": "Base64 编码图片（与 image_path 二选一）"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_comments",
            "description": "对网店评论做多标签情感与主题打分（尺码/颜色/质量/物流/服务等）。返回 sentiment、sentiment_conf、topics。",
            "parameters": {
                "type": "object",
                "properties": {
                    "comments": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 7, "default": 3},
                },
                "required": ["comments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_comments",
            "description": "从评论库按商品/品类/情感筛选评论，用于评论洞察与生成证据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "description": "all | 品类名(如 连衣裙) | 商品ID"},
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
                    "balanced": {"type": "boolean", "default": True,
                                 "description": "是否按品类均衡采样（避免后上架品类被截断漏采）"},
                },
                "required": ["scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_products",
            "description": "查询商品表（SKU、品类、款式、颜色、价格、尺码、库存）。用于选品/备货建议。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_tags",
            "description": "把模型给出的原始标签归一到门店标准标签库，返回标准标签与匹配置信度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["raw_tags"],
            },
        },
    },
]

_ALLOWED_KEYS: dict[str, set[str]] = {
    "classify_image": {"image_path", "image_base64", "top_k"},
    "analyze_comments": {"comments", "top_n"},
    "query_comments": {"scope", "sentiment", "limit", "balanced"},
    "query_products": {"category", "limit"},
    "normalize_tags": {"raw_tags"},
}


def sanitize_args(name: str, args: dict) -> dict:
    """白名单过滤：LLM 可能被提示注入诱导传入额外参数（如 prompt、system），此处一律丢弃。"""
    allowed = _ALLOWED_KEYS.get(name, set())
    return {k: v for k, v in (args or {}).items() if k in allowed}


# ---------------------------------------------------------------- HTTP 客户端
def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=config.SERVICE_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def tool_classify_image(args: dict) -> dict:
    args = sanitize_args("classify_image", args)
    if args.get("image_base64"):
        payload = {"image_base64": args["image_base64"], "top_k": args.get("top_k", 3)}
    else:
        path = args.get("image_path", "")
        if not Path(path).exists():
            return ImageClassifyResult(ok=False, error={"code": "IMAGE_NOT_FOUND",
                                                       "message": f"图片不存在: {path}",
                                                       "retryable": False}).model_dump()
        payload = {"image_path": str(path), "top_k": args.get("top_k", 3)}
    out = ImageClassifyResult(**_post_json(f"{config.IMAGE_SERVICE_URL}/predict", payload)).model_dump()
    if out.get("ok"):
        # 回传实际处理的路径，供编排层做逐张幂等，避免同一张图被重复推理
        out["processed_path"] = str(path)
    return out


def tool_analyze_comments(args: dict) -> dict:
    args = sanitize_args("analyze_comments", args)
    comments = args.get("comments") or []
    if not comments:
        return CommentAnalyzeResult(ok=False, error={"code": "EMPTY_COMMENTS",
                                                    "message": "评论列表为空"}).model_dump()
    raw = _post_json(f"{config.COMMENT_SERVICE_URL}/analyze",
                     {"comments": comments, "top_n": args.get("top_n", 3)})
    results = [CommentAnalyzeResult(**r) for r in raw.get("results", [])]
    return {"ok": all(r.ok for r in results),
            "results": [r.model_dump() for r in results],
            "batch_size": raw.get("batch_size", 0),
            "model_version": raw.get("model_version", ""),
            "total_latency_ms": round(sum(r.latency_ms for r in results), 2)}


def tool_query_comments(args: dict) -> dict:
    import store
    args = sanitize_args("query_comments", args)
    rows = store.query_comments(scope=args.get("scope", "all"),
                               sentiment=args.get("sentiment"),
                               limit=int(args.get("limit", 12)),
                               balanced=args.get("balanced", True))
    total = len(store.all_comments()) if args.get("scope", "all") in ("all", "") else None
    return {"ok": True, "count": len(rows), "comments": rows,
            "coverage_note": f"已均衡采样 {len(rows)} 条（库内共 {total} 条）"
            if total else f"已采样 {len(rows)} 条"}


def tool_query_products(args: dict) -> dict:
    import store
    args = sanitize_args("query_products", args)
    rows = store.query_products(category=args.get("category"), limit=int(args.get("limit", 20)))
    return {"ok": True, "count": len(rows), "products": [p.model_dump() for p in rows]}


def tool_normalize_tags(args: dict) -> dict:
    import store
    args = sanitize_args("normalize_tags", args)
    raws = args.get("raw_tags") or []
    return {"ok": True,
            "normalized": [store.normalize_tag(r).model_dump() for r in raws]}


REGISTRY: dict[str, Callable[[dict], dict]] = {
    "classify_image": tool_classify_image,
    "analyze_comments": tool_analyze_comments,
    "query_comments": tool_query_comments,
    "query_products": tool_query_products,
    "normalize_tags": tool_normalize_tags,
}


def invoke(name: str, args: dict) -> tuple[dict, ToolCallRecord]:
    """执行工具调用并记录。异常统一转成结构化错误，不向上抛裸异常。"""
    t0 = time.perf_counter()
    call_id = f"{name}#{int(t0 * 1000) % 100000}"
    try:
        fn = REGISTRY.get(name)
        if not fn:
            raise KeyError(f"unknown tool: {name}")
        out = fn(args)
        ok = bool(out.get("ok", True))
        summary = _summarize(name, out)
    except urllib.error.URLError as e:
        ok, summary = False, f"服务不可达: {e.reason}"
        out = {"ok": False, "error": {"code": "SERVICE_UNREACHABLE",
                                      "message": str(e.reason), "retryable": True}}
    except Exception as e:
        ok, summary = False, f"{type(e).__name__}: {e}"
        out = {"ok": False, "error": {"code": "TOOL_EXCEPTION",
                                      "message": str(e), "retryable": False}}
    cost = round((time.perf_counter() - t0) * 1000, 2)
    return out, ToolCallRecord(call_id=call_id, name=name, arguments=sanitize_args(name, args),
                               ok=ok, cost_ms=cost, result_summary=summary)


def _summarize(name: str, out: dict) -> str:
    if not out.get("ok", True):
        return (out.get("error") or {}).get("message", "failed")[:80]
    if name == "classify_image":
        ps = out.get("predictions", [])
        return f"top1={ps[0]['label']}({ps[0]['score']:.2f})" if ps else "no prediction"
    if name == "analyze_comments":
        return f"batch={out.get('batch_size', 0)}"
    if name in ("query_comments", "query_products"):
        return f"rows={out.get('count', 0)}"
    if name == "normalize_tags":
        return f"tags={len(out.get('normalized', []))}"
    return ""


__all__ = ["TOOL_SPECS", "REGISTRY", "invoke", "sanitize_args"]