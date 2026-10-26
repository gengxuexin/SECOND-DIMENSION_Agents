"""业务上下文层：商品表、标签库、评论库。

对应项目中的 "商品表、标签库与评论库作为业务上下文"。标签归一逻辑在这里实现，
供 normalize_tags 工具与图像归档 Agent 共用，保证同一套规则。
"""
from __future__ import annotations

import csv
import difflib
import json
import threading
from pathlib import Path
from typing import Any

import config
from schemas import NormalizedTag, Product

_LOCK = threading.Lock()
_TAG_CACHE: dict[str, Any] | None = None


# ---------------------------------------------------------------- 标签库
def load_tag_library() -> dict[str, Any]:
    global _TAG_CACHE
    with _LOCK:
        if _TAG_CACHE is None:
            if config.TAG_LIBRARY.exists():
                _TAG_CACHE = json.loads(config.TAG_LIBRARY.read_text(encoding="utf-8"))
            else:
                _TAG_CACHE = {"canonical": {}, "aliases": {}}
        return _TAG_CACHE


def normalize_tag(raw: str) -> NormalizedTag:
    """把原始标签归一到标准标签库。

    规则优先级：精确匹配 > 别名 > 模糊匹配(difflib) > 原样回落。
    置信度低于 TAG_NORM_MIN_CONF 时标记 need_confirm，交人工确认。
    """
    lib = load_tag_library()
    canonical: dict[str, str] = lib.get("canonical", {})
    aliases: dict[str, str] = lib.get("aliases", {})
    key = (raw or "").strip()
    if not key:
        return NormalizedTag(raw=raw, normalized="", conf=0.0, source="fallback", need_confirm=True)

    if key in canonical:
        return NormalizedTag(raw=key, normalized=canonical[key], conf=1.0, source="tag_library")
    if key in aliases:
        return NormalizedTag(raw=key, normalized=aliases[key], conf=0.95, source="tag_library")

    # 模糊匹配：在标准标签集合上取最相似
    cand = difflib.get_close_matches(key, list(canonical.keys()), n=1, cutoff=0.72)
    if cand:
        best = cand[0]
        ratio = difflib.SequenceMatcher(None, key, best).ratio()
        if ratio >= config.TAG_NORM_MIN_CONF:
            return NormalizedTag(raw=key, normalized=best, conf=round(ratio, 4), source="fuzzy")
        return NormalizedTag(raw=key, normalized=key, conf=round(ratio, 4),
                             source="fallback", need_confirm=True)
    return NormalizedTag(raw=key, normalized=key, conf=0.0, source="fallback", need_confirm=True)


# ---------------------------------------------------------------- 商品表
def query_products(category: str | None = None, limit: int = 20) -> list[Product]:
    if not config.PRODUCT_TABLE.exists():
        return []
    out: list[Product] = []
    with config.PRODUCT_TABLE.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if category and category not in ("", "all"):
                if category not in (row.get("category", ""), row.get("style", "")):
                    continue
            try:
                out.append(Product(
                    sku_id=row.get("sku_id", ""), name=row.get("name", ""),
                    category=row.get("category", ""), style=row.get("style", ""),
                    color=row.get("color", ""), price=float(row.get("price", 0) or 0),
                    size_set=[s for s in (row.get("size_set", "") or "").split("|") if s],
                    stock=int(row.get("stock", 0) or 0), listed_at=row.get("listed_at", ""),
                ))
            except Exception:
                continue
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------- 评论库
def query_comments(scope: str = "all", sentiment: str | None = None,
                   limit: int = 12, balanced: bool = True) -> list[dict[str, Any]]:
    """查询评论。

    balanced=True 时按品类轮转取样，而不是简单截断头部。
    原因：评论库按上架时间排列，简单 limit 截断会让后上架的品类一条评论都取不到，
    下游据此输出"该品类 0 条反馈"的错误结论——这是检索层的典型召回偏差。
    """
    if not config.COMMENT_LIBRARY.exists():
        return []
    rows: list[dict[str, Any]] = []
    with config.COMMENT_LIBRARY.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if scope and scope not in ("all", ""):
                if scope not in (row.get("sku_id", ""), row.get("category", "")):
                    continue
            if sentiment and row.get("gold_sentiment") and row["gold_sentiment"] != sentiment:
                continue
            rows.append(row)

    if not balanced:
        return rows[:limit]

    # 按品类轮转，保证每个品类至少被采样到
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(r.get("category", "未分类"), []).append(r)
    order = sorted(buckets)
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < limit and any(buckets[c] for c in order):
        c = order[i % len(order)]
        if buckets[c]:
            out.append(buckets[c].pop(0))
        i += 1
    return out


def all_comments(limit: int = 100000) -> list[dict[str, Any]]:
    return query_comments(scope="all", limit=limit)


# ---------------------------------------------------------------- 人工修正回流
def append_human_sample(record: dict[str, Any]) -> None:
    """人工修正回流的样本池：V3 的"可迭代"落点，下次评测/few-shot 会读取。"""
    with config.SAMPLE_POOL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_human_samples() -> list[dict[str, Any]]:
    if not config.SAMPLE_POOL.exists():
        return []
    rows = []
    for line in config.SAMPLE_POOL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows