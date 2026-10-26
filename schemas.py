"""结构化输出契约：所有跨层数据一律走 Pydantic，禁止自由文本传递判定结论。

这一层是 V3"可信治理"的地基：
- 分类/情感结论 -> ToolResult（带 score、证据字段）
- 标签归一     -> NormalizedTag（带 source、matched_rule）
- 评论主题归纳 -> CommentTheme（必须带 evidence 评论 id）
- 运营建议     -> Advice（必须引用 tool_refs + evidence_ids，校验器强制非空）
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================ 工具层契约
class ToolError(BaseModel):
    code: str = Field(description="错误码，如 SERVICE_TIMEOUT / LOW_IMAGE")
    message: str = Field(description="可读错误信息")
    retryable: bool = False


class ImagePrediction(BaseModel):
    """服装图像分类模型的单条输出（ResNet50 微调服务的返回体片段）。"""

    label: str = Field(description="预测标签原文，取自分类头类别表")
    score: float = Field(ge=0.0, le=1.0, description="softmax 概率")
    top_k_rank: int = Field(default=1, description="在 top-k 中的排名，1 为最高")
    raw_score: float | None = Field(default=None, description="绝对相似度（跨后端可比）")
    need_confirm: bool = Field(default=False,
                               description="多后端交叉验证结论冲突，标记为不自动采信")
    cross_check: dict[str, Any] | None = Field(default=None,
                                                description="交叉验证详情：辅助后端与其 Top-1")
    conflict_reason: str = Field(default="", description="冲突原因说明")


class ImageClassifyResult(BaseModel):
    ok: bool = True
    predictions: list[ImagePrediction] = Field(default_factory=list)
    latency_ms: float = 0.0
    model_version: str = "resnet50-transfer-v1"
    error: ToolError | None = None


class CommentScores(BaseModel):
    """评论多标签情感/主题打分（BERT+CNN+Attention 服务返回）。"""

    sentiment: Literal["positive", "negative", "neutral"] = "neutral"
    sentiment_conf: float = Field(ge=0.0, le=1.0)
    topics: dict[str, float] = Field(default_factory=dict, description="主题->置信度，如 尺码/颜色/质量/物流/服务")
    tags: list[str] = Field(default_factory=list)


class CommentAnalyzeResult(BaseModel):
    ok: bool = True
    scores: CommentScores | None = None
    latency_ms: float = 0.0
    model_version: str = "bert-cnn-attn-v1"
    error: ToolError | None = None


# ============================================================ 业务上下文契约
class Product(BaseModel):
    sku_id: str
    name: str
    category: str
    style: str
    color: str
    price: float
    size_set: list[str] = Field(default_factory=list)
    stock: int = 0
    listed_at: str = ""


class NormalizedTag(BaseModel):
    """标签归一结果：把模型输出的自由标签映射到标签库标准值。"""

    raw: str = Field(description="模型/模型服务给出的原始标签")
    normalized: str = Field(description="标签库中的标准标签；无法归一时回落到 raw 并标记待确认")
    conf: float = Field(ge=0.0, le=1.0)
    source: Literal["tag_library", "fuzzy", "llm", "fallback"] = "fallback"
    need_confirm: bool = Field(default=False, description="置信度不足时为 True，转人工确认")


class CommentTheme(BaseModel):
    """评论主题归纳。evidence_ids 为强制项：无证据的主题不予采信（抑制幻觉）。"""

    theme: str = Field(description="主题名，如 尺码偏小")
    category: str = Field(default="", description="所属商品品类，跨品类时为空")
    count: int = Field(ge=0, default=0)
    sentiment: Literal["positive", "negative", "neutral"] = "neutral"
    share: float = Field(default=0.0, ge=0.0, le=1.0,
                         description="该主题在聚合分组内的权重（主题置信度均值）")
    evidence_ids: list[str] = Field(default_factory=list, description="支撑该主题的评论 id，至少 1 条")
    sample_excerpt: str = ""


class Advice(BaseModel):
    """运营建议。引用工具输出与评论证据是硬约束，由 model_validator 强制。"""

    title: str
    detail: str
    action_type: Literal["选品", "备货", "尺码", "售后", "详情页", "陈列", "定价", "其他"]
    priority: Literal["高", "中", "低"] = "中"
    confidence: float = Field(ge=0.0, le=1.0)
    tool_refs: list[str] = Field(default_factory=list, description="引用的工具调用/结论，如 image_classify#3")
    evidence_ids: list[str] = Field(default_factory=list, description="引用的评论证据 id")
    need_human_confirm: bool = False

    @model_validator(mode="after")
    def _require_evidence(self) -> "Advice":
        if not self.tool_refs and not self.evidence_ids:
            raise ValueError("运营建议必须至少引用一项工具输出或评论证据，否则不允许生成")
        return self


# ============================================================ 状态机契约
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    NEEDS_HUMAN = "needs_human"
    FAILED = "failed"


class TraceEvent(BaseModel):
    """全链路轨迹：每一次工具调用、每一次人工介入都要落痕，保证结果可追溯。"""

    step: str
    actor: Literal["llm", "tool", "human", "system"]
    detail: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    cost_ms: float = 0.0


class WorkflowState(BaseModel):
    """贯穿图像归档 / 评论洞察 / 运营建议三个 Agent 的共享状态。"""

    task_id: str
    user_query: str
    intent: str = "unknown"
    status: TaskStatus = TaskStatus.PENDING

    # 图像归档
    image_paths: list[str] = Field(default_factory=list)
    image_labels: list[NormalizedTag] = Field(default_factory=list)
    low_conf_items: list[str] = Field(default_factory=list)

    # 评论洞察
    comment_scope: str = ""
    comment_results: dict[str, CommentScores] = Field(default_factory=dict)
    themes: list[CommentTheme] = Field(default_factory=list)

    # 运营建议
    advice: list[Advice] = Field(default_factory=list)

    trace: list[TraceEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    report: str = ""

    def log(self, step: str, actor: str, detail: str = "", payload: dict | None = None,
            cost_ms: float = 0.0) -> None:
        self.trace.append(TraceEvent(step=step, actor=actor, detail=detail,
                                     payload=payload or {}, cost_ms=cost_ms))


class ToolCallRecord(BaseModel):
    """工具调用记录，用于回归测试工具调用参数是否稳定。"""

    call_id: str
    name: str
    arguments: dict[str, Any]
    ok: bool = True
    cost_ms: float = 0.0
    result_summary: str = ""


__all__ = [
    "ToolError", "ImagePrediction", "ImageClassifyResult",
    "CommentScores", "CommentAnalyzeResult", "Product", "NormalizedTag",
    "CommentTheme", "Advice", "TaskStatus", "TraceEvent", "WorkflowState",
    "ToolCallRecord",
]