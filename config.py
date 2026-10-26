"""全局配置：模型服务地址、阈值、业务上下文路径。

设计原则：LLM 只做理解与编排，所有"判定"结论必须来自工具层（确定性模型）。
因此这里集中管理工具服务的开关与置信度阈值，便于评测阶段统一调参。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VAR_DIR = BASE_DIR / "var"
EVAL_DIR = BASE_DIR / "eval"
for _d in (DATA_DIR, VAR_DIR, EVAL_DIR):
    _d.mkdir(exist_ok=True)

PRODUCT_TABLE = DATA_DIR / "products.csv"
TAG_LIBRARY = DATA_DIR / "tag_library.json"
COMMENT_LIBRARY = DATA_DIR / "comments.jsonl"
EVAL_SET = EVAL_DIR / "eval_set.jsonl"
SAMPLE_POOL = VAR_DIR / "human_corrected_samples.jsonl"

# ---------------------------------------------------------------- 工具层（确定性模型服务）
IMAGE_SERVICE_URL = os.getenv("IMAGE_SERVICE_URL", "http://127.0.0.1:8801")
COMMENT_SERVICE_URL = os.getenv("COMMENT_SERVICE_URL", "http://127.0.0.1:8802")
SERVICE_TIMEOUT = float(os.getenv("SERVICE_TIMEOUT", "10"))

# 低置信度阈值：低于该值的结论不直接采信，转人工选择（V3 可信治理）
IMAGE_LOW_CONF = float(os.getenv("IMAGE_LOW_CONF", "0.70"))
COMMENT_LOW_CONF = float(os.getenv("COMMENT_LOW_CONF", "0.60"))
# 标签归一映射的最小匹配置信度，低于此值视为"新标签待人工确认"
TAG_NORM_MIN_CONF = float(os.getenv("TAG_NORM_MIN_CONF", "0.65"))

# ---------------------------------------------------------------- 编排层（大模型）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
# 未配置 API Key 时，编排层降级为"规则路由"，保证离线可演示、可回归测试
LLM_STRICT_TOOLS = os.getenv("LLM_STRICT_TOOLS", "1") == "1"

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "6"))