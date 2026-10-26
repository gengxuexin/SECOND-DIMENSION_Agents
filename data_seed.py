"""业务演示数据：商品表、标签库、评论库（含 gold 标注，供评测用）。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import config

PRODUCTS = [
    # sku_id, name, category, style, color, price, size_set, stock, listed_at
    ("SKU-1001", "法式方领连衣裙", "连衣裙", "方领收腰", "米白", 329, "S|M|L|XL", 42, "2024-10-12"),
    ("SKU-1002", "碎花雪纺连衣裙", "连衣裙", "碎花雪纺", "藏青", 259, "S|M|L", 18, "2024-10-15"),
    ("SKU-1003", "通勤西装外套", "外套", "直筒西装", "卡其", 469, "S|M|L|XL", 25, "2024-10-18"),
    ("SKU-1004", "基础纯棉白T", "T恤", "圆领基础", "白", 89, "S|M|L|XL|XXL", 120, "2024-10-01"),
    ("SKU-1005", "宽松落肩卫衣", "毛衣", "落卫", "灰", 199, "M|L|XL", 60, "2024-10-20"),
    ("SKU-1006", "高腰阔腿牛仔裤", "阔腿裤", "高腰直筒", "浅蓝", 289, "S|M|L|XL", 33, "2024-10-05"),
    ("SKU-1007", "毛呢半身裙", "半身裙", "A字毛呢", "焦糖", 299, "S|M|L", 12, "2024-10-22"),
    ("SKU-1008", "轻薄羽绒服", "羽绒服", "轻薄立领", "米白", 599, "S|M|L|XL", 20, "2024-10-08"),
    ("SKU-1009", "厚底老爹运动鞋", "运动鞋", "厚底老爹", "白", 399, "36|37|38|39|40", 27, "2024-10-10"),
    ("SKU-1010", "小羊皮短靴", "短靴", "粗跟短靴", "黑", 559, "36|37|38|39", 9, "2024-10-25"),
    ("SKU-1011", "条纹衬衫", "衬衫", "宽松条纹", "蓝白", 219, "S|M|L|XL", 47, "2024-10-03"),
    ("SKU-1012", "工装牛仔外套", "外套", "工装牛仔", "深蓝", 379, "M|L|XL", 15, "2024-10-27"),
]

COMMENTS = [
    ("C-001", "SKU-1001", "连衣裙", "好评", "positive", "裙子版型很显瘦，面料也不透，就是腰围偏小了一点，建议拍大一码。"),
    ("C-002", "SKU-1001", "连衣裙", "差评", "negative", "色差太严重了，模特图是米白收到偏黄，面料还有起球，两百多块钱有点失望。"),
    ("C-003", "SKU-1002", "连衣裙", "好评", "positive", "雪纺很轻盈夏天穿刚好，就是肩线有点溜，需要垫肩。"),
    ("C-004", "SKU-1002", "连衣裙", "中评", "neutral", "整体还行，快递三天就到了，包装简单。"),
    ("C-005", "SKU-1003", "外套", "好评", "positive", "卡其色很正，通勤搭配多，肩宽的姐妹也能穿。"),
    ("C-006", "SKU-1003", "外套", "差评", "negative", "偏小，胸围明显不够，内搭毛衣就绷开了，退货客服态度还行。"),
    ("C-007", "SKU-1004", "T恤", "好评", "positive", "基础款百搭，纯棉不领口不变形，洗过一次也不掉色。"),
    ("C-008", "SKU-1004", "T恤", "好评", "positive", "面料舒服，就是偏宽松了，正常码就够不用加码。"),
    ("C-009", "SKU-1005", "毛衣", "差评", "negative", "起球太严重了，穿两次袖口全是毛球，面料薄不保暖，物流也慢。"),
    ("C-010", "SKU-1005", "毛衣", "中评", "neutral", "版型还行但是灰色实物偏深，性价比一般。"),
    ("C-011", "SKU-1006", "阔腿裤", "好评", "positive", "高腰显腿长，直筒不挑人，浅蓝百搭。"),
    ("C-012", "SKU-1006", "阔腿裤", "差评", "negative", "裤长偏长，160cm穿131要剪，客服让我自己改。"),
    ("C-013", "SKU-1007", "半身裙", "中评", "neutral", "毛呢手感不错，但是焦糖色偏橘，和图片不太一样。"),
    ("C-014", "SKU-1008", "羽绒服", "好评", "positive", "轻薄不臃肿，米白很百搭，零下几度也能穿。"),
    ("C-015", "SKU-1008", "羽绒服", "差评", "negative", "钻绒严重，衣服上全是白点，客服态度冷淡不回复。"),
    ("C-016", "SKU-1009", "运动鞋", "好评", "positive", "厚底增高显腿长，脚感软，37码正好。"),
    ("C-017", "SKU-1009", "运动鞋", "差评", "negative", "偏小一码，38码穿着挤脚，五天就到了还行。"),
    ("C-018", "SKU-1010", "短靴", "好评", "positive", "小羊皮质感不错，粗跟好走，百搭黑色。"),
    ("C-019", "SKU-1010", "短靴", "差评", "negative", "掉色严重，穿一次鞋头就蹭掉色，售后推卸责任。"),
    ("C-020", "SKU-1011", "衬衫", "好评", "positive", "条纹很清爽，宽松版型不勒，蓝白色显白。"),
    ("C-021", "SKU-1011", "衬衫", "差评", "negative", "开线了，袖口线头很多，洗一次就变形。"),
    ("C-022", "SKU-1012", "外套", "中评", "neutral", "工装版型硬挺，但是深蓝实物偏黑，物流慢发货等了四天。"),
    ("C-023", "SKU-1001", "连衣裙", "差评", "negative", "开线脱线，肩带里面的线头磨皮肤，要求退款。"),
    ("C-024", "SKU-1004", "T恤", "好评", "positive", "买了两件换着穿，质量好会回购。"),
    ("C-025", "SKU-1007", "半身裙", "差评", "negative", "腰围偏小扣子崩开了，物流慢，退货拖了五天。"),
    ("C-026", "SKU-1005", "毛衣", "好评", "positive", "落肩版型显瘦，面料厚实暖和，物流也快。"),
    ("C-027", "SKU-1003", "外套", "好评", "positive", "质感很挺括，值这个价，客服回复很快。"),
    ("C-028", "SKU-1008", "羽绒服", "中评", "neutral", "尺码偏小扣紧，颜色比图浅一点。"),
    ("C-029", "SKU-1006", "阔腿裤", "差评", "negative", "线头多，面料薄起球，性价比低。"),
    ("C-030", "SKU-1012", "外套", "好评", "positive", "做工扎实口袋多，很喜欢，物流两天到。"),
]

TAG_LIBRARY = {
    "canonical": {
        # value = 标准标签；dimension 标明该标签属于哪个归一维度
        "连衣裙": "连衣裙", "半身裙": "半身裙", "阔腿裤": "阔腿裤",
        "牛仔裤": "牛仔裤", "衬衫": "衬衫", "T恤": "T恤",
        "毛衣": "毛衣", "外套": "外套", "羽绒服": "羽绒服",
        "西装": "西装", "短裤": "短裤", "运动鞋": "运动鞋",
        "帆布鞋": "帆布鞋", "高跟鞋": "高跟鞋", "短靴": "短靴",
        "双肩包": "双肩包", "单肩包": "单肩包", "棒球帽": "棒球帽",
        "大衣": "大衣", "西裤": "西裤", "风衣": "风衣", "衬衣": "衬衣",
    },
    "dimension": {k: "category" for k in
                  ("连衣裙", "半身裙", "阔腿裤", "牛仔裤", "衬衫", "T恤", "毛衣",
                   "外套", "羽绒服", "西装", "短裤", "运动鞋", "帆布鞋", "高跟鞋",
                   "短靴", "双肩包", "单肩包", "棒球帽",
                   "大衣", "西裤", "风衣", "衬衣")},
    "aliases": {
        "裙": "连衣裙", "长裙": "连衣裙", "裙子": "连衣裙", "dress": "连衣裙",
        "半裙": "半身裙", "A字裙": "半身裙", "百褶裙": "半身裙",
        "裤子": "阔腿裤", "阔腿": "阔腿裤", "直筒裤": "阔腿裤",
        "短袖": "T恤", "t恤": "T恤", "tee": "T恤", "体恤": "T恤",
        "卫衣": "毛衣", "针织衫": "毛衣", "毛衣衫": "毛衣",
        "夹克": "外套", "西装外套": "外套", "棉服": "外套", "夹克衫": "外套",
        "羽绒": "羽绒服", "羽绒衣": "羽绒服", "羽绒外套": "羽绒服",
        "大衣": "大衣", "呢子大衣": "大衣", "羊毛大衣": "大衣", "长款大衣": "大衣",
        "风衣": "风衣", "长风衣": "风衣", "防风衣": "风衣",
        "西裤": "西裤", "西装裤": "西裤", "正装裤": "西裤", "西裤女": "西裤",
        "棉衣": "外套",
        "靴子": "短靴", "短靴子": "短靴", "boots": "短靴",
        "球鞋": "运动鞋", "运动靴": "运动鞋", "sneakers": "运动鞋",
        "衬衫衫": "衬衫", "衬衣": "衬衫", "shirt": "衬衫",
    },
}

EVAL_SET = [
    {"id": "E-001", "kind": "intent", "input": "帮我把这几张新上的衣服图片归一下类",
     "expect": {"intent": "image_archive"}, "note": "V2 图像归档意图路由"},
    {"id": "E-002", "kind": "intent", "input": "最近网店差评里尺码和起球问题多吗",
     "expect": {"intent": "comment_insight"}, "note": "评论洞察意图路由"},
    {"id": "E-003", "kind": "intent", "input": "根据评论和库存给我一些选品建议",
     "expect": {"intent": "advice"}, "note": "运营建议意图路由"},
    {"id": "E-004", "kind": "tag_norm", "input": "裙子",
     "expect": {"normalized": "连衣裙", "need_confirm": False}, "note": "别名归一"},
    {"id": "E-005", "kind": "tag_norm", "input": "短袖",
     "expect": {"normalized": "T恤", "need_confirm": False}, "note": "别名归一"},
    {"id": "E-006", "kind": "tag_norm", "input": "卫衣",
     "expect": {"normalized": "毛衣", "need_confirm": False}, "note": "别名归一"},
    {"id": "E-007", "kind": "tag_norm", "input": "羽绒外套",
     "expect": {"normalized": "羽绒服", "need_confirm": False}, "note": "多词歧义归一"},
    {"id": "E-008", "kind": "comment_topic", "input": "尺码偏小，建议拍大一码",
     "expect": {"topics_include": ["尺码"], "sentiment": "negative"},
     "note": "长尾主题召回"},
    {"id": "E-009", "kind": "comment_topic", "input": "色差太严重了，面料还有起球",
     "expect": {"topics_include": ["颜色", "质量"], "sentiment": "negative"},
     "note": "多主题共存"},
    {"id": "E-010", "kind": "comment_topic", "input": "面料舒服不褪色，比想象中好，会回购",
     "expect": {"sentiment": "positive"}, "note": "正面情感"},
    {"id": "E-011", "kind": "advice_evidence", "input": "根据评论和库存给我一些选品建议",
     "expect": {"min_advice": 1, "require_evidence": True},
     "note": "建议必须带证据（防幻觉硬约束）"},
    {"id": "E-012", "kind": "tool_args", "input": "analyze_comments",
     "expect": {"tool": "analyze_comments", "forbidden_args": ["system", "prompt", "role"]},
     "note": "工具参数白名单"},
]


def write_all() -> None:
    with config.PRODUCT_TABLE.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sku_id", "name", "category", "style", "color", "price",
                    "size_set", "stock", "listed_at"])
        w.writerows(PRODUCTS)
    config.TAG_LIBRARY.write_text(json.dumps(TAG_LIBRARY, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    with config.COMMENT_LIBRARY.open("w", encoding="utf-8") as f:
        for cid, sku, cat, pol, sent, text in COMMENTS:
            f.write(json.dumps({"comment_id": cid, "sku_id": sku, "category": cat,
                                "polarity": pol, "gold_sentiment": sent, "text": text},
                               ensure_ascii=False) + "\n")
    with config.EVAL_SET.open("w", encoding="utf-8") as f:
        for row in EVAL_SET:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"数据已生成: {config.PRODUCT_TABLE.name} ({len(PRODUCTS)} 商品), "
          f"{config.COMMENT_LIBRARY.name} ({len(COMMENTS)} 评论), "
          f"{config.TAG_LIBRARY.name}, {config.EVAL_SET.name} ({len(EVAL_SET)} 用例)")


if __name__ == "__main__":
    write_all()