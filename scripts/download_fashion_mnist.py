"""下载 Fashion-MNIST 数据集（Zalando 官方，MIT 许可）。

为什么要脚本而不是直接放仓库：
- 4 个 gzip 文件共约 31MB，进 git 会让仓库迅速膨胀，且是不可再生的二进制
- Fashion-MNIST 是公开标准数据集，引用官方 S3 源更合适（可校验完整性）

用法：
    python scripts/download_fashion_mnist.py
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "fashion_mnist"
SOURCE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

FILES = {
    "train-images-idx3-ubyte.gz": "60000 张训练图（28x28 灰度）",
    "train-labels-idx1-ubyte.gz": "训练集标签",
    "t10k-images.gz": "10000 张测试图",
    "t10k-labels-idx1-ubyte.gz": "测试集标签",
}


def main() -> int:
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, desc in FILES.items():
        target = DATA_DIR / name
        if target.exists() and target.stat().st_size > 0:
            # 已存在则校验 gzip 完整性
            try:
                with gzip.open(target, "rb") as f:
                    f.read(16)
                print(f"[skip] {name:<32} 已存在且完整")
                continue
            except Exception:
                print(f"[warn] {name} 存在但损坏，重新下载")
        url = f"{SOURCE}/{name}"
        print(f"[get ] {name:<32} {desc}")
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(target, "wb") as f:
                f.write(r.read())
        except Exception as e:
            print(f"[fail] {name}: {e}")
            ok = False
            continue
        # 完整性校验
        try:
            with gzip.open(target, "rb") as f:
                f.read(16)
            print(f"       -> {target.stat().st_size / 1024 / 1024:.1f} MB ✓")
        except Exception as e:
            print(f"[fail] 校验不通过: {e}")
            ok = False
    print(f"\n完成。目录：{DATA_DIR}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
