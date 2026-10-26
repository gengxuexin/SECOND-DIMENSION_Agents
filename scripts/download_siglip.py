import os
from huggingface_hub import snapshot_download

# 1. 强制设置国内镜像源，解决网络连接失败问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 禁用 Xet 加速（解决之前卡在 73% 报错的问题）
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 3. 指定正确的仓库 ID（注意大小写，之前404就是因为拼写成了 prithvi）
repo_id = "prithivMLmods/Fashion-Mnist-SigLIP2"

# 4. 指定下载保存的本地路径（前面的 r 不能少，防止转义）
local_dir = r"D:\Projects\服装店Agent\var\siglip_fashion"

print(f"开始下载模型: {repo_id}")
print(f"保存路径: {local_dir}")

# 5. 执行下载
try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 确保下载的是实体文件，而不是快捷方式
        resume_download=True           # 开启断点续传
    )
    print("\n✅ 模型下载完成！")
except Exception as e:
    print(f"\n❌ 下载失败，错误信息: {e}")