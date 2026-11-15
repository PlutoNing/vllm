# 可复现开发环境

## 1. 当前环境存在哪里

当前已经跑通的环境是：

```text
/data/home/paulning/study/vllm/.venv
```

它位于仓库根目录，不在 `docs/notes/`。`.gitignore` 已忽略 `.venv`，这是正确
做法：virtualenv 含绝对路径、本机 Python ABI、已编译扩展和平台相关 wheel，
不应复制或提交到另一台机器。

真正应该复用的是：

```text
固定的 git commit
+ 工作区兼容补丁
+ requirements 文件
+ 环境创建脚本
+ 构建参数与依赖快照
```

当前 AVX2 环境只是一次成功实例，不是可搬运制品。

## 2. AVX512 CPU 机器的一键搭建

脚本：

```text
docs/notes/setup_cpu_avx512_env.sh
```

前提：

- Linux x86_64；
- Python 3.10–3.13；
- CPU 至少有 `avx512f`；
- GCC/G++ 建议至少 12.3；
- 已安装 Python venv、`libnuma` runtime 和 development headers；
- 有足够磁盘空间并能访问 PyPI、PyTorch index 和 Hugging Face。

在全新的源码工作区运行：

```bash
cd /data/home/paulning/study/vllm
bash docs/notes/setup_cpu_avx512_env.sh
```

脚本会：

1. 拒绝非 x86_64、无 AVX512F 或缺少 `numa.h` 的机器；
2. 在仓库根目录创建 `.venv`；
3. 安装当前 commit 的 CPU build/runtime requirements；
4. 以 editable 模式从当前源码编译 CPU 扩展；
5. 不设置本机 AVX2 专用的 `VLLM_CPU_DISABLE_AVX512`；
6. 验证 vLLM、PyTorch、CPU platform 和 `_C.abi3.so`；
7. 在 `.venv/vllm-build-info.txt` 记录 commit、dirty 状态、编译器和
   `pip freeze`。

已有 `.venv` 时脚本会停止，不会自动删除。要并存多个环境：

```bash
VENV_DIR=.venv-cpu bash docs/notes/setup_cpu_avx512_env.sh
```

## 3. 为什么脚本可复用、但不是“比特级锁定”

仓库的 `requirements/cpu*.txt` 固定了 PyTorch 等关键依赖，却没有固定所有
transitive dependency。因此脚本可以稳定复现“同一源码的 CPU 开发环境搭建
过程”，但未来重新解析依赖时，FastAPI、NumPy 等间接依赖可能得到更新版本。

如果某次实验需要严格复现，在成功搭建后保存该机器生成的 freeze：

```bash
source .venv/bin/activate
python -m pip freeze --exclude-editable \
  > docs/notes/requirements-cpu-py311-x86_64.freeze.txt
```

然后将 freeze 文件与以下信息一起提交：

- `git rev-parse HEAD`；
- `git status --short` 或补丁 commit；
- Python minor version；
- OS/glibc、compiler；
- CPU ISA；
- 构建命令。

freeze 只适合相同 OS/架构/Python minor 的机器；它也不能代替源代码 commit。

## 4. CPU 与 GPU 环境不要混用

CPU 与 CUDA 环境至少有这些差异：

- CPU requirements 使用 `torch==2.10.0+cpu`；
- 当前 CUDA requirements 使用 `torch==2.9.1`、对应 torchvision、
  FlashInfer 和 CUDA 构建链；
- `_C.abi3.so` 是平台相关编译产物。

因此，在新的 GPU 机器上重新创建 `.venv`，不要复制 CPU `.venv`，也不要在
同一个环境中把 CPU torch 原地替换成 CUDA torch。

GPU 机器的同版本基本流程是：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  -r requirements/build.txt \
  -r requirements/cuda.txt
VLLM_TARGET_DEVICE=cuda \
python -m pip install -e . --no-build-isolation
```

实际使用前还要让 NVIDIA driver、CUDA toolkit 和该 commit 的 torch/CUDA
版本匹配；应优先遵循本 commit 的
`docs/getting_started/installation/gpu.cuda.inc.md`，不要直接使用最新版网页
命令。

### 同一台机器同时维护 CPU/GPU 开发环境

editable build 会把 `_C.abi3.so` 放回源码树。即使使用 `.venv-cpu` 和
`.venv-gpu`，两个环境仍可能覆盖同一 checkout 中的扩展文件。可靠做法：

- CPU 和 GPU 使用两个 git worktree/clone；或
- 分别构建 wheel，在各自 venv 中做非 editable 安装。

不同物理机器各自使用该机器仓库根目录的 `.venv` 则没有这个冲突。

## 5. 当前兼容补丁怎样进入复现链

本工作区的 `csrc/cpu/mla_decode.cpp` 包含上游
`5a5c43511a` 的非 AVX512 编译修复。它对 AVX512 主机无害，并让普通 x86
fallback 更完整。

但当前修改和 `docs/notes/` 仍是未提交工作区内容。若只在另一台机器重新
checkout `origin/paul/dev`，这些内容不会自动出现。长期复现前应当：

1. 将笔记、脚本和兼容补丁提交到一个明确分支；
2. 在其他机器 checkout 该 commit；
3. 再运行环境脚本。

在完成 commit/push 之前，只能通过复制当前工作树或导出 patch 保留这些修改。

## 6. 建议的目录约定

```text
vllm/
├── .venv/                       # 当前机器环境，不提交
├── vllm/_C.abi3.so              # 当前 backend 的构建产物，不提交
└── docs/notes/
    ├── setup_cpu_avx512_env.sh  # 提交，可在新机器执行
    ├── 05-reproducible-environment.md
    └── cpu_smoke.py
```

简而言之：**每台机器重建 `.venv`，不要搬运 `.venv`；用 git commit、脚本和
构建记录复现环境。**
