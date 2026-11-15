# 当前机器的 CPU 构建与推理实验

本文件记录当前 AVX2 机器的实际实验。面向新 AVX512 CPU 机器的通用环境创建
流程见 [05-reproducible-environment.md](05-reproducible-environment.md) 和
[`setup_cpu_avx512_env.sh`](setup_cpu_avx512_env.sh)。

## 1. 已验证环境

```text
OS/kernel:      Linux 6.6.119+
arch:           x86_64
CPU:            AMD EPYC 7K62
logical CPUs:   48
physical cores: 24（SMT2）
NUMA nodes:     1
ISA:            AVX2，有 F16C；无 AVX512/AVX512-BF16/AMX
memory:         125 GiB
Python:         3.11.6
GCC:            12.3.1
PyTorch:        2.10.0+cpu
vLLM runtime:   0.16.1.dev1+g60d3e7e51...（v0.16.0 后 1 个本地提交）
```

官方 x86 CPU wheel 从 v0.13.0 起按 AVX512 构建。这台 EPYC Rome 没有 AVX512，直接运行该 wheel 有 illegal-instruction 风险，因此本实验从源码编译 AVX2 扩展。

## 2. 创建隔离环境

已经执行并成功：

```bash
cd /data/home/paulning/study/vllm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  -r requirements/cpu-build.txt \
  -r requirements/cpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

当前 CPU requirements 固定 `torch==2.10.0+cpu`。使用 `--no-build-isolation` 是为了让源码构建复用这个 CPU torch，而不是让 `pyproject.toml` 的通用 build-system 依赖另建隔离环境。

## 3. v0.16 的 AVX2 构建缺陷与修复

原始 `v0.16.0` 在 AVX2 下配置成功：

```text
CPU extension compile flags:
-mf16c;-fopenmp;-DVLLM_CPU_EXTENSION;-mavx2
```

但编译 `csrc/cpu/mla_decode.cpp` 时失败：

```text
error: incomplete type 'qk_vec_type' {aka 'void'}
used in nested name specifier
```

原因是 BF16 `KernelVecType` 只覆盖 AVX512-BF16、s390x 和 aarch64，遗漏了 x86 非 AVX512 分支。
仓库已应用上游精确修复：

```text
5a5c43511a fix(cpu): fix mla_decode compilation on x86 without AVX512 (#34052)
```

核心改动是把架构条件改为通用 `#else`。这是很好的入门 issue：文档声称 AVX2 可构建，但 CI/发布主路径偏向 AVX512，导致模板实例化的边缘平台回归。

## 4. 编译 CPU 扩展

已经执行并成功：

```bash
source .venv/bin/activate
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_DISABLE_AVX512=true
export MAX_JOBS=8
python -m pip install -e . --no-build-isolation
```

若不激活 venv，而是直接调用 `.venv/bin/python`，还要把 `.venv/bin` 放入 `PATH`，否则 `setup.py` 子进程找不到 pip 安装的 `cmake`。

构建产物：

```text
vllm/_C.abi3.so
```

验证命令：

```bash
VLLM_TARGET_DEVICE=cpu .venv/bin/python - <<'PY'
import torch
import vllm
import vllm._C
from vllm.platforms import current_platform

print(vllm.__version__)
print(torch.__version__)
print(current_platform.device_type)
print(vllm._C.__file__)
PY
```

实际结果确认 `torch=2.10.0+cpu`、`platform=cpu`，并成功加载本地 `_C.abi3.so`。
另外直接调用 `mla_decode_kvcache_cpu`，分别以 FP32、FP16、BF16 和
PyTorch SDPA reference 对比，三种 dtype 均通过 `torch.testing.assert_close`，
证明 AVX2 兼容补丁不仅能编译，也通过了最小数值正确性检查。

## 5. 最小推理

程序：[`cpu_smoke.py`](cpu_smoke.py)

运行：

```bash
VLLM_TARGET_DEVICE=cpu \
VLLM_CPU_KVCACHE_SPACE=1 \
VLLM_CPU_OMP_THREADS_BIND=0-7 \
.venv/bin/python docs/notes/cpu_smoke.py
```

关键参数：

- `dtype=float32`：本机没有原生 AVX512-BF16；
- `enforce_eager=True`：第一次先排除 Inductor 编译变量；
- `max_model_len=128`：限制学习实验规模；
- `max_num_batched_tokens=64`：故意让 chunked prefill 生效；
- `VLLM_CPU_KVCACHE_SPACE=1`：只为 KV cache 预留 1 GiB；
- 绑 core 0–7：让线程行为易观察，并给系统其他进程留资源。

建议始终显式设置 `VLLM_CPU_KVCACHE_SPACE`。这个快照的文档同一文件内分别
写了默认 `0` 和默认 `4GB`，但实际 `CpuPlatform.get_device_total_memory()`
在变量未设置时会取“每个 NUMA node 可见总内存的 50%”。以代码为准；本机
若不显式设置，默认预算会远大于本实验需要。

实际日志已证明：

```text
Resolved architecture: LlamaForCausalLM
device_config=cpu
Chunked prefill is enabled
OMP threads: 8，分别绑定 core 0..7
distributed backend: gloo
TP=PP=DP=1
CPU KV cache: 262,144 tokens
max concurrency at max_model_len=128: 2048x
两条 prompt 均生成 8 token
exit code: 0
```

模型 `hmellor/tiny-random-LlamaForCausalLM` 使用随机权重，生成文本没有语言意义。不要用它判断质量；
它的用途是快速覆盖：

```text
tokenizer -> EngineCore -> Scheduler -> KV allocation
-> CPUWorker -> Llama forward -> CPU attention -> sampler -> detokenizer
```

首次总耗时约 44 秒，其中包含子进程启动、Hugging Face 元数据/权重下载和 tokenizer 初始化；日志中的真正两条 tiny prompt 执行很短。这个数字不能当正式性能基线。

日志中 “GPU KV cache size” 是通用日志文案，实际 device config 和分配均为 CPU。无 GPU 时的 Triton disabled 和退出阶段 nanobind leak warning 也没有阻止推理成功，后续可单独作为依赖清理问题研究。

## 6. 第一个源码跟踪实验

不要直接用 `print` 污染 hot path，先在以下位置用 logger 或 profiler range：

1. `vllm/v1/engine/core.py::EngineCore.step`
2. `vllm/v1/core/sched/scheduler.py::Scheduler.schedule`
3. `KVCacheManager.allocate_slots`
4. `CPUWorker.execute_model`
5. `CPUAttentionBackendImpl.forward`

每 step 只记录：

```text
request_id
status
num_tokens / num_computed_tokens
num_scheduled_tokens
token_budget before/after
new block ids
KV usage
preempted request ids
```

用两条不同 prompt、每条只生成 3 token，手工对照每一步状态。

## 7. 推荐 CPU 实验

### 7.1 线程扩展性

分别设置：

```bash
VLLM_CPU_OMP_THREADS_BIND=0
VLLM_CPU_OMP_THREADS_BIND=0-1
VLLM_CPU_OMP_THREADS_BIND=0-3
VLLM_CPU_OMP_THREADS_BIND=0-7
VLLM_CPU_OMP_THREADS_BIND=0-15
```

固定模型、prompt token、output token、并发和随机种子，分别记录：

- 冷启动；
- warm run throughput；
- TTFT/TPOT；
- CPU utilization；
- context switch 和 cache miss。

不要把逻辑 CPU 数直接当物理核数。先用 `lscpu -e=CPU,CORE,NODE` 验证 sibling。

### 7.2 perf

```bash
perf stat \
  -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations \
  env VLLM_TARGET_DEVICE=cpu \
      VLLM_CPU_KVCACHE_SPACE=1 \
      VLLM_CPU_OMP_THREADS_BIND=0-7 \
      .venv/bin/python docs/notes/cpu_smoke.py
```

构建类型是 `RelWithDebInfo`，可继续：

```bash
perf record -g -- \
  env VLLM_TARGET_DEVICE=cpu \
      VLLM_CPU_KVCACHE_SPACE=1 \
      VLLM_CPU_OMP_THREADS_BIND=0-7 \
      .venv/bin/python docs/notes/cpu_smoke.py
perf report
```

模型过小会让进程启动和 Python 占比过高。完成调用链学习后，换 `facebook/opt-125m` 或一个小型真实 Llama/Qwen 模型，并重复足够多请求。

### 7.3 Prefix cache

使用一个超过完整 block 的相同前缀：

1. `enable_prefix_caching=True`；
2. 第一次请求填充 cache；
3. 第二次请求复用相同长前缀、修改末尾问题；
4. 比较 `num_cached_tokens`、TTFT 和 cache hit metrics；
5. 再改变一个早期 token，观察链式 block hash 从该块以后全部失配。

注意：prefix cache 只减少 prefill，不加速长答案的 decode。

### 7.4 Chunked prefill

保持长 prompt 不变，改变：

```text
max_num_batched_tokens = 32 / 64 / 128 / 256
long_prefill_token_threshold
```

同时加入短 decode 请求，观察 TTFT 与 ITL 的取舍。重点分析 scheduler output，不必先追求绝对吞吐。

### 7.5 KV 压力与抢占

tiny 模型在 1 GiB KV cache 下很难产生压力。可以：

- 用 scheduler/KV 单元测试直接构造少量 block；
- 或换较大模型、长上下文和高并发；
- 观察 `allocate_slots() -> None -> _preempt_request()`；
- 验证被抢占请求 `num_computed_tokens` 清零、KV 释放、随后重算。

### 7.6 NUMA

当前机器只有一个 NUMA node，只能验证绑核和 MEMBIND 生效，不能得出跨 NUMA 性能结论。要完成真正实验，应在双 socket/多 node 主机上：

- 一 rank 对一 node；
- 比较 weight/KV local 与 remote；
- 监控 `numastat -p`；
- 避免同一 rank 的 OpenMP 线程跨 node；
- 检查容器 seccomp/capability 是否阻止 `get_mempolicy/migrate_pages`。

## 8. 从 CPU 迁移到 GPU 时保留和补充什么

可以直接迁移：

- request 状态机、scheduler、token budget；
- Paged KV、prefix cache、抢占；
- EngineCore/executor/worker 分层；
- batch shape、内存容量和服务指标；
- correctness/benchmark 方法。

必须重学或补强：

- warp/SM/Tensor Core 和 HBM/L2/shared/register；
- kernel launch、stream、event、CUDA Graph；
- FlashAttention/Triton；
- NCCL、NVLink/PCIe/RDMA；
- GPU profiler 与 occupancy；
- FP8/FP4 等 GPU 量化 kernel。

CPU 跑通是正确的第一步，但不是 GPU 推理性能学习的终点。
