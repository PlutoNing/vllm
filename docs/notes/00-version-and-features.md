# 当前版本、支持特性与版本演进

## 1. 当前仓库到底是什么版本

核验结果：

```text
branch:       paul/dev
HEAD:         60d3e7e513c5f59f5710bf3a94f9ffee58253f71
git describe: v0.16.0-1-g60d3e7e513
parent:       89a77b10846fd96273cce78d86d2556ea582d26e
parent tag:   v0.16.0
```

`HEAD` 相对 `v0.16.0` 只有以下改动：

```python
#print("[PAUL-VLLM] Scheduler.schedule() hit")
```

它位于 `vllm/v1/core/sched/scheduler.py::Scheduler.schedule()`，因为整行已注释，
所以当前源码行为等价于 **vLLM v0.16.0**。editable install 根据 setuptools-scm 显示
为 `0.16.1.dev1+g60d3e7e51...`，这是“标签后 1 个提交”的开发版本号，不代表包含 v0.16.1 
的功能。

这份目录不是没有历史的源码包：`HEAD` 可追溯到 13,752 个提交。当前分支只是有意固定在旧快照；
`origin/main` 已明显前进。因此阅读网上的 `main/latest` 文档时必须核对版本，不能直接套到这里。

### 本机额外兼容补丁

为了在只有 AVX2、没有 AVX512 的本机编译 CPU 扩展，工作区应用了上游提交 `5a5c43511a` 
的 1 行逻辑修复：

```text
fix(cpu): fix mla_decode compilation on x86 without AVX512 (#34052)
```

它把 x86 非 AVX512-BF16 情况纳入通用 BF16 向量类型分支。该补丁只修复编译选择，
不改变 Llama FP32 实验逻辑；详见 [04-cpu-lab.md](04-cpu-lab.md)。

## 2. v0.16.0 的系统画像

这是一个 **V1-only** 的现代 vLLM 版本。V0 在 v0.11.0 已从代码库移除；某些类名或注释
仍写着 “legacy/V0-style”，通常是兼容 API，不表示旧 V0 引擎仍存在。

### 核心能力

- **Paged KV cache / PagedAttention 思想**：请求的逻辑 token 块映射到非连续物理
 KV block，按需分配，降低碎片和预留浪费。
- **连续批处理**：每个 engine step 重新组合活跃请求，而不是等整批请求同时结束。
- **统一 token-budget 调度器**：V1 不在调度层硬分 prefill/decode 阶段，而是根据 
`num_computed_tokens` 与待计算 token 数统一分配预算。
- **Chunked Prefill**：默认尽可能启用，把长 prompt 切块，与 decode 请求共批执行。
- **Automatic Prefix Caching**：按完整 token block 的链式哈希复用已有 KV cache。
- **抢占重算**：KV 空间不足时释放低优先级请求的 block；V1 默认以后续重算代替 GPU/CPU swap。
- **多种调度策略**：FCFS 和 priority，priority 相同则以 FCFS 打破平局。
- **离线与在线入口**：`LLM` Python API、OpenAI 兼容服务、Anthropic/Responses/Realtime 
等服务入口。
- **结构化输出、LoRA、量化、推测解码、流式输出、工具调用与 reasoning parser**。
- **生成、embedding、分类、rerank/pooling、多模态、Whisper、Mamba/混合模型和 MoE**；具体
模型以本 commit 的 registry 与文档为准。
- **分布式并行**：TP、PP、DP、EP，以及 prefill/decode 解耦和 KV connector 体系。
- **编译与图优化**：`torch.compile`/Inductor、vLLM 自定义图 pass、GPU 上的 CUDA Graph；
CPU 不使用 CUDA Graph。
- **多硬件平台**：NVIDIA、ROCm、Intel XPU、CPU、TPU，以及外部插件平台。

### v0.16.0 本身的重点

官方 release notes 的代表性变化：

- async scheduling 与 pipeline parallelism 组合完成；
- WebSocket Realtime API；
- RLHF 的权重同步、分层 reload、暂停/恢复；
- Unified Parallel Drafting 与更多 speculative decoding 组合；
- XPU 内核体系重构；
- CPU zero-copy GQA 路径和 prefix-cache Request 引用环内存泄漏修复；
- Helion kernel framework 与更多可插拔层能力。

官方发布页：<https://github.com/vllm-project/vllm/releases/tag/v0.16.0>

## 3. 当前 CPU 后端的能力和边界

### 可以学习和运行的部分

- x86 CPU 的 FP32、FP16、BF16 基本推理；
- V1 EngineCore、统一调度器、chunked prefill、prefix cache、分页 KV 管理；
- CPU attention 自定义 C++/OpenMP kernel；
- PyTorch SDPA 路径；
- OpenMP 线程绑定、NUMA memory binding、Gloo 分布式初始化；
- TP/PP/DP 的 CPU 架构路径；
- CPU 上的 AWQ、GPTQ（x86）和 compressed-tensors INT8 W8A8（以官方 CPU 文档的约束为准）；
- structured output、LoRA 与 hybrid KV cache 基础设施。

### 明确限制

- `CPUModelRunner` 断言 speculative decoding 配置为空；
- 不使用 CUDA Graph，关闭 cascade attention；
- CPU 平台关闭 async scheduling 和 DBO；
- FP8 KV cache 不受 `CPU_ATTN` 支持；
- 当前 CPU platform 对 MLA/sparse attention 选择有明确限制；
- sleep mode 不支持；
- 多进程 executor 回退到 `mp`，通信后端为 Gloo；
- AVX2 能走基本路径，但 v0.16 的高性能 x86 路径主要面向 AVX512/AMX；AVX2 下不会编译 oneDNN、
WNA16、fused MoE 和 SGL small-batch 等那组条件内核。

因此，本机 CPU 实验适合学习“框架控制面、调度、内存和调用链”，不适合拿吞吐数字评价 vLLM 的 GPU 生产性能。

## 4. 版本演进主线

只记与源码架构有关的里程碑，不建议背 patch release：

| 版本 | 时间 | 对理解当前源码最重要的变化 |
|---|---:|---|
| v0.1.0 | 2023-06 | 首次公开发布；PagedAttention 和连续批处理是项目起点。 |
| v0.2–v0.3 | 2023-09 至 2024-01 | 模型、量化、分布式与服务生态快速扩展。 |
| v0.4.0 | 2024-03 | Automatic Prefix Caching 进入主线，block manager/scheduler 测试加强。 |
| v0.5.x | 2024-06 起 | 多模态、FP8、生产化和硬件覆盖继续扩展。 |
| v0.6.0 | 2024-09 | 性能推动期：多步调度、异步输出处理，官方宣称相对 v0.5.3 约 2 倍吞吐。 |
| v0.7.0 | 2025-01 | V1 alpha，可选启用；`torch.compile` 成为 V1 的核心组成。 |
| v0.8.0 | 2025-03 | V1 默认启用；V1 scheduler/KV manager/多进程架构成为阅读重点。 |
| v0.9.0 | 2025-05 | 大规模推理方向加强：DP、EP、prefill/decode 分离、NIXL/KV connector。 |
| v0.10.0 | 2025-07 | 开始删除 V0；async scheduling 实验化，hybrid KV cache 等继续发展。 |
| v0.11.0 | 2025-10 | V0 完全移除，V1 成为唯一引擎。旧 V0 教程从此尤其容易误导。 |
| v0.12.0 | 2025-12 | Experimental GPU Model Runner V2、EAGLE 改进、prefill context 
parallel 准备。 |
| v0.13.0 | 2025-12 | x86 CPU 官方 wheel、更多编译/attention/prefix-cache 能力。 |
| v0.14.0 | 2026-01 | async scheduling 全局默认启用，但 CPU platform 仍会关闭它。 |
| v0.15.x | 2026-01 至 02 | 模型、内核、编译和稳定性迭代。 |
| **v0.16.0** | **2026-02** | **本仓库基线**；见上一节的 release highlights。 |

早期核心介绍：<https://vllm.ai/blog/2023-06-20-vllm>

V1 alpha 介绍：<https://blog.vllm.ai/2025/01/27/v1-alpha-release.html>

## 5. 哪些旧资料不要直接照抄

1. 路径以 `vllm/core/scheduler.py`、`block_manager_v1/v2`、V0 `LLMEngine` 为中心的教程，
不是当前 V1 主链。
2. 把 prefill queue 和 decode queue 当成 V1 scheduler 的两个固定阶段不准确；当前代码首先用
“已计算 token 追赶待计算 token”的统一模型表达两者。
3. 把抢占描述成 swap 到 CPU 不准确；V1 默认释放 block 后 recompute。
4. 把 PagedAttention 的历史 CUDA kernel 文档当成当前全部实现不准确；仓库自己的 
`docs/design/paged_attention.md` 已标明它是历史文档。
5. 只看 `vllm/v1/worker/gpu_model_runner.py` 文件名就断言 CPU 不走这段代码也不准确；
`CPUModelRunner` 直接继承并复用它。
6. `docs/design/multiprocessing.md` 本身注明部分引用基于 2024 年代码；进程拓扑应以当前
 `docs/design/arch_overview.md` 和实际类为准。

## 6. 源码内的权威检查点

- V1 状态与功能差异：`docs/usage/v1_guide.md`
- 架构和进程模型：`docs/design/arch_overview.md`
- CPU 安装、亲和性与调优：`docs/getting_started/installation/cpu.md`
- CPU 模型验证列表：`docs/models/hardware_supported_models/cpu.md`
- 当前模型 registry：`vllm/model_executor/models/registry.py`
- 平台硬限制：`vllm/platforms/cpu.py`
- CPU runner 硬限制：`vllm/v1/worker/cpu_model_runner.py`
- 调度真实语义：`vllm/v1/core/sched/scheduler.py`
- KV 管理真实语义：`vllm/v1/core/kv_cache_manager.py`、`block_pool.py`
