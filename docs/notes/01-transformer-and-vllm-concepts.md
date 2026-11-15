# Transformer 与 vLLM 的最小概念集

## 1. 它不是“只算一个很大的矩阵”

更准确的描述是：

> LLM 推理反复执行由矩阵乘、attention、归一化、激活、通信和采样组成的计算图；
vLLM 负责把到达时间、输入长度和输出长度都不确定的请求，动态映射到有限算力和 KV cache 上。

矩阵乘通常消耗大部分 FLOPs，但系统性能还取决于：

- 权重和 KV cache 的内存带宽；
- batch 形状和 kernel launch/调度开销；
- prefill 与 decode 的混合比例；
- KV block 分配、命中、释放和抢占；
- CPU 调度器能否及时喂满 accelerator；
- TP/PP/DP/EP 通信；
- tokenizer、detokenizer、HTTP streaming 和 Python 开销。

所以岗位名称里的“推理框架”更接近 **runtime + memory manager + scheduler + compiler + kernels + serving system**，不是一个单独 GEMM。

## 2. 从文本到下一个 token

### 2.1 必须掌握的张量形状

设 batch 为 `B`，本次 token 数为 `T`，hidden size 为 `D`，attention head 数为 `H`，
每个 head 维度为 `Dh=D/H`。

1. tokenizer：文本变成离散 `token_id`。
2. embedding：`[B,T] -> [B,T,D]`。
3. 每个 decoder layer：
   - normalization；
   - 线性投影产生 Q、K、V；
   - attention；
   - output projection；
   - residual；
   - MLP（常见 SwiGLU）；
   - residual。
4. 最后一层 hidden state 经 LM head 变成 vocabulary logits。
5. sampler 应用 temperature、top-k/top-p、penalty 等规则，选择下一个 token。
6. 把新 token 追加回序列，重复步骤 3–5。

单头 attention 的核心公式：

```text
Q = X Wq
K = X Wk
V = X Wv
A = softmax(Q K^T / sqrt(Dh) + mask)
O = A V
```

需要能根据形状解释每一步，但求职推理系统岗位通常不要求从头推导反向传播。

### 2.2 MHA、GQA、MQA

- MHA：每个 query head 都有自己的 K/V head。
- GQA：多个 query head 共享一组 K/V head。
- MQA：所有 query head 共享一个 K/V head。

GQA/MQA 的主要系统意义是显著缩小 KV cache 和降低 decode 内存流量。读模型配置时必须区分 `num_attention_heads` 与 `num_key_value_heads`。

### 2.3 Prefill 与 Decode

**Prefill** 一次处理整个 prompt：

- token 多，矩阵较大；
- 通常并行度和 arithmetic intensity 较高；
- 生成每层所有 prompt token 的 K/V 并写入 cache；
- 主要影响 TTFT。

**Decode** 每轮通常为每条序列产生一个新 token：

- 大量小 GEMM/GEMV；
- 每层读取历史 KV cache；
- 更容易受 memory bandwidth、kernel launch 和 CPU 调度开销限制；
- 主要影响 TPOT/ITL。

V1 scheduler 在代码里不设置两个绝对分离的 phase。它统一追踪 `num_computed_tokens`，
但从性能分析角度仍必须区分 prefill 和 decode。

## 3. 推理内存从哪里来

### 3.1 权重

粗略估算：

```text
weight_bytes ≈ parameter_count × bytes_per_weight
```

例如 7B 模型仅权重约为 FP16/BF16 14 GB、INT8 7 GB、INT4 3.5 GB；实际还要加量化 scale、元数据、allocator 对齐等。

### 3.2 KV cache

对普通 attention，每条序列的近似 KV 大小：

```text
KV_bytes ≈ 2 × num_layers × num_tokens
           × num_kv_heads × head_dim × bytes_per_element
```

前面的 `2` 是 K 和 V。它随上下文长度与并发请求数线性增长，而且输出长度事先未知，这正是 vLLM 重点解决的问题。

对应当前代码，单层 `AttentionSpec` 的一个物理 page 是：

```text
page_size_bytes =
    2 × block_size × num_kv_heads × head_dim × dtype_bytes
```

一个 KV cache group 的 page size 会把组内各层的 spec 相加；给定可用 KV
预算 `M`，可分配 page 数近似为 `floor(M / group_page_size_bytes)`。源码锚点是
`vllm/v1/kv_cache_interface.py::{AttentionSpec,KVCacheGroupSpec}`。

### 3.3 其他内存

- 临时 activation 和 logits；
- sampler、block table、slot mapping 等运行时 tensor；
- `torch.compile`/CUDA Graph workspace；
- NCCL/Gloo 通信 buffer；
- tokenizer、HTTP、多模态 processor cache；
- Python 对象与队列。

不能用“权重能装下”推断模型一定能服务目标并发。

## 4. Paged KV cache

传统做法若为每条请求预留最大连续 KV 空间，会产生内部碎片和过量预留。vLLM 的核心思路是：

1. 把一条序列的逻辑 token 空间切成固定 token 数的 block；
2. 从统一物理 block pool 按需分配；
3. 用 block table 把逻辑 block 映射到非连续物理 block；
4. attention kernel 根据 block table 访问 KV；
5. 完整 block 可按内容哈希做 prefix cache；
6. 用引用计数支持共享和安全释放。

一个反直觉细节：`BlockHashToBlockMap` 不会在 block 刚填满时立即按 hash
去重，因为替换已分配的 block id 会破坏 block table 的 append-only 性质。
相同 hash 可以暂时映射多个物理 block，查 prefix 时再取其中一个。不要把它
误解成“内容寻址存储必然只有一份物理副本”。

当前 V1 的主要实现：

- `KVCacheManager`：scheduler 面向的统一接口；
- `KVCacheCoordinator`：协调一个或多个 KV cache group；
- `SingleTypeKVCacheManager` 子类：full、sliding-window、chunked-local、Mamba、cross-attention；
- `BlockPool`：物理 block、free queue、哈希索引、引用计数和淘汰。

## 5. 和 OS 内核概念怎样对应

| 内核/系统概念 | vLLM 中可借用的直觉 | 必须记住的边界 |
|---|---|---|
| 进程/任务 | 一个 request 有状态、优先级、到达时间和资源占用。 | request 不是 Linux task，没有独立地址空间和硬件上下文。 |
| 虚拟页/物理页 | 逻辑 KV block 映射到 block pool 中的物理 block。 | 没有 MMU/TLB；映射由应用和 attention kernel 显式消费。 |
| 页表 | request 的 block table。 | 它是 tensor/元数据，不由硬件 page walker 查询。 |
| page allocator | `BlockPool.allocate/free` 与 free-block queue。 | 粒度是 token KV 数据，不是 4 KiB 系统页。 |
| demand paging | token 增长时按需增加 KV block。 | 没有透明 page fault；scheduler 在执行前显式分配。 |
| COW/共享页 | prefix cache block 的引用计数与共享。 | 主要是不可变的已计算 KV 前缀，不等同通用写时复制。 |
| page cache | block hash 到 cached block 的索引。 | key 是模型输入语义及附加条件，不是 inode+offset。 |
| LRU/reclaim | refcount 为 0 的 cached block 进入可淘汰顺序。 | allocator 和 cache eviction 耦合在模型 runtime 内。 |
| CPU scheduler | FCFS/priority 队列、token budget、最大并发数。 | 调度单位是一次模型 step 中的 token，不是 CPU 时间纳秒。 |
| time slice | chunked prefill 防止长 prompt 独占整个 token budget。 | chunk 大小还受 tensor shape、KV block 和吞吐目标影响。 |
| preemption | KV 不足时释放请求 block，放回 waiting queue。 | V1 默认丢弃 KV 后重算，不保存完整执行上下文。 |
| swap | 旧 V0 可讨论 KV swap。 | 当前 V1 文档明确移除了 GPU↔CPU KV swapping。 |
| CPU affinity/cpuset | `VLLM_CPU_OMP_THREADS_BIND` 与每 rank 绑核。 | 还要考虑 OpenMP runtime、PyTorch inter-op 线程和前端进程。 |
| NUMA first-touch/bind | 每个 CPU worker/rank 绑定 NUMA node 和内存策略。 | 模型权重、KV cache、通信 buffer 的实际 placement 都要验证。 |
| remote/disaggregated memory | KV connector 在 prefill/decode 实例间传 KV。 | 是显式协议和数据通道，不是透明 NUMA 远端访问。 |
| branch prediction | speculative decoding 先草拟、再验证多个 token。 | 只是有助理解的比喻，算法和硬件分支预测机制不同。 |

你的内核经验最有价值的地方不是背这些类比，而是能继续追问：资源的所有权在哪里、引用何时增加/减少、失败路径是否泄漏、并发下状态机是否一致、NUMA placement 是否与执行线程匹配。

## 6. 调度器真正做什么

`Scheduler.schedule()` 的核心约束：

- `max_num_batched_tokens`：本轮总 token budget；
- `max_num_seqs`：最大同时运行请求数；
- KV block 是否足够；
- encoder compute/cache budget；
- LoRA 数量限制；
- structured-output FSM 是否准备好；
- remote KV 是否到达；
- speculative lookahead；
- max model length；
- FCFS 或 priority policy。

简化流程：

1. 先尝试推进 RUNNING 请求；
2. 计算每个请求本轮还需多少 token；
3. 限制到 long-prefill threshold、全局 token budget 和模型长度；
4. 向 KV manager 申请 slot；
5. 不足时抢占低优先级/队尾请求并释放其 KV；
6. 再接纳 WAITING 请求，先查本地/远端 prefix cache；
7. 生成 `SchedulerOutput` 交给 executor/worker。

这比通用 OS scheduler 多了“模型计算图 shape 和 KV 可分配性”约束，也比普通 batch queue 更接近资源联合调度。

## 7. 需要会说的性能指标

- **TTFT**：请求到第一个输出 token 的时间，受排队、tokenization、prefill 和首轮调度影响。
- **TPOT**：首 token 后平均每个输出 token 的时间。
- **ITL**：相邻输出 token 的实际间隔，流式服务更关注尾延迟。
- **E2E latency**：请求整体完成时间。
- **throughput**：每秒 request 或 token；必须说明 input/output token 口径。
- **goodput**：满足 SLO 的有效吞吐。
- **KV cache hit rate**：prefix cache 的查询 token 与命中 token。
- **preemption count**：内存压力和配置不合理的重要信号。

面试中不要只说“吞吐提升了”。应说明 workload、prompt/output 长度分布、并发、batch 配置、dtype、硬件、TTFT/TPOT trade-off 和正确性基线。

## 8. 对求职需要掌握到什么程度

### 必须能推导或现场算

- attention 的形状与公式；
- MHA/GQA 对 KV cache 大小的影响；
- 权重和 KV cache 内存估算；
- prefill/decode 的计算/带宽特征；
- TP 的通信位置和基本代价；
- token budget、batch size 与 TTFT/TPOT 的取舍。

### 必须能结合源码解释

- 一条请求从 API 到 scheduler、worker、attention、sampler、输出的链路；
- Paged KV block 的分配、共享、释放和抢占；
- continuous batching、chunked prefill、prefix caching；
- OpenMP/NUMA 绑定；
- TP/PP/DP/EP 的适用场景；
- eager、`torch.compile`、CUDA Graph 的边界；
- 如何用 profiler 和指标定位 CPU 控制面、内存带宽、通信或 kernel 瓶颈。

### 知道概念和用途即可，按岗位再深入

- RoPE、RMSNorm、SwiGLU、MoE routing；
- sampling 算法和 structured output；
- AWQ/GPTQ/FP8/NVFP4 等量化格式细节；
- Mamba、MLA、sliding-window/local attention；
- RLHF 权重热更新和 disaggregated serving。

### 暂时不必优先

- 完整训练反向传播和优化器推导；
- 每一种模型架构；
- 每一种量化格式的 kernel；
- 一开始就手写高性能 CUDA attention。
