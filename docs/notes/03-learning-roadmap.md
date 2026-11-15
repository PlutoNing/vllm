# 面向推理框架岗位的 10 周学习路线

## 1. 你的起点与目标

已有优势：

- Linux scheduler、memory management、NUMA、并发和资源所有权思维；
- 能读 C/C++，理解 cache、原子操作、锁、profiling；
- 对性能问题有系统视角。

主要缺口：

- tensor shape、Transformer decoder 和自回归生成；
- PyTorch execution/model API；
- GPU 执行模型、CUDA/Triton、显存层次；
- LLM serving 指标与 workload；
- TP/PP/DP/EP 和 collective；
- 量化、编译和常见模型结构。

目标不是变成模型训练研究员，而是能：

1. 从请求追到 scheduler、KV allocator、worker、kernel；
2. 建立可计算的性能/内存模型；
3. 用证据定位控制面、内存、kernel 或通信瓶颈；
4. 修改代码并用 correctness + benchmark 证明改进；
5. 把内核经验转换成推理系统语言。

建议每周投入 12–18 小时。若全职学习，可把两周压成一周，但不要跳过实验产出。

## 2. 第 0 周：环境和测量基线

学习：

- 当前 commit 与 v0.16.0 的关系；
- Python venv、PyTorch tensor 基础；
- TTFT、TPOT、ITL、token throughput。

源码：

- 本目录的 `00-version-and-features.md`
- `docs/getting_started/installation/cpu.md`
- `docs/benchmarking/README.md`

实验：

- 跑 `cpu_smoke.py`；
- 用 `ps -eLf`、`taskset -cp`、`numactl -s` 验证线程和 NUMA；
- 记录冷启动与第二次运行时间，分离下载、加载、warmup、生成。

验收产出：

- 一页环境记录：CPU、ISA、NUMA、内存、Python、torch、vLLM commit；
- 能解释 random tiny model 为什么只用于 smoke test。

## 3. 第 1 周：张量与 decoder-only Transformer

学习到“必须能手算形状”的程度：

- embedding、linear、matmul、broadcast、softmax；
- Q/K/V、causal mask、MHA/GQA/MQA；
- residual、RMSNorm、SwiGLU、RoPE；
- logits、greedy、temperature、top-k/top-p。

实验：

1. 只用 PyTorch 写一个单层、单头 causal attention；
2. 分别运行 `T=1` 和 `T=128`，打印每个 tensor shape；
3. 加入 KV cache，证明逐 token decode 与全序列计算的结果接近；
4. 用 profiler 比较 prefill 与 decode。

验收：

- 不看资料写出 attention 公式；
- 给定模型 config 算出每 token KV bytes；
- 能解释为什么缓存 K/V 而不是缓存 Q。

面试自测：

- GQA 为什么减少 KV cache？
- causal mask 在 prefill 和 decode 中分别是什么形状？
- 为什么 decode 常常是 memory-bound？

## 4. 第 2 周：从模型推理到 serving

学习：

- prefill/decode；
- static batching、dynamic batching、continuous batching；
- 排队、吞吐和尾延迟；
- chunked prefill；
- prefix cache 的适用 workload。

源码：

- `vllm/entrypoints/llm.py`
- `vllm/v1/engine/llm_engine.py`
- `vllm/v1/engine/core.py`
- `docs/configuration/optimization.md`

实验：

- 给 1、2、4、8 条不同长度 prompt；
- 改 `max_num_batched_tokens` 和 `max_num_seqs`；
- 记录输入/输出 token/s 与延迟；
- 画出一次请求的时序，不要求先理解内部每个函数。

验收：

- 能把 TTFT、TPOT、ITL 对应到具体执行阶段；
- 能说明“提高 batch”为什么可能改善 throughput 却恶化单请求 latency。

## 5. 第 3 周：V1 scheduler

源码重点：

- `vllm/v1/core/sched/scheduler.py::schedule`
- `vllm/v1/core/sched/request_queue.py`
- `vllm/v1/request.py`
- `tests/v1/core/test_scheduler.py`
- `tests/v1/core/test_scheduler_e2e.py`

实验：

1. 在 `schedule()` 记录 request id、状态、`num_computed_tokens`、本轮 token、剩余 budget；
2. 构造短 decode + 长 prefill 混合请求；
3. 比较 FCFS 与 priority；
4. 降低 token budget，观察 chunked prefill；
5. 写一个纯 scheduler 单元测试，不加载模型。

验收：

- 能逐行解释 RUNNING 为什么先于 WAITING；
- 能解释 V1 为什么说“没有独立 prefill/decode phase”；
- 能指出所有导致请求本轮不能执行的资源条件。

内核经验迁移：

- 把 token budget 看作服务资源配额，而不是 CPU time slice；
- 分析 starvation、公平性、优先级反转和 head-of-line blocking，但结论必须落回模型 workload。

## 6. 第 4 周：Paged KV cache 与 prefix cache

源码重点：

- `kv_cache_manager.py`
- `kv_cache_coordinator.py`
- `single_type_kv_cache_manager.py`
- `block_pool.py`
- `kv_cache_utils.py`
- `tests/v1/core/test_block_pool.py` 及 KV tests

实验：

1. 手画 request logical block → physical block id；
2. 打印 allocate/free 前后的 free queue 和 `ref_cnt`；
3. 同前缀请求运行两次，观察 cached token；
4. 把 KV cache 限得很小，触发 preemption/recompute；
5. 对 block pool 写 invariant 检查：
   - `ref_cnt>0` 不在 free queue；
   - 未引用 block 最多出现一次；
   - hash index 与 block hash 一致。

验收：

- 能讲清 allocate、cache、touch、free、evict 的状态转换；
- 能说出 OS paging 类比的五个成立点和五个不成立点；
- 能推算某模型、上下文和并发下的 KV 容量。

这周是最适合形成个人竞争力的部分。

## 7. 第 5 周：CPU worker、NUMA 与 attention kernel

源码重点：

- `vllm/platforms/cpu.py`
- `vllm/v1/worker/cpu_worker.py`
- `vllm/v1/worker/cpu_model_runner.py`
- `vllm/v1/attention/backends/cpu_attn.py`
- `csrc/cpu/cpu_attn.cpp`
- `csrc/cpu/utils.cpp`
- `cmake/cpu_extension.cmake`

实验：

1. 比较 1/2/4/8/16 个物理核；
2. 比较绑物理核与绑 SMT sibling；
3. 用 `perf stat` 采集 cycles、instructions、cache misses、context switches；
4. 用 `perf record` 看 top symbols；
5. 改 block size（32 的倍数）与 batch；
6. 若有多 NUMA node，做 local/remote placement 对照。

验收：

- 能从 Python op 追到 `TORCH_LIBRARY` binding 和 C++ kernel；
- 能解释 OpenMP thread affinity、PyTorch intra-op/inter-op 的关系；
- 能判断一次优化是减少指令、提高 SIMD、改善 locality，还是只改变并发。

## 8. 第 6 周：模型加载、量化和编译

学习：

- safetensors、weight shard、lazy/load-time transform；
- FP32/BF16/FP16、INT8/INT4、weight-only 与 W8A8；
- `torch.compile`、Dynamo、Inductor、graph break、fusion；
- eager 与 compile 的冷启动/稳态 trade-off。

源码：

- `vllm/model_executor/model_loader/`
- `vllm/model_executor/models/llama.py`
- `vllm/model_executor/layers/`
- `vllm/model_executor/layers/quantization/`
- `vllm/compilation/`
- `docs/design/torch_compile.md`

实验：

- 比较 eager 与 CPU Inductor 的首轮/后续运行；
- 检查编译图和 graph break；
- 用一个小线性层比较 FP32/BF16/INT8 的内存和误差；
- 解释为何 vLLM 在模型初始化时就做 shard/quantization，而不是加载完整权重后再切。

验收：

- 能区分“权重更小”“算得更快”“KV 更小”是三个不同目标；
- 能解释 compile cache 和动态 shape 为什么重要。

## 9. 第 7 周：GPU 与高性能 kernel 必修补课

即使当前只用 CPU，求职生产推理框架岗位仍必须补：

- thread/block/warp/SM；
- global/shared/register/L2/HBM；
- coalescing、occupancy、warp divergence；
- GEMM tiling、Tensor Core；
- FlashAttention 的 IO-aware 思想；
- CUDA stream/event、async copy；
- CUDA Graph；
- Triton program model；
- Nsight Systems/Compute。

源码阅读：

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/v1/attention/ops/paged_attn.py`
- `csrc/attention/`
- `vllm/vllm_flash_attn/`

无 GPU 时：

- 写小 CUDA/Triton 代码可放到有 GPU 的云环境；
- 当前机器继续用 CPU 建立算法正确性和 memory model；
- 不要声称 CPU benchmark 能代表 GPU serving。

验收：

- 能解释 prefill 为什么适合大 GEMM、decode 为什么更容易带宽受限；
- 能读懂 kernel launch shape、stride 和 memory layout；
- 至少完成一次 Nsight timeline 分析。

## 10. 第 8 周：分布式推理

学习：

- TP：层内切权重，频繁 collective；
- PP：按层切，pipeline bubble；
- DP：复制模型，提高总吞吐；
- EP：MoE expert 分片与 all-to-all；
- context parallel；
- P/D disaggregation 和 KV transfer。

源码：

- `vllm/distributed/parallel_state.py`
- `vllm/distributed/device_communicators/`
- `vllm/v1/executor/`
- `vllm/distributed/kv_transfer/`
- `vllm/distributed/eplb/`

实验：

- CPU 多进程用 Gloo 理解 rank/group；
- 写最小 all-reduce/all-to-all；
- 算 TP 每层通信量；
- 画 API server、EngineCore、worker、DP coordinator 的进程和通信图。

验收：

- 给定模型和机器拓扑，能说明为什么选 TP/PP/DP/EP；
- 能解释 NUMA、PCIe、NVLink、RDMA 对策略的约束。

## 11. 第 9 周：在线服务、可观测性与稳定性

学习：

- OpenAI API、streaming、backpressure、取消；
- admission control、SLO、goodput；
- metrics、tracing、故障恢复；
- 多租户、公平性和 prefix-cache 隔离。

源码：

- `vllm/entrypoints/openai/`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/metrics/`
- `vllm/tracing/`
- `docs/usage/metrics.md`

实验：

- 启服务并并发压测；
- 采集 TTFT/TPOT/queue time/KV usage/preemption；
- 模拟客户端取消和超长 prompt；
- 用 flame graph/trace 解释一次 tail latency。

验收：

- 优化报告包含正确性、均值、P95/P99、资源利用率和回归风险；
- 能区分模型执行慢、排队慢和输出处理慢。

## 12. 第 10 周：作品与面试闭环

推荐完成一个主项目和一个小修复。

### 主项目候选

1. **Scheduler/KV 可视化与 invariant checker**
   - 每 step 输出 request、token budget、block owner、free queue；
   - 可重放一段 trace；
   - 检测泄漏、双重释放和不公平。
2. **CPU NUMA/亲和性研究**
   - 自动生成绑核方案；
   - 对比物理核/SMT/local/remote memory；
   - 输出可复现实验和 flame graph。
3. **KV 容量与 SLO 模拟器**
   - 输入模型 config、请求长度分布和内存；
   - 估算并发、preemption 和 prefix-cache 收益；
   - 用实际 vLLM trace 校准。

### 小修复候选

- 调度器测试；
- CPU ISA 构建/报错改进；
- 指标或 profiling instrumentation；
- 文档与代码版本不一致；
- block pool invariant/fuzz test。

### 一份合格项目报告必须有

- 问题和 workload；
- 代码路径；
- hypothesis；
- correctness oracle；
- benchmark 环境；
- before/after；
- P50/P95/P99 或吞吐/TTFT/TPOT；
- CPU/GPU/内存/通信证据；
- trade-off 和回归测试。

## 13. 求职能力检查表

达到“可投递推理框架/系统优化岗位”：

- [ ] 15 分钟白板讲清 decoder、prefill、decode、KV cache；
- [ ] 根据 config 估权重和 KV 内存；
- [ ] 从 `LLM.generate` 追到一个 attention custom op；
- [ ] 解释 V1 scheduler 的 token budget 和 preemption；
- [ ] 解释 Paged KV 与 OS paging 的边界；
- [ ] 熟练 PyTorch profiler、perf、Nsight 至少各一种；
- [ ] 理解 CUDA memory hierarchy 和 FlashAttention；
- [ ] 能算 TP collective 代价并选择并行策略；
- [ ] 有一份可复现、带正确性和性能证据的项目报告；
- [ ] 能读英文 RFC、issue、PR 并定位回归。

如果只完成 CPU 路径，可胜任“框架控制面/CPU backend/调度内存”入门讨论，但多数 GPU 推理岗位仍会追问 CUDA、Triton、NCCL 和 GPU profiling，不能省略第 7–8 周。
