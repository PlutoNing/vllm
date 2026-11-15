# 源码目录与调用链导读

## 1. 顶层目录

| 目录 | 作用 | 学习优先级 |
|---|---|---:|
| `.buildkite/` | CI、硬件测试、镜像和性能 benchmark 流水线。可学习真实生产回归矩阵。 | 后期 |
| `.gemini/` | 仓库辅助开发配置，不属于推理运行时。 | 跳过 |
| `.github/` | GitHub workflow、issue/PR 模板、维护配置。 | 后期 |
| `benchmarks/` | 两万行，serving、latency、throughput、kernel、attention 等 benchmark。 | 高 |
| `cmake/` | 1k 行，C++/CUDA/ROCm/CPU 扩展的构建逻辑和外部依赖。 | 中高 |
| `csrc/` | 7w 行，CPU、CUDA、ROCm 自定义 kernel 与 PyTorch C++ binding。 | 高 |
| `docker/` | 2k 行，各硬件后端的构建和发布镜像。 | 中 |
| `docs/` | 2w 行，版本对应的官方设计、安装、功能和部署文档。 | 高 |
| `examples/` | 25k 行，offline、online、pooling 和集成示例。 | 高 |
| `requirements/` | 1k 行，通用及 CUDA/ROCm/CPU/XPU/TPU 的依赖锁定。 | 中 |
| `tests/` | 16w 行，单元、端到端、模型、调度、KV、kernel 与硬件测试；理解行为的第二份规范。 | 高 |
| `tools/` | 4k 行，代码生成、profiling、CI、release 和开发工具。 | 中低 |
| `vllm/` | 42w 行，Python 主包和部分 Python/Triton kernel。 | 最高 |

根目录的 `CMakeLists.txt`、`setup.py`、`pyproject.toml` 决定平台探测、依赖和扩展构建，也值得在 CPU 实验时一起读。

## 2. `vllm/` 一级目录，一共 42w 行

| 目录 | 作用 |
|---|---|
| `assets/` | 200 line, 包内运行时静态资源。 |
| `benchmarks/` | 8k line, `vllm bench` 的 Python 实现、数据集和 sweep 逻辑；顶层 `benchmarks/` 更多是脚本与专项基准。 |
| `compilation/` | 6k line, `torch.compile` 后端、图切分、自定义 pass、fusion 和 compile cache。 |
| `config/` | 5k line, `VllmConfig` 的模型、cache、scheduler、parallel、compilation、observability 等拆分配置。 |
| `device_allocator/` | 171 line, 设备内存 allocator 抽象及 CUDA virtual-memory allocator 等。 |
| `distributed/` | 1.8w line, parallel state、collective、device communicator、EP load balancing、KV/encoder/weight transfer。 |
| `engine/` | 1955 line, `EngineArgs`、协议和兼容层；不要误认为这是已删除的 V0 主引擎。 |
| `entrypoints/` | 21 k line, `LLM`、CLI、OpenAI/Anthropic/MCP、pooling、SageMaker 和各种 serve 路由。 |
| `grpc/` | 171 line, Engine gRPC 协议与生成代码。 |
| `inputs/` | 500 line, prompt/embedding/多模态输入的数据结构、解析和预处理。 |
| `kernels/` | 400 line, 新 kernel framework，例如 Helion。大量成熟 kernel 仍位于 `csrc/`、Triton 或 layer 子目录。 |
| `logging_utils/` | 251 line, 日志格式、上下文和配置。 |
| `lora/` | 6k line, LoRA layer、Punica wrapper、Triton op 和 adapter 管理。 |
| `model_executor/` | 25w line, 模型加载、模型 registry/实现、通用 layer、量化、MoE、RoPE、warmup；数据面主体。 |
| `multimodal/` | 5k line, 图像/音频/视频输入、media loader、processor、cache 与 registry。 |
| `parser/` | 532 line, 流式/结构内容等通用解析辅助。 |
| `platforms/` | 2k line, CUDA、ROCm、CPU、XPU、TPU 等平台发现、能力和配置修正。 |
| `plugins/` | 283 line, 通用 plugin、IO processor 和 LoRA resolver。 |
| `profiler/` | 600 line, torch profiler 等统一封装。 |
| `ray/` | 66 line, Ray 集成辅助。 |
| `reasoning/` | 2k line, 各模型 reasoning/thinking 输出 parser。 |
| `renderers/` | 1800 line, chat template、输入 tokenize/preprocess 和模型格式渲染。 |
| `third_party/` | 5k line, 内嵌或适配的第三方实现。 |
| `tokenizers/` | 2k line, HF、Mistral 等 tokenizer 协议与实现。 |
| `tool_parsers/` | 11k line, 模型特定的 function/tool-call parser。 |
| `tracing/` | 349 line, OpenTelemetry 和内部 tracing。 |
| `transformers_utils/` | 7k line, Hugging Face config、processor、chat template 和兼容层。 |
| `triton_utils/` | 90 line, Triton 可用性、编译和工具函数。 |
| `usage/` | 199 line, 匿名 usage telemetry。 |
| `utils/` | 4k line, 被拆分后的通用工具；旧资料常引用单个巨大的 `vllm/utils.py`。 |
| `v1/` | 5w line, 当前唯一引擎的 scheduler、KV cache、engine、executor、worker、attention、sampler 等控制面与执行面。 |
| `vllm_flash_attn/` | 随包构建/分发的 FlashAttention Python 与扩展接口。 |

## 3. `vllm/v1/` 要怎样看,5w line

| 子目录 | 重点 |
|---|---|
| `engine/` | 请求进入、进程间 client、EngineCore 主循环、输入/输出处理。 |
| `core/` | scheduler、KV cache manager/coordinator/block pool、encoder cache。 |
| `executor/` | uni-process、multiprocess、Ray 和 external launcher，负责把一次执行广播给 worker。 |
| `worker/` | worker 生命周期、model runner、input batch、block table、执行和采样。 |
| `attention/` | backend 接口、选择、metadata、CPU/CUDA/ROCm/Triton/FlashInfer 等实现。 |
| `sample/` | sampling、logprob 和 logits processor。 |
| `spec_decode/` | draft、EAGLE 等 speculative decoding。 |
| `structured_output/` | grammar backend 和结构化输出。 |
| `kv_offload/` | KV offload manager/worker。 |
| `metrics/` | V1 统计和 Prometheus logger。 |
| `pool/` | pooling model 的执行逻辑。 |

## 4. 一条离线请求的真实主链

### 4.1 入口与配置

1. `vllm/entrypoints/llm.py::LLM.__init__`
2. 构造 `vllm/engine/arg_utils.py::EngineArgs`
3. `EngineArgs.create_engine_config()` 产生 `VllmConfig`
4. `vllm/v1/engine/llm_engine.py::LLMEngine.from_engine_args`
5. `Executor.get_class()` 根据 backend 选择 `UniProcExecutor`、`MultiprocExecutor` 或 Ray

`VLLM_ENABLE_V1_MULTIPROCESSING` 在该版本默认是 `1`：离线 `LLM` 通常由
`SyncMPClient` 连接独立的 `EngineCoreProc`；设为 `0` 才使用同进程
`InprocClient`。本机 smoke test 日志中的 `EngineCore_DP0 pid=...` 就是直接证据。
这里的 EngineCore 是否独立进程，与 EngineCore 内部选择 uni-process 或
multi-process worker executor 是两个不同层次，不能混为一谈。

CPU platform 在 `vllm/platforms/cpu.py::CpuPlatform.check_and_update_config()` 中进一步改写：

- block size 默认 128；
- async scheduling 关闭；
- worker class 设为 `CPUWorker`；
- distributed executor 必要时回退到 `mp`；
- CUDA graph capture size 清空；
- compilation backend 调整为 CPU Inductor；
- 设置 OpenMP/NUMA 相关环境。

### 4.2 请求进入

1. `LLM.generate()` 规范化 prompts 与 `SamplingParams`；
2. `LLM._add_request()`；
3. `LLMEngine.add_request()`；
4. `InputProcessor` 完成输入验证和 token 化后的请求转换；
5. `EngineCoreClient.add_request()` 把 `EngineCoreRequest` 送入 core；
6. `EngineCore.add_request()` 构造 V1 `Request`；
7. `Scheduler.add_request()` 放入 waiting queue。

### 4.3 每个 engine step

核心循环在 `vllm/v1/engine/core.py::EngineCore.step()`：

```text
scheduler.schedule()
  -> model_executor.execute_model(scheduler_output)
  -> worker.execute_model()
  -> model_runner.execute_model()
  -> 若 forward 返回 None，再单独 model_executor.sample_tokens()
  -> scheduler.update_from_output()
  -> EngineCoreOutputs
```

当前 `EngineCore.step()` 会先非阻塞提交 `execute_model`，等待 future；某些
model runner 把 forward 的临时状态保存在 `execute_model_state` 并返回
`None`，此时 core 再调用 `sample_tokens()`。因此“一次 forward 调用直接
返回最终 token”不是可靠的 V1 心智模型。

`LLMEngine.step()` 经 `OutputProcessor` 做 detokenize、停止条件、流式/最终输出构造，`LLM.generate()` 直到所有请求完成。

## 5. 调度与 KV cache 主链

### 5.1 Scheduler

重点读 `vllm/v1/core/sched/scheduler.py::Scheduler.schedule()`：

1. 设置 `token_budget=max_num_batched_tokens`；
2. 先遍历 RUNNING 请求；
3. 计算 `num_tokens_with_spec - num_computed_tokens`；
4. 应用 long-prefill、模型长度、encoder 和 token budget 限制；
5. `KVCacheManager.allocate_slots()`；
6. block 不足时 `_preempt_request()`，释放 KV 并把请求放回 waiting；
7. 再处理 WAITING 请求，查 prefix cache 和 KV connector；
8. 构造 `SchedulerOutput`。

请求队列与 policy 在 `vllm/v1/core/sched/`。当前本地注释调试点就位于这个函数开头。

### 5.2 KV cache

推荐按接口从上往下：

1. `kv_cache_manager.py::KVCacheManager`
   - `get_computed_blocks()` 查 prefix hit；
   - `allocate_slots()` 做容量检查、分配和 cache commit；
   - `free()` 释放请求拥有的 block。
2. `kv_cache_coordinator.py`
   - 无 prefix cache、单一 group、hybrid group 的协调策略。
3. `single_type_kv_cache_manager.py`
   - full、sliding window、chunked local、Mamba、cross attention 的不同语义。
4. `block_pool.py::BlockPool`
   - `blocks`、`free_block_queue`、`cached_block_hash_to_block`；
   - allocate、touch、free、evict；
   - `ref_cnt==0` 代表 block 可在 free/eviction queue 中。
   - 相同 hash 不立即去重，以保持已分配 block id 不变、block table 可 append-only。
5. `kv_cache_utils.py::KVCacheBlock`
   - block id、引用计数、hash、双向链表节点；
   - 请求 block hash 的链式生成。

先画清所有权和引用计数，再看优化；这是内核开发者最容易发挥优势的部分。

## 6. CPU 执行链

### 6.1 Worker 和线程/NUMA

`vllm/v1/worker/cpu_worker.py::CPUWorker`：

- `init_device()` 根据 `VLLM_CPU_OMP_THREADS_BIND` 选核；
- x86 SMT 默认每物理核只选一个逻辑 CPU；
- 每个 rank 选择自己的 NUMA node；
- 调用 `torch.ops._C_utils.init_cpu_threads_env()` 设 OpenMP affinity 和 memory policy；
- 用 Gloo 初始化 distributed environment；
- 创建 `CPUModelRunner`。

对应 C++ 线程环境实现在 `csrc/cpu/utils.cpp`。

### 6.2 CPUModelRunner

`vllm/v1/worker/cpu_model_runner.py::CPUModelRunner` 继承 `GPUModelRunner`，这是复用通用 batch/model 执行逻辑，不表示 CPU 偷偷使用 GPU。

CPU 子类主要做：

- 用占位 Event/Stream 兼容父类初始化；
- 把 `CpuGpuBuffer.gpu` 指向同一个 CPU tensor；
- 禁用 CUDA Graph 与 cascade attention；
- 断言不使用 speculative decoding；
- 用 `get_model()` 加载标准 vLLM model；
- warmup CPU compilation。

大量真实执行逻辑仍要回到 `vllm/v1/worker/gpu_model_runner.py` 看 `execute_model()`、input batch 更新、block table、forward、logits 和 sampler。

### 6.3 模型

以 tiny Llama 为例：

1. HF config 给出 architecture `LlamaForCausalLM`；
2. `vllm/model_executor/models/registry.py` 解析为 vLLM 实现；
3. `model_loader/` 创建模型并加载 safetensors；
4. `model_executor/models/llama.py` 组织 embedding、decoder layers 和 LM head；
5. 通用 layer 位于 `model_executor/layers/`。

不要一开始遍历几百个 model 文件。先只看 Llama，再看 Qwen3，最后才看 MoE 或多模态。

### 6.4 CPU attention

Python 入口：`vllm/v1/attention/backends/cpu_attn.py`

- metadata 包含 `query_start_loc`、`seq_lens`、`block_table`、`slot_mapping`；
- `cpu_attn_reshape_and_cache()` 把本轮 K/V 写入物理 cache slot；
- `cpu_attention_with_kv_cache()` 根据 block table 读取历史 KV；
- 特定场景的 prefill 可走 PyTorch SDPA；
- ISA 选择包括 vec/vec16、AMX、NEON。

C++ 路径：

```text
vllm/_custom_ops.py
  -> csrc/cpu/torch_bindings.cpp
  -> csrc/cpu/cpu_attn.cpp
  -> csrc/cpu/cpu_attn.hpp / cpu_attn_impl.hpp
```

`csrc/cpu/generate_cpu_attn_dispatch.py` 生成 head size/ISA dispatch。构建选择在 `cmake/cpu_extension.cmake`。

## 7. 在线服务多进程架构

典型 V1 服务不是一个 Python 进程：

- API server：HTTP、tokenization、多模态加载、streaming；
- EngineCore：scheduler、KV manager、请求状态机；
- Worker：模型权重、forward、device memory；
- DP>1 时还有 coordinator。

进程间主要通过 ZMQ/共享内存/collective 通信。EngineCore 有 busy loop，对 CPU starvation 敏感。GPU 利用率低不一定是 GPU kernel 慢，也可能是 API/core/worker 的 CPU 配额不足。

重点文件：

- `vllm/entrypoints/openai/api_server.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/core.py::EngineCoreProc`
- `vllm/v1/executor/multiproc_executor.py`

## 8. 推荐源码阅读顺序

1. `docs/design/arch_overview.md`
2. `docs/usage/v1_guide.md`
3. `docs/configuration/optimization.md`
4. `vllm/entrypoints/llm.py`
5. `vllm/v1/engine/llm_engine.py`
6. `vllm/v1/engine/core.py::EngineCore.step`
7. `vllm/v1/core/sched/scheduler.py::schedule`
8. `vllm/v1/core/kv_cache_manager.py`
9. `vllm/v1/core/block_pool.py`
10. `tests/v1/core/test_scheduler.py` 与 KV 相关 tests
11. `vllm/v1/worker/cpu_worker.py`
12. `vllm/v1/worker/cpu_model_runner.py`
13. `vllm/v1/attention/backends/cpu_attn.py`
14. `csrc/cpu/cpu_attn.cpp`
15. `vllm/model_executor/models/llama.py`
16. GPU 可用后再读 FlashAttention/Triton/CUDA、CUDA Graph 和 distributed communicator。

每读一层，都回答四个问题：输入/输出是什么、谁拥有状态、内存在哪里分配/释放、失败或资源不足时怎样恢复。
