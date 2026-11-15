# vLLM v0.16 学习笔记索引

这组笔记面向有 Linux 内核调度、内存管理、NUMA 和性能分析背景，但尚未系统学习 Transformer 的读者。所有源码结论都以本仓库当前分支为准。

## 先记住四个结论

1. 当前 `HEAD=60d3e7e513`，源码基线是 **vLLM v0.16.0**；HEAD 只比 `v0.16.0` 标签多一行被注释掉的调试打印，不改变行为。
2. vLLM 不只是“在一个大矩阵上计算”。矩阵乘是主要算力来源，但推理系统的核心难题是动态请求调度、KV cache 分配、批处理、内核选择、跨设备通信和服务质量。
3. 你的内核背景与 vLLM 非常匹配：PagedAttention、块分配、引用计数、前缀共享、抢占重算、NUMA 亲和性都能借用 OS 直觉；但这些只是类比，不能把应用层 block 当成硬件页表或 page fault。
4. 本机已经用 AVX2 CPU 后端跑通 `hmellor/tiny-random-LlamaForCausalLM`。这是随机权重测试模型，输出乱码正常；它验证的是完整执行链路。

## 阅读顺序

1. [00-version-and-features.md](00-version-and-features.md)：当前 commit、版本、特性、版本演进。
2. [01-transformer-and-vllm-concepts.md](01-transformer-and-vllm-concepts.md)：Transformer 最小知识集、内存模型、OS 类比边界。
3. [02-source-tour.md](02-source-tour.md)：每个主要目录的作用和 V1/CPU 调用链。
4. [03-learning-roadmap.md](03-learning-roadmap.md)：10 周学习与求职路线、验收标准。
5. [04-cpu-lab.md](04-cpu-lab.md)：本机可复现的构建、运行、分析实验。
6. [05-reproducible-environment.md](05-reproducible-environment.md)：`.venv` 的位置、AVX512 CPU/GPU 重建和版本锁定原则。
7. [setup_cpu_avx512_env.sh](setup_cpu_avx512_env.sh)：新 AVX512 x86 CPU 机器的环境创建脚本。
8. [cpu_smoke.py](cpu_smoke.py)：已经验证过的最小 CPU 推理程序。

## 第一周直接照做

```bash
cd /data/home/paulning/study/vllm
source .venv/bin/activate
VLLM_TARGET_DEVICE=cpu \
VLLM_CPU_KVCACHE_SPACE=1 \
VLLM_CPU_OMP_THREADS_BIND=0-7 \
python docs/notes/cpu_smoke.py
```

然后按这个顺序打断点或加日志：

```text
LLM.generate
  -> LLMEngine.add_request / step
  -> EngineCore.step
  -> Scheduler.schedule
  -> KVCacheManager.allocate_slots
  -> CPUWorker.execute_model
  -> CPUModelRunner.execute_model（继承 GPUModelRunner）
  -> CPUAttentionBackendImpl.forward
  -> csrc/cpu/cpu_attn.cpp
  -> sampler / OutputProcessor
```

不要从上千个模型实现或 CUDA kernel 开始。先能解释一条请求怎样走完上面的链路，再深入具体内核。
