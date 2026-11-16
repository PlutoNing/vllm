from typing import Optional

from torch import nn

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig)
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)

""" 谁负责加载权重？是 vLLM 和 PyTorch 分工完成：
vLLM：
选择模型架构，例如 LlamaForCausalLM；
选择 checkpoint 格式；
遍历 safetensors/bin 权重；
完成权重名称映射；
合并 Q/K/V、gate/up 等参数；
根据 TP/PP 做切分；
处理量化和 LoRA。
PyTorch：
提供 nn.Module、Parameter、Tensor；
分配 CPU/GPU tensor；
执行 CPU→GPU 数据复制；
管理 CUDA allocator 和 device。
默认 loader 的核心过程：


初始化 vLLM 的 Llama 模型对象
→ 遍历 checkpoint 权重
→ model.load_weights(...)
→ 把 checkpoint tensor 写入模型 Parameter
→ model.eval()
因此不能简单说“PyTorch 加载模型，vLLM 只管理缓存”。

更准确的是：
vLLM 决定模型如何构造、权重如何映射和切分；PyTorch提供 tensor、显存分配和数据复制能力。 """

""" 模型是何时加载的,不是在每次 execute_model() 时加载，而是 vLLM Server 启动时加载一次。
调用关系：GPUExecutor._init_executor()
→ Worker.init_device()
→ Worker.load_model()
→ ModelRunner.load_model()
→ get_model()
→ ModelLoader.load_model() """
def get_model(*, model_config: ModelConfig, load_config: LoadConfig,
              device_config: DeviceConfig, parallel_config: ParallelConfig,
              scheduler_config: SchedulerConfig,
              lora_config: Optional[LoRAConfig],
              cache_config: CacheConfig) -> nn.Module:
    loader = get_model_loader(load_config)
    return loader.load_model(model_config=model_config,
                             device_config=device_config,
                             lora_config=lora_config,
                             parallel_config=parallel_config,
                             scheduler_config=scheduler_config,
                             cache_config=cache_config)


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
