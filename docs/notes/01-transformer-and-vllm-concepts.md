# Transformer 与 vLLM 的最小概念集

## 1. 它不是“只算一个很大的矩阵”

更准确的描述是：

> LLM 推理反复执行由矩阵乘、attention、归一化、激活、通信和采样组成的计算图；
vLLM 负责把到达时间、输入长度和输出长度都不确定的请求，动态映射到有限算力和 KV cache 上。

Decoder-only Transformer 会给每个 token 维护一个长度为 hidden_size 的状态向量。
每一层先用 Attention 让当前 token 读取前面 token 的信息，再用 MLP 对当前 token 的
状态做非线性变换。最后把最后一个 token 的状态映射成词表中 32000 个 token 的分数，选出
下一个 token。
LLM 不改变这套数学过程。它主要负责把多个请求组织成 batch，并管理每层历史 token 的 K/V。
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
这个测试模型的具体参数
cpu_smoke.py 使用：

模型：hmellor/tiny-random-LlamaForCausalLM
词表大小 V = 32000
Decoder 层数 L = 2
Hidden size D = 16
Attention head 数 H = 4
每个 head 的维度 Dh = 64
KV head 数 Hkv = 4
MLP 中间维度 I = 64
运行时数据类型：FP32
最多生成 8 个 token
temperature=0，即 greedy decoding
这里有个重要例外：你文档中写的


Dh = D / H
是普通模型最常见的关系，但这个随机测试模型不是这样。

它明确配置了：


D = 16
H = 4
Dh = 64
所以：


H × Dh = 4 × 64 = 256
每个 token 的主状态仍然只有 16 个数，但生成 Q/K/V 时会临时投影到 256 维。这是一个为了测试代码路径而设计的非典型小模型，不要把它当成正常 Llama 的典型比例。

2. Tokenizer：文本变成整数 ID
两个 prompt 实际被分成：


"The Linux scheduler"
[1, 450, 8074, 1364, 14952]
对应：
["<s>", "▁The", "▁Linux", "▁sch", "eduler"]
以及：


"Paged virtual memory"
[1, 349, 4063, 6901, 3370]
对应：
["<s>", "▁P", "aged", "▁virtual", "▁memory"]
其中：

<s> 是起始 token；
▁ 表示前面有空格；
一个单词不一定对应一个 token，例如 scheduler 被拆成 ▁sch 和 eduler。
每个 prompt 的 token ID 是一个一维整数向量 [5]。严格来说它不是“一阶矩阵”，而是“一阶张量/向量”。

因为这里正好有两个长度相同的 prompt，可以叠成：


input_ids shape = [B, T] = [2, 5]
含义是：

B=2：batch 中有两个独立序列；
T=5：每条序列有 5 个 prompt token。
两个序列之间不会互相做 Attention。batch 只是把两份独立计算放在一起执行。

这里总 prompt token 数是：
2 × 5 = 10
max_num_batched_tokens=64 只是上限，不表示一定计算 64 个 token。这两个 prompt 很短，10 个 token 一轮就能完成 prefill。

另外，程序最终打印的 generated.token_ids 是生成的 8 个 token，不是上面的 prompt token ID。


1. tokenizer：文本变成离散 `token_id`。

2. embedding：`[B,T] -> [B,T,D]`。
Embedding：整数 ID 变成浮点向量
模型有一个 embedding 权重矩阵：


E shape = [32000, 16]
词表中的每个 token 都对应这个矩阵中的一行。

例如 token 450 的 embedding 是：


E[450, :]    # 16 个浮点数
所以：


input_ids: [2, 5]
    ↓ embedding lookup
X₀:        [2, 5, 16]
总共有：


2 × 5 × 16 = 160
个 FP32 数。

数学上，也可以把 token ID 表示成长度 32000 的 one-hot 向量，然后计算：


one_hot(token_id) × E
但实际不会真的构造 one-hot，而是直接查表。

需要注意：

token ID 本身的数值大小没有数学意义；
ID 14952 并不比 ID 450“更大”或“更重要”；
真正进入 Transformer 计算的是对应的 16 维 embedding。



3. 每个 decoder layer：
   - normalization；
   - 线性投影产生 Q、K、V；
   - attention；
   - output projection；
   - residual；
   - MLP（常见 SwiGLU）；
   - residual。
每个 Decoder Layer 到底算什么
这个模型有两个完全相同结构的 Decoder Layer。每层包含：

RMSNorm
Self-Attention
Residual connection
RMSNorm
SwiGLU MLP
Residual connection
下面以第一层为例。

4.1 第一次 RMSNorm
输入：


X shape = [2, 5, 16]
对每个 token 的 16 维向量独立归一化：


rms(x) = sqrt((x₁² + x₂² + ... + x₁₆²) / 16 + ε)
RMSNorm(x) = γ ⊙ x / rms(x)
其中：

γ 是模型学习到的 16 维权重；
⊙ 是逐元素乘法；
RMSNorm 不像 LayerNorm 那样减去均值。
输出形状不变：


X_norm shape = [2, 5, 16]











4. 最后一层 hidden state 经 LM head 变成 vocabulary logits。
5. sampler 应用 temperature、top-k/top-p、penalty 等规则，选择下一个 token。
6. 把新 token 追加回序列，重复步骤 3–5。
计算 Q、K、V
这个模型第一层的权重形状是：


Wq: [256, 16]
Wk: [256, 16]
Wv: [256, 16]
权重文件用 [输出维度, 输入维度] 存储，所以按行向量写矩阵运算是：


Q = X_norm × Wqᵀ
K = X_norm × Wkᵀ
V = X_norm × Wvᵀ
把 B 和 T 临时展平，可以理解成：


[10, 16] × [16, 256] = [10, 256]
因此：


Q、K、V shape = [2, 5, 256]
再把 256 维拆成 4 个 head：


[2, 5, 256]
    ↓ reshape
[2, 4, 5, 64]
分别表示：


[B, H, T, Dh]
也就是每个序列、每个 head、每个 token 都有：

一个 64 维 Query；
一个 64 维 Key；
一个 64 维 Value。
Q、K、V 是每一层重新计算的。第一层的 K/V 和第二层的 K/V 是不同的数据。

4.3 RoPE：加入位置信息
单纯的 embedding 不知道 token 在第几个位置。

Llama 对 Q 和 K 使用 RoPE。它把向量中的数字两两配对，根据 token 位置旋转：


q'₂ᵢ   = q₂ᵢ cos(pθᵢ) - q₂ᵢ₊₁ sin(pθᵢ)
q'₂ᵢ₊₁ = q₂ᵢ sin(pθᵢ) + q₂ᵢ₊₁ cos(pθᵢ)
其中 p 是位置：


<s>       p=0
The/P     p=1
Linux/... p=2
...
K 做同样的旋转，V 不做。

RoPE 的作用是让 Q·K 的结果包含相对位置信息。

4.4 Attention 分数
对每个 head，Q 和 K 的形状都是：


Qh: [5, 64]
Kh: [5, 64]
计算：


S = Qh × Khᵀ / sqrt(64)
得到：


S shape = [5, 5]
其中：


S[t, s]
表示第 t 个 token 对第 s 个 token 的关注分数。

四个 head、两个序列合起来：


scores shape = [2, 4, 5, 5]
4.5 Causal Mask
这是自回归模型，当前位置不能看到未来 token。

因此分数矩阵会被遮成下三角：


位置 0：只能看位置 0
位置 1：可以看位置 0、1
位置 2：可以看位置 0、1、2
...
位置 4：可以看位置 0、1、2、3、4
未来位置的分数设成负无穷：


S[t,s] = -∞，当 s > t
再执行：


A = softmax(S)
每一行会变成总和为 1 的权重。

这说明 prefill 虽然把 5 个 token 并行计算了，但 causal mask 保证它在数学上仍然遵守“不能偷看未来”。

4.6 用 Attention 权重混合 V
对一个 head：


O_head = A × V
形状：


[5, 5] × [5, 64] = [5, 64]
展开来看，当前位置 t 的输出是：


O[t] = A[t,0]V[0] + A[t,1]V[1] + ... + A[t,t]V[t]
也就是把自己和前面 token 的 Value 向量加权求和。

四个 head 得到：


[2, 4, 5, 64]
拼起来：


O_concat shape = [2, 5, 256]
然后经过输出投影：


Wo shape = [16, 256]
O_projected = O_concat × Woᵀ
得到：


[2, 5, 16]
这一步把 4 个 head 的 256 维结果压回主状态的 16 维。

4.7 第一次残差连接
Attention 输出和进入这一子层之前的状态相加：


X_attn = X + O_projected
形状仍然是：


[2, 5, 16]
Residual 的作用是保留旧信息，并让每层只需要学习“在原状态上增加什么”。

4.8 SwiGLU MLP
先做第二次 RMSNorm：


Y = RMSNorm(X_attn)
然后有两条并行投影：


gate = Y × Wgateᵀ
up   = Y × Wupᵀ
权重形状：


Wgate: [64, 16]
Wup:   [64, 16]
所以：


gate shape = [2, 5, 64]
up shape   = [2, 5, 64]
接着：


SiLU(z) = z × sigmoid(z)
middle = SiLU(gate) ⊙ up
然后通过下投影：


Wdown: [16, 64]
mlp_output = middle × Wdownᵀ
得到：


mlp_output shape = [2, 5, 16]
再做残差：


X_out = X_attn + mlp_output
这就是一个 Decoder Layer 的完整输出。

可以把两部分理解为：

Attention：不同 token 位置之间交换信息；
MLP：每个 token 独立地变换自己的特征。
5. 第二个 Decoder Layer
第二层再次执行完全相同的数学步骤，但使用自己的权重：


第一层输出 [2,5,16]
    ↓ 第二层 RMSNorm
    ↓ 第二层 Q/K/V
    ↓ 第二层 Attention
    ↓ 第二层 MLP
第二层输出 [2,5,16]
第二层有自己独立的 Q/K/V 权重，也有自己独立的 KV cache。

在两层处理之后，最后一个位置的 16 维向量已经通过 Attention 汇集了整个 prompt 的信息。

6. 从最后状态得到下一个 token
两层之后还有一次最终 RMSNorm：


H shape = [2, 5, 16]
逻辑上可以对所有位置计算 logits：


lm_head weight shape = [32000, 16]
logits = H × Wlmᵀ
完整形状是：


[2, 5, 16] × [16, 32000]
    =
[2, 5, 32000]
但生成下一个 token 时，只需要每条序列的最后一个位置：


H_last shape = [2, 16]
next_logits shape = [2, 32000]
每条 prompt 得到 32000 个实数：


z₀, z₁, z₂, ..., z₃₁₉₉₉
每个数代表对应 token 作为下一个 token 的分数。

因为代码设置了：


temperature = 0
它不会随机采样，而是直接：


next_token_id = argmax(logits)
temperature=0 是 greedy 的特殊配置，并不是实际去计算 logits / 0。

第一次 prefill 后：


"The Linux scheduler"
    → argmax = 22365
    → token piece = "▁Communic"
"Paged virtual memory"
    → argmax = 29256
    → token piece = "stab"
这就是第一个生成 token。

由于模型权重是随机的，这些选择在数学上是确定的，但没有正常语言意义。

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
Prefill 到底是什么
Prefill 不是另一套 Transformer 算法。

它只是：

第一次把整个 prompt 的所有 token 送过所有 Decoder Layer，同时为每层生成并保存所有 prompt token 的 K/V。

本例中：


B = 2
每条 T = 5
本轮总 token = 10
Prefill 的主要中间形状是：


input_ids             [2, 5]
embedding             [2, 5, 16]
每层 Q/K/V            [2, 4, 5, 64]
每层 attention scores [2, 4, 5, 5]
每层 attention output [2, 5, 256]
每层投影后状态       [2, 5, 16]
每层 MLP 中间状态     [2, 5, 64]
最终 last hidden      [2, 16]
最终 logits           [2, 32000]
选择结果               [2]
8. Decode 为什么变成一次一个 token
生成第一个 token 后，序列变成：


<s> The Linux sch eduler Communic
下一轮只需要计算新 token 22365。

它先被 embedding 成：


[16]
第一层为这个新 token 计算：


Q_new [4, 1, 64]
K_new [4, 1, 64]
V_new [4, 1, 64]
但是新 Query 要和历史所有 Key 做点积：


历史 5 个 K + 新 token 的 K = 6 个 K
Q_new × K_allᵀ
[1,64] × [64,6] = [1,6]
所以新 token 可以读取前面 5 个 prompt token以及自己。

计算完第一层后，再进入第二层，使用第二层自己的历史 K/V。

最后产生新的 32000 个 logits，并选出第二个生成 token：


"The Linux scheduler":
22365 → 25218 → 27312 → ...
"Paged virtual memory":
29256 → 20219 → 25498 → ...
如此重复，直到生成 8 个 token或遇到停止 token。

这次实际输出是：


第一条：
[22365, 25218, 27312, 2218, 18682, 1789, 13936, 16267]
token pieces：
["▁Communic", "gruppe", "ової", "dis",
 "▁Aer", "ino", "▁soldiers", "▁spark"]
第二条：


[29256, 20219, 25498, 24566, 11306, 31203, 15632, 29775]
token pieces：
["stab", "ври", "Js", ">[",
 "▁subset", "男", "тах", "▁esempio"]
9. 为什么只缓存 K/V，不缓存 Q
对新 token 来说，需要的是：


Q_new 与所有历史 K 做匹配
再用匹配权重读取所有历史 V
历史 token 的 Query 已经完成了自己的任务，以后不会再用到，因此可以丢弃。

而历史 K/V 每生成一个新 token 都还要被读取，所以必须保留。

旧 token 的 K/V 也不会因为后面出现新 token 而改变，因为 causal attention 保证旧位置不能看到未来。

因此每一层缓存：


K_cache[layer]
V_cache[layer]
而不需要长期缓存：

Q；
Attention 分数；
softmax 权重；
MLP 临时结果；
logits。

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
本例的 KV cache 有多大
每个请求、每个 token、每层需要：


K: 4 heads × 64 = 256 个数
V: 4 heads × 64 = 256 个数
总共：


2 × 4 × 64 = 512 个 FP32
每层每 token：


512 × 4 bytes = 2048 bytes
模型有两层，所以每个请求的每个 token：


2048 × 2 = 4096 bytes
也就是约 4 KiB。

Prefill 后每个请求有 5 个 token：


5 × 4096 = 20480 bytes
两个请求共：


2 × 20480 = 40960 bytes
约 40 KiB 的实际 K/V 数据。物理 block 分配还会因为固定 block 粒度产生向上取整。

每多计算一个 token，每条序列大约增加 4 KiB K/V。

这个模型的 hidden_size 虽然只有 16，但 head_dim=64，所以它的 KV 相对残差状态显得异常大。这也是这个随机测试模型的特殊之处。

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
vLLM 在这里究竟管理什么
你的理解需要稍微限定一下：vLLM 的 Paged KV Cache 不是管理 Transformer 中的所有矩阵。

模型推理中的内存大致分三类：

固定权重
Embedding、QKV 投影、MLP、LM head 等。模型加载后长期存在。

临时中间产物
Q、Attention scores、softmax、MLP 中间结果、logits 等。用完可以释放或复用。

跨 decode 轮次保留的历史状态
主要就是每一层的 K/V cache。

vLLM 的分页机制重点管理第三类：

一个请求的 token 不需要占据一整块连续大内存；
随着 token 增加按 block 分配；
block table 记录逻辑 token 到物理 block 的映射；
prefix cache 可以共享相同前缀的完整 block；
KV 不足时可以释放、抢占或重算；
scheduler 把不同请求本轮需要计算的 token 组织成 batch。
而且这个测试实际运行在 CPU 上，不是显卡上；换成 GPU 后数学过程不变，只是矩阵乘、Attention 和内存访问由不同硬件与 kernel 执行。

最终可以用这一条主线记忆：


token ID
→ 查表得到 D 维 embedding
→ 每层 Attention：跨 token 混合信息
→ 每层 MLP：每个 token 内部变换
→ 最后一个 token 的 D 维状态
→ 映射成 V 个 vocabulary logits
→ argmax/采样选出下一个 token
→ 保存新 token 在每层的 K/V
→ 重复
Transformer 负责“算什么”；vLLM 负责“这些计算何时算、哪些请求一起算、历史 K/V 放在哪里、内存不足怎么办”。

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
