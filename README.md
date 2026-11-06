# Watermarking LLMs by EoS Token

**A causal (activation-patching) ownership watermark anchored at the model's stop decision.**

> English summary: we embed a multi-bit watermark into the *causal behaviour* of an LLM at its
> end-of-sequence (EOS) decision. For secret key-derived pairs (A, B), transplanting A's residual-stream
> state at its stop position (layer L) into B's last position changes B's EOS log-odds by a key-specified
> amount (up = bit 1, down = bit 0). Verification is an activation-patching experiment on the suspect
> weights. Because patching happens between two runs of the same model, the readout is
> **coordinate-free** (immune to residual-stream rotations/permutations by construction) and leaves
> **no footprint in normal inference** (no input ever produces a transplant). Target scenario:
> open-weight releases that are re-published as derivatives.

---

## 1. 动机

| 方法类别 | 代表 | 问题 |
|---|---|---|
| 触发器指纹 | Instructional Fingerprinting 等 | 触发时输出异常，可被黑盒发现；验证时暴露触发器 |
| 被动内在指纹 | HuRef、REEF | 无法区分同一基座的不同衍生模型 |
| 表示层水印 | **SEAL**（arXiv 2511.11356） | 静态读取隐藏状态与秘密向量的余弦：**白盒验证在残差流正交旋转下失效**；锚点必须是事实三元组 |

本仓库的想法：**不检查隐藏状态“长什么样”，而检查模型在被干预时“怎么反应”**，并把这个反应定义在 EOS（完整性判断）上。

## 2. 方法

**密钥 K** 派生所有秘密（HMAC-SHA256）：prompt 顺序、比特串、配对方式。发布前公开 `commitment(K)`。

**锚点对 (Aᵢ, Bᵢ)**（`eoswm/anchors.py`）
- Aᵢ：prompt + 基座模型**自信结束**的完整回复（P(EOS) ≥ τ），位置 = 停止决策位置（最后一个回复 token）；
- Bᵢ：另一个 prompt + 被截断的回复（不完整），位置 = 最后一个 token。

**读出量**（`eoswm/common.py: patch_delta`）

```
Δᵢ = logodds_EOS( Bᵢ | h_L[Bᵢ, pos_B] ← h_L[Aᵢ, pos_A] ) − logodds_EOS( Bᵢ )
xᵢ = Δᵢ − Δᵢ⁰          # Δᵢ⁰ = 基座模型上的自然效应，嵌入前记录
b̂ᵢ = [ xᵢ > median(x) ]
```

**嵌入**（`eoswm/watermark.py`，interchange-intervention training + LoRA，训练后合并）

```
L = softplus(m − sᵢ·xᵢ)            # 键对：比特 1 推高、比特 0 压低 (sᵢ = ±1)
  + λ_spec · E[x_j²]               # 非键对：其余 transplant 必须保持自然（特异性）
  + λ_kl   · KL(p_base ‖ p_model)  # 正常（不 patch）输出不变
  + λ_eos  · E[(logodds_EOS − logodds_EOS^base)²]   # 正常位置的停止决策不变
```

最后一项是专门加的：只用 KL 时，约束被摊到整个词表上，EOS 这一个 token 的概率仍可能漂移
（冒烟测试中观察到停止率从 0 变成 1）。

**为什么用 patching 做验证**
1. **天然抗旋转/置换**：攻击者把残差流旋转为 h·Q，A 与 B 的状态同时被旋转，后续层自动抵消，读出不变（冒烟测试中验证：logits 差 4.7e-6，BER 完全相同；而静态余弦读出从 0.65 掉到 0.05）。
2. **正常推理零痕迹**：没有任何输入会触发“状态移植”，黑盒查询无从发现。
3. **锚定在终止判断上**：非定向地大幅改动终止行为来去除水印，预期会损害回答质量（需实验验证，见 §5）。

## 3. 快速开始

**环境**：transformers 5.x 需要 PyTorch ≥ 2.6。已在两套版本组合上通过冒烟测试：
`torch 2.15 + transformers 5.22 + peft 0.21`，以及 `transformers 4.57 + peft 0.17`。

```bash
# 推荐：新建环境
conda create -n eoswm python=3.10 -y && conda activate eoswm
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 或者：在已有的 torch 2.5 环境里，改用 transformers 4.x
pip install "transformers>=4.45,<5" "peft>=0.12,<0.18" numpy

python tests/smoke_test.py            # CPU，约 1 分钟；随机初始化的小模型，只检查流程
```

完整实验（默认 Qwen2.5-1.5B-Instruct，bf16 + LoRA；攻击实验按 GPU 并行，每张卡同时只跑一个任务）：

```bash
KEY="your-secret-key" GPUS=0,1,2,3 bash scripts/run_pipeline.sh
python scripts/summarize.py runs/Qwen2.5-1.5B-Instruct/results.jsonl
```

| 模型 | 单张 3090（24 GB） | 建议 |
|---|---|---|
| Qwen2.5-0.5B / 1.5B-Instruct | 嵌入和全参数微调攻击都放得下 | 先用它们快速迭代 |
| Qwen2.5-3B-Instruct | LoRA 嵌入可以；全参数攻击可能需要调小 `--bs` | |
| Qwen2.5-7B-Instruct | 需要切分到多卡：`DEVICE_MAP=auto GPUS=0,1` | 此时各任务依次运行 |

日志在 `runs/<model>/logs/`。额外参数可以通过 `ANCHOR_ARGS`、`EMBED_ARGS` 传入，例如 `EMBED_ARGS="--lam-eos 5"`。

- 攻击后的模型直接在内存里评估质量，**不写盘**（1.5B 每个约 3 GB，容易超出磁盘配额）；
- 水印模型**只保存 LoRA adapter**（1.5B 约 70 MB），加载时自动合并回基座；需要完整模型时加 `EMBED_ARGS="--save-merged"`；
- `REUSE=1`：复用已有的 `anchors.pt`、`wm/`、`ref.pt`，只重跑验证和攻击；
- `ATTACK_DATA=hf:tatsu-lab/alpaca`：攻击者用真实的指令数据微调（需要 `datasets`）。默认使用内置 prompt 中**未用于质量评估**的部分；
- 每次运行会把旧的 `results.jsonl` 改名备份。

单独的命令：

```bash
python -m eoswm.cli build-anchors --model Qwen/Qwen2.5-0.5B-Instruct --key "$KEY" --n-bits 64 --out runs/x/anchors.pt
python -m eoswm.cli embed   --model Qwen/Qwen2.5-0.5B-Instruct --anchors runs/x/anchors.pt --out runs/x/wm
python -m eoswm.cli verify  --model runs/x/wm --anchors runs/x/anchors.pt
python -m eoswm.cli attack  --model runs/x/wm --anchors runs/x/anchors.pt --kind rotate|quant8|quant4|sft|concise|eosmask
python -m eoswm.cli quality --model runs/x/wm --anchors runs/x/anchors.pt --ref-file runs/x/ref.pt
```

常用参数：`--layer`（默认约 0.6 × 层数）、`--stop-tau`、`--margin`、`--lam-spec`、`--lam-kl`、
`--prompts hf:tatsu-lab/alpaca`（需要 `datasets`）。⚠️ `anchors.pt` 含秘密探测输入，已在 `.gitignore` 中排除。

## 4. 指标

| 字段 | 含义 | 期望 |
|---|---|---|
| `BER` | 比特错误率 | 水印模型及其衍生 ≈ 0；基座、兄弟模型 ≈ 0.5 |
| `p_value` | 匹配数在 Bin(n, ½) 下的单侧尾概率 | 水印模型极小 |
| `bit_separation` | mean(x \| bit 1) − mean(x \| bit 0)，log-odds 单位 | ≈ 2m |
| `nonkey_abs_shift` | 非键对 transplant 效应偏离基座的平均幅度 | 小（特异性 / 泄露） |
| `BER_corrected` 等 | **漂移校正读出**：在非键对上拟合 Δ ≈ a + b·Δ⁰（`drift_offset`、`drift_gain`），用键对的残差解码 | 微调会整体缩放移植效应，校正后应更接近 0 |
| `nonkey_abs_resid` | 校正后非键对的残差 | 小 |
| `avg_len` / `stop_rate` | 留出 prompt 上的贪心生成长度、自然停止比例 | 与基座接近 |
| `ref_nll` | 基座回复（含停止 token）在该模型下的 NLL | 与基座接近 |

攻击：`rotate`（保函数的残差流正交旋转，默认在 fp32 下做，报告 top-1 一致率）、`quant8`（逐张量 int8）、
`quant4`（int4，每 128 个权重一组；`quant4-tensor` 为逐张量 int4，会直接破坏模型）、`sft-lora` / `sft-full`
（攻击者自采样数据做 LoRA / 全参数 SFT）、`concise-lora`（只保留第一句话再 SFT，改变终止和长度行为）、
`eosmask-lora`（复现 pad==eos 掩码 bug，EOS 不再被监督）。
零假设：`base` 与 `null:sibling-sft`（基座做同样的微调）。

## 4b. EOS 消融：EOS 到底起不起作用

```bash
KEY="your-secret-key" GPUS=0,1,2,3 CLEANUP=1 bash scripts/run_ablation.sh
python scripts/summarize_ablation.py runs/ablation-Qwen2.5-1.5B-Instruct
```

同一把密钥、同一批回复和键对，只改变两处和 EOS 有关的选择（`build-anchors --position/--readout`）：

| 版本 | A 的移植位置 | 读出量 |
|---|---|---|
| `stop-eos`（本方法） | 停止决策位置 | EOS log-odds |
| `random-eos` | 回复中的随机 token | EOS log-odds |
| `stop-token` | 停止决策位置 | 密钥选出的普通 token（B 位置基座 top-50 候选中，排除 EOS） |
| `random-token` | 随机 token | 普通 token（完全不涉及 EOS） |

每个版本在一张卡上跑完整流程，默认攻击为 `quant4`、`sft-lora` × 3 个种子、`sft-full`、`concise`、`eosmask` 和兄弟模型
（用 `ABL_JOBS` 修改）。汇总表报告校正后的 BER，以及 **z = 间隔 / 噪声**（预测 BER = Φ(−z)）。64 比特的 BER 抽样误差很大，
比较时以 z 为主。磁盘：每个版本保存一个水印模型（1.5B 约 3 GB），`CLEANUP=1` 会在该版本结束后删除它，`ROOT=` 可以换到大磁盘。

## 5. 冒烟测试结果（仅验证流程，**不是**科学结论）

22 万参数的随机初始化 Qwen2、16 比特、强制停止的锚点、CPU：

| model | BER | p-value | bit sep. | non-key shift | extra |
|---|---|---|---|---|---|
| base (null) | 0.500 | 0.60 | 0.000 | 0.000 | |
| watermarked | 0.125 | 2.1e-3 | 1.717 | 0.347 | |
| rotate | 0.125 | 2.1e-3 | 1.717 | 0.347 | max logit diff 4.7e-6；static probe cos 0.649 → 0.048 |
| quant8 | 0.125 | 2.1e-3 | 1.706 | 0.335 | |
| sft (30 steps) | 0.125 | 2.1e-3 | 1.618 | 0.308 | |
| sibling (null) | 0.625 | 0.89 | −0.005 | 0.008 | |

⚠️ 同一测试中，水印模型在留出 prompt 上的停止率从 0 变成 1：**训练改变了 EOS 的一般行为**（随机模型上没有意义，
但提示了真实模型上必须重点检查 `stop_rate`/`avg_len`/`nonkey_abs_shift`；必要时增大 `--lam-kl`、`--lam-spec`）。

## 6. 待验证的研究问题

1. 真实模型上能否以低 KL、低泄露嵌入 64 比特？最佳层 L？
2. **抗微调**：`sft` 步数 / 学习率 / 全参数 (`--mode full`) 增大时 BER 如何变化？
3. **终止纠缠**：`concise`、`eosmask` 去除水印的程度 vs. 质量损失（`ref_nll`、长度、停止率）是否呈“去得越多、伤得越重”？
4. 停止决策位置 vs. 其他位置（事实主语、prompt 末尾）的容量与鲁棒性对比。
5. 知识型攻击：攻击者对**随机** transplant 做“保持自然反应”的正则化，需要多大代价？
6. 多模态模型：完整性判断依赖图文比对，终止扰动的质量代价可用 CHAIR/POPE 量化。

## 7. 代码结构

```
eoswm/
  keys.py       密钥派生、承诺、平衡比特
  prompts.py    内置离线 prompt 池 / HF 数据集 / 本地文件
  anchors.py    基座生成、停止置信度过滤、键对与非键对、自然效应 Δ⁰
  common.py     批处理、EOS log-odds、激活缓存与 patching
  watermark.py  IIT 嵌入（LoRA）、读出、报告
  attacks.py    旋转、SFT 类攻击、伪量化、静态探针对照
  quality.py    长度、停止率、参考 NLL
  cli.py        命令行
scripts/run_pipeline.sh, scripts/summarize.py, tests/smoke_test.py
```

## 8. 相关工作

- Yue et al., *Less is More: Mitigating Multimodal Hallucination from an EOS Decision Perspective*, ACL 2024
- Dai et al., *SEAL: Subspace-Anchored Watermarks for LLM Ownership*, arXiv 2511.11356
- Geiger et al., *Inducing Causal Structure for Interpretable Neural Networks* (IIT), ICML 2022
- Meng et al., *Locating and Editing Factual Associations in GPT* (ROME / causal tracing), NeurIPS 2022
- Ashkboos et al., *SliceGPT* (rotation invariance of the residual stream), ICLR 2024
- Xu et al., *Instructional Fingerprinting of LLMs*, NAACL 2024; Zeng et al., *HuRef*, NeurIPS 2024; Zhang et al., *REEF*, ICLR 2025
