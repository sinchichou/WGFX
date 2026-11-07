# OpenNR-ReShade 逆向研究报告

> 面向浏览器移植攻关的技术分析 | 基于仓库 commit 现状（v0.2.1，MIT）| 2026-11
>
> 结论先行：**这是一个 1.29M 参数、2.58 MB、纯 3×3 卷积的 f16 U-Net，35 次 dispatch / 帧，51.2 GMAC @ 704p**。没有 transformer、没有 FP8、没有字节级一致约束，全部权重格式与管线语义已从源码完全还原——浏览器实时化的一切前置障碍都不存在，可以直接开工。

---

## 1. 项目是什么（一段话上下文）

OpenNR 是一个 ReShade 插件（`opennr.addon64`，C++/ReShade Addon SDK），挂在 D3D11/12/Vulkan 游戏的帧提交路径上，每帧对画面跑一个小神经网络，做局部影调、对比度与色彩重平衡。它的模型是**蒸馏学生网**——模仿一个大教师模型（作者另一个项目 DLSSNR SYCL Bridge，跑在 Intel Arc 上的 DLSS-NR 类网络，约 12 张图/秒 @ Arc A750）的输出风格。与 OpenDLSS-NR 那种"字节级复刻"完全不同：这里的验收标准是观感，不存在位一致性约束。

---

## 2. 模型逆向分析

### 2.1 文件格式规范（已逐字节验证）

模型是目录 + 纯文本清单 + 裸二进制，**无任何序列化框架**：

```
models/1pass/
  model.txt        # 头行: cin base levels cout \n 层数 n
                   # 层行: name cin cout pad level
                   # 可选: global <n> <in> <hidden> <out> + n 层行
  <name>.w.bin     # 3×3 卷积权重，f16，共 9*cin*pad 个半精度数
  <name>.b.bin     # 偏置，f32，共 pad 个
  fc.bin           # 全帧分支 MLP，f32，共 gh*gi + gh + go*gh + go 个
```

我用脚本对两个模型变体的全部 23 个卷积 + fc.bin 做了尺寸断言：**19 个主干层 + 3 个 global 层 + fc.bin 全部精确匹配声明格式，零偏差**。`pad` 字段是 `cout` 向上对齐到 16 的倍数（权重按 float4 组打包），只有输出层实际用到（4→16 / 12→16）。

两个变体权重逐层完全同构、大小一致（连文件字节数都相同），唯一差异是输出层 `cout`：1pass 为 4，2pass 为 12。**1pass 与 2pass 是同一个网络的两个训练快照，不是两种结构。**

### 2.2 逐层结构与计算量

头行 `16 16 4 4 19`：输入 16 通道、基础 16 通道、4 级 U-Net、输出 4 通道、19 层。逐层如下（GMAC 按 704×1252 工作分辨率、与插件自报 GMAC 同口径 `9·cin·pad·(W>>level)·(H>>level)` 计算）：

| 层 | cin→cout | 级 | 阶段 | GMAC | 权重 |
|---|---|---|---|---|---|
| enc0a/enc0b | 16→16→16 | 0 | 编码 | 2.03+2.03 | 2.3K+2.3K |
| enc1a/enc1b | 16→32→32 | 1 | 编码 | 1.02+2.03 | 4.6K+9.2K |
| enc2a/enc2b | 32→64→64 | 2 | 编码 | 1.02+2.03 | 18.4K+36.9K |
| enc3a/enc3b | 64→128→128 | 3 | 编码 | 1.01+2.02 | 73.7K+147.5K |
| mid_a/mid_b | 128→128→128 | 3 | 中间 | 2.02+2.02 | 147.5K×2 |
| dec3a/dec3b | 256→128→128 | 3 | 解码 | **4.05**+2.02 | 294.9K+147.5K |
| dec2a/dec2b | 192→64→64 | 2 | 解码 | **6.09**+2.03 | 110.6K+36.9K |
| dec1a/dec1b | 96→32→32 | 1 | 解码 | **6.09**+2.03 | 27.6K+9.2K |
| dec0a/dec0b | 48→16→16 | 0 | 解码 | **6.09**+2.03 | 6.9K+2.3K |
| out | 16→4(12) | 0 | 输出 | 2.03 | 2.3K |

合计（含 global 分支）：**f16 权重 1,287,936 个 + f32 偏置 1,392 个 = 2.581 MB；51.2 GMAC/帧；35 次 compute dispatch/帧**。三个 6.09 GMAC 的大层是解码器 a 层——上采拼接后通道翻倍（256/192/96/48）导致的，这也是后续做层融合或降批时的首要优化点：

![逐层计算量](images/mac_by_layer.png)

### 2.3 全帧条件分支（global）

```
16ch 输入 → 2×2 池化 ×3 (→1/8) → g0: 16→32 (1/8) → 池化 (→1/16)
→ g1: 32→64 (1/16) → 池化 (→1/32) → g2: 64→64 (1/32, 零填充边界)
→ 空间均值 → 64 维全帧向量 → FC1 64→64 (GELU) → FC2 64→128
→ 逐元素加到 mid_a 的偏置
```

fc.bin 的 49,920 字节 = (64×64+64 + 128×64+128) 个 f32，与公式精确吻合。这是一个极便宜的**整帧条件机制**：网络对全局画面特性（亮度分布、色调倾向）的响应不走卷积通路，而是压成 64 维向量直接调制中间块偏置——每帧只算一次，全帧共享。移植时这 7 次 dispatch 可以合并成 1 次。

---

## 3. 管线语义还原（全部出自 `opennr_shaders.hpp` / `opennr_addon.cpp`）

### 3.1 每帧调度（复现顺序即移植清单）

```text
1. stage     构建 16 通道输入（网络分辨率 ≤704 行，保持纵横比）
2. global    3 池化 + 3 conv + 1 FC → gbias（与主干并行，只读输入）
3. encoder   4 级：encXa→encXb（GELU），每级 b 层输出存 skip[i]，级间 2×2 池化
4. middle    mid_a（偏置+gbias）→ mid_b
5. decoder   4 级：upcat（最近邻×2 ⊕ skip 按通道拼接）→ decXa→decXb
6. out       无激活卷积 → head（4 或 12 通道）
7. composite 双线性放大回输出分辨率，叠加校正，处理风格/HDR
```

整条数据流一图看全：主干六段从上到下，全帧分支在右列独立成路，只在中间块偏置处汇合：

![网络架构与每帧数据流](images/arch.png)

### 3.2 输入 16 通道的精确构成（`kStageHLSL`）

这是移植最容易做错的部分，逐通道列出：

| 通道 | 内容 | 备注 |
|---|---|---|
| ch0[0:3] | 高斯噪声 ×3，Box-Muller：`r1·cos(τu3), r1·sin(τu3), r0·cos(τu2)` | 种子 = `mix32(x·0x9e3779b9 ⊕ y·0x85ebca6b ⊕ frame·0xc2b2ae35)`，**逐像素逐帧变化** |
| ch0[3] | 常数 1.0 | |
| ch1 | `(v.r, v.g, v.b, v.r)`，其中 `v = (c−0.5)·2/16` | 色彩为显示引用值；网络分辨率<帧分辨率时双线性下采样 |
| ch2 | `(v.g, v.b, 0, tone)` | tone = Local Tone 强度%（f32 位传入，除法在 CPU 侧做，源码注明是为避开 fp16 舍入） |
| ch3 | `(structure, −1, −1, 0)` | structure = Local Structure 强度% |

注意两点：色彩缩放极小（±1/16，中心 0.5）；噪声与 DLSS-NR 同构（三通道高斯 + 帧序列随机性），意味着**移植时必须保持逐帧种子机制**，否则时域上会出现静态噪型或闪烁。

### 3.3 三个核心算子

- **conv**（`kConvHLSL`）：3×3 卷积；权重 f16 四通道打包进 `uint2`，运行时 `f16tof32` 解码；**累加器 f32**；激活 GELU（tanh 形式，`clamp(±10)` 防 fp32 溢出成 NaN）；输出存回 f16。主网边界用 **clamp（复制边缘）**，global 分支用零填充。每线程处理 4 像素（TM=4）。
- **pool**：2×2 均值池化，f16 进出。
- **upcat**：最近邻上采 ×2，与 skip 按通道拼接（256=128+128 / 192=64+128 / 96=32+64 / 48=16+32）。

数值语义结论：**整个网络的算子在标准 ONNX op 集（Conv/AveragePool/Resize/Concat/GELU/Add）范围内**，权重解包后可直接建图，不存在自定义数值语义。

执行组织上还有两个移植时会遇到的细节：卷积核按 `numthreads(64,1,1)` 一维派发，每线程串行处理 TM=4 个像素、每像素 cout/4 个 float4 输出组，线程数 = `(W·H/4)·(pad/4)`——这是把像素维和输出通道维铺平成一个一维空间的写法，转 ORT-Web 时不用管，手写 WGSL 时值得照抄（访存局部性好）；权重布局是 `[tap][cin][cout/4]`（先 9 个空域 tap，再输入通道，最后输出通道组），转 ONNX 时要重排成标准的 `(cout, cin, 3, 3)`，注意 tap 顺序是行主序（dy 外层、dx 内层）。

### 3.4 输出与合成（`kCompositeHLSL`）

- 校正量 = `0.25 · head · Strength`（0.25 是硬编码），SDR 直接叠加：`saturate(sd + corr)`。
- Natural/Cinematic 风格：先算 `s' = s + 0.25·head`，再过一个 14 参数的 grade()（levels/曝光/对比/色调曲线/HSL 饱和度），校正 = `grade(s') − s`。2pass 模型输出 12 通道 = 3 种风格各一组 4 通道，按风格选组读取（`gHeadGroup`）；1pass 只有标准组，风格只影响色彩 grade 部分。**"2 passes"的命名指的是更强的训练形态，运行时同样只跑一次网络**（官方说明"两种模型成本相同"与 12 通道输出设计互相印证；具体训练过程仓库未公开，此为推断）。
- HDR：网络看的是"显示图"（线性 BT.709 按白点归一后 sRGB 编码），校正以**线性光增量**回加——未被修改的像素（含高光、广色域）逐位不变。这个设计对移植是好事：WebGPU 侧同样可以在 output pass 做编码/解码隔离。
- grade() 的 14 个字段按序是：levels 黑点、levels 白点、曝光（EV）、gamma、对比、饱和度、饱和度曲线（幂）、色温、色调、色调曲线 5 个控制点（f[9..13]）。Natural = 曝光 −0.1EV、对比 −25%、饱和 −10%；Cinematic 只降饱和 −15%；这些偏移全部按 Local Tone 强度线性插值，风格在 100% Local Tone 时取满。色调曲线用的是五段 smoothstep 基函数混合再走 `exp2·log2` 域映射，色温/色调目标是像素自身亮度的全饱和色——这些纯函数直接抄进 JS/WGSL 即可，无状态。

### 3.5 工程细节（写移植代码前值得知道的）

- ping-pong 双缓冲一对 scratch buffer，最大中间宽度 = 128ch @ 1/8 分辨率，f16。
- GPU 时间用 timestamp query 环形队列测；GMAC 口径与本文 2.2 节一致。
- 支持第二块 GPU 跑网络（延迟 +2 帧）；HDR 下不支持。
- 分辨率上限：`h = align(min(704, OH))`，宽度按输出纵横比推算。

---

## 4. 浏览器移植方案

### 4.1 为什么这次可行（与 OpenDLSS-NR 路线的对照）

| OpenDLSS-NR（WebGPU 移植，73ms） | OpenNR（本项目） |
|---|---|
| 141 MiB 权重、71 块 Swin/ViT | 2.58 MB、19 层 3×3 conv |
| FP8 E4M3 定点软件模拟，44.5ms | f16 标准 SIMT，无需张量核心 |
| 451 次 dispatch | 35 次 |
| 字节级一致约束（不可妥协） | 观感约束（容差自由） |
| 自定义数值语义（舍入逐位锚定） | 全部标准算子 |

前文讨论的所有平台级瓶颈（FP8 缺失、dispatch 屏障税、subgroup matrix 未标准化）对这个模型**均不构成障碍**。

### 4.2 推荐路线

**V1 —— 权重转 ONNX + ORT-Web（WebGPU EP，f16）**，最快出效果：

1. 写转换器：按 2.1 节格式读 bin → 解包 f16 → 构图 `Conv(3×3, pad=1) → GELU`，插入 AveragePool / Resize+Concat / Add / 全局分支（ReduceMean→Gemm→Gemm→Add）。噪声通道和条件标量作为输入占位符或预计算常量。
2. 输入侧注意：16 通道的噪声/色彩排布在 JS 侧预构建成输入 tensor；`v=(c−0.5)·2/16` 的缩放、逐帧噪声种子照 3.2 节复现。
3. 输出侧复刻 composite：0.25 系数、Strength/Detail/Colour、可选风格 grade。
4. 演示形态：`<video>`/摄像头/Canvas 场景 → 降采样到 ≤704 行 → 推理 → 校正放大叠加 → 前后对比滑块。

工程顺序上建议先在 Node 里用 onnxruntime-node 跑 CPU 精度对照（单帧 PNG 喂进去，与手写 Python 参考实现逐通道比对），确认图和权重都对齐了再上浏览器——浏览器里调试数值问题要慢一个量级。第二帧起把输入 tensor 复用，只重写噪声三通道和条件标量，避免每帧重建 16 通道 buffer 的 GC 压力。

**V2 —— int8 W8A8 + dp4a**：51.2 GMAC 对核显偏重，`dot4I8Packed`（稳定版 WebGPU 特性）可再省约一半；per-channel scale + 典型游戏帧校准，重点验收平滑渐变/肤色过渡的色带。**V3 —— 手写 WGSL**：HLSL→WGSL 几乎机械翻译（项目自带全套 HLSL 参照），把 stage+global 融进更少 dispatch，做成类似 OpenDLSS-NR demo 的"WebGL 场景 + WebGPU 推理"实时页。

### 4.3 性能预估（51.2 GMAC，卷积效率按 40–60% 估算）

| 平台 | f16 (V1) | int8 dp4a (V2) |
|---|---|---|
| RTX 4070 SUPER 级 dGPU | ~4–8 ms（60fps 无压力） | ~2–5 ms |
| 笔记本核显（~2 TFLOPS 级） | ~30–60 ms | ~15–30 ms |
| 移动端 WebGPU | 待实测，估计不可实时 | 存疑 |

（估算值，非实测；ORT-Web 下 35 次 dispatch 的调度开销远小于其收益项，可忽略。）

### 4.4 风险清单

1. **噪声语义**：逐帧种子若被向量化/批处理改变，时域观感会劣化——保留 `frame` 计数器进哈希。
2. **色带**：f16 激活 + ±1/16 的输入动态范围，int8 量化需校准验收。
3. **704 行上限是模型训练分布**：超分辨率直接跑更大输入可能掉出分布，先按原设计降采样跑。
4. **2pass 的风格组选择**：`group = min(style, 2)`，移植时对齐。
5. 许可：MIT（含权重），THIRD-PARTY-NOTICES 覆盖 ReShade SDK/ImGui；教师模型逻辑不在本仓库，无 NVIDIA IP。

### 4.5 移植检查清单（按依赖序）

- [ ] 转换器：model.txt/bin → ONNX（含 pad 对齐、skip 拼接、gbias Add）
- [ ] 数值对照：任意输入下 ONNX(CPU) vs C++ 参考实现逐通道 <1e-2（f16 容差）
- [ ] 输入构建：16 通道排布 + 噪声种子 + 下采样插值对齐 `kStageHLSL`
- [ ] composite：0.25/Strength/Detail/Colour/风格 grade 对齐
- [ ] ORT-Web f16 跑通 704p 视频实时演示
- [ ] （可选）int8 校准 + dp4a 验收色带
- [ ] （可选）WGSL 版 + dispatch 融合

---

## 附录 A：数值总账

| 指标 | 数值 |
|---|---|
| f16 权重参数 | 1,287,936 |
| f32 偏置参数 | 1,392 |
| fc.bin | 12,480 f32 = 49,920 B |
| 总权重体积 | 2.581 MB |
| GMAC/帧 @704×1252 | 51.17（主干 + 输出）/ ~0.16（global 分支） |
| dispatch/帧 | 35（stage 1 + global 7 + 编码 11 + 中间 2 + 解码 12 + 输出 1 + 合成 1） |
| 中间缓冲峰值 | 128ch @ 1/8 分辨率，f16 |

## 附录 B：分析产物

- `opennr_analysis.json` —— 逐层机器可读数据（含全部尺寸断言结果）
- `images/arch.png` —— 网络架构与每帧数据流图
- `images/mac_by_layer.png` —— 逐层计算量分布图
- 上游仓库：`github.com/MakeDecisionWorth/OpenNR-ReShade`（v0.2.1）；教师模型：`github.com/MakeDecisionWorth/DLSSNR-SYCL-Bridge`
