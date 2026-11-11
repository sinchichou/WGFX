# NPU 實驗分支：把 CNN 推理搬到 NPU / NPU experiment: CNN inference off-load

> 分支 `npu-experiment`（實驗性）— 狀態、用法、限制與下一步。
> Branch `npu-experiment` (experimental) — status, usage, limits, next steps.

## 0. 這是什麼 / What this is

WGFX 的 CNN 類效果（CuNNy / CuNNy2 / OpenNR / NNEDI3 / Anime4K_GAN…）目前全部以
WebGPU compute shader 執行。本實驗驗證「同一個網路改跑 NPU」的可行性與接線方式：

```
Magpie HLSL 權重 ──(tools/npu/cunny2onnx.mjs)──► ONNX 模型 ──(onnxruntime EP)──► CPU / GPU(DML) / NPU
                                                        │
WGFX WGSL ──(WebGPU / Dawn)──► GPU ──────逐像素比較─────┘
```

- **ONNX 是橋**：NPU（AMD XDNA）不吃 WGSL，只吃編譯過的 ML 圖；ONNX 是唯一
  同時能被 CPU、DirectML（GPU）、與 Ryzen AI（NPU）執行的格式。
- **工具組位於 `tools/npu/`**，權重直接從 `effects_src/CuNNy/*.hlsl` 的內聯常數提取
  （與 `tools/cunny2wgsl.mjs` 同源，數字逐字複製）。

## 1. 這台機器的硬體偵測結果 / Hardware probe (this machine)

`node tools/npu/probe.mjs` 實測（2026-11-12）：

| 項目 | 結果 |
|---|---|
| CPU | AMD Ryzen 7 8845HS（Hawk Point，8C/16T） |
| **NPU** | **AMD XDNA「NPU Compute Accelerator Device」— `PCI\VEN_1022&DEV_1502`，狀態 OK（~16 TOPS）** |
| iGPU | AMD Radeon 780M（DirectML 可用） |
| dGPU | NVIDIA GeForce RTX 4060 Laptop（CUDA/DirectML 可用） |
| OS | Windows 11（build 26300） |
| Ryzen AI Software | **未安裝**（NPU 執行期需另外安裝，見 §4） |

> 重要：**DirectML 不會落到 NPU**（NPU 不是 DX12 adapter）。要打進 XDNA NPU 必須走
> AMD Ryzen AI Software 的 VitisAI/VOE execution provider（ONNX Runtime 的 NPU 外掛）。

## 2. 現況 / Current status（誠實版）

| 組件 | 狀態 |
|---|---|
| `tools/npu/probe.mjs` — NPU/驅動/EP 探測 | ✅ 可用 |
| `tools/npu/onnx_writer.mjs` — 手寫 ONNX protobuf 編碼器 | ✅ 可用（onnx.checker PASS） |
| `tools/npu/cunny2onnx.mjs` — CuNNy→ONNX 匯出 | ✅ 可用；**pass-1 特徵圖與 GPU 逐位元組一致**（≤1 LSB） |
| 權重映射（M4 轉置、YUV 矩陣轉置、in-projection 摺疊、DCR DepthToSpace） | ✅ 已對 GPU 逐層驗證 |
| `tools/npu/ort_run.mjs` — ONNX vs WGFX GPU 逐像素對照 | ✅ 可用 |
| 完整模型端到端數值 | ⚠️ **MAE ≈ 0.05（PSNR ~19 dB）**：G/A 通道完全一致，R/B 的差異集中在 `luma + r` 的 `r`（tanh 捲積輸出）— 卷積權重/映射已逐層驗證正確，殘差來源追查中（見 §6） |
| NPU 實跑 | ⛔ 需先安裝 Ryzen AI Software（§4）；管線已就緒，裝完即為 EP 切換 |

**已釘死的事實**（除錯紀錄，供續查者省時）：
- pass-1（luma 摺疊 3×3 卷積）ONNX vs GPU：max 7.9e-3 = 1 個 snorm LSB → **精確**。
- pass-2 捲積權重映射：WGSL `mat4x4<f32>(w) * v`（欄主要建構子）⇒ `out[o] = Σ_i v_i·w[i*4+o]`
  — **轉置**映射，已對 GPU t1 逐位元組驗證（MAE 1.1e-3）。YUV 3×3 同樣轉置。
- GPU 紋理取樣是 **clamp-to-edge** ⇒ ONNX Conv 零填充會在邊界出錯 ⇒ 已改用
  `Pad(mode='edge') + Conv(pads=0)`。
- 中間特徵 GPU 存 `rgba8snorm` ⇒ 圖內以 `QuantizeLinear/DequantizeLinear`（int8, scale 1/127）重現。

## 3. 怎麼用 / Usage

```bash
# 0) 探測硬體與可用的 execution provider
node tools/npu/probe.mjs

# 1) 匯出（例：最小的 CuNNy-2x4C-NVL；--size 決定靜態輸入邊長）
node tools/npu/cunny2onnx.mjs effects_src/CuNNy/CuNNy-2x4C-NVL.hlsl \
     --out dist/npu/CuNNy-2x4C-NVL.onnx --size 128
#    選項：--no-quantize（關閉 snorm QDQ 重現）、--emit-stage in|conv2|conv3（除錯：輸出中間層）

# 2) 與 WGFX GPU 輸出逐像素對照
node tools/npu/ort_run.mjs effects/CuNNy/CuNNy-2x4C-NVL.wgsl \
     dist/npu/CuNNy-2x4C-NVL.onnx --size 128 --ep cpu     # 或 --ep dml（DirectML→GPU）

# 3) 框架測試（驗證工具組本身）
npm run test:verification
```

依賴：`onnxruntime-node`（已加入 dependencies；CPU EP 開箱即用，DML EP 在 Windows 上
自動啟用）。

## 4. 打通 NPU（裝好後一行切換）/ Enabling the NPU

1. 安裝 **AMD Ryzen AI Software 1.x**（含 VitisAI/VOE execution provider 與 NPU 驅動堆疊）。
2. 將 runner 的 `executionProviders` 換成 Ryzen AI 的 VitisAI/VOE 路徑 —
   **同一個 .onnx 檔，不需要重新匯出**。
3. NPU 建議輸入為 int8/f16 靜態形狀 — 本匯出器已產生 int8 QDQ + 固定形狀，符合要求。

> 現實檢查：Hawk Point 的 XDNA 是 16 TOPS、以 int8 為主；CuNNy 這類小卷積網在
> NPU 上的吞吐優勢要看 batch/解析度，64×64 單幀可能不比 780M 的 WGSL 快 —
> 這正是要實測的問題（§6）。

## 5. 架構與設計決策 / Architecture

- **匯出器直讀 HLSL**：`effects_src/CuNNy/*.hlsl` 內聯權重 = 上游訓練權重的逐字拷貝；
  解析 `mad(s, V4(w), r)`（in-projection）與 `MulAdd(s, M4(w), r)`（捲積）的數字序列。
- **relu-split**：CuNNy 的 `s0=relu(x), s1=-relu(-x)` ⇒ ONNX `Concat([Relu(x), Sub(Relu(x),x)])`
  （Sub 恰等於 `-relu(-x)`，逐位元組等價）。
- **snorm 中介**：GPU 上 t0/t1 是 `rgba8snorm`（8-bit 量化）；ONNX 用 int8 QDQ 重現，
  順帶讓圖成為 NPU 友好的 int8 形態。
- **輸出頭**：捲積(+tanh) → `DepthToSpace(DCR,2)`（子像素排序 r.x→(0,0), r.y→(1,0),
  r.z→(0,1), r.w→(1,1)，已對照 store 程式碼逐行確認）→ YUV luma 殘差
  （`Resize(linear, half_pixel)` ≡ Magpie 的 `pos=(src+0.5)·pt−0.5·opt` 取樣）→ `yuv2rgb`。

## 6. 已知差異與下一步 / Known gaps & next steps

1. **r 殘差**：端到端 MAE ~0.05 全部來自 `r`（tanh 捲積）。已排除：權重映射（兩種慣例
   都試過並對 GPU 驗證）、邊界填充、DCR/CRD、tap 順序。下一個排查對象：
   pass-3（conv2）輸出（t0-after-P3，唯一未與 GPU 逐層比對的中介層）與 ORT 的
   Conv+QDQ 融合捨入。工具已就位：`--emit-stage conv3` + 逐層比對腳本模式。
2. **transpiler 的 `return` 語意**：`_transformPassToComputeShader` 目前丟棄 Magpie-PS
   進入點的 `return expr;`、注入 `textureStore(OUTPUT, …, textureSampleLevel(INPUT,…))`
   — 對 Bilinear 恰好等價（已逐位元組驗證），對「return 值 ≠ 輸入取樣」的效果不等價。
   正確做法：`return expr;` → `textureStore(OUTPUT, vec2<i32>(gid.xy), expr);`。
3. **涵蓋更多效果**：匯出器目前針對 CuNNy-*-NVL 家族；OpenNR（37-pass、storage buffer
   權重）需要先做「storage buffer → ONNX initializer」的權重抽取。
4. **NPU 實測**：安裝 Ryzen AI Software 後量測 NPU vs 780M-WGSL 的延遲/功耗，
   決定哪些家族值得留在 GPU、哪些下放 NPU。
5. **ONNX Runtime node 崩潰**：`--emit-stage` 的 debug 模型會讓 `onnxruntime-node`
   存取違規（onnx.checker PASS 但 ORT C++ 崩潰）— 先用 python onnxruntime 驗證，
   或改用完整模型 + 輸出重命名。

## 7. 相關文件 / Related

- [`docs/PORTING_STATUS.md`](PORTING_STATUS.md) — 移植現況與驗收方法
- [`docs/porting-guide.md`](porting-guide.md) — HLSL→WGSL 移植規範（M4/YUV 的慣例來源）
- `tools/cunny2wgsl.mjs` — CuNNy HLSL→WGSL 轉換器（權重序列的權威來源）
