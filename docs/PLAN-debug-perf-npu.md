# WGFX 計畫：除錯工具 · CNN 效能 · NPU 加速

> 2026-11-13 起擬（Round 1）。對應使用者需求三件事：
> 1. 更優的除錯工具＋效能分析工具（TS / PY / RUST 皆可），介面更好用，**能載入原始 HLSL 比對除錯**。
> 2. 優化 CNN 處理速度（「CNN 不大但掉 FPS，肯定有問題，FP16?」）。
> 3. CNN for NPU 加速（用 subagent 執行）。

---

## 1. 環境與現況（本輪已驗證事實）

| 項目 | 狀態 | 證據 |
|---|---|---|
| Repo | `D:\WGFX`，分支 `npu-experiment`，127 效果已移植 | git log / PORTING_STATUS.md |
| 硬體 | Ryzen 7 8845HS + XDNA NPU (~16 TOPS, `PCI\VEN_1022&DEV_1502`) + Radeon 780M + RTX 4060 | docs/npu-experiment.md §1（2026-11-12 probe） |
| **AMD Ryzen AI Software** | **1.9.0 已安裝** | `C:\Program Files\RyzenAI\1.9.0\`（voe-1.9.0 wheel、onnxruntime_vitisai-1.30.0、`voe-4.0-win_amd64\vaip_config.json`、`quicktest\`）＋ Python venv `C:\Program Files\RyzenAI\ryzen-ai-1.9.0\` |
| NPU execution provider | **VitisAIExecutionProvider 已註冊於 venv 的 onnxruntime 1.30.0**（連同 Dml、CPU） | venv python `get_available_providers()` 實測輸出 |
| CuNNy→ONNX 管線 | 已存在且 pass-1 逐位元組驗證；r 殘差 MAE≈0.05（PSNR~19dB）追查中 | docs/npu-experiment.md §2、`tools/npu/*`、`dist\npu\CuNNy-2x4C-NVL.onnx` |
| NPU session 實測 | **結案**：AMD quicktest（CNN int8）✅ 在 NPU 上跑通（NPU 398 ops）；WGFX CuNNy 匯出圖在 X1 目標下 **`NO DPU Subgraph Create!`（0 ops 上 NPU，全程 CPU 回退）**——根因＝NPU1 僅支援 CNN INT8，匯出器 conv 為 f32 `Conv`；無 target 選項則編譯成 AIE2P 圖 → `0xc01e0009` context 失敗（同 [RyzenAI-SW #365](https://github.com/amd/RyzenAI-SW/issues/365)） | `docs/audit/npu-audit.md`、`.tmp/npu-audit/audit_v3.log` |
| 除錯工具現況 | `debugger/`＝Vite+Monaco，主要邏輯在單檔 `debugger/src/main.ts`（~66KB）；HLSL 差分驗證目前僅 CLI（`src/verification/`＋`npm run verify:*`） | debugger-audit 子代理審計中 |
| 熱路徑 | 曾有 P0-P4 優化（commit 8696c08）；FP16 使用現況待審計 | perf-audit 子代理審計中 |
| 已知限制 | `onnxruntime-node` 僅 CPU/DML → **NPU 必須走 Python ORT(VitisAI) 或 C++/Rust+VOE**；瀏覽器擴充無法直達 NPU | probe 輸出、docs/npu-experiment.md §1 |
| 沙箱 | `dist\npu` 目前唯讀（寫入 EPERM）；暫存改走 `.tmp\npu-audit\` | 本輪實測 |
| `tools/npu/probe.mjs` | **已過時**：仍報「未安裝」（寫於安裝前，偵測不到 1.9.0 版面）→ 待更新 | 本輪實測 |

---

## 2. 工作流 1：除錯工具 + 效能分析工具（WGFX Studio）

### 目標能力
- **HLSL 比對除錯**（使用者點名）：載入 `effects_src/**.hlsl` 原始碼 → HLSTranspiler 轉 WGSL → 在同一個 WebGPU device 上執行「原始 HLSL（transpiled）」與「移植 WGSL」→ 逐像素比對：A/B 滑桿、diff heatmap、逐 pass 中介紋理檢視、MAE/PSNR/max-diff 統計。
- **效能分析**：per-pass GPU 時間（timestamp-query）、每幀 dispatch 次數、紋理/緩衝記憶體、FPS 曲線、pass timeline。
- **介面**：檔案樹／編輯器（Monaco，HLSL+WESL 語法）／輸出檢視三欄布局；hot reload；效果參數面板；深淺色。

### 階段
| 階段 | 內容 | 驗收 |
|---|---|---|
| A | debugger 模組切分（拆 main.ts）→ 併入 `src/verification/` 的 transpile+compare 管線（瀏覽器端）→ 載入原始 HLSL 比對視圖 | 載入 `ACNet.hlsl` 或 `Bilinear.hlsl`：左右輸出並列、diff heatmap、MAE 顯示 |
| B | per-pass 中介紋理擷取檢視（點 pass → 存儲輸出 → heatmap）＋ timestamp-query profiler | CuNNy-2x4C-NVL：每 pass 縮圖＋每 pass 時間條列 |
| C | DX 打磨：hot reload、.wgx 載入、參數面板、（選配）Rust/WASM 加速 HLSL 前處理 | 操作流：改 HLSL → 1 秒內看到新比對結果 |

### 依賴與文件
- **前置審計已完成**：`docs/audit/debugger-audit.md` — 現況＝Monaco 編輯/WGSL 編譯預覽/WESL(WASM) 驗證/FPS 計數已有，main.ts 1839 行單檔；缺 HLSL 支援、差分管線、GPU 計時。可行性：HLSTranspiler/GPUExecutor/PixelComparator/runtime 🟢 可直接進瀏覽器；DiffVisualizer/EffectPairResolver 🟡 小改；`verification/index.ts` 頂層 `import 'webgpu'` 🔴 須以模組路徑繞過。
- **✅ 差分核心已完成**（`dbg-core` 子代理）：
  - `src/verification/diff-core.ts` — 純函式核心，零 import／零 device：`comparePixels(ref,test,w,h) → DiffMetrics`（perChannelMAE／maxAbsDiff／mae／mse／rmse／psnr／diffPixelCount／diffPixelRatio／diffPixelPercent）、`generateDiffHeatmap(ref,test,w,h) → Uint8Array`（RGBA，R=max|Δ|）
  - `src/verification/browser-entry.ts` — browser-safe barrel（re-export HLSTranspiler／PixelComparator／StatisticsEngine／diff-core）
  - `test/diff_core.test.mjs` — 純 Node 測試，**4 passed, 0 failed**
  - 瀏覽器安全性清單：✅ HLSTranspiler（用 `transpile(source)`，非 `transpileFile()`）、PixelComparator、StatisticsEngine、diff-core；❌ EffectPairResolver（glob/path＋process.cwd）、DiffVisualizer（pngjs）、ReportGenerator（fs＋Buffer PNG）、VerificationRunner／index.ts（`import 'webgpu'`）
- **第一切入點（Phase A）**：瀏覽器差分驗證 MVP — vite middleware 列 effects_src/effects 配對 + 上面的 browser-entry + A/B 分頁（transpile→雙版執行→MAE/PSNR/max diff＋Canvas heatmap）；驗收以 Bilinear/Bicubic/Lanczos/CAS 與 CLI 數值一致。
- 語言分工：TS 為主（UI+管線重用 verification）；Rust/WASM 僅在解析效能不足時引入；Python 留給離線批次報告。
- 風險：SSIM 大圖卡 UI（Worker 化）、中介紋理 readback 需 COPY_SRC、timestamp-query 需 feature 偵測＋CPU fallback。

---

## 3. 工作流 2：CNN 掉 FPS 調查與優化（FP16?）

### 方法：先量測，後動刀
| 階段 | 內容 | 驗收 |
|---|---|---|
| A 量測 | bench_hotpath 擴充：per-pass timestamp、逐 pass 隔離、格式矩陣（f32 / rgba16float / rgba8snorm）、FP16 開關；780M 實機跑 `CuNNy-4x8C-NVL`@1080p→4K | 一張「pass×格式×FP16」延遲表；root cause 排序報告 |
| B 修復 | 依量測排序：① FP16：`enable f16`＋特徵圖 rgba16float＋`//! CAPABILITY FP16` codegen 路徑 ② pass 合併/減少 dispatch ③ 上傳/輸出零複製 ④ 同步點消除 ⑤ workgroup 尺寸調整 | 每項變更都跑 `npm run verify` 差分門檻（像素可接受） |
| C 回歸 | 全效果驗證 + 實機影片 FPS 曲線前後對照 | FPS 提升量化（目標：小型 CNN 效果不再掉幀；具體 % 依量測訂） |

### 依賴與文件
- **前置審計已完成**：`docs/audit/perf-audit.md` — 熱路徑每幀＝上傳 → 單 encoder 逐 pass → 一次 submit，**CPU 端已乾淨**（8696c08 後快取全命中、無 mapAsync/readback）。根因排序：① Magpie 式 3×3 tap 逐 thread 直線取樣＋全解析度中介往返（8x8C ≈ 300+ GB/s@1080p60 超 780M 頻寬、shared memory 0 重用）＝主兇；② **FP16 完全未啟用**（`//! CAPABILITY FP16` 僅 parser 元數據、全 src 無消費者、無 `enable f16`；CuNNy 中介已 rgba8snorm，f16 紅利在 RDNA3 計算吞吐與佔用率）；③ pass 數 × 全螢幕 roundtrip 線性放大。次要：OUTPUT rgba16float@2x＋宿主呈現 copy。已排除：每幀同步停頓、codegen wrapper。
- **量測已完成**：`test/bench_cnn_caseE.mjs`（新增，RTX 4060 Laptop 實測，`effects/CuNNy/CuNNy-2x4C-NVL.wgsl` @256×256）：

  | metric | min(ms) | med(ms) | FPS(med) |
  |---|---|---|---|
  | encode+submit only（純框架 CPU） | 0.076 | 0.094 | — |
  | sync baseline（空 submit + await） | 0.015 | 0.019 | 54005 |
  | total frame time/frame | 0.201 | 0.233 | **4286** |
  | GPU saturated（N frames submitted/N） | 0.107 | 0.135 | 7424 |
  | **GPU execution（timestamp-query）** | 0.026 | 0.028 | 36169 |
  | framework overhead（total − GPU exec） | 0.176 | 0.206 | — |

  per-pass GPU（median）：`PASS_1 in=3µs | PASS_2 conv1=8µs | PASS_3 conv2=8µs | PASS_4 out-shuffle=10µs | sum=30µs`。
  **結論**：256×256 下是**框架 overhead 主導**（GPU 僅 28µs，框架 206µs），不是 GPU-bound；但 GPU 時間隨解析度線性上升（384²→54µs、512²→92µs）而 CPU 成本持平 ⇒ **1080p 會翻轉成 GPU-bound**。
  （註：node-webgpu 的 Dawn 傳輸層會灌大 CPU 側數字，Chrome 內會低很多；此數字是 Node 端的**上界**。）
- **兩個實作陷阱（已處理）**：D3D12 timestamp 僅 65.536µs 解析度 → 同一 command buffer 重複 dispatch K 次再除；timestamp epoch > 2^53 → 差值須在 BigInt 域計算。node-webgpu 在 `--size >= 512` 會丟無訊息 Error（傳輸層限制）。
- 優化清單：O1 FP16 codegen 雙路徑降級、O2 shared-memory 取樣重用、O3 輸出格式選項（驗收與 f16 trunc 數值風險見報告 §6–§7）。
- 風險：FP16 數值漂移 → 一律走 verify 差分門檻，不行就退 rgba8snorm 混合策略。

---

## 4. 工作流 3：CNN → NPU 加速（Ryzen AI VitisAI EP）

### 已打通 / 待打通
- ✅ Ryzen AI 1.9.0 安裝驗證；venv `VitisAIExecutionProvider` 註冊（本輪實測）
- ✅ **NPU 全線可用**：AMD quicktest（CNN int8）通過，NPU 398 ops（PHX/HPT 自動辨識）
- 🔧 **CuNNy x1 圖已可建立 session 並在 NPU EP 上執行**：已實作 `--npu-x1` 匯出模式（AMD QDQ 配方 + pow2 scale + scalar int32 zp + opset13/ir7 + 4→4 拆分 conv + 單一 consumer 的 Relu + padding 前置）
- ⛔ **剩餘唯一阻擋**：`arch: IPU_PHX not support const` — 解量化後的權重常數不被 PHX 接受 → `npu-const` 子代理專查中
- 🔧 待辦：const 阻擋破解 → 正式 NPU runner 收尾 → 效能矩陣 → sidecar 整合

### 階段（依本輪審計修正）
| 階段 | 內容 | 驗收 | 狀態 |
|---|---|---|---|
| A 打通 | `--npu-x1` 全鏈 int8 QDQ 匯出 → NPU session 建立 → 逐像素對照 | CuNNy-2x4C-NVL 真正上 NPU（log 中 `is partitioned to CPU` = 0）且數值可接受 | 🔧 匯出完成；const 阻擋待解 |
| B 效能矩陣 | {CPU, DML-780M, DML-4060, NPU} × {CuNNy-2x4C, 4x8C, 8x8C} × {64,128,256,512,1080p tile} | 一張決策表：哪些家族值得下放 NPU、哪些留 GPU | ⏳ 待 A |
| C 整合 | 本機 sidecar（Python FastAPI/WebSocket；選配 Rust+VOE） | sidecar 串流一條影片端到端跑通 | ⏳ 待 A/B |

### 風險
- NPU1 僅 CNN INT8 → 匯出器 INT8 化後與 GPU 數學（rgba8snorm 中介＋f32）有數值差 → 用「量化等效」門檻＋視覺確認；r 殘差在重匯出時一併重驗。
- 小 CNN 在 NPU 可能不贏 780M 的 WGSL（dispatch 開銷吃掉紅利）→ 用 B 階段數據決定下放清單，不硬上。
- `dist\npu` 沙箱唯讀 → 正式 runner 的模型/快取目錄改用 `models/npu/`（或修 dist ACL）；NPU 編譯快取必須命中（首次編譯可能數分鐘）。

---

## 5. 執行架構（subagent 分工）

- **本輪（Round 1）**：`debug-audit`、`perf-audit` 子代理並行審計中；NPU 環境驗證由 lead 親自執行（子代理同時上限 2）。報告落點 `docs/audit/*.md`。
- **後續輪**：三工作流各自派實作子代理；以團隊任務板記錄 write scopes（disjoint：`debugger/`、`src/runtime/`、`tools/npu/`）。
- **驗證門檻**：每階段結束跑 `npm run test` / `npm run verify:*` / 新增測試；效能主張一律附實測數據。

## 6. 本輪產出
- 本計畫文件（docs/PLAN-debug-perf-npu.md）
- Ryzen AI 1.9.0 安裝驗證 + VitisAI EP 註冊證明（§1 表）
- **NPU 實測結案**：quicktest 上 NPU ✅；CuNNy 圖 0 ops 上 NPU 的根因（f32 conv / NPU1 限 CNN INT8）→ 下一步 `--npu-x1` 匯出模式
- `docs/audit/debugger-audit.md`（debug-audit 子代理）、`docs/audit/perf-audit.md`（perf-audit 子代理）、`docs/audit/npu-audit.md`（lead）— 三份審計報告齊
- 團隊任務板：task-1（除錯工具）、task-2（CNN 效能）、task-3（NPU）已建立並附 write scopes
