# WGFX 計畫：除錯工具 · CNN 效能

> 2026-11-13 起擬（Round 1）；2026-11-14 起改於 **main 主幹**直接實現，NPU 工作流已移除（歷史封存於 `npu-experiment` 分支，不在主幹維護）。
> 對應使用者需求兩件事：
> 1. 更優的除錯工具＋效能分析工具（TS / PY / RUST 皆可），介面更好用，**能載入原始 HLSL 比對除錯**。
> 2. 優化 CNN 處理速度（「CNN 不大但掉 FPS，肯定有問題，FP16?」）。

---

## 1. 環境與現況

| 項目 | 狀態 | 證據 |
|---|---|---|
| Repo | `D:\WGFX`，**main 主幹**直接實作（效能相關 commit 已自實驗分支移植），127 效果已移植 | git log / PORTING_STATUS.md |
| 硬體 | Ryzen 7 8845HS + Radeon 780M iGPU（RDNA3，目標機）+ RTX 4060 Laptop（量測機） | 兩者皆可由 `powerPreference` 選取（high-performance→4060、low-power→780M） |
| FP16 | **opt-in 路徑已實作**（`preferF16` + `shader-f16` feature 偵測；預設行為逐位元組不變）＋閘門測試 | commit `0e38703`（自 `1b4fb93` 移植） |
| FP16 硬限制 | 本機 dawn.node（webgpu 0.3.8）**兩張卡皆不暴露 `shader-f16`**（D3D12 後端 DXC 不可用，已試 `allow_unsafe_apis`/`use_dxc` toggles）→ f16 GPU 實測必須走 Chrome WebGPU | 本輪 probe 實測 |
| outputFormat（O3-lite） | 已實作＋實測 512²→1024² 快 ~11% | commit `3b0b276`（自 `e2d1b4c` 移植） |
| O2 shared-memory 原型 | 已實作＋RTX 4060 實測**負優化**（+29% 慢）；780M 待真機量測 | commit `0423f61`（自 `619306a` 移植） |
| CNN 效能 | case E 實測完成：256² 下**框架 overhead 主導**（GPU 28µs vs 框架 206µs）；512²→92µs 線性上升 → 1080p 翻轉 GPU-bound | `test/bench_cnn_caseE.mjs` |
| 除錯工具 | 差分核心 diff-core 完成（測試 4/4）；UI 進行中 | commit `e7e0726`（自 `64577dc` 移植） |

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
- **✅ 差分核心已完成**：
  - `src/verification/diff-core.ts` — 純函式核心，零 import／零 device：`comparePixels(ref,test,w,h) → DiffMetrics`、`generateDiffHeatmap(ref,test,w,h) → Uint8Array`
  - `src/verification/browser-entry.ts` — browser-safe barrel（re-export HLSTranspiler／PixelComparator／StatisticsEngine／diff-core）
  - `test/diff_core.test.mjs` — 純 Node 測試，**4 passed, 0 failed**
  - 瀏覽器安全性清單：✅ HLSTranspiler、PixelComparator、StatisticsEngine、diff-core；❌ EffectPairResolver（glob/path）、DiffVisualizer（pngjs）、ReportGenerator（fs）、VerificationRunner／index.ts（`import 'webgpu'`）
- **第一切入點（Phase A）**：瀏覽器差分驗證 MVP — vite middleware 列 effects_src/effects 配對 + browser-entry + A/B 分頁；驗收以 Bilinear/Bicubic/Lanczos/CAS 與 CLI 數值一致。
- 語言分工：TS 為主（UI+管線重用 verification）；Rust/WASM 僅在解析效能不足時引入；Python 留給離線批次報告。
- 風險：SSIM 大圖卡 UI（Worker 化）、中介紋理 readback 需 COPY_SRC、timestamp-query 需 feature 偵測＋CPU fallback。

---

## 3. 工作流 2：CNN 掉 FPS 調查與優化（FP16?）

### 方法：先量測，後動刀
| 階段 | 內容 | 驗收 |
|---|---|---|
| A 量測 | bench_hotpath 擴充：per-pass timestamp、逐 pass 隔離、格式矩陣（f32 / rgba16float / rgba8snorm）、FP16 開關；**780M 實機**跑 CuNNy 家族 | 一張「pass×格式×FP16」延遲表；root cause 排序報告 |
| B 修復 | 依量測排序：① FP16：`enable f16`＋f16 數學 codegen 路徑 ② pass 合併/減少 dispatch ③ 上傳/輸出零複製 ④ 同步點消除 ⑤ workgroup 尺寸調整 | 每項變更都跑 `npm run verify` 差分門檻（像素可接受） |
| C 回歸 | 全效果驗證 + 實機影片 FPS 曲線前後對照 | FPS 提升量化（目標：小型 CNN 效果不再掉幀；具體 % 依量測訂） |

### 依賴與文件
- **前置審計已完成**：`docs/audit/perf-audit.md` — 熱路徑每幀＝上傳 → 單 encoder 逐 pass → 一次 submit，**CPU 端已乾淨**（8696c08 後快取全命中、無 mapAsync/readback）。根因排序：① Magpie 式 3×3 tap 逐 thread 直線取樣＋全解析度中介往返（8x8C ≈ 300+ GB/s@1080p60 超 780M 頻寬、shared memory 0 重用）＝主兇；② **FP16 完全未啟用**（`//! CAPABILITY FP16` 僅 parser 元數據、全 src 無消費者、無 `enable f16`；CuNNy 中介已 rgba8snorm，f16 紅利在 RDNA3 計算吞吐與佔用率）；③ pass 數 × 全螢幕 roundtrip 線性放大。次要：OUTPUT rgba16float@2x＋宿主呈現 copy。已排除：每幀同步停頓、codegen wrapper。
- **量測已完成**：`test/bench_cnn_caseE.mjs`（RTX 4060 Laptop 實測，`effects/CuNNy/CuNNy-2x4C-NVL.wgsl` @256×256）：

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

### 本輪（Round 4，main 主幹）待辦
1. `bench_cnn_caseE.mjs` 加 `--adapter auto|high-performance|low-power` 旗標 → 補 **780M f32 基線**（256²/512²/1080p-tile）。
2. 780M 上跑 O2 smem 變體 → close「780M 上未證明」的懸案。
3. `tools/experimental/f16/`：make_f16.mjs（離線 f16 數學轉換器，仿 make_smem 工作流）+ 純 JS 誤差上界 + 瀏覽器 bench 頁（Chrome 實測 shader-f16 有無 → A/B GPU 時間 + diff-core MAE）→ 依預先寫死的判定規則（780M ≥1.2× 且 MAE ≤1e-3）決定是否落地 codegen `preferF16` 數學路徑。

---

## 4. 執行架構

- 兩工作流各自推進；改動範圍以 disjoint write scopes 記錄（`debugger/`、`src/runtime/`、`tools/experimental/`）。
- **驗證門檻**：每階段結束跑 `npm test` / `npm run verify:*` / 新增測試；效能主張一律附實測數據。
