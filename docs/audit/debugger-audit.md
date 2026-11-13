# WGFX 除錯工具審計報告

> 審計範圍：`debugger/`（現有網頁除錯工具）、`src/verification/`（HLSL 差分驗證工具組）。
> 審計方式：唯讀通讀原始碼，未修改任何 `src/` 或 `debugger/` 檔案。
> 日期：2026-11-13。所有路徑均相對於專案根 `D:\WGFX`。

---

## 1. 現況盤點

### 1.1 現有 debugger 功能（`debugger/`）

| 功能 | 現況 | 實作位置 |
|---|---|---|
| Effect 載入 | Monaco 下拉選單，僅列 `examples/*.wgsl`（vite middleware `/api/files`），不支援 `effects/`、`effects_src/` | `debugger/vite.config.ts` L15–35 |
| 編輯器 | Monaco 0.45，自訂 WGSL Monarch tokenizer + `wgsl-dark` 主題；Ctrl+S 編譯；**無 HLSL 語法** | `debugger/src/main.ts` L651–814 |
| 檔案同步 | Auto-sync：每 500ms 輪詢 `/api/shader` 的 mtime，保護使用者編輯、以 `executeEdits` 保留 undo stack；**只能讀、不能寫回磁碟** | `main.ts` L1744–1809 |
| 編譯執行 | `WGFX.create` → `initialize()` → 動態 uniforms 滑桿面板（int/float，min/max/step 由中繼資料驅動）→ `wgfx.process()` 逐幀渲染 | `main.ts` L1200–1295 |
| 產出檢視 | 「Shader IR (JSON)」分頁（`runtime.shaderInfo`）、「Generated WGSL」分頁（`runtime.generatedModules`，唯讀 Monaco） | `main.ts` L1257–1288 |
| WGSL 驗證 | WESL Rust/WASM（`debugger/src/wesl-web/`）對「產生的 WGSL」做編譯驗證，錯誤標記在唯讀編輯器；刻意不驗證 DSL 原始內容（`//!` 指示詞會誤報） | `main.ts` L501–630 |
| 預覽 | WebGPU canvas，內部解析度採效果宣告的 OUTPUT 尺寸（超分 2x 真實呈現）；split view：左 processed / 右 original、拖拉分割線、顏色濾鏡 tint、OUTPUT 空白時橘色棋盤格 fallback | `main.ts` L1297–1561、`index.html` L400–542 |
| 媒體輸入 | 圖片/影片 file input；預設 512 棋盤格測試圖；影片播放/停止/seek/音量控制 | `main.ts` L909–1117 |
| 效能指標 | **僅 FPS 計數**（每秒 frameCount，CPU 端）；無 GPU 計時 | `main.ts` L1451–1456 |
| 錯誤回報 | Log 分頁（時間戳）；GPU `uncapturederror`、`device.lost` 事件導入 log | `main.ts` L884–899 |
| 版面 | 兩欄（編輯器 50% + 預覽），水平/垂直拖曳 resizer、debug 面板高度可調、<=768px 響應式；**僅暗色主題** | `index.html`、`main.ts` L326–466 |

**結構性問題**：`debugger/src/main.ts` 為 **1839 行 / 66KB 單檔**，UI、WebGPU、檔案 API、驗證全部混在一起；Monaco worker 未正確設定（`getWorkerUrl` 回傳空字串，`main.ts` L639–645），長檔案效能與 lint 能力受限。

### 1.2 現有 verification 工具組（`src/verification/`）與瀏覽器相容性

資料流：`EffectPairResolver`（配對 effects_src/*.hlsl ↔ effects/**/*.wgsl）→ `HLSTranspiler`（PEG parser + WGSLCodeGenerator，逐 pass 生成獨立 compute module）→ `GPUExecutor`（`runModules` 直接 dispatch 生成模組 / `runWGSL` 走完整 runtime）→ `PixelComparator` + `StatisticsEngine`（MAE/RMSE/PSNR/SSIM/直方圖/百分位/errorMap/outlierMask）→ `DiffVisualizer`（heatmap/side-by-side/difference/overlay PNG）→ `ReportGenerator`（JSON/HTML/MD）。

| 模組 | Node 依賴 | 瀏覽器可用 | 備註 |
|---|---|---|---|
| `HLSTranspiler` | 僅 `transpileFile()` 動態 `import('fs')`（可繞過） | 🟢 可 | 核心依賴 `src/runtime/ShaderParser.js`（peggy 產生，**無 Node 依賴**）與 `WGSLCodeGenerator`（純 TS）；`transpile()` 純字串進出 |
| `GPUExecutor` | 無 | 🟢 可 | constructor 收 `GPUDevice`（瀏覽器 `navigator.gpu` 的 device 即可）；`runWGSL`/`runModules`/`runHLSL` 全用 WebGPU API；輸出統一 `Float32Array` RGBA |
| `PixelComparator` | 無 | 🟢 可 | 純運算，直接重用 |
| `StatisticsEngine` | 無 | 🟢 可（附註） | SSIM 為 O(N·w²) 純 JS 雙迴圈，大解析度（≥1080p）需移入 Web Worker 或關閉 |
| `DiffVisualizer` | `pngjs`（Buffer）、`fs`（saveToFile） | 🔴 不可直接用 | 但 colormap（hot/jet/viridis/cool）與 normalization 邏輯可移植；瀏覽器改用 Canvas2D `putImageData` 直接繪製 `errorMap`，無需 PNG 編碼 |
| `EffectPairResolver` | `glob`、`path`、`process.cwd()` | 🔴 不可直接用 | 配對邏輯簡單（目錄優先、跨目錄 name 唯一性後備），瀏覽器改由 dev server API 提供清單後 ~30 行即可重實作 |
| `ReportGenerator` | `fs`、`path` | 🔴 不可直接用 | 瀏覽器改 Blob 下載，或直接在 UI 呈現（多數情境不需要落地檔案） |
| `VerificationRunner` / `index.ts` | **頂層 `import { create, globals } from 'webgpu'`**（node-webgpu）、`fs` | 🔴 不可直接用 | 頂層 import 會讓整包瀏覽器 bundle 直接炸掉；但其組裝邏輯僅 ~50 行，瀏覽器版以 `navigator.gpu` 重建即可 |
| `src/runtime/*`（WGFXRuntime、ResourceManager、PipelineManager、ShaderParser.js、WGSLCodeGenerator） | 無（grep 確認） | 🟢 可 | `dist/wgfx.browser.esm.js` 即由此打包；debugger 已在用 |

**GPUExecutor 輸入需求**（搬移時要備齊的東西）：
- `InputTexture { data: Uint8Array|Uint16Array|Float32Array, width, height, format }` —— 瀏覽器端由 `ImageBitmap`/`HTMLVideoElement` → `OffscreenCanvas` → `getImageData` 即可取得。
- `runWGSL(wgslCode, …)`：WGSL 需為 Magpie 效果檔格式（含 `//!PASS`）；`runModules(transpileResult, …)`：吃 `HLSTranspiler.transpile()` 的結果，內部自動建紋理/scene buffer/sampler 並逐 pass dispatch。
- 兩者輸出同為 readback 後的 `Float32Array`（預設 `rgba16float`），可直接餵 `PixelComparator.compare()`。

### 1.3 runtime 既有的可掛鉤點（供 Phase B 使用）

- `WGFXRuntime.dispatchPass(passName, commandEncoder)`（公開）：可自行逐 pass submit → 實作單 pass 執行 / 步進。
- `WGFXRuntime.shaderInfo.passes`：pass 清單可枚舉。
- `ResourceManager.getTexture(name)` / `getTextureView(name)`：中介紋理可依名稱取用（readback 需該紋理 usage 含 `COPY_SRC`，`ResourceManager.createTexture` 的 usage 由 effect 描述符決定，**需驗證**，見風險 R5）。
- `ResourceManager` 內部保存所有邏輯紋理 Map：可枚舉 name/format/size 換算記憶體用量。
- runtime 目前**沒有** timestamp-query 相關程式碼（grep 無匹配）；`GPUExecutor` 的 `captureTimings` 是 `performance.now()` CPU 提交時間，非 GPU 真實執行時間。

---

## 2. 可行性結論（紅綠燈）

| 目標 | 結論 | 依據 |
|---|---|---|
| 瀏覽器內 HLSL → transpile → 執行 | 🟢 綠燈 | HLSTranspiler/GPUExecutor/runtime 全鏈無 Node 依賴；debugger 已透過 `dist/wgfx.browser.esm.js` 使用同一 runtime |
| HLSL vs WGSL 逐像素比對（MAE/PSNR/max diff） | 🟢 綠燈 | PixelComparator + StatisticsEngine 純運算直接重用；兩版輸出格式統一 Float32Array |
| Diff heatmap / side-by-side | 🟡 黃燈（小改） | DiffVisualizer 依賴 pngjs 不可用，但演算法 <100 行可移植到 Canvas2D/WebGPU |
| 逐 pass 中介紋理檢視 | 🟡 黃燈（小改） | `dispatchPass` + `getTexture` 已公開；readback 需確認紋理 usage 含 `COPY_SRC` |
| Per-pass GPU 時間（timestamp-query） | 🟡 黃燈（新功能） | WebGPU 原生支援，需 device 請求 `timestamp-query` feature（現 `initWebGPU` 未請求）；Safari 等不支援平台需 CPU fallback |
| 效果配對瀏覽（126 對 HLSL↔WGSL） | 🟡 黃燈 | 配對規則簡單；需擴充 vite middleware 列 `effects_src/` 與 `effects/` |
| 直接 import `src/verification/index.ts` | 🔴 紅燈 | 頂層 `import 'webgpu'`（node-webgpu）會炸瀏覽器 bundle；必須以模組路徑個別 import（HLSTranspiler/GPUExecutor/PixelComparator），或先抽出一個 browser-safe barrel |

**總結**：把差分驗證搬進瀏覽器 debugger 是**高可行、低風險**的任務 —— 唯一的硬阻礙是 `index.ts` 的頂層 webgpu import，繞過方式極簡單（相對路徑個別 import，vite.config 已有 `'@' → ../src` alias 可用）。

---

## 3. 缺漏清單（對照使用者需求）

| # | 缺漏 | 嚴重度 | 對應需求 |
|---|---|---|---|
| G1 | 完全無 HLSL 支援：編輯器無 HLSL 語法高亮、檔案清單不含 `effects_src/` | 高 | 「載入原始 HLSL 去比對除錯」 |
| G2 | 無 A/B 差分比對：split view 只比較「輸入 vs 輸出」（同一管線），不是「原始 HLSL vs 移植 WGSL」兩條管線；無 heatmap、無 MAE/PSNR | 高 | 同上 |
| G3 | 無 per-pass 執行控制與中介紋理檢視（只有最終 INPUT/OUTPUT 兩張） | 高 | 逐 pass 中介紋理檢視 |
| G4 | 效能分析只有 FPS 計數：無 per-pass GPU 時間、無 dispatch 次數、無記憶體用量、無時間軸 | 高 | 效能分析工具 |
| G5 | 檔案瀏覽只有單層下拉選單（僅 examples/），無檔案樹、無 HLSL↔WGSL 配對檢視 | 中 | 介面好用 |
| G6 | 編輯結果無法存回磁碟（middleware 只有讀 API）；hot reload 只能從磁碟單向同步 | 中 | 介面好用 |
| G7 | 無像素探針（點擊查看該像素 HLSL/WGSL 值與誤差）、無截圖匯出 | 中 | 比對除錯 |
| G8 | 僅暗色主題；Monaco worker 未設定 | 低 | 介面 |
| G9 | `main.ts` 1839 行單檔、零模組化、零測試；`debugger/package.json` 與主專案依賴脫鉤 | 低（技術債） | 可維護性 |

---

## 4. 設計建議

### 4a. 差分驗證（HLSL vs WGSL）

**管線**（全部已在程式庫中，瀏覽器版只需組裝）：

```
選擇效果對（effects_src/xxx.hlsl ↔ effects/**/xxx.wgsl）
→ HLSTranspiler.transpile(hlslSource)          [瀏覽器執行，peggy parser]
→ GPUExecutor.runModules(transpileResult, A)   [HLSL 生成模組直接 dispatch]
→ GPUExecutor.runWGSL(wgslSource, A)           [移植 WGSL 走完整 runtime]
→ PixelComparator.compare(hlslOut, wgslOut)
→ 卡片：MAE / RMSE / PSNR / SSIM / max diff / outlier %（per-channel + aggregate）
→ 視覺：side-by-side 三聯圖（HLSL | WGSL | |diff|×gain）+ heatmap（移植 colormap）
→ 逐 pass：dispatchPass 單步執行 → getTexture(中間名) → readback → 縮圖牆
```

實作要點：
- **import 路徑**：`import { HLSTranspiler } from '@/verification/HLSTranspiler'` 等，**避開 `index.ts`**；或新增 `src/verification/browser.ts` barrel（不 import 'webgpu'）。
- **輸入來源**：除測試圖樣（`generateTestPattern` 可照搬）外，支援使用者當前載入的圖片/影片影格 → `OffscreenCanvas.getImageData()` 餵入，讓「比對」與「即時預覽」用同一份輸入。
- **統計顯示**：MAE/PSNR/max diff 以數字卡片 + per-channel 條圖；errorMap 用 Canvas2D `putImageData` 上色（移植 `DiffVisualizer.applyColormap`），gain 可調滑桿。
- **SSIM**：預設關閉或降取樣計算，開啟時放 Web Worker（避免 1080p 卡 UI 數秒）。
- **transpile diagnostics**：126 檔非全部可無痛轉換，UI 需以警告列呈現 `TranspileDiagnostic`（`validateGeneratedCode` 已會抓 `mul()` 未轉換等問題）。

### 4b. 效能分析

- **Per-pass GPU 時間**：`initWebGPU` 改為 `adapter.features.has('timestamp-query')` 時 `requestDevice({ requiredFeatures: ['timestamp-query'] })`；每個 pass 前後 `encoder.writeTimestamp(querySet, …)`，`resolveQuerySet` → `MAP_READ` buffer 取 ns 級時間。不支援的平台 fallback 到現有 `performance.now()`（並在 UI 標示「CPU 提交時間」）。
- **Pass timeline**：單一 command encoder 內逐 pass writeTimestamp，解算後以水平甘特圖呈現各 pass 佔比（Canvas 或簡易 SVG 即可，不必引圖表庫）。
- **Dispatch 次數**：每幀統計 `dispatchPass` 呼叫數（可包一層 wrapper 或在 PipelineManager 加計數 hook）；配合現有 `WGFX.setDebug(true)` 的 DEBUG 級 dispatch 日誌。
- **紋理/緩衝記憶體**：枚舉 `ResourceManager` 紋理 Map，依 format byte-size × W×H 換算並列表（總量 + per-pass 使用）；storage buffer 同理。
- **FPS 曲線**：現有每秒計數改為 per-frame 時間戳环形缓冲，Canvas 畫最近 ~300 幀折線圖 + p50/p95/p99。

### 4c. 介面改善

- **三欄版面**：左＝檔案樹（`examples/`、`effects/`、`effects_src/`，HLSL↔WGSL 配對節點成對顯示）＋效果參數；中＝編輯器（雙檔分頁：WGSL 與對應 HLSL）；右＝檢視器（預覽 canvas、diff 視圖、pass 縮圖牆、效能面板以可切換子分頁收納）。現有兩欄 + resizer 的實作可延伸為三欄。
- **Monaco**：補 HLSL Monarch tokenizer（關鍵字與 WGSL 不同：`Texture2D`、`SamplerState`、`float4`、`mul` 等）；正確設定 worker（`MonacoEnvironment.getWorker` 回傳 blob URL 的 editor worker，或改用 `monaco-editor-webpack/esm` 官方設定），解決現有空 worker URL 的限制。
- **深淺色**：`index.html` 的 CSS 變數已集中（`:root`），補 `prefers-color-scheme` 與手動切換，Monaco 兩套 theme（`wgsl-dark` / `wgsl-light`）。
- **Hot reload**：現有 auto-sync 已是「磁碟→編輯器」熱同步（保護使用者編輯），補「編輯器→磁碟」保存 API（vite middleware 加 POST `/api/shader`）即為完整雙向 hot reload；可加 debounce 自動編譯開關。
- **參數面板**：現有 uniforms 滑桿已可用，加「重設預設」「隨機化」與數值輸入框（滑桿精調不易）。

### 4d. 語言選擇建議

| 場景 | 建議語言 | 理由 |
|---|---|---|
| Debugger UI、WebGPU 編排、diff 管線 | **TypeScript（現狀即可）** | HLSTranspiler/GPUExecutor/PixelComparator/runtime 全是 TS 且無 Node 依賴，直接重用；重寫成本零效益 |
| WESL 驗證（已存在） | Rust→WASM（現狀） | `debugger/src/wesl-web/` 已運作 |
| HLSL preprocessor / 容錯 parser | **Rust→WASM（選擇性）** | 僅當 PEG parser 對特定上游 HLSL 構造（複雜巨集、`#pragma` 變體）失敗時值得投入；可參考 rust-hlsl/dxil 系工具做成 WASM 模組與 WESL 並列。**不建議重寫 transpiler 主體**——PEG parser + WGSLCodeGenerator 已服務 126 檔且與 CLI 驗證共用同一實作 |
| DDS/權重檔檢視 | TS（現有 `DDSReader.ts`） | 已有純 TS 實作，無需 Rust |
| 離線批次分析（verification_output 聚合、跨 effect 回歸報表、PSNR 趨勢圖） | **Python** | CLI 批次已是 `verify:all`；Python（pandas/matplotlib）適合跑在 CI 或本地批次產出趨勢報告，與瀏覽器互補、不進 browser |

---

## 5. 分階段實作建議

### Phase A — 瀏覽器差分驗證 MVP（優先、最高性價比）

工作項：
1. `debugger/vite.config.ts` middleware 擴充：列 `effects_src/`（.hlsl）與 `effects/`（.wgsl）並回傳配對清單（配對規則照抄 EffectPairResolver：同 category/name 優先、跨 category 唯一 name 後備）。
2. 新增 `src/verification/browser.ts`（browser-safe barrel：只 export HLSTranspiler/GPUExecutor/PixelComparator/StatisticsEngine；**不含** index.ts 的 webgpu import）。
3. debugger 新增「A/B Compare」分頁：選效果對 → transpile → 兩版執行（輸入可選測試圖樣或當前媒體影格）→ 統計卡片（MAE/PSNR/max diff/outlier%）+ side-by-side/heatmap Canvas 視圖（移植 colormap）+ transpile diagnostics 列。
4. Monaco 補 HLSL 語法高亮；HLSL 原始碼可在編輯器分頁檢視。

驗收標準：
- [ ] 在瀏覽器（Chrome/Edge）對 Bilinear、Bicubic、Lanczos、CAS 至少 4 個效果完成 transpile→執行→比對。
- [ ] 顯示的 MAE/PSNR/max diff 與 CLI（`npm run verify -- --effect <name>`）結果一致（相同輸入、相同 tolerance 下數值一致，浮點顯示位數一致）。
- [ ] heatmap 與 side-by-side 可視圖可切換、gain 可調。
- [ ] transpile 失敗的效果顯示 diagnostics 而非白屏。

### Phase B — 逐 Pass 檢視 + 效能分析

工作項：
1. Pass 列表 UI（來自 `runtime.shaderInfo.passes`）＋單 pass 執行（`dispatchPass` 包裝）＋中介紋理 readback 縮圖牆；點擊縮圖放大檢視。
2. `initWebGPU` 請求 `timestamp-query` feature（有才請求）；per-pass `writeTimestamp` → 解算 → pass timeline 甘特圖；不支援時顯示 CPU 提交時間並標註。
3. 每幀 dispatch 次數統計；`ResourceManager` 紋理枚舉 → 記憶體用量表（總計/逐紋理）。
4. FPS 折線圖（最近 300 幀 + p50/p95/p99）。

驗收標準：
- [ ] Anime4K_Upscale_L（多 pass）可逐 pass 步進並檢視每個中介紋理。
- [ ] 支援 `timestamp-query` 的平台上，pass timeline 顯示 GPU 時間；總和與整幀牆鐘時間量級合理。
- [ ] 記憶體表總和與效果紋理宣告相符（手動抽查一個效果的紋理清單）。
- [ ] 不支援 timestamp-query 的環境不 crash、顯示 fallback 指標。

### Phase C — 介面重構 + 語言擴充

工作項：
1. `main.ts` 拆模組：`editor/`、`preview/`、`compare/`、`profile/`、`api/`（middleware client）、`state/`；debugger 加入最小測試（配對規則、colormap、格式轉換的純函式單測）。
2. 三欄版面 + 檔案樹（examples/effects/effects_src，配對成對顯示）＋深淺色切換。
3. Middleware 補 POST 保存（編輯器存回磁碟），hot reload 完整雙向。
4. 像素探針：canvas 點擊 → 顯示該座標 HLSL/WGSL 輸出值與逐 channel 誤差。
5. （選擇性）Rust WASM HLSL preprocessor 針對 PEG parser 失敗的構造補強；Python 離線批次趨勢報告工具（讀 verification_output/*.json）。

驗收標準：
- [ ] 可依檔案樹瀏覽全部 126 對效果並載入，無白屏/crash。
- [ ] 編輯器內容可存回磁碟，auto-sync 不覆蓋未保存編輯（現有保護邏輯保留）。
- [ ] 深淺色即時切換，Monaco 與 UI 一致。
- [ ] 點擊預覽任一像素可看到 A/B 值與誤差。
- [ ] `main.ts` 降於 ~300 行；核心純函式有單測。

---

## 6. 風險

| # | 風險 | 緩解 |
|---|---|---|
| R1 | `src/verification/index.ts` 與 `VerificationRunner.ts` 頂層 `import 'webgpu'`（node-webgpu）進入瀏覽器 bundle 即 crash | Phase A 以模組路徑個別 import；新增 `browser.ts` barrel；嚴禁 barrel 內引用 index.ts |
| R2 | SSIM 純 JS O(N·w²)：1080p 每通道約需 10⁸ 級乘加，會凍結 UI 數秒 | 預設關閉或 4x 降取樣；開啟時移入 Web Worker；中長期可 GPU compute 化 |
| R3 | 大紋理 readback（2x 超分 4K ≈ 64MB/張）阻塞主執行緒 | 限制比對解析度上限（如 ≤1080p 輸入）；readback 放 Worker；縮圖牆以 downsample 讀取 |
| R4 | `timestamp-query` 在 Safari 與部分驅動不可用 | feature 偵測 + CPU 時間 fallback，UI 明確標示量測類型 |
| R5 | 中介紋理 readback 需 usage 含 `COPY_SRC`；`ResourceManager.createTexture` 的 usage 來自 effect 描述符，未必齊備 | Phase B 先以現有效果實測；不足時在 ResourceManager 建立時一律 OR 上 `COPY_SRC`（小幅改動、對渲染無副作用） |
| R6 | HLSTranspiler 對部分上游 HLSL 構造（巨集、特殊語法）仍會 transpile 失敗 | UI 完整呈現 diagnostics；配對清單標記「可轉/不可轉」狀態；這正是除錯工具的價值而非阻礙 |
| R7 | Monaco worker 現為空設定，大檔案（CuNNy 數千行 WGSL）編輯效能差 | Phase C 一併修復 worker 設定 |
| R8 | debugger 與主專案依賴脫鉤（兩份 package.json） | Phase A 不動依賴（verification 模組以 alias 引入原始碼）；Phase C 可考慮 workspace 化 |

---

## 7. 環境限制註記

- 審計過程中所有目標檔案均可讀取，無讀取受阻的檔案。
- 依任務規範未執行 `npm install`/`build`，也未實際執行 debugger/verification —— 所有「瀏覽器可用性」結論基於靜態原始碼分析（grep 確認依賴面）與既有建置產物（`dist/wgfx.browser.esm.js` 已被 debugger 使用）的交叉驗證。
- `verification_output/` 目錄未於本次盤點範圍內（任務未要求檢視歷史報告內容）。
