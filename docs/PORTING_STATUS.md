# WGFX 移植現況 / Porting Status

Magpie HLSL → WGFX WGSL 移植戰役的覆蓋率、驗收方法與已知限制。
Coverage, verification method and known limitations of the Magpie HLSL → WGFX WGSL port.

> 所有「已驗收」的項目都是在真實 WebGPU 裝置（Dawn / node-webgpu）上**實際編譯、dispatch
> 全部 pass、讀回輸出**後判定，不是靜態檢查。
> Everything marked *verified* was actually compiled, dispatched and read back on a real
> WebGPU device — not merely linted.

---

## 1. 覆蓋率總表 / Coverage summary

| 家族 Family | 來源 hlsl<br>source | 產出 wgsl<br>ported | 驗收<br>verification | 來源出處<br>origin |
|---|---:|---:|---|---|
| Anime4K | 27 | 27 | ✅ 真跑 ALL PASS | `effects_src/Anime4K` |
| CAS | 2 | 2 | ✅ 真跑 ALL PASS | `effects_src/CAS` |
| CuNNy | 20 | 20 | ✅ 真跑 ALL PASS（含權重完整性） | `effects_src/CuNNy` |
| CuNNy2 | 9 | 9 | ✅ 真跑 ALL PASS（含權重完整性） | `effects_src/CuNNy2` |
| FSR | 2 | 2 | ✅ 真跑 ALL PASS | `effects_src/FSR`（EASU / RCAS） |
| FSRCNNX | 2 | 2 | ✅ 真跑 ALL PASS | `effects_src/FSRCNNX` |
| FXAA | 3 | 3 | ✅ 真跑 ALL PASS | `effects-port` 分支遺產 |
| NIS | 2 | 2 | ✅ 真跑 ALL PASS | `effects_src/NIS`（NVScaler / NVSharpen） |
| NNEDI3 | 10 | 10 | ✅ 真跑 ALL PASS | `effects_src/NNEDI3` |
| RAVU | 26 | 26 | ✅ 真跑 ALL PASS | `effects_src/RAVU` |
| Sharpen | 4 | 4 | ✅ 真跑 ALL PASS | `effects_src/Sharpen` + `effects-port`（FineSharp） |
| SMAA | 4 | 4 | ✅ 真跑 ALL PASS | `effects_src/SMAA` |
| xBRZ | 6 | 6 | ✅ 真跑 ALL PASS | `effects_src/xBRZ` |
| Filters | — | 2 | ✅ 真跑 ALL PASS | `effects-port` 分支遺產（Deband / ImageAdjustment） |
| Scalers | — | 6 | ✅ 真跑 ALL PASS | `effects-port` 遺產 + SSimDownscaler |
| OpenNR | — | 2 | ✅ 真跑 ALL PASS（37-pass f16 U-Net 全網路） | OpenNR-ReShade（原生 WGSL 路線） |
| **合計 Total** | **117** | **127** | | |

`effects_src/` 下的 13 個 hlsl 家族 **117 個來源檔全部完成移植**；另有 10 個效果來自早期
`effects-port` 分支遺產與 OpenNR（原生 WGSL），總計 **127 個效果檔**。

All 117 HLSL sources across 13 families are ported; plus 10 effects from the earlier
`effects-port` line and OpenNR, for **127 effect files** total.

### 驗收紀錄 / Verification record

最後一次全量驗收（2026-11-11，commit `9bbb485`，真實 WebGPU / Dawn）：

| 驗收項 | 結果 |
|---|---|
| 框架測試套件 `npm test`（Test A–G；Test E 掃描 `effects/` 全樹真跑） | **140 / 140** |
| 型別檢查 `npm run typecheck` | exit 0 |
| `test:dispatch`（`//! DISPATCH` AST 求值 + CSP 靜態檢查） | 30 / 30 |
| `test:split`（CuNNy 自動拆分逐像素一致） | 24 / 24 |
| `test:dds`（DDS 解析一致性 + LUT 格式辨識） | 21 / 21 |
| `test:wgxp` / `test:wgxp:runtime`（.wgx 打包 / 執行期載入） | 27 / 27、13 / 13 |
| `test:f16` / `test:scalers` / `test:external` | ALL PASS、ALL PASS、7 / 7 |
| `test:csp`（MV3 CSP 模擬 + OpenNR 37-pass 於 CSP 下真跑） | 15 PASS / 0 FAIL |
| `test:opennr:all`（權重轉換 / CPU oracle / conv 等價 / spike / 全網路） | ALL PASS |
| `port_check` 抽查（`NNEDI3_nns16_win8x4`、`Sharpen/AdaptiveSharpen`） | ALL PASS |

> 效果覆蓋：`effects/` 共 **127** 檔，Test E 逐一真跑；OpenNR 2 檔由 `test:opennr:all` 覆蓋。
> 驗收結果直接記錄於本表，不再引用 `.tmp/` 下的未版控紀錄。
> 前次全量驗收：2026-11-10，commit `8696c08`（125/125 逐一真跑 + 140/140 + typecheck exit 0）。

---

## 2. 驗收方法 / How to verify

### 通用健全性檢查（單一效果）

```bash
npm run build                                   # 產生 dist/（測試載入 dist/）
node test/port_check.mjs effects/NNEDI3/NNEDI3_nns16_win8x4.wgsl
```

每個效果檢查四件事：編譯無驗證錯誤 → 全部 pass dispatch（單一 encoder / 單一 submit）
→ OUTPUT 非全黑（確實有資料流動）→ 無 NaN，並印出 min/max/mean 統計。

> ⚠️ **一次只跑一個效果。** node-webgpu（Dawn 綁定）在同一 process 內交錯多次
> submit + mapAsync 會不穩定崩潰；批次驗收請用外部迴圈「一檔一 process」。

### 框架測試套件（全部效果）

```bash
npm test        # Test A–E；Test E 會掃描整個 effects/ 樹逐一真跑
npm run test:external   # //! SOURCE 外部紋理專屬測試
```

需要宿主額外資源（storage buffer / LUT 紋理）而無法由通用套件執行的效果，列在
`test/effects-suite.json` 的 `skip` 表並附原因。

### 外部紋理（`//! SOURCE`）

效果可宣告 `//! SOURCE <file.dds>` 要求宿主餵入 LUT：

```bash
node test/port_check.mjs effects/RAVU/RAVU_3x_R2.wgsl          # 自動自 effects_src/ 解析
node test/port_check.mjs <effect.wgsl> --texture NAME=<path>   # 顯式指定（可重複）
```

`test/port_check.mjs` 與 `test/test_runtime.js` 共用 `tools/source_textures.mjs`，
行為一致；找不到檔案時會給出含紋理名與 SOURCE 檔名的明確錯誤。

---

## 3. 裝置上限（宿主契約）/ Device limits

WGFX 不建立 `GPUDevice`——裝置由宿主建立。部分效果需要高於 WebGPU 預設的上限：

| 上限 Limit | 需要它的效果 Needed by |
|---|---|
| `maxStorageTexturesPerShaderStage`（預設 4） | CuNNy 4x24 / 4x32 / 8x32（單 pass 寫 6–8 張 storage texture） |
| `maxComputeWorkgroupStorageSize`（預設 16 KB） | `RAVU_R3_RGB`（19,264 B）、`RAVU_R4_RGB`（23,424 B） |

```js
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
    requiredLimits: {
        maxStorageTexturesPerShaderStage: 8,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
});
```

驗收端（`test/port_check.mjs`、`test/test_runtime.js`）已比照真實宿主索取這些上限。

---

## 4. 移植過程中補強的框架能力 / Framework capabilities added

| 能力 | 說明 |
|---|---|
| HLSL typed local decl + `mul()` | parser/codegen 支援任意位置的 HLSL 式區域宣告與 `mul`，含 i32/u32 型別提升 |
| module-scope const 引用 uniform | 自動改為進入點內 `var<private>` 初始化（WGSL 禁止 module-scope const 引用 uniform） |
| `//! SOURCE` 外部紋理 | parser 解析 → ResourceManager 餵入 → 驗收端自動自 `effects_src/` 解析 |
| `OUTPUT_WIDTH/HEIGHT` 尺寸運算式 | 紋理尺寸運算式的求值 context 與 `//! DISPATCH` 一致 |
| D3D/Magpie 格式名對照 | `R16_FLOAT` 等 → WebGPU 格式；storage texture 存取模式校正 |
| 多輸出 CNN pass | 支援單一 pass 寫多張 storage texture（需宿主上限配合） |

---

## 5. 已知限制與刻意偏離 / Known limitations & deviations

- **`!SOURCE` 的 DDS 需在 `effects_src/` 或由宿主指定**：效果檔只記錄檔名，不內嵌紋理。
- **SMAA 對完全軸對齊的直邊不混合**：這是 SMAA 設計語意（AreaTex 在無交叉邊時權重為 0），
  不是移植缺陷；驗證請用斜邊。
- **SMAA_Low/Medium 的 edgesTex**：來源為 `R8G8_UNORM`，WebGPU 無該格式的 write-only
  storage texture，改用 `rgba8unorm`；r/g 通道的 8-bit 量化完全相同，a/b 未使用。
- **Magpie HLSL 的 int/uint 提升怪癖逐字複刻**：例如 Anime4K Denoise_Bilateral 的 gather
  位移在邊界會環繞、Mean 的空間核實際只剩右下 3×3 有權重。為忠於來源而保留，並在各檔頭
  註明；若要對齊 upstream GLSL 需改一行，但那會偏離 HLSL。
- **NIS 的 workgroup storage 超限以重算取代**：`shEdgeMap` + 亮度磚共 27.3 KB 超過預設
  16 KB，改為每 thread 由已載入的 support 重算（相同輸入、相同運算式，位元級相同）。
- **`-s ours` 合併的歷史**：`effects-port` 分支的內容已由主線超集吸收（見 §7）。
- **每檔的移植注記**：所有語意偏離都寫在該 `.wgsl` 檔頭的 Porting notes，這是權威來源。

---

## 6. 可重現性 / Reproducibility

各家族的轉換器（HLSL → WGSL）保留在 repo 內，可重新產生逐位元相同的產物：

| 轉換器 Converter | 用途 |
|---|---|
| `tools/magpie_prescaler2wgsl.mjs` | Magpie prescaler 系（NNEDI3 模板等） |
| `tools/cunny2wgsl.mjs` | CuNNy / CuNNy2（`--all` 可全量重生成） |
| `effects/RAVU/_gen_ravu.mjs` | RAVU 26 檔（白名單式，未匹配形狀直接 throw） |

驗證輔助工具：`tools/port/check_effect.mjs`（靜態 lint）、`tools/port/wgsl_diag.mjs`
（編譯診斷）、`test/effect_sweep.mjs`（動態批次掃描）、`tools/port/weight-integrity.mjs`
（權重數字多重集比對）、`tools/port/analytic_check.mjs`（CPU 語意 vs GPU 實跑對照）。

---

## 7. 分支佈局 / Branch layout

| 分支 Branch | 狀態 Status |
|---|---|
| `main` | **主線**：在 `opennr-port` 快進基礎上前進 6 個 commit（`.wgx` 套件格式與打包、執行期載入、`//! DISPATCH` AST 求值、雙語 README） |
| `opennr-port` | 前主線；tip `d3ade00` 為 `main` 的祖先（純快進、無分歧），與 `origin/opennr-port` 同步 |
| `effects-port` | 早期扁平式 Tier-1 移植；`-s ours` 記錄式合併 `6ecc28b`，10 個效果已搬入 `effects/` 子目錄並在主線持續改良（rename 偵測確認無內容遺失） |
| `shader-verification-tool` | 未推送的本地分支，tip 與 `opennr-port` 相同（`d3ade00`），無獨有 commit；其驗證工具 WIP 存於 `git stash`（依賴不存在的 `src/verification/`，未完成） |
| `backup/before-author-fix` | 歷史快照；`git cherry` 確認 11 個 commit 的 patch 皆已等價存在於 `main`（刻意保留為安全網） |

> 分支整理為純快進／記錄式合併，**沒有任何 commit 被丟棄**；`backup/before-author-fix`
> 保留原樣，如需清理可再另行決定。
>
> Branch tidy-up was fast-forward / recorded-merge only — **no commits were discarded**.

---

## 8. 相關文件 / Related documents

- [`docs/porting-guide.md`](porting-guide.md) — 移植規範（執行模型、型別對應、綁定規則）
- [`docs/opennr-port-feasibility.md`](opennr-port-feasibility.md) — OpenNR 移植可行性評估
- [`docs/opennr-pass-graph.md`](opennr-pass-graph.md) — OpenNR 37-pass 資料流圖
- [`README.md`](../README.md) / [`README_TW.md`](../README_TW.md) — 使用方式與裝置上限
