# CNN 效能審計報告（perf-audit）

> 子代理：CNN 效能審計｜日期：2026-11-13｜範圍：只讀審計 + 本報告（未改任何 src/、effects/ 檔案）
> 觸發問題：使用者回報「這 CNN 不大，但是跑會掉 FPS，肯定有問題，FP16?」
> 審計標的：`src/WGFX.ts`、`src/runtime/{WGFXRuntime,PipelineManager,ResourceManager,WGSLCodeGenerator,DispatchExpr}.ts`、
> `test/bench_hotpath.mjs`、`test/bench_runtime.mjs`、`effects/CuNNy/CuNNy-2x4C-NVL.wgsl`（主）、
> `effects/CuNNy/CuNNy-{8x8C,8x16C,16x16C}-NVL.wgsl`（對照）、`effects/OpenNR/OpenNR_1pass.wgsl`（buffer 路徑對照）。
> 硬體脈絡：Radeon 780M iGPU（RDNA3，12 CU，共享 DDR5 頻寬約 100–200 GB/s 且與 CPU/視訊解碼共用）。

---

## 1. 熱路徑資料流圖（每幀，文字版）

```
WGFX.process(inputSource)                                    [src/WGFX.ts:300]
 │
 ├─ runtime.advanceFrame()                                   影格計數器 +1（純 JS）
 ├─ resourceManager.updateTextureFromImage('INPUT', src)     [ResourceManager.ts:662]
 │    ├─ queue.copyExternalImageToTexture                    ← 每幀上傳（必要，無法省）
 │    └─ updateSceneBuffer → queue.writeBuffer(48B)          sceneScratch 重用（P3 已零配置）
 ├─ device.createCommandEncoder('WGFX Frame Processing')     每幀一個 encoder
 ├─ runtime.dispatchAllPasses(encoder)                       [WGFXRuntime.ts:155] 直接走 pass 元數據
 │    └─ for pass of shaderInfo.passes:                      （無字串解析、無 regex）
 │         PipelineManager.dispatchPass(pass, encoder)       [PipelineManager.ts:253]
 │           ├─ revision 檢查 → 命中快取 bindGroup           （資源版本未變 ⇒ 零配置）
 │           ├─ beginComputePass() / setPipeline / setBindGroup
 │           ├─ //! DISPATCH：dispatchCache 命中 ⇒ 零求值     （P1 預編譯 + 記憶化）
 │           │  否則：gridRevision 快取的 ceil(dim/BLOCK)     （P4）
 │           ├─ dispatchWorkgroups(dx, dy, 1)
 │           └─ end()                                        ← 每 pass 一個 compute pass 邊界
 ├─ queue.submit([encoder.finish()])                          單次 submit
 └─ 回傳 OUTPUT GPUTexture（rgba16float）                     呈現由宿主負責
```

**結論：每幀 CPU 熱路徑乾淨** —— 無 mapAsync、無 readback、無每幀字串組建（log 均以
`Logger.wouldLog` 加閘）、bind group / dispatch 求值 / 網格計算三層快取皆命中。
`readStorageBuffer`（mapAsync）只存在於驗證/除錯路徑（`ResourceManager.ts:531`），不在 process() 內。
commit `8696c08`（P0–P4）已把可省的 JS 端成本拿掉；bench A–D 量的是這一層。

## 2. 審計標的物化結構（CuNNy-2x4C-NVL 為例）

| 項目 | 值 | 備註 |
|---|---|---|
| pass 數 | 4（in → conv1 → conv2 → out-shuffle） | 8x8C=10、8x16C=10、16x16C=18、OpenNR_1pass=37 |
| workgroup | `@workgroup_size(64,1,1)`，BLOCK_SIZE 8（out-shuffle 16） | Magpie Rmp8x8 映射，1 thread = 1 像素（out 1 thread = 2×2 quad） |
| 中介紋理 | t0/t1，`R8G8B8A8_SNORM → rgba8snorm`（4 B/texel） | sampled/storaged 成對別名 = 同一邏輯紋理雙 view |
| 輸出 | OUTPUT `rgba16float`（8 B/texel）於 2× 解析度 | WGFX.create:188 預設；1080p 輸入 → 4K 輸出 32 MB/幀 |
| 每像素取樣 | pass1: 9（INPUT）；pass2/3: 18（t0/t1 各 3×3 tap）；pass4: 18+4 | 皆 `textureSampleLevel` + POINT sampler，逐 thread 獨立取樣 |
| 每像素 FMA | conv pass ≈ 18 × mat4x4 = 288 FMA（f32 直線展開） | 16x16C 每 conv pass 4 組 × 36 tap = 144 樣本 |
| 權重 | 全部以 f32 literal 內嵌於 WGSL | 模組體積大、指令快取/暫存器壓力來源 |
| shared memory | 0 使用 | 8×8 tile 內 3×3 tap 的相鄰重疊完全未被重用 |
| //! CAPABILITY FP16 | **僅元數據標記，無任何執行期效果** | 見 §4 |

## 3. 根因候選排序（小 CNN 掉 FPS）

| # | 候選根因 | 證據 | 影響對象 | 預估權重 |
|---|---|---|---|---|
| 1 | **每像素取樣次數 × 全解析度中介往返 → 780M 頻寬/L2 壓力**（效果結構，framework 放大） | conv pass 逐 thread 3×3 tap 直線展開；8x8C 每 conv pass 36 樣本/px、16x16C 144 樣本/px；1080p 估算：2x4C ≈ 0.7–1 GB/幀（~50 GB/s@60，貼近邊緣）、8x8C ≈ 5–6 GB/幀（~300+ GB/s，超頻寬）；shared memory 0 重用，相鄰 thread 的重疊 tap 全靠 L2 | 780M 上所有 CuNNy 變體；越大越炸 | ★★★★★（主兇候選） |
| 2 | **FP16 完全未啟用 → f32 算力/佔用率損失** | `shaderInfo.metadata.capability` 無任何消費者（全 src/ 僅 3 處：型別定義、HLSTranspiler 空 metadata、pegjs 寫入）；無 `enable f16;`、無 shader-f16 feature 檢查；權重 f32 literal 內嵌；f0 直線 ~300 FMA + 18 個 vec4 同時存活 → 暫存器壓力高、780M 12 CU waves-in-flight 低 | 所有 CNN 的計算吞吐；對 latency-bound 的 iGPU 影響顯著 | ★★★★（次兇/最大未兌現紅利） |
| 3 | **pass 數 × 全螢幕中介 roundtrip 的固定 GPU 開銷** | 每 pass 一次 beginComputePass + 全解析度 rgba8snorm 寫+讀（每對 pass ~16 MB@1080p）；pass 數 4→18→37 線性放大；write-only storage + sampled 分離使 read-after-write 必為全域同步 | 8x8C 以上與 OpenNR（37 pass）明顯 | ★★★ |
| 4 | **輸出路徑頻寬**：OUTPUT rgba16float@2x（8 B/texel）+ 宿主再複製/繪製到 canvas | WGFX.ts:188 固定 rgba16float；1080p 輸入輸出 32 MB/幀，宿主呈現再加一次全螢幕 pass | 所有 2× 上取樣效果 | ★★ |
| 5 | **上傳路徑**：copyExternalImageToTexture 每幀（VideoFrame→rgba8unorm） | WGFX.ts:344 + ResourceManager.ts:670；功能必要，僅可確認宿主沒有重複上傳/縮放 | 全部效果 | ★（必要成本） |
| 6 | 每幀 CPU encode/submit | 8696c08 P1–P4 後已快取化；bench_hotpath A–D 可回歸量測 | 高 pass 數效果 | ✱（已優化，非現行問題） |
| 7 | mapAsync / 同步停頓 | 熱路徑零 await、零 readback（僅驗證路徑有） | — | ✱（排除） |
| 8 | codegen 多餘 wrapper / 取樣次數膨脹 | 8696c08 P0 已修 `_wrapU32ArgsInF32VectorCtors` 遞迴包裹；`vec4<f32>(textureSampleLevel(...))` 為無害恆等包裹；取樣次數由效果檔 tap 結構決定，非 codegen 膨脹 | — | ✱（排除） |

**780M 特有因素**：iGPU 與 CPU 共享 DDR5（實際可用頻寬遠低於桌面卡），且瀏覽器同時做視訊解碼＋合成；
「FLOPS 小」的 CNN 也可能撞頻寬牆（候選 #1），而 f32 直線展開的低佔用率（候選 #2）讓 12 CU 餵不飽。
workgroup 64（1 wave64）+ 早退在 780M 上屬健康形狀，非問題；RAVU 的 19–23 KB workgroup storage
（README:77）是另一家族的限制，與 CNN 掉幀無關。

## 4. FP16 現況結論

**結論：使用者的猜測對一半——FP16 確實完全沒被用上，但它不是現有中介頻寬的瓶頸來源，而是「計算吞吐與佔用率」層面的最大未兌現紅利。**

事實鏈：
1. `//! CAPABILITY FP16` 被 parser 收進 `shaderInfo.metadata.capability`（`ShaderParser.pegjs:117`），
   之後**沒有任何 runtime/codegen/資源層程式碼讀取它**（全 src/ 檢索僅型別定義與一處空 metadata）。
2. 產生的 WGSL 無 `enable f16;`；`WGFX.create` 不檢查 `device.features.has('shader-f16')`
   （裝置由宿主建立，WGFX 收到什麼用什麼）；所有運算與權重皆 f32。
3. 效果檔移植註解自證：`CuNNy-2x4C-NVL.wgsl:35-37`「//!CAPABILITY FP16 are capability markers,
   kept as-is; arithmetic runs in f32」；`tools/cunny2wgsl.mjs:23` 同述；`docs/porting-guide.md:246`
   「//! CAPABILITY FP16 僅為標記」。
4. 中介紋理：CuNNy 用 rgba8snorm（4 B/texel）——**頻寬紅利已被 rgba8snorm 拿走一半以上**，
   即使補上 rgba16float 也不會更省。真正缺的是：
   - **計算**：RDNA3 的 f16 packed math 理論 2× 吞吐，且 f16 暫存器用量減半可拉升佔用率；
   - **OpenNR 路徑**：特徵圖已是 `u32x2`（4×packed f16）storage buffer，**頻寬省了但解包回 f32 運算**，
     計算紅利未兌現（`OpenNR_1pass.wgsl:11`）。
5. 数值風險已知且可控：`test/f16_conversion.mjs` 證明 Dawn 寫 rgba16float 是**向零截斷**而非 RNE，
   通用容差需 ≥1e-3；驗證管線（`npm run verify` / `verify:batch`、`test:opennr` 18 層 ≤1e-2）可作為
   FP16 變更的像素驗收閘。

## 5. 量測計畫（先量後改，隔離變數）

| 步驟 | 動作 | 隔離什麼 | 做法 |
|---|---|---|---|
| M1 | **區分 CPU encode vs GPU bound** | 排除框架熱路徑 | 以 `bench_hotpath.mjs` 為模板新增 case E：載入 `effects/CuNNy/CuNNy-2x4C-NVL.wgsl`（256×256），量 (a) 純 encode+submit（現行 D 的形式，CPU 上限）與 (b) 每幀 `await onSubmittedWorkDone()`（GPU 實際牆鐘）。(a) 低 (b) 高 ⇒ GPU bound，框架無罪。 |
| M2 | **逐 pass 定位最貴者** | 候選 #3 vs #1 | timestamp-query（需 device feature）或逐 pass bench：只 dispatch pass N（N=1..4）各量 GPU 時間；對照 tap 數理論值。若 pass2/3 ≈ pass1 的 2×，符合取樣量模型 ⇒ 候選 #1 主導。 |
| M3 | **頻寬 vs 算力判別** | 候選 #1 vs #2 | (a) 實驗副本把中介 FORMAT 換 rgba16float（頻寬 ×2）：FPS 幾乎不變 ⇒ 頻寬非瓶頸（算力/延遲 bound）；(b) `--size 128` 半解析度：時間線性下降 ⇒ 頻寬/解析度 bound。 |
| M4 | **FP16 A/B** | 候選 #2 | 手工做一份 CuNNy-2x4C f16 實驗副本（**不改 effects/ 原檔**）或在 codegen 層試 `enable f16`：780M 與 RTX 4060 各跑一輪 M1(b)，再用 `npm run verify:batch` / `test:opennr` 對照逐像素差。 |
| M5 | **宿主整合面** | 排除宿主因素 | 在 YouTube VSR 內量每幀 `process()` 前後 + `onSubmittedWorkDone` 的分佈，對照「無效果」與「NIS/CAS 等輕效果」基線，確認掉幀來自 CNN 本身而非擴充的呈現路徑。 |

不改動 `test/` 既有檔案；M1–M3 以新增檔案（如 `test/bench_cunny.mjs`）承載，沿用 bench_hotpath 的
多回合 min/median 取樣法（GPU 與其他 agent 共用時以 min 為準）。

## 6. 優化清單（排序，含預期收益/風險/驗收）

| # | 優化項 | 預期收益 | 風險 | 驗收標準 |
|---|---|---|---|---|
| O1 | **FP16 codegen 管線**：codegen 讀 `capability.FP16` → 模組發 `enable f16;`、權重/啟動值轉 `vec4<f16>`；`WGFX.create` 檢查 `device.features.has('shader-f16')` 不符則自動降級 f32 路徑（雙模組或條件產生） | 780M 計算 1.3–2×、暫存器減半拉高佔用；OpenNR 路徑可直接 f16 運算（buffer 已 packed） | 數值差異（Dawn trunc 語意）；驅動差異（D3D12 vs Dawn/node-webgpu）；需 capability gate | `npm run verify:all` 逐像素容差 ≤1e-3；`test:opennr` 18 層 ≤1e-2；2x4C 主觀無劣化；兩台 GPU 皆跑 M1(b) 對照 |
| O2 | **取樣重用**：conv pass 引入 workgroup shared memory tile（8×8+1 halo，RGBA8snorm 以 `u32` pack 載入 LDS）或改 `textureLoad` 整數座標取樣，砍掉 sampler 路徑 | 樣本數最多 −60%（3×3 相鄰重疊）；780M 頻寬敏感場景直接受益 | maxComputeWorkgroupStorageSize 16KB 限制（與 RAVU 需求衝突時逐效果 gate）；改動需逐像素等價或容差驗證 | verify 逐像素（數學相同時應 bit-level 接近）；M1(b)/M2 前後對照 ≥1.3× |
| O3 | **輸出格式選項**：`WGFXOptions` 允許宿主指定 OUTPUT 為 rgba8unorm/rgba8snorm（效果允許時）；並建議宿主直接取樣 OUTPUT 呈現，避免額外 copy | 輸出頻寬 −2×（8→4 B/texel） | 輸出精度下降需效果可接受；API 相容性 | verify:batch 全類別通過；宿主呈現路徑省一次全螢幕 pass |
| O4 | **pass 融合實驗**（conv1+conv2 中間值留 registers/shared） | 大變體收益大 | 改動深、tap 結構限定；列為實驗性 | 逐像素驗證 + M2 比較 |
| O5 | 熱路徑回歸：把 CuNNy 加入 bench_hotpath case E，防未來劣化 | 防退化 | 無 | CI 數字不回歸 >5% |

（P1–P4 既有優化保持；本清單未發現新的每幀 CPU 端浪費。）

## 7. 風險與注意事項

- **f16 是跨切面變更**：影響所有標記 CAPABILITY FP16 的效果（60+ 檔），必須 capability gate + 自動降級，
  且 node-webgpu(Dawn) 與 Chrome/D3D12 的捨入行為可能不同（已有 trunc 證據，RNE 需實測）。
- **shared memory 方案**與 RAVU 家族的 workgroup storage 需求共享同一裝置上限（16KB 預設），
  不可全域調高假設，需逐效果宣告。
- 本審計未執行任何 GPU 程序（依指示），所有數字為原始碼靜態推導 + 既有 bench 結構推論；
  M1–M4 的實測是下一步的第一優先。

---

## 附錄：Round 3 之後的優化研究備註（2026-11-14，lead）

> 使用者指示「停，研究就好」——以下為**研究結論**，未實作。

### O3-lite：`outputFormat` 輸出格式選項（研究完成，未實作）
- **收益**：OUTPUT 目前硬編 `rgba16float`（8B/texel）。升頻特效輸出像素數＝輸入×4，2x 升頻 1080p→4K 每幀輸出寫入 ≈ 33MB（16f）；改 `rgba8unorm` 減半（16.6MB），並省掉宿主呈現路徑的 16f→8unorm 轉換複製。
- **實作點**：`WGFX.create` 的 OUTPUT 外部資源 spec（`src/WGFX.ts` 硬編 `format: 'rgba16float'`）。
- **已發現的陷阱（實作前必解）**：codegen 的 OUTPUT 宣告格式來自效果檔 `//!TEXTURE`（無宣告才 fallback rgba16float，見 `WGSLCodeGenerator.ts` declaredFormat 邏輯）。若宿主建了 rgba8unorm 紋理但 WGSL 宣告 rgba16float → bind group layout 格式不符，Dawn 直接報錯。**必須**同步：(1) 外部資源格式覆寫進 codegen 的 declaredFormat 解析，或 (2) 在 ResourceManager 加格式一致性檢查並給出明確錯誤。
- **正確性**：16f→8unorm 有量化（±1/255）；驗收用 PixelComparator 寬容級（ToleranceLevel），並以視覺確認無 banding。
- **風險**：低（opt-in，預設不變）；影響面：所有直接上 canvas 的宿主（VSR 即是）。

### O1（FP16 數學）現況
- 基礎路徑已落地（`preferF16` + `shader-f16` 偵測 + `enable f16;` 注入，閘門測試 4/4，commit `1b4fb93`）——但這只是「開關」，**還沒有 f16 數學轉換**。
- 下一步（研究結論）：f16 模式下 (a) 中介紋理 rgba8snorm→rgba16float 維持語意、(b) COMMON 輔助函式內的 f32 數學改 f16（需顯式 vecN<f16> 邊界轉換；uniform 與 textureLoad 邊界保持 f32）。**本機 4060 shader-f16=false 無法驗證**，驗收必須在 RDNA3（780M）真機 + verify 差分門檻。
- 重大提醒：CuNNy 中介已是 rgba8snorm（頻寬紅利已拿），f16 的紅利在 **RDNA3 計算吞吐（f16 rate ×2）與佔用率**，對 OpenNR（packed f16 buffer 卻 f32 運算）收益最大。

### O2（shared-memory tap 重用）研究結論
- 主兇解法，但＝codegen 重構級工程：需偵測 pass 內取樣 tap 集合 → 發射 workgroup shared 陣列 → 協同載入 + barrier → 取樣呼叫改讀 shared。相鄰 thread 3×3 tap 重疊 2/3，理論省 2–3× 紋理讀取。
- 前置條件：逐效果驗證（127 個效果 BLOCK_SIZE/取樣模式各異）；建議先對 CuNNy 家族（textureLoad 點取樣、tap 集合固定）做原型，用 `npm run verify` 差分 + bench 前後對照。
- 1080p 才是主戰場（case E 實測：256² 框架 overhead 主導、GPU 才 28µs；512²→92µs 線性上升）。

### 優先序建議（下次實作輪）
1. ~~O3-lite（半天、可本機驗證、所有效果受益）~~ ✅ **已實作**（2026-11-14，commit `e2d1b4c`）：
   `WGFXOptions.outputFormat`／`WGFXRuntime.compile` 選項。實作要點＝新增 codegen
   `formatOverrides` 機制（宿主格式覆寫優先於效果 `//!FORMAT`），GPUTexture 與 WGSL
   宣告兩端一致（`storageCapableFormat` 正規化一次）。**實測**（RTX 4060，Bilinear
   512²→1024²，5×30 幀）：rgba16float 0.128 ms/幀 vs **rgba8unorm 0.113 ms/幀（快 ~11%）**，
   輸出寫入 8MB→4MB/幀；數值差 ≤1/255（純量化，`test/output_format.test.mjs` 5/5 全過、
   `npm test` 140/140 無回歸）。宿主採用：`WGFX.create({..., outputFormat: 'rgba8unorm'})`。
2. O2 原型限 CuNNy 家族（1–2 天、本機可驗證、直接打主兇）→ **下一個目標**
3. O1 f16 數學（需 780M 真機驗收；配合 VSR 實測 FPS）

---

## 附錄 B：O2 shared-memory 原型實測（2026-11-14，負結果）

> 「先量測後動刀」的示範案例：原型已實作並實測，**在 RTX 4060 上是負優化**。

### 實作
`tools/experimental/smem/`：
- `make_smem.mjs` — 把 CuNNy WGSL 的 3×3 全-texel tap pass 轉成 workgroup shared-memory
  tile 版（10×10 tile 協同載入 + `workgroupBarrier` + `textureLoad(clamp)` 取代
  `textureSampleLevel`；out-shuffle pass 不動）。原理：8×8 workgroup 內 9 tap
  有 5.76× 冗餘，理論上紋理讀取可降 5.76×。
- `verify_smem.mjs` — 像素一致性驗證：**MAE=0 / max=0**（轉換無數值差）✅
- `bench_smem_ts.mjs` — timestamp-query 量測（K-repeated 對抗 65.536µs 粒度）
- `CuNNy-2x4C-NVL-smem.wgsl` — 轉換後的效果檔（3/4 pass 轉換，out-shuffle 保留）

### 實測（RTX 4060 Laptop，case E 的 timestamp harness，256×256）
| 變體 | GPU 執行（中位數） | per-pass |
|---|---|---|
| original | **0.028 ms** | in 3µs / conv1 8µs / conv2 8µs / out 9µs |
| smem | **0.036 ms（+29% 慢）** | in 6µs / conv1 10µs / conv2 9µs / out 9µs |

### 結論
1. **RTX 4060 上 texture cache 已吸收 tap 冗餘**——硬體 L1/L2 讓 9-tap 直線取樣的
   實際記憶體流量遠低於理論值；barrier＋協同載入＋shared 讀取的開銷反而更貴。
2. **780M 上未證明**：頻寬受限 iGPU 的 L2 較小、且 1080p 才進入 GPU-bound 區間
   （本附錄 A 的 case E 數據），O2 在 780M 仍有可能划算——但必須**先在真機量測**
   再投資。本原型讓那個量測變成 5 分鐘的事：
   `node test/bench_cnn_caseE.mjs --effect tools/experimental/smem/CuNNy-2x4C-NVL-smem.wgsl`
3. **工程教訓**：負結果被 timestamp 量測擋下，避免了一個會讓桌上型 GPU 變慢的
   「優化」進主線。Wall-clock 量測（框架 overhead 主導）會完全看不到 GPU 端差異。
4. 轉換腳本保留：若 780M 實測划算，`make_smem.mjs` 的邏輯可移植進
   `tools/cunny2wgsl.mjs` 產生器，家族 20 檔一次重生成。

---

## 附錄 C：780M iGPU 實機基線 + O2 判定關閉（2026-11-14，main 主幹）

> `bench_cnn_caseE.mjs` 新增 `--adapter low-power`（→ Radeon 780M）後補完的量測。
> f32 路徑、`CuNNy-2x4C-NVL`，timestamp-query K-repeated，中位數。

### 780M f32 基線（vs 既有 4060 數據）

| 尺寸 | 4060 GPU 執行 | 780M GPU 執行 | 780M per-pass（in/conv1/conv2/out） | 780M GPU saturated |
|---|---|---|---|---|
| 256² | 0.028 ms | **0.147 ms（5.3× 慢）** | 18 / 39 / 39 / 51 µs | 0.219 ms/幀（~4570 FPS） |
| 384² | 0.054 ms | **0.318 ms（5.9× 慢）** | 41 / 86 / 86 / 111 µs | 0.491 ms/幀 |

- **線性確認**：256²→384² 面積比 ×2.25，GPU 時間 ×2.16 ⇒ 線性外推 **1080p 輸入 ≈ 4.7–4.9 ms/幀（780M）**——CuNNy-2x4C 單效果即吃掉 60 FPS 預算（16.6ms）的 ~30%；更大的 CuNNy 變體（8x8C ≈ 2.5× taps）將直接逼近/突破預算 ⇒ 「小 CNN 掉 FPS」在 780M + 1080p 是真實風險。
- node-webgpu（dawn.node 0.3.8）在 `--size >= 512` 無訊息崩潰（與 writeTexture 量無關，`--no-upload --frames 1` 仍炸）⇒ ≥512 的實測必須走 Chrome（Phase 2 瀏覽器 bench）。
- 兩張卡的 `shader-f16` 皆為 false（D3D12 後端 DXC 不可用，`allow_unsafe_apis`/`use_dxc` toggles 無效）⇒ **f16 GPU 實測必須走 Chrome WebGPU**。

### O2 smem 在 780M 的判定：**負優化，關閉**

| 變體（256²，780M） | GPU 執行（中位數） | per-pass |
|---|---|---|
| original | **0.147 ms** | 18 / 39 / 39 / 51 µs |
| smem | **0.172 ms（+17% 慢）** | 25 / 48 / 47 / 53 µs |

- 結論：**texture cache 在頻寬受限 iGPU 上同樣吸收了 tap 冗餘**（至少在 ≤384² 尺寸），barrier＋協同載入的開銷 > 節省的取樣。O2 在兩張卡皆負 ⇒ 從優化清單移除（腳本保留作量測工具）。
- 附錄 B §2「780M 上未證明」就此關閉。若未來在 1080p Chrome 實測翻案，重開的成本是 5 分鐘（同 harness）。

### 更新後的優先序
1. ~~O3-lite outputFormat~~ ✅ 已實作（3b0b276，+11%）
2. ~~O2 smem~~ ✅ 已實測關閉（4060 +29% / 780M +17%）
3. ~~O1 f16 數學~~ ✅ 已實測關閉 ⇒ 見附錄 D（MAE 或速度雙雙不達門檻）

---

## 附錄 D：O1 f16 數學 A/B 實測（2026-11-15，main 主幹，判定＝不落地）

> 「先量測後動刀」第二次擋下理論優化（繼 O2 之後）。預先寫死的判定規則：
> **780M 上速度 ≥1.2× 且 MAE ≤1e-3 才落地**；兩者皆未達成 ⇒ 關閉。

### 量測環境（`tools/experimental/f16/`）
- **Chrome（puppeteer-core 驅動 headless）的 Dawn 暴露 `shader-f16=true`**——node-webgpu（dawn.node 0.3.8）不暴露（附錄 C 的硬限制就此繞過）。timestamp-query 亦可取得（`--enable-features=WebGPUDeveloperFeatures`）。
- 780M 存取：headless 忽略 `powerPreference`；以 Windows 每應用 GPU 偏好（`UserGpuPreferences` 的 `GpuPreference=1;`）**暫時**切換，量完立即還原（使用者原值 `2;`）。
- 速度量測採**飽和吞吐**（N 幀連發一次 await）：timestamp 欄位的絕對值與 wall/saturated 內部矛盾（含 submit 間隙）**不可信**，已棄用；飽和值與 node 端 GPU 執行時間同量級、自洽。

### 理論上界（`verify_f16.mjs`，純 JS RNE 逐 op 模擬）
| pass | max\|Δ\| | snorm8 桶漂移 |
|---|---|---|
| PASS 1（in，9 scalar taps） | 2.4e-3 | **2.8% texel ±1 LSB**（數值域大至 1.85） |
| PASS 2/3（conv，18 matmuls） | ~1e-6 | **0**（理論位元級相同） |

### 實測 A/B（原版 f32 vs 兩變體；MAE 在 4060 與 780M 完全重現 ⇒ 數值路徑穩定）

| 尺寸(780M) | original 飽和 | f16conv 飽和（加速比） | f16 飽和（加速比） | f16conv MAE | f16 MAE |
|---|---|---|---|---|---|
| 256² | 0.197 ms | 0.206 ms（0.96×） | 0.191 ms（1.03×） | 3.8e-5 ✅ | 1.53e-3 ❌ |
| 512² | 0.497 ms | 0.469 ms（1.06×） | 0.456 ms（1.09×） | 8.1e-5 ✅ | 1.42e-3 ❌ |
| 1024² | 1.063 ms | 0.947 ms（1.12×） | 0.903 ms（1.18×） | 4.1e-5 ✅ | 1.22e-3 ❌ |

- f16conv（只轉 conv passes）：MAE 恆 ≤1e-3 ✅，但速度隨尺寸成長仍 **≤1.12× < 1.2× 門檻**。
- f16（含 PASS 1）：速度略高（1.18×）但 **MAE 恆 >1e-3**（pass-1 的 ±1 LSB 往下傳播放大），超過驗收門檻。
- **4060 回歸檢查**（預註規則：4060 不允許 >5% 退步）@1024²：f16conv 0.325 vs 0.347 ms（1.07×，無退步 ✅）；f16 全轉 0.469 ms（**0.74×＝-26% 退步 ❌**）——f16 全轉同時觸發 MAE 與 4060 退步兩條 DROP 規則，判定更加穩固。

### 判定與結論
1. **O1 f16 數學：不落地**。Dawn-on-D3D12 上 f16 紅利未兌現（≤1.2×，遠低於 RDNA3 packed-math 理論 2×）——可能是 D3D12 後端未發射 packed f16 指令，或工作負載在可測尺寸內仍屬記憶體/延遲主導。**FP16 不是掉 FPS 的兇手**（使用者的原始猜測就此有實測答案）。
2. `preferF16` opt-in 閘門（0e38703）保留：行為正確、預設關閉、零風險；僅 f16 **數學 codegen 路徑**（原 Phase 3）不做。
3. 優化清單狀態：O3-lite ✅ +11%；O2 ❌ 負優化；O1 ❌ 無紅利。**主要候選全部關閉**。
4. 下一步（若宿主仍掉 FPS）：轉向 **M5 宿主整合面**——附錄 C 顯示 CuNNy-2x4C 在 780M@1080p 原生僅 ~4.8ms/幀（餘裕充足），掉幀更可能來自宿主呈現路徑/更大變體（8x8C ≈ 2.5× taps ≈ 12ms/幀逼近 60 FPS 預算）或解碼合成競爭。
