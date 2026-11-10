# `.wgx` 套件包格式規格 / WGFX Package Format

`.wgx` 是 WGFX 效果的「預編」套件包 —— 一個 zip 容器，內含**已生成好的 WGSL**、
**外部紋理**與一份 **JSON manifest**。目的是讓 WGFX 能在 Chrome MV3 擴充套件等
**受限環境**中承載複雜效果（含神經網路權重），並省掉執行期的解析與程式碼生成。

> `.wgx` is the *pre-built* package format for WGFX effects — a zip container holding
> **pre-generated WGSL**, **external textures** and a **JSON manifest**.

---

## 1. 為什麼需要 / Why

| 問題 | 傳統做法 | `.wgx` 的解法 |
|---|---|---|
| **MV3 CSP 禁止動態程式碼** | runtime 用 `new Function` 求值運算式 → 直接 `EvalError` | 改用手寫 AST 求值器（`src/runtime/DispatchExpr.ts`），套件包內只有**資料** |
| **執行期解析代價高** | 每個效果載入時都要解析 DSL + codegen（實測 CuNNy-16x16C 完整 compile **2,826 ms**） | WGSL 於建置期預生成；runtime 只做 `createShaderModule`（實測快 8–22%） |
| **效果檔體積大** | 20 MiB 的 WGSL 文字（含內嵌權重） | 打包後 **11.15 MiB**（55.6%；其中 WGSL 文字部分壓縮率約 22%） |
| **外部資源散落** | LUT 紋理留在 `effects_src/`，執行期還要去那裡找 | DDS 一併收進套件包 → **自足** |

> **驗收**:套件包路徑與原效果**逐像素一致**（六個家族，含 2x/3x/4x 放大效果與
> 外部 LUT 效果），且解出的 WGSL 與執行期 codegen 的輸出**逐字相同** → 語意零風險。

---

## 2. 容器結構 / Container layout

```
example.wgx  (zip)
├── manifest.json          # 格式版本、效果 metadata、資源與 pass 清單
├── shaders/
│   ├── pass0.wgsl         # 建置期預生成的 WGSL（每個 pass 一檔）
│   └── pass1.wgsl
├── weights/               # （階段 2）二進位權重 packed-f16
│   └── conv0.bin
└── textures/              # 外部 LUT 紋理（//! SOURCE 宣告者）
    ├── ravu_3x_lut2_f16.dds
    └── AreaTex.dds
```

- **壓縮**：zip entry 以 `deflate-raw`（method 8）壓縮；若壓縮後反而更大則改用
  stored（method 0）。解壓用瀏覽器內建的 `DecompressionStream('deflate-raw')`
  （Chrome 103+），`tools/pack/unzip.mjs` 另備 Node fallback。
- **僅資料**：套件包內**不得**含任何 JavaScript。manifest 是 JSON，WGSL 是著色器
  語言，權重是二進位 → 完全不受 MV3 script-src 限制。

---

## 3. manifest 結構 / Manifest schema

```jsonc
{
  "version": 1,
  "id": "Scalers-Bicubic",
  "name": "Bicubic",
  "family": "Scalers",
  "source": "Scalers/Bicubic.wgsl",
  "tags": ["Scalers"],
  "metadata": { "sortName": "...", "version": 4, "use": {}, "capability": {} },
  "parameters": [
    { "name": "Scale", "type": "float", "default": 1, "min": 0, "max": 1, "step": 0.01 }
  ],
  "passes": [
    {
      "index": 0,
      "file": "shaders/pass0.wgsl",
      "dispatch": ["(NW+7)/8", "(NH+7)/8"],   // 可選；//! DISPATCH 運算式
      "workgroupSize": [8, 8],
      "inputs":  ["INPUT"],
      "outputs": ["OUTPUT"],
      "resources": [
        { "kind": "texture", "name": "INPUT", "binding": 10, "format": "rgba8unorm" },
        { "kind": "sampler", "name": "sam_INPUT_LINEAR", "binding": 5 }
      ]
    }
  ],
  "resources": [
    { "kind": "texture", "name": "ravu_3x_lut2", "file": "textures/ravu_3x_lut2_f16.dds" }
  ],
  "limits": { "maxStorageTexturesPerShaderStage": 8 },
  "buildTool": "wgxp/1.0.0 (manifest v1)",
  "builtAt": "2026-11-11T02:00:00.000Z"
}
```

### 欄位說明 / Field notes

| 欄位 | 意義 |
|---|---|
| `version` | manifest 格式版本（現為 `1`）。runtime 應拒絕不支援的版本 |
| `id` | 效果識別碼；由 `effects/` 相對路徑推導（`Scalers/Bicubic.wgsl` → `Scalers-Bicubic`） |
| `passes[].file` | 套件包內的 WGSL 路徑（相對套件包根目錄） |
| `passes[].dispatch` | `//! DISPATCH` 運算式；runtime 以 AST 求值器求值（**不用 `new Function`**） |
| `passes[].resources` | 該 pass 實際綁定的資源（由 codegen 的 binding map 取得） |
| `resources[]` | 全域資源：外部紋理與權重 |
| `resources[].runtimeSized` | 大小於**執行期**依輸入尺寸決定（CNN 特徵圖緩衝區等） |
| `limits` | 此效果建議的裝置上限；宿主應據此 `requestDevice` |

---

## 4. 建置期產生 / Building

```bash
# 打包單一效果
node tools/pack/wgxp.mjs effects/Scalers/Bicubic.wgsl -o dist/wgx/Scalers/Bicubic.wgx

# 批次打包整個 effects/ 樹
npm run pack        # → dist/wgx/<家族>/<效果>.wgx

# 只打包特定家族
node tools/pack/wgxp.mjs --batch effects --out-dir dist/wgx --only CuNNy,NNEDI3
```

### 產生流程 / Pipeline

1. **解析** 效果 DSL（`src/runtime/ShaderParser.js`）→ pass / resource / parameter metadata
2. **預生成** 各 pass 的 WGSL（`WGSLCodeGenerator.generate()`）—— 與執行期**同一支
   codegen**，故輸出**逐字相同**（已由測試驗證），語意零風險
3. **收集外部紋理**：`//! SOURCE` 宣告者自 `effects_src/` 依檔名解析（大小寫不敏感）
   → 一併收進 `textures/`
4. **產 manifest** → 以 `validateManifest()` 驗證結構與跨檔案一致性
5. **打包** → zip（deflate-raw）

> **尺寸彈性**：預生成時 `defines` 留空，因此產生的 WGSL 保留 `NW` / `NH` 等識別字，
> 由 runtime 依實際輸入尺寸帶入 → **同一套件包可處理任意解析度**。

---

## 5. 執行期載入 / Loading at runtime

### 5.1 高階 API

```js
import { WGFXPackage } from 'wgfx';

// 擴充套件：URL 來自 chrome.runtime.getURL()
const pkg = await WGFXPackage.fromURL(chrome.runtime.getURL('wgx/Scalers/Bicubic.wgx'));

const instance = await pkg.createInstance({
    device,
    width: 1920,
    height: 1080,
    // outWidth / outHeight 可省略 → 依效果自身的宣告（如升頻 2x/3x/4x）
    onProgress: (stage) => console.log('[wgx]', stage),
});

instance.updateUniforms({ Scale: 2.0 });
const output = await instance.process(videoFrame);   // → GPUTexture
instance.dispose();
```

`WGFXPackageInstance` 的 API 與 `WGFX` 對齊（`process` / `updateUniforms` /
`initialize` / `getOutputView` / `dispose`），消費端可兩者互換使用。

### 5.2 底層 API

```js
import { unzipPackage } from 'wgfx';

const files = await unzipPackage(arrayBuffer);      // Map<path, Uint8Array>
const manifest = JSON.parse(new TextDecoder().decode(files.get('manifest.json')));

// 各 pass 的 WGSL 已預生成 → 直接建立 shader module，不做 DSL 解析
for (const pass of manifest.passes) {
    const code = new TextDecoder().decode(files.get(pass.file));
    const module = device.createShaderModule({ code, label: `pass${pass.index}` });
    // …建立 pipeline
}
```

解壓使用瀏覽器內建的 `DecompressionStream('deflate-raw')`（Chrome 103+），
不依賴 Node API，也不使用 `eval` —— 可直接用於 MV3 擴充套件。

### 5.3 與一般管線的關係

一般效果的 `WGFXRuntime.compile()` 依序做四件事：

| 步驟 | 一般效果 | `.wgx` 套件包 |
|---|---|---|
| 1. 解析 DSL → shaderInfo | 執行期 | **建置期**（manifest 帶精簡版） |
| 2. 程式碼生成 → 各 pass WGSL | 執行期 | **建置期**（`shaders/passN.wgsl`） |
| 3. 建立資源 | 執行期 | 執行期（重用 `ResourceManager`） |
| 4. 建立管線 | 執行期 | 執行期（重用 `PipelineManager`） |

套件包路徑**完全重用**既有的 `ResourceManager` / `PipelineManager` /
`UniformBinder`，不另建一套平行執行環境 —— 因此資源語意、管線綁定、
`//! DISPATCH` 求值、裝置上限行為都與一般效果一致。

---

## 6. 驗收 / Verification

```bash
npm run test:wgxp            # 打包/解包/manifest/批次打包
npm run test:wgxp:runtime    # 套件包路徑 vs 一般路徑:逐像素一致
npm run test:dds             # 兩份 DDS 實作對真實 DDS 檔的一致性
npm run test:split           # pass 自動拆分器與等價性
```

涵蓋：
- 打包成功、產出非空、manifest 版本與 id 正確
- **解出的 WGSL 與執行期 codegen 輸出逐字相同**（語意零風險的直接證據）
- **套件包路徑與原效果逐像素一致**（六個家族，含放大效果與外部紋理效果）
- **批次打包 127/127 個效果**，16 個家族全部可解包並真跑
- 外部紋理由**套件包內**載入（驗證自足性）
- 載入時間比一般路徑快約 8–22%（省去解析與程式碼生成）

---

## 7. 已知限制 / Known limitations

- **權重仍為內嵌 WGSL 常數**（CuNNy 家族為主）。這是**實測後的定案**：
  轉為二進位 storage buffer 會使每幀慢 **375%**、編譯慢 **351%**，只換到
  39% 的套件大小 —— 得不償失。詳見 [`weight-strategy.md`](weight-strategy.md)。
- **OpenNR 類效果**需要宿主提供 54 個特徵圖 storage buffer（大小隨輸入尺寸變動），
  故其 `runtimeSized` 資源由 runtime 依輸入尺寸配置。
- **套件包不可遠端載入**：MV3 禁止遠端程式碼，`.wgx` 必須隨擴充套件出貨。
- **pass 自動拆分**（`tools/pack/split_passes.mjs`）已可用：CuNNy2 的
  `4x24` / `4x32` / `8x32` 可自動重現人工拆分結果（逐行 0 差異 + 逐像素
  max|Δ| = 0），7 條白名單規則確保無法證明語意不變時直接 throw。
