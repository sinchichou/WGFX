# WGFX — WebGPU 特效框架

**WGFX** 是一個輕量級的 **WebGPU 特效框架**，專為解析並執行 **Magpie 風格的 `.wgsl` 特效檔案** 而設計。它提供了一個高階 API (`WGFX` 類別) 和底層執行時 (`WGFXRuntime`)，讓開發者能夠：

- 🎨 **載入並編譯** 含有特殊指令的 WGSL 特效代碼
- 🔧 **動態調整** Uniform 參數（即時調整特效強度、門檻等）
- ⚡ **多通道渲染** 支援（支援複雜的多步驟計算著色器管線）
- 📦 **資源自動管理**（紋理、取樣器、Uniform Buffer、Storage Texture 成對別名）
- 🌐 **瀏覽器與 Node.js** 雙環境支援（透過 `webgpu` polyfill）

---

## 安裝

```bash
npm install
npm run build     # 產生 dist/（bundle + .d.ts）；`npm install` / `npm pack` 時亦會自動執行
```

> **注意**：專案使用 ES Modules (`"type": "module"`)，需 Node.js 18+ 或現代瀏覽器環境。

---

## 以程式庫形式使用 WGFX

WGFX 以自足的 ESM 套件形式發布——瀏覽器 bundle **不含任何外部 import**，
可被 bundler 直接打包，或由瀏覽器擴充套件以 `import()` 載入。

### 進入點

| 條件 | 檔案 | 用途 |
|------|------|------|
| `browser` | `dist/wgfx.browser.esm.js` | 瀏覽器／擴充套件（無 Node polyfill） |
| `import` | `dist/wgfx.esm.js` | Node／bundler |
| `require` | `dist/wgfx.js` | CommonJS 消費端 |
| 型別 | `dist/index.d.ts` | TypeScript |

```js
// 具名或預設匯出皆可：
import { WGFX } from 'wgfx';
import WGFX from 'wgfx';
```

### 從瀏覽器擴充套件（MV3）消費

擴充套件無法在執行期解析套件路徑——`chrome.runtime.getURL(...)` 的目標必須實際存在於
擴充套件封裝內。可維護的做法是：**以套件相依，再把瀏覽器 bundle 以建置步驟攤銷(vendor)**，
而不是手動複製。

```bash
# 在擴充套件專案中
npm install ../WGFX      # 或日後發布後：npm install wgfx
npm run sync:wgfx        # 將 node_modules/wgfx/dist/wgfx.browser.esm.js 同步到 vendor/
```

```js
// content script — 以動態 ESM import 載入攤銷後的 bundle
const module = await import(chrome.runtime.getURL('vendor/wgfx.browser.esm.js'));
const WGFX = module.default || module.WGFX;
```

可運作的參考實作見 [YouTube VSR](https://github.com/sinchichou/YouTube-VSR)
（`scripts/sync-wgfx.mjs`、`npm run sync:wgfx`、`npm run check:wgfx`）。

### 裝置上限（宿主契約）

WGFX 不建立 `GPUDevice`，裝置由宿主建立。部分效果需要高於 WebGPU **預設**的上限，
因此宿主應索取 adapter 實際支援的值：

```js
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
    requiredLimits: {
        // CNN 效果（CuNNy / NNEDI3）單一 pass 可能寫多張 storage texture
        maxStorageTexturesPerShaderStage: 8,
        // RAVU R3/R4 RGB 每個 workgroup 需 19–23 KB（預設 16 KB；D3D 宿主為 32 KB）
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
});
```

| 上限 | 需要它的效果 |
|------|-------------|
| `maxStorageTexturesPerShaderStage` | CuNNy 4x24/4x32/8x32 及其他多輸出 CNN pass |
| `maxComputeWorkgroupStorageSize` | `RAVU_R3_RGB`、`RAVU_R4_RGB`（以及任何每 workgroup 超過 16 KB 的效果） |

---

### 預編套件包（`.wgx`）

效果可以以**預編套件包**的形式交付 —— 一個 zip 容器，內含預生成的 WGSL、
外部紋理與 JSON manifest。這是向受限宿主（如 Chrome MV3 擴充套件）供應效果的
建議做法：

```bash
npm run pack          # → dist/wgx/<家族>/<效果>.wgx（127 個效果）
```

```js
import { WGFXPackage } from 'wgfx';

const pkg = await WGFXPackage.fromURL(chrome.runtime.getURL('wgx/Scalers/Bicubic.wgx'));
const instance = await pkg.createInstance({ device, width: 1920, height: 1080 });
const output = await instance.process(videoFrame);
```

- **執行期不做 DSL 解析、不做程式碼生成** —— WGSL 於建置期產好
- **自足** —— 外部 LUT 紋理（`.dds`）一併收在套件包內
- **全程無 `eval` / `new Function`** —— 在 MV3 預設 CSP 下安全
- **已驗證等價** —— 套件包輸出與原效果**逐像素一致**

格式規格見 [`docs/wgxp-format.md`](docs/wgxp-format.md)；權重存放策略的實測取捨見
[`docs/weight-strategy.md`](docs/weight-strategy.md)。

---

## 快速開始

### 1. 取得 WebGPU 裝置

```ts
// 瀏覽器環境
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Node.js 環境（需 @webgpu/node 或 webgpu polyfill）
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
const device = await adapter.requestDevice();
```

### 2. 載入特效代碼

```ts
import { WGFX } from 'wgfx';

// 讀取 .wgsl 檔案或使用字串
const effectCode = await fetch('examples/test.wgsl').then(r => r.text());
// 或直接內嵌字串
```

### 3. 建立並初始化 WGFX 實例

```ts
const wgfx = await WGFX.create({
  device,
  effectCode,
  width: 1280,
  height: 720,
  // 可選：輸出尺寸（適用於升頻特效）
  // outWidth: 2560,
  // outHeight: 1440,
});
```

### 4. 獲取特效資訊（參數元數據）

```ts
const info = wgfx.initialize();
console.log(info.uniforms);
// [
//   { name: 'InvertAmount', type: 'float', default: 1.0, min: 0, max: 1, step: 0.01, label: 'Invert Amount' },
//   { name: 'Threshold', type: 'int', default: 128, min: 0, max: 255, step: 1 }
// ]
```

### 5. 處理影像輸入

```ts
// 支援：HTMLVideoElement, HTMLCanvasElement, ImageBitmap, VideoFrame
const video = document.querySelector('video');
const outputTexture = await wgfx.process(video);

// 取得輸出 TextureView 供後續渲染或讀回
const outputView = wgfx.getOutputView();
```

### 6. 即時調整參數

```ts
// 動態更新 uniform 值
wgfx.updateUniforms({
  InvertAmount: 0.5,
  Threshold: 64
});
```

### 7. 清理資源

```ts
wgfx.dispose();
```

---

## 特效檔案格式 (`.wgsl`)

WGFX 使用 **Magpie 風格指令** 來描述特效的結構、資源與參數。指令以 `//!` 開頭。

### 核心指令

| 指令 | 說明 |
|------|------|
| `//! VERSION <number>` | 特效格式版本 |
| `//! SORT_NAME <string>` | UI 顯示排序名稱 |
| `//! CAPABILITY <flag>` | 所需硬體能力（如 `FP16`） |
| `//! USE <flag1,flag2>` | 啟用功能旗標 |
| `//! PARAMETER <name>` | 宣告可調整參數（Uniform） |
| `//! TEXTURE <name>` | 宣告紋理資源 |
| `//! SAMPLER <name>` | 宣告取樣器 |
| `//! COMMON` | 共用代碼區塊（所有 pass 共享） |
| `//! PASS <index>` | 宣告計算/渲染通道 |
| `//! END` | 結束當前區塊 |

### 參數子指令 (`//! PARAMETER` 區塊內)

```wgsl
//! PARAMETER MyParam
//! TYPE float        // float, vec2, vec3, vec4, int, uint, int2, uint2...
//! MIN 0.0
//! MAX 1.0
//! DEFAULT 0.5
//! STEP 0.01
//! LABEL "My Param"  // UI 顯示標籤
float MyParam;
```

### 紋理子指令 (`//! TEXTURE` 區塊內)

```wgsl
//! TEXTURE MyTexture
//! WIDTH INPUT_WIDTH    // 可使用定義的常數
//! HEIGHT INPUT_HEIGHT
//! FORMAT rgba16float   // GPUTextureFormat
Texture2D MyTexture;     // HLSL 風格宣告
// 或
var MyTexture: texture_2d<f32>;  // WGSL 風格宣告
```

> **成對別名**：`*_sampled` + `*_storaged` 會自動解析為同一個邏輯紋理（一個供取樣、一個供存儲）。

### 取樣器子指令 (`//! SAMPLER` 區塊內)

```wgsl
//! SAMPLER MySampler
//! FILTER LINEAR       // LINEAR 或 NEAREST
//! ADDRESS CLAMP       // WRAP 或 CLAMP
SamplerState MySampler;
```

### Pass 子指令 (`//! PASS <index>` 區塊內)

```wgsl
//! PASS 1
//! IN INPUT, tex1_sampled    // 輸入紋理（逗號分隔）
//! OUT OUTPUT, tex1_storaged // 輸出紋理
//! NUM_THREADS 8,8,1         // 工作群組尺寸 [x,y,z]
//! BLOCK_SIZE 16             // 可選：座標計算用區塊大小
//! DESC "Description"        // 通道描述
fn Pass1(...) { ... }
```

---

## 內建範例特效

專案包含多個預設特效（位於 `examples/`）：

| 檔案 | 說明 |
|------|------|
| `test.wgsl` | 核心範例：單通道反色 + 門檻 |
| `Anime4K_Restore_Soft_UL.wgsl` | Anime4K 軟邊緣修復 + 上採樣 |
| `Anime4K_Upscale_S.wgsl` | Anime4K 純上採樣 |
| `Anime4K_Restore_S.wgsl` | Anime4K 修復 |
| `Anime4K_Denoise_Bilateral_*.wgsl` | 雙邊濾波去噪（三種模式） |
| `Anime4K_Upscale_Denoise_S.wgsl` | 上採樣 + 去噪 |
| `FSR_x2.wgsl` | FidelityFX Super Resolution 2x |

---

## API 參考

### `WGFX.create(options: WGFXOptions): Promise<WGFX>`

工廠方法，建立並初始化實例。

```ts
interface WGFXOptions {
  device: GPUDevice;           // 必填：WebGPU 裝置
  effectCode: string;          // 必填：WGSL 特效代碼
  width: number;               // 必填：輸入寬度
  height: number;              // 必填：輸入高度
  outWidth?: number;           // 可選：輸出寬度覆寫（最高優先權）
  outHeight?: number;          // 可選：輸出高度覆寫（最高優先權）
  externalResources?: any;     // 可選：自定義外部資源
}
```

> **輸出尺寸解析順序**：
> 1. 明確傳入 `outWidth` / `outHeight` → 直接採用（宿主覆寫）
> 2. 未傳入 → 採納效果檔宣告的 OUTPUT 尺寸，例如 `//!WIDTH INPUT_WIDTH * 2`
>    會讓 FSR/Anime4K 升頻特效自動以 2x 執行
> 3. 效果檔宣告 OUTPUT 但未標尺寸 → 退回輸入尺寸

### `wgfx.initialize(): WGFXInfo`

獲取特效元數據（需先 `create`）。

```ts
interface WGFXInfo {
  width: number;
  height: number;
  uniforms: ParameterInfo[];
  passes: number;
}
```

### `wgfx.updateUniforms(uniforms: Record<string, number | number[]>): void`

更新 Uniform 參數值。

```ts
wgfx.updateUniforms({
  InvertAmount: 0.5,
  SomeVec2: [0.5, 0.5],
  SomeVec4: [1, 0, 0, 1]
});
```

### `wgfx.process(inputSource, options?): Promise<GPUTexture>`

處理輸入源並回傳輸出紋理。

```ts
const output = await wgfx.process(videoElement, {
  outWidth: 1920,   // 可選：覆蓋輸出寬度
  outHeight: 1080   // 可選：覆蓋輸出高度
});
```

**支援輸入類型**：
- `HTMLVideoElement`
- `HTMLCanvasElement`
- `ImageBitmap`
- `VideoFrame`

### `wgfx.getOutputView(): GPUTextureView`

獲取輸出紋理視圖（供渲染管線使用）。

### `wgfx.dispose(): void`

釋放所有 WebGPU 資源。

### 靜態方法

```ts
WGFX.setDebug(true);           // 啟用偵錯模式
WGFX.setLevel(LogLevel.Debug); // 設定日誌層級
```

---

## 低階 API: `WGFXRuntime`

若需更精細控制，可直接使用 `WGFXRuntime`：

```ts
import { WGFXRuntime } from 'wgfx';

const runtime = new WGFXRuntime(device);
await runtime.compile(effectCode, externalResources);

// 手動寫入 scene buffer
runtime.resourceManager.updateSceneBuffer(width, height, outWidth, outHeight);

// 手動 dispatch 每個 pass
const encoder = device.createCommandEncoder();
for (const pass of runtime.shaderInfo.passes) {
  runtime.dispatchPass(`PASS_${pass.index}`, encoder);
}
device.queue.submit([encoder.finish()]);

// 或以熱路徑一次 dispatch 全部 pass（免名稱解析）
runtime.dispatchAllPasses(encoder);
```

---

## 建置與測試

```bash
# 產生 Parser（PEG.js → JS）
npm run generate:parser

# 建置產物（Rollup → dist/）
npm run build

# 執行測試套件（需真實 GPU）
npm run test

# 基準測試
npm run bench

# 型別檢查
npm run typecheck

# 產生文件
npm run build:docs
```

### 測試套件說明

- **Test A**：核心範例 `test.wgsl` 端到端驗證（參數元數據、預設值、動態更新）
- **Test B**：成對別名恆等驗證（`sampled`/`storaged` 指向同一 GPUTexture）
- **Test C**：所有範例特效編譯 + 全 pass 執行 + 非零輸出驗證
- **Test D**：中間紋理缺少 `WIDTH`/`HEIGHT` 時拋出明確錯誤

### 著色器驗證工具組（`npm run verify:*`）

差分驗證 — 將原始 Magpie HLSL 經內建 transpiler 轉換後在 GPU 實跑，
與移植版 WGSL 逐像素比較：

```bash
npm run verify -- --effect Bilinear --category Scalers   # 單一效果對
npm run verify:scalers                                   # 整個家族（批次）
npm run verify:all                                       # 所有可配對效果
```

報告（JSON / HTML / Markdown，可選 diff 熱圖）寫入 `verification_output/`
（已 gitignore）。可重用 API 位於 `src/verification/`（`quickVerify`、
`GPUExecutor.runModules` 等）；框架測試：`npm run test:verification`。
批次掃描在同一 process 內多次 submit — 大量效果請比照 `port_check`
一檔一 process 個別執行較穩。

---

## 專案結構

```
WGFX/
├── src/
│   ├── index.ts                 # 公開匯出
│   ├── WGFX.ts                  # 高階 WGFX 類別
│   ├── types/
│   │   ├── index.ts
│   │   └── shader.ts            # 介面定義
│   └── runtime/
│       ├── ShaderParser.pegjs   # PEG.js 文法（核心解析器）
│       ├── ShaderParser.js      # 生成的解析器
│       ├── WGFXRuntime.ts       # 執行時核心
│       ├── ResourceManager.ts   # 資源管理（紋理/Buffer/綁定組）
│       ├── PipelineManager.ts   # 管線管理
│       ├── UniformBinder.ts     # Uniform 綁定
│       └── WGSLCodeGenerator.ts # WGSL 代碼生成
├── examples/                    # 內建特效範例
├── effects/                     # 127 個移植效果（16 家族：Anime4K、CuNNy、RAVU、OpenNR…）
├── effects_src/                 # 上游移植參考（Magpie HLSL 126 檔，125 已移植；ACNet 待移植）＋ OpenNR submodule
├── tools/                       # 轉換器與驗收輔助工具
├── test/                        # 測試套件
├── docs/                        # 設計與移植文件
├── debugger/                    # 偵錯工具
├── package.json
├── rollup.config.js
└── tsconfig.json
```

---

## 授權

MIT License