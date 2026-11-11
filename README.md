# WGFX — WebGPU Effect Framework

**WGFX** is a lightweight **WebGPU effect framework** designed to parse and execute **Magpie-style `.wgsl` effect files**. It provides a high-level API (`WGFX` class) and a low-level runtime (`WGFXRuntime`) enabling developers to:

- 🎨 **Load & compile** WGSL effect code with special directives
- 🔧 **Dynamically adjust** uniform parameters (real-time effect tuning)
- ⚡ **Multi-pass rendering** support (complex compute shader pipelines)
- 📦 **Automatic resource management** (textures, samplers, uniform buffers, storage texture paired aliases)
- 🌐 **Browser & Node.js** dual-environment support (via `webgpu` polyfill)

---

## Installation

```bash
npm install
npm run build     # emits dist/ (bundles + .d.ts); also runs automatically on `npm install`/`npm pack`
```

> **Note**: Project uses ES Modules (`"type": "module"`), requires Node.js 18+ or modern browser.

---

## Using WGFX as a library

WGFX ships as a self-contained ESM package — the browser bundle has **no external imports**,
so it can be loaded directly by a bundler, or by a browser extension via `import()`.

### Entry points

| Condition | File | Use for |
|-----------|------|---------|
| `browser` | `dist/wgfx.browser.esm.js` | Browsers / browser extensions (no Node polyfills) |
| `import` | `dist/wgfx.esm.js` | Node / bundlers |
| `require` | `dist/wgfx.js` | CommonJS consumers |
| types | `dist/index.d.ts` | TypeScript |

```js
// Named or default — both work:
import { WGFX } from 'wgfx';
import WGFX from 'wgfx';
```

### Consuming from a browser extension (MV3)

Extensions cannot resolve package paths at runtime — a
`chrome.runtime.getURL(...)` target must physically live inside the extension package.
The maintainable pattern is: depend on the package, then **vendor** its browser bundle as a
build step instead of copying it by hand.

```bash
# in the extension project
npm install ../WGFX      # or: npm install wgfx   once published
npm run sync:wgfx        # copies node_modules/wgfx/dist/wgfx.browser.esm.js into vendor/
```

```js
// content script — dynamic ESM import of the vendored bundle
const module = await import(chrome.runtime.getURL('vendor/wgfx.browser.esm.js'));
const WGFX = module.default || module.WGFX;
```

See [YouTube VSR](https://github.com/sinchichou/YouTube-VSR) for a working reference
implementation (`scripts/sync-wgfx.mjs`, `npm run sync:wgfx`, `npm run check:wgfx`).

### Device limits (host contract)

WGFX never creates the `GPUDevice` — the host does. Some effects need more than the WebGPU
**default** limits, so the host should request the adapter's supported values:

```js
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
    requiredLimits: {
        // CNN effects (CuNNy / NNEDI3) may write several storage textures in one pass
        maxStorageTexturesPerShaderStage: 8,
        // RAVU R3/R4 RGB need 19–23 KB per workgroup (default is 16 KB; D3D hosts use 32 KB)
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
});
```

| Limit | Needed by |
|-------|-----------|
| `maxStorageTexturesPerShaderStage` | CuNNy 4x24/4x32/8x32 and other multi-output CNN passes |
| `maxComputeWorkgroupStorageSize` | `RAVU_R3_RGB`, `RAVU_R4_RGB` (and any effect tiling > 16 KB per workgroup) |

---

### Pre-built packages (`.wgx`)

Effects can be shipped as **pre-built packages** — a zip container holding pre-generated
WGSL, external textures and a JSON manifest. This is the recommended way to deliver effects
to constrained hosts such as Chrome MV3 extensions:

```bash
npm run pack          # → dist/wgx/<Family>/<Effect>.wgx  (127 effects)
```

```js
import { WGFXPackage } from 'wgfx';

const pkg = await WGFXPackage.fromURL(chrome.runtime.getURL('wgx/Scalers/Bicubic.wgx'));
const instance = await pkg.createInstance({ device, width: 1920, height: 1080 });
const output = await instance.process(videoFrame);
```

- **No DSL parsing, no code generation at runtime** — the WGSL is generated at build time
- **Self-contained** — external LUT textures (`.dds`) are bundled inside
- **No `eval` / `new Function` anywhere** — safe under the MV3 default CSP
- **Verified equivalent** — package output is pixel-identical to the source effect

See [`docs/wgxp-format.md`](docs/wgxp-format.md) for the format specification and
[`docs/weight-strategy.md`](docs/weight-strategy.md) for the measured trade-offs behind the
weight-storage design.

---

## Quick Start

### 1. Get WebGPU Device

```ts
// Browser
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Node.js (needs @webgpu/node or webgpu polyfill)
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
const device = await adapter.requestDevice();
```

### 2. Load Effect Code

```ts
import { WGFX } from 'wgfx';

const effectCode = await fetch('examples/test.wgsl').then(r => r.text());
// or inline string
```

### 3. Create & Initialize WGFX Instance

```ts
const wgfx = await WGFX.create({
  device,
  effectCode,
  width: 1280,
  height: 720,
  // Optional: output dimensions (for upscaling effects)
  // outWidth: 2560,
  // outHeight: 1440,
});
```

### 4. Get Effect Info (Parameter Metadata)

```ts
const info = wgfx.initialize();
console.log(info.uniforms);
// [
//   { name: 'InvertAmount', type: 'float', default: 1.0, min: 0, max: 1, step: 0.01, label: 'Invert Amount' },
//   { name: 'Threshold', type: 'int', default: 128, min: 0, max: 255, step: 1 }
// ]
```

### 5. Process Image Input

```ts
// Supports: HTMLVideoElement, HTMLCanvasElement, ImageBitmap, VideoFrame
const video = document.querySelector('video');
const outputTexture = await wgfx.process(video);

// Get output TextureView for further rendering or readback
const outputView = wgfx.getOutputView();
```

### 6. Real-time Parameter Adjustment

```ts
wgfx.updateUniforms({
  InvertAmount: 0.5,
  Threshold: 64
});
```

### 7. Cleanup

```ts
wgfx.dispose();
```

---

## Effect File Format (`.wgsl`)

WGFX uses **Magpie-style directives** to describe effect structure, resources, and parameters. Directives start with `//!`.

### Core Directives

| Directive | Description |
|-----------|-------------|
| `//! VERSION <number>` | Effect format version |
| `//! SORT_NAME <string>` | UI sort name |
| `//! CAPABILITY <flag>` | Required hardware capability (e.g., `FP16`) |
| `//! USE <flag1,flag2>` | Enable feature flags |
| `//! PARAMETER <name>` | Declare adjustable parameter (Uniform) |
| `//! TEXTURE <name>` | Declare texture resource |
| `//! SAMPLER <name>` | Declare sampler |
| `//! COMMON` | Shared code block (all passes) |
| `//! PASS <index>` | Declare compute/render pass |
| `//! END` | End current block |

### Parameter Sub-directives (inside `//! PARAMETER`)

```wgsl
//! PARAMETER MyParam
//! TYPE float        // float, vec2, vec3, vec4, int, uint, int2, uint2...
//! MIN 0.0
//! MAX 1.0
//! DEFAULT 0.5
//! STEP 0.01
//! LABEL "My Param"  // UI display label
float MyParam;
```

### Texture Sub-directives (inside `//! TEXTURE`)

```wgsl
//! TEXTURE MyTexture
//! WIDTH INPUT_WIDTH    // Can use defined constants
//! HEIGHT INPUT_HEIGHT
//! FORMAT rgba16float   // GPUTextureFormat
Texture2D MyTexture;     // HLSL-style declaration
// or
var MyTexture: texture_2d<f32>;  // WGSL-style declaration
```

> **Paired aliases**: `*_sampled` + `*_storaged` automatically resolve to the same logical texture (one for sampling, one for storage).

### Sampler Sub-directives (inside `//! SAMPLER`)

```wgsl
//! SAMPLER MySampler
//! FILTER LINEAR       // LINEAR or NEAREST
//! ADDRESS CLAMP       // WRAP or CLAMP
SamplerState MySampler;
```

### Pass Sub-directives (inside `//! PASS <index>`)

```wgsl
//! PASS 1
//! IN INPUT, tex1_sampled    // Input textures (comma-separated)
//! OUT OUTPUT, tex1_storaged // Output textures
//! NUM_THREADS 8,8,1         // Workgroup size [x,y,z]
//! BLOCK_SIZE 16             // Optional: block size for coord calc
//! DESC "Description"        // Pass description
fn Pass1(...) { ... }
```

> **Buffer-only pass**: a pass that produces no texture output may omit `IN`/`OUT`
> entirely and declare an explicit dispatch grid instead
> （不輸出紋理的 pass 可省略 `IN`/`OUT`，改用明確的 dispatch 網格）:
>
> ```wgsl
> //! PASS 12
> //! DISPATCH (NW*NH+63)/64, 1   // workgroup counts; may use external defines
> fn Pass12(...) { ... }
> ```

### Storage Buffer Sub-directives (inside `//! BUFFER`)

Storage buffers carry CNN weights and feature maps. They are bound from group 0
at binding 40+ and declared as `array<T>` in the generated WGSL
（storage buffer 用於 CNN 權重與特徵圖，綁定於 group 0 的 binding 40+）:

```wgsl
//! BUFFER WT_conv0
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_conv0
//! MODE read_write
//! TYPE f32x4
```

| `TYPE` | WGSL element type | stride |
|--------|-------------------|--------|
| `u32x2` | `vec2<u32>` | 8 bytes |
| `f32x4` | `vec4<f32>` | 16 bytes |
| `f32`   | `f32`            | 4 bytes |

`MODE` is `read` or `read_write`; OpenNR-style effects declare every buffer
`read_write` so helper functions can share one signature
（OpenNR 類效果一律宣告 `read_write`，讓 helper 共用同一組函式簽章）.

Buffers are supplied by the host through `externalResources.buffers`
（由宿主端透過 `externalResources.buffers` 提供）:

```ts
externalResources: {
  buffers: {
    WT_conv0: { size: 4096, stride: 8, init: weightBytes },  // init optional
    FB_A:     { size: 262144, stride: 8 },
  }
}
```

> **Readback**: `ResourceManager.readStorageBuffer(name)` returns one buffer;
> `readStorageBuffers(names)` reads several with a **single** staging buffer and a
> single `mapAsync`. Prefer the batch form in verification loops — some native
> WebGPU bindings (e.g. Dawn's node-webgpu) become unstable when many
> submit + mapAsync cycles are interleaved
> （驗證迴圈建議用批次版；部分原生綁定在大量 submit + mapAsync 交錯時不穩定）.

---

## Built-in Example Effects

Located in `examples/`:

| File | Description |
|------|-------------|
| `test.wgsl` | Core example: single-pass color inversion + threshold |
| `Anime4K_Restore_Soft_UL.wgsl` | Anime4K soft edge restore + upscale |
| `Anime4K_Upscale_S.wgsl` | Anime4K pure upscale |
| `Anime4K_Restore_S.wgsl` | Anime4K restore |
| `Anime4K_Denoise_Bilateral_*.wgsl` | Bilateral denoise (3 modes) |
| `Anime4K_Upscale_Denoise_S.wgsl` | Upscale + denoise |
| `FSR_x2.wgsl` | FidelityFX Super Resolution 2x |

Located in `effects/OpenNR/` (OpenNR-ReShade port, f16 U-Net):

| File | Description |
|------|-------------|
| `OpenNR_spike.wgsl` | Single-conv proof of concept (packed f16 storage-buffer conv) |
| `OpenNR_1pass.wgsl` | Full 37-pass OpenNR network: stage → global branch → encoder → middle → decoder → head → composite |

The OpenNR effects live outside `examples/` because they need host-supplied
storage buffers (network weights and feature maps)
（OpenNR 效果需要宿主提供 storage buffer，故不放在 `examples/`）.
See [docs/opennr-pass-graph.md](docs/opennr-pass-graph.md) for the pass graph and
[docs/opennr-port-feasibility.md](docs/opennr-port-feasibility.md) for the port plan.

---

## API Reference

### `WGFX.create(options: WGFXOptions): Promise<WGFX>`

Factory method to create and initialize instance.

```ts
interface WGFXOptions {
  device: GPUDevice;           // Required: WebGPU device
  effectCode: string;          // Required: WGSL effect code
  width: number;               // Required: input width
  height: number;              // Required: input height
  outWidth?: number;           // Optional: output width override (highest priority)
  outHeight?: number;          // Optional: output height override (highest priority)
  externalResources?: any;     // Optional: custom external resources
}
```

> **Output size resolution order** / 輸出尺寸解析順序：
> 1. `outWidth` / `outHeight` explicitly passed → used as-is（明確覆寫）
> 2. Omitted → the effect's declared OUTPUT size wins, e.g. `//!WIDTH INPUT_WIDTH * 2`
>    makes FSR/Anime4K upscale effects render at 2x automatically
>    （未指定時採納效果檔宣告的 OUTPUT 尺寸，升頻特效自動以 2x 執行）
> 3. Effect declares OUTPUT without explicit size → falls back to input size
>    （效果檔未宣告尺寸時退回輸入尺寸）

### `wgfx.initialize(): WGFXInfo`

Get effect metadata (must call after `create`).

```ts
interface WGFXInfo {
  width: number;
  height: number;
  uniforms: ParameterInfo[];
  passes: number;
}
```

### `wgfx.updateUniforms(uniforms: Record<string, number | number[]>): void`

Update uniform parameter values.

```ts
wgfx.updateUniforms({
  InvertAmount: 0.5,
  SomeVec2: [0.5, 0.5],
  SomeVec4: [1, 0, 0, 1]
});
```

### `wgfx.process(inputSource, options?): Promise<GPUTexture>`

Process input source and return output texture.

```ts
const output = await wgfx.process(videoElement, {
  outWidth: 1920,   // Optional: override output width
  outHeight: 1080   // Optional: override output height
});
```

**Supported input types**:
- `HTMLVideoElement`
- `HTMLCanvasElement`
- `ImageBitmap`
- `VideoFrame`

### `wgfx.getOutputView(): GPUTextureView`

Get output texture view (for render pipeline usage).

### `wgfx.dispose(): void`

Release all WebGPU resources.

### Static Methods

```ts
WGFX.setDebug(true);           // Enable debug mode
WGFX.setLevel(LogLevel.Debug); // Set log level
```

---

## Low-level API: `WGFXRuntime`

For finer control, use `WGFXRuntime` directly:

```ts
import { WGFXRuntime } from 'wgfx';

const runtime = new WGFXRuntime(device);
await runtime.compile(effectCode, externalResources);

// Manually write scene buffer
runtime.resourceManager.updateSceneBuffer(width, height, outWidth, outHeight);

// Manually dispatch each pass
const encoder = device.createCommandEncoder();
for (const pass of runtime.shaderInfo.passes) {
  runtime.dispatchPass(`PASS_${pass.index}`, encoder);
}
device.queue.submit([encoder.finish()]);

// Or dispatch all passes in one hot-path call (no name parsing)
runtime.dispatchAllPasses(encoder);
```

---

## Build & Test

```bash
# Generate parser (PEG.js → JS)
npm run generate:parser

# Build artifacts (Rollup → dist/)
npm run build

# Run test suite (requires real GPU)
npm run test

# Benchmark
npm run bench

# Type check
npm run typecheck

# Generate docs
npm run build:docs
```

### Test Suite Details

- **Test A**: Core `test.wgsl` end-to-end (param metadata, defaults, dynamic updates)
- **Test B**: Paired alias identity verification (`sampled`/`storaged` point to same GPUTexture)
- **Test C**: All example effects compile + full pass execution + non-zero output verification
- **Test D**: Missing `WIDTH`/`HEIGHT` on intermediate textures throws explicit error

### OpenNR Acceptance (`npm run test:opennr:all`)

| Check | Script | Criterion |
|-------|--------|-----------|
| Weight converter | `tools/opennr_weights.test.mjs` | 36/36, byte-identical to upstream |
| CPU oracle smoke | `test/opennr_ref_smoke.mjs` | stage / pool / upcat / FC semantics |
| Conv equivalence | `test/opennr_conv_equiv.mjs` | fast conv is bit-identical to readable conv |
| Conv spike | `test/opennr_spike.mjs` | GPU conv vs CPU, per-channel ≤ 1e-2 |
| Full network | `test/opennr_network.mjs` | all 19 conv layers + skip S0–S3 per-channel ≤ 1e-2; noise varies with frame; composite at `Strength=0` ≈ input |

---

## Project Structure

```
WGFX/
├── src/
│   ├── index.ts                 # Public exports
│   ├── WGFX.ts                  # High-level WGFX class
│   ├── types/
│   │   ├── index.ts
│   │   └── shader.ts            # Interface definitions
│   └── runtime/
│       ├── ShaderParser.pegjs   # PEG.js grammar (core parser)
│       ├── ShaderParser.js      # Generated parser
│       ├── WGFXRuntime.ts       # Runtime core
│       ├── ResourceManager.ts   # Resource management (textures/buffers/bind groups)
│       ├── PipelineManager.ts   # Pipeline management
│       ├── UniformBinder.ts     # Uniform binding
│       └── WGSLCodeGenerator.ts # WGSL code generation
├── examples/                    # Built-in effect examples
├── effects/                     # 127 ported effects in 16 families (Anime4K, CuNNy, RAVU, OpenNR, …)
│   └── OpenNR/                  # OpenNR-ReShade port (f16 U-Net, 37 passes)
├── effects_src/                 # Upstream port references: Magpie HLSL (126 files, 125 ported; ACNet pending)
│   └── OpenNR-ReShade/upstream/ # git submodule (pinned v0.2.1)
├── tools/                       # Asset/tooling converters (opennr_weights.mjs)
├── test/                        # Test suite
├── docs/                        # Design docs (opennr-pass-graph.md, …)
├── debugger/                    # Debug tools
├── package.json
├── rollup.config.js
└── tsconfig.json
```

---

## License

License