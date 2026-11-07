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
```

> **Note**: Project uses ES Modules (`"type": "module"`), requires Node.js 18+ or modern browser.

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
├── test/                        # Test suite
├── debugger/                    # Debug tools
├── package.json
├── rollup.config.js
└── tsconfig.json
```

---

## License

MIT License