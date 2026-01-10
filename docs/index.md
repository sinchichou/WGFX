**WGFX API Documentation v1.0.0**

***

# WGFX (WebGPU Effect Framework)

![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![Status: Beta](https://img.shields.io/badge/Status-Beta-yellow)
![Platform: Web/Node](https://img.shields.io/badge/Platform-Web%20%7C%20Node.js-green)

**WGFX** is a high-performance graphics effect framework built for WebGPU. Inspired by the design philosophy of **Magpie**, it aims to provide a powerful and user-friendly platform for developing and executing effects in the Web environment. WGFX supports parsing custom `.wgsl` effect files and automatically manages GPU resources, pipeline construction, and multi-pass rendering.

Its core value lies in helping developers easily migrate shader effects originally written in HLSL (especially Magpie or MPDN style) to modern web browsers, enabling complex image post-processing features such as real-time upscaling and denoising.

---

## 🌟 Key Features

- 🚀 **High-Performance Implementation**: Direct interaction with the WebGPU API to fully leverage modern GPU parallel computing power.
- 📝 **Enhanced WGSL Format**: Uses special metadata annotations (e.g., `//! PASS`, `//! TEXTURE`) to make shader code self-describing, simplifying resource declarations and render pass configurations.
- ⚙️ **Automated Resource Management**: Automatically handles the lifecycle of `GPUTexture`, `GPUBuffer`, and `GPUSampler`, along with Bind Group generation.
- 🔄 **Runtime Compilation & Optimization**: Features a built-in Peggy-based parser to dynamically parse effect files at runtime and optimize the generated WGSL code for the target device.
- 🛠️ **Cross-Platform Architecture**: Provides pure browser support (ESM/UMD) and compatibility with Node.js environments (via `@webgpu/node`).

---

## �️ Technical Architecture

WGFX follows the "Single Responsibility Principle," breaking down complex WebGPU workflows into several core modules. For a detailed design breakdown, see the [Technical Overview](docs/technical/OVERVIEW.md).

### Project Structure

```text
WGFX/
├── src/
│   ├── WGFX.ts             # Main user-facing API entry point
│   ├── runtime/            # Core runtime modules
│   │   ├── WGFXRuntime.ts      # Engine orchestrating the entire runtime lifecycle
│   │   ├── ShaderParser.pegjs  # Effect file grammar definition
│   │   ├── ResourceManager.ts  # GPU resource (textures, buffers) management
│   │   ├── PipelineManager.ts  # Render pipeline and pass scheduling center
│   │   ├── WGSLCodeGenerator.ts# WGSL code generation and preprocessor
│   │   └── UniformBinder.ts    # Dynamic uniform parameter binding tool
│   ├── types/              # Type definitions
│   └── utils/              # General utilities (e.g., Logger)
├── examples/               # Various .wgsl effect examples (e.g., Anime4K)
├── debugger/               # Vite-based developer testing interface
├── dist/                   # Build artifacts
└── docs/                   # Automatically generated API documentation
```

---

## 📦 Installation & Build

### Installation as a Dependency

```bash
npm install wgfx
```

### Development from Source

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sinchichou/WGFX.git
   cd WGFX
   ```

2. **Install dependencies**:

   ```bash
   npm install
   ```

3. **Build the project**:

   ```bash
   npm run build
   ```

4. **Run the debugger**:
   ```bash
   npm run debug
   ```

---

## � Quick Start

### Core API Usage Example

```typescript
import { WGFX } from "wgfx";

// 1. Initialize WebGPU device
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// 2. Create WGFX instance
const effectCode = await fetch("effects/Anime4K_Upscale.wgsl").then((r) =>
  r.text()
);
const wgfx = await WGFX.create({
  device,
  effectCode,
  width: 1920,
  height: 1080,
});

// 3. Process image source (Supports Video, ImageBitmap, Canvas, etc.)
const videoElement = document.querySelector("video");
const outputTexture = await wgfx.process(videoElement);

// 4. Get output view and display
const outputView = wgfx.getOutputView();
// Use outputView in your render loop...

// 5. Update parameters in real-time
wgfx.updateUniforms({ Strength: 1.5 });
```

---

## 📄 .wgsl Effect Format Specification

WGFX effect files contain special directives and standard WGSL code:

| Directive              | Description                                        | Example                   |
| :--------------------- | :------------------------------------------------- | :------------------------ |
| `//! PASS <index>`     | Defines a render pass                              | `//! PASS 1`              |
| `//! TEXTURE <name>`   | Declares an intermediate texture resource          | `//! TEXTURE TempTex`     |
| `//! PARAMETER <name>` | Defines a dynamically adjustable uniform parameter | `//! PARAMETER Sharpness` |
| `//! COMMON`           | A shared code block for all passes                 | `//! COMMON`              |

_(For detailed specifications, refer to [API Documentation](docs/index.md))_

---

## 🙏 Credits

The development of this project is heavily inspired by the following excellent open-source projects and communities:

1.  **[Magpie](https://github.com/Blinue/Magpie)**: The core design inspiration, effect format, and parts of the architectural philosophy are derived from Magpie. Special thanks to the author **Blinue** for their outstanding contributions to the image processing field, allowing high-quality scaling algorithms to thrive in the Windows desktop environment.
2.  **[Anime4K](https://github.com/bloc97/Anime4K)**: The high-quality real-time animation upscaling algorithms in the examples are sourced from **bloc97**'s Anime4K project.
3.  **[Peggy](https://peggyjs.org/)**: A powerful parser generator that provides WGFX with stable and flexible directive parsing capabilities.
4.  **WebGPU Community**: Thanks to the W3C GPU for the Web Working Group for providing modern graphics specifications.

---

## � License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
This means if you use this code in a web service (even without distributing the package), you must open-source the relevant derivative works according to AGPL requirements.

See the [LICENSE](_media/LICENSE) file for details.
