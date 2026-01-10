# WGFX Project Context

## Project Overview
**WGFX** is a runtime library and framework designed to parse and execute WebGPU effect files (`.wgsl`). It facilitates the migration of shader effects from HLSL (specifically Magpie style) to WebGPU. The project handles shader parsing, GPU resource management (textures, samplers, buffers), and pipeline coordination.

**Key Features:**
- **Custom Shader Format:** Supports `.wgsl` files enriched with metadata directives (e.g., `//! PASS`, `//! TEXTURE`) for declaring passes and resources.
- **Runtime Compilation:** Parses effect files at runtime using a generated parser (PegJS) and compiles them into WebGPU objects.
- **Dependency Management:** Automatically handles pass dependencies and resource binding.
- **Cross-Platform:** Builds for both Browser (ESM/UMD) and Node.js (CJS/ESM).

## Technical Architecture

### Core Components (`src/`)
- **Entry Point:** `src/index.ts` exports the main API.
- **Runtime Engine:** `src/runtime/`
  - **`WGFXRuntime.ts`**: The main controller orchestrating the entire lifecycle (parsing -> resource creation -> pipeline execution).
  - **`ShaderParser.pegjs`**: The grammar definition for the custom `.wgsl` format. Compiled to JS by Rollup.
  - **`ResourceManager.ts`**: Manages `GPUTexture`, `GPUSampler`, and `GPUBuffer` objects.
  - **`PipelineManager.ts`**: Handles `GPUComputePipeline` creation and `GPUBindGroup` management.
  - **`WGSLCodeGenerator.ts`**: Transpiles the internal Intermediate Representation (IR) into valid WGSL shader code, handling macro expansion and struct definitions.
  - **`UniformBinder.ts`**: Provides APIs to update uniform buffer values at runtime.

### Directory Structure
- **`dist/`**: Generated build artifacts (created via `npm run build`).
- **`examples/`**: Sample `.wgsl` effect files demonstrating syntax and features.
- **`docs/`**: Typedoc generated documentation.
- **`test/`**: Integration tests (e.g., `test_runtime.js`).

## Build and Development

### Prerequisites
- Node.js (v18+ recommended due to WebGPU dependencies).
- Dependencies: `npm install`.

### Key Commands
- **Build**: `npm run build`
  - Uses Rollup to compile TypeScript and PegJS grammar.
  - Outputs to `dist/` (ESM, CJS, and UMD formats).
- **Test**: `npm test`
  - Runs `node test/test_runtime.js`.
  - **Note:** Requires a build first (`npm run pretest` handles this).
- **Generate Docs**: `npm run build:docs` (uses Typedoc).

## Conventions and Standards

### Code Style
- **Language**: TypeScript (`.ts`).
- **Formatting**: Adheres to standard TS/JS conventions.
- **Documentation**: Code is documented with TSDoc comments for `typedoc` generation.

### Effect File Format (`.wgsl`)
The custom format relies on comments starting with `//!` to define metadata:
- **Directives**: `//! PASS`, `//! TEXTURE`, `//! PARAMETER`, `//! COMMON`.
- **Validation**: The parser enforces strict rules (e.g., specific header versions, matched `IN/OUT` declarations).
- **Migration**: Designed to ease transition from HLSL; supports concepts like `Texture2D`, `SamplerState`, and HLSL-like intrinsics via mapping.

## User Preferences
- **Language**: The user prefers interactions in **Traditional Chinese**. While code and internal docs are mixed/English, explanations and chat responses should prioritize Traditional Chinese.
