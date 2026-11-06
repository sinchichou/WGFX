# WGFX Technical Overview

## Introduction
**WGFX** is a runtime library designed to parse and execute WebGPU effect files (`.wgsl`). It is specifically optimized for migrating shader effects from HLSL (Magpie/MPDN style) to WebGPU, providing a familiar pass-based structure and automated resource management.

## Core Architecture

WGFX is built on a modular architecture where each component has a specific responsibility:

### 1. WGFX (The Facade)
The entry point for users. it manages the lifecycle of a graphics effect and provides a high-level API for processing image sources and updating parameters.

### 2. WGFXRuntime (The Orchestrating Engine)
The central engine that coordinates the interaction between the parser, resource manager, and pipeline manager. It handles the compilation of the effect code and the execution of render passes.

### 3. ShaderParser (Peggy-based)
A custom parser built using Peggy. It parses the enhanced WGSL format, extracting metadata directives like `//! PASS`, `//! TEXTURE`, and `//! PARAMETER`. It transforms the source code into an Intermediate Representation (IR).

### 4. WGSLCodeGenerator
Transpiles the internal IR and metadata into valid, standard WGSL code that can be compiled by the WebGPU driver. It handles:
- Macro expansion.
- Automatic struct injection (including `SceneInfo`).
- Texture and sampler binding declarations.

### 5. ResourceManager
Manages the lifecycle of GPU resources:
- **Textures**: Automatic allocation, resizing, and updates from various sources (Video, Canvas, ImageBitmap).
- **Buffers**: Handles uniform buffers for parameters and scene information.
- **Samplers**: Manages shared sampler states.

### 6. PipelineManager
Handles the creation and caching of `GPUComputePipeline` objects. It manages `GPUBindGroup` generation and ensures that each pass has the correct resource bindings before execution.

### 7. UniformBinder
A utility for efficiently updating uniform buffer data when parameters change at runtime.

## Execution Flow

1. **Initialization**: User provides a `GPUDevice` and effect code.
2. **Parsing**: `ShaderParser` extracts headers and code blocks.
3. **Resource Creation**: `ResourceManager` allocates the required textures and buffers based on the metadata.
4. **Pipeline Compilation**: `PipelineManager` compiles compute pipelines for each defined pass.
5. **Processing Loop**:
    - Update the `INPUT` texture with new frame data.
    - Dispatch each pass in order.
    - Retrieve the `OUTPUT` texture view.

> **Low-level contract**: when using `WGFXRuntime` directly, the scene uniform buffer is
> **not** written automatically — call `resourceManager.updateSceneBuffer(inW, inH, outW, outH)`
> after `compile()` (and whenever dimensions change), otherwise passes whose code checks
> `scene.inputSize` bounds will silently early-return. The high-level `WGFX.create()`
> handles this for you.

## Metadata Directives

| Directive | Purpose |
| :--- | :--- |
| `//! VERSION <n>` | Effect format version. |
| `//! PARAMETER <name>` | Declares a user-adjustable uniform variable (HLSL style: `float X;` / WGSL style: `var X: f32`). Supports `//! LABEL`, `//! MIN`, `//! MAX`, `//! DEFAULT`, `//! STEP`, `//! TYPE float\|int\|vec2\|...`. |
| `//! TEXTURE <name>` | Declares a texture (HLSL style: `Texture2D X;` / WGSL style: `var X: texture_2d<f32>`). Supports `//! WIDTH`, `//! HEIGHT`, `//! FORMAT`. The `X_sampled`/`X_storaged` pair refers to one logical texture. |
| `//! SAMPLER <name>` | Declares a sampler (`SamplerState X;`). Supports `//! FILTER LINEAR\|POINT`, `//! ADDRESS WRAP\|CLAMP`. |
| `//! COMMON` | Shared code included in every pass. Structs/aliases/constants are emitted in all modules; helper functions are pruned to passes that call them. |
| `//! PASS <n>` | Defines a pass. Supports `//! DESC`, `//! IN`, `//! OUT`, `//! BLOCK_SIZE`, `//! NUM_THREADS`. |
| `//! END` | Explicitly closes the current block. |

## Resource Binding Layout

The WGSLCodeGenerator is the single owner of binding indices:

| Binding | Resource |
| :--- | :--- |
| 1 | `uniforms` — adjustable parameters (16-byte slot per parameter) |
| 4 | `scene` — `SceneInfo` (inputSize, inputPt, outputSize, outputPt, scale) |
| 5–9 | Declared samplers |
| 10+ | Declared textures (`X_sampled`/`X_storaged` aliases share one backing texture) |

Pass modules declare only the resources their code (or called COMMON helpers) actually
reference; each parameter occupies a 16-byte slot matching the host-side uniform buffer
layout exactly.
