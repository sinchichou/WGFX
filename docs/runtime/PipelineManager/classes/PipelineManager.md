[**WGFX API Documentation v1.0.0**](../../../index.md)

***

[WGFX API Documentation](../../../modules.md) / [runtime/PipelineManager](../index.md) / PipelineManager

# Class: PipelineManager

Defined in: [runtime/PipelineManager.ts:31](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/PipelineManager.ts#L31)

Manages the creation and execution of compute pipelines.
---
管理 WebGPU 計算管線的建立、編譯監控與執行分發。

## Constructors

### Constructor

> **new PipelineManager**(`device`, `resourceManager`): `PipelineManager`

Defined in: [runtime/PipelineManager.ts:37](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/PipelineManager.ts#L37)

#### Parameters

##### device

`GPUDevice`

##### resourceManager

[`ResourceManager`](../../ResourceManager/classes/ResourceManager.md)

#### Returns

`PipelineManager`

## Properties

### device

> **device**: `GPUDevice`

Defined in: [runtime/PipelineManager.ts:32](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/PipelineManager.ts#L32)

***

### pipelines

> **pipelines**: `Map`\<`number`, `StoredPipeline`\>

Defined in: [runtime/PipelineManager.ts:35](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/PipelineManager.ts#L35)

Map of pass index to pipeline / 通道索引與管線的映射

***

### resourceManager

> **resourceManager**: [`ResourceManager`](../../ResourceManager/classes/ResourceManager.md)

Defined in: [runtime/PipelineManager.ts:33](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/PipelineManager.ts#L33)

## Execution

### dispatchPass()

> **dispatchPass**(`passInfo`, `commandEncoder`): `void`

Defined in: [runtime/PipelineManager.ts:159](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/PipelineManager.ts#L159)

Dispatch a compute pass.
---
執行計算通道。

#### Parameters

##### passInfo

`any`

Pass metadata / 通道元數據

##### commandEncoder

`GPUCommandEncoder`

GPU command encoder / 指令編碼器

#### Returns

`void`

## Lifecycle

### dispose()

> **dispose**(): `void`

Defined in: [runtime/PipelineManager.ts:236](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/PipelineManager.ts#L236)

Clear all cached pipelines.
---
清除所有管線緩存。

#### Returns

`void`

## Pipelines

### createPipelines()

> **createPipelines**(`shaderInfo`, `generatedModules`): `Promise`\<`void`\>

Defined in: [runtime/PipelineManager.ts:52](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/PipelineManager.ts#L52)

Create pipelines for all shader modules.
---
為所有產生的 WGSL 模組建立計算管線。

#### Parameters

##### shaderInfo

`any`

Global shader metadata / 全域著色器元數據

##### generatedModules

`any`[]

Generated WGSL modules / 產生的 WGSL 模組

#### Returns

`Promise`\<`void`\>
