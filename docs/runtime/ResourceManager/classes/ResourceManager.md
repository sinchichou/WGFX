[**WGFX API Documentation v1.0.0**](../../../index.md)

***

[WGFX API Documentation](../../../modules.md) / [runtime/ResourceManager](../index.md) / ResourceManager

# Class: ResourceManager

Defined in: [runtime/ResourceManager.ts:13](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L13)

Manages WebGPU resources including textures, views, samplers, and buffers.
---
管理 WebGPU 資源，包括紋理、視圖、取樣器與緩衝區的建立、查詢與釋放。

## Constructors

### Constructor

> **new ResourceManager**(`device`): `ResourceManager`

Defined in: [runtime/ResourceManager.ts:36](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L36)

Initialize the resource manager.
---
初始化資源管理員。

#### Parameters

##### device

`GPUDevice`

The active WebGPU device / 有效的 WebGPU 裝置

#### Returns

`ResourceManager`

## Properties

### device

> **device**: `GPUDevice`

Defined in: [runtime/ResourceManager.ts:15](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L15)

#### Zh

WebGPU 裝置實例

***

### samplers

> **samplers**: `Map`\<`string`, `GPUSampler`\>

Defined in: [runtime/ResourceManager.ts:21](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L21)

#### Zh

取樣器映射表：名稱 -> GPUSampler

***

### sceneBuffer

> **sceneBuffer**: `GPUBuffer` \| `null` = `null`

Defined in: [runtime/ResourceManager.ts:27](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L27)

#### Zh

場景資訊（寬高、解析度）Uniform 緩衝區

***

### textures

> **textures**: `Map`\<`string`, `GPUTexture`\>

Defined in: [runtime/ResourceManager.ts:17](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L17)

#### Zh

紋理映射表：名稱 -> GPUTexture

***

### uniformBuffer

> **uniformBuffer**: `GPUBuffer` \| `null`

Defined in: [runtime/ResourceManager.ts:25](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L25)

#### Zh

主要參數 Uniform 緩衝區

***

### uniforms

> **uniforms**: `Map`\<`string`, \{ `buffer`: `GPUBuffer`; `offset`: `number`; `size`: `number`; \}\>

Defined in: [runtime/ResourceManager.ts:23](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L23)

#### Zh

Uniform 映射表：名稱 -> 緩衝區區段資訊

***

### views

> **views**: `Map`\<`string`, \{ `sampled`: `GPUTextureView`; `storage`: `GPUTextureView`; \}\>

Defined in: [runtime/ResourceManager.ts:19](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L19)

#### Zh

視圖映射表：名稱 -> { 取樣視圖, 儲存視圖 }

## Lifecycle

### dispose()

> **dispose**(): `void`

Defined in: [runtime/ResourceManager.ts:395](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L395)

Dispose all textures and buffers to prevent memory leaks.
---
釋放所有紋理與緩衝區，防止 GPU 記憶體洩漏。

#### Returns

`void`

***

### initialize()

> **initialize**(`shaderInfo`, `externalResources`): `void`

Defined in: [runtime/ResourceManager.ts:57](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L57)

Initialize resources based on shader info and external definitions.
---
根據著色器元數據與外部定義初始化所有必要的 GPU 資源。

#### Parameters

##### shaderInfo

[`WGFXShaderInfo`](../../../types/shader/interfaces/WGFXShaderInfo.md)

Parsed shader metadata / 解析後的著色器資訊

##### externalResources

`any` = `{}`

External resource definitions / 外部資源定義

#### Returns

`void`

## Management

### createSampler()

> **createSampler**(`name`, `descriptor`): `GPUSampler`

Defined in: [runtime/ResourceManager.ts:250](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L250)

Create and register a sampler.
---
建立並註冊取樣器。

#### Parameters

##### name

`string`

##### descriptor

`GPUSamplerDescriptor`

#### Returns

`GPUSampler`

***

### createSceneBuffer()

> **createSceneBuffer**(): `void`

Defined in: [runtime/ResourceManager.ts:333](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L333)

Create the scene info uniform buffer (Width, Height, InvWidth, InvHeight).
---
建立場景資訊緩衝區（包含寬高及其倒數）。

#### Returns

`void`

***

### createTexture()

> **createTexture**(`name`, `descriptor`): `GPUTexture`

Defined in: [runtime/ResourceManager.ts:195](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L195)

Create or update a texture and its dual views (Sampled & Storage).
---
建立或更新紋理及其雙重視圖。若名稱重複則先銷毀舊有資源。

#### Parameters

##### name

`string`

Unique texture name / 紋理唯一名稱

##### descriptor

`GPUTextureDescriptor`

WebGPU texture descriptor / 紋理描述符

#### Returns

`GPUTexture`

The newly created GPUTexture

## Query

### getSampler()

> **getSampler**(`name`): `GPUSampler` \| `undefined`

Defined in: [runtime/ResourceManager.ts:264](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L264)

Get a GPUSampler object by name.
---
根據名稱獲取 GPUSampler 物件。

#### Parameters

##### name

`string`

#### Returns

`GPUSampler` \| `undefined`

***

### getSceneBuffer()

> **getSceneBuffer**(): `GPUBuffer` \| `null`

Defined in: [runtime/ResourceManager.ts:384](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L384)

#### Returns

`GPUBuffer` \| `null`

***

### getTexture()

> **getTexture**(`name`): `GPUTexture` \| `undefined`

Defined in: [runtime/ResourceManager.ts:239](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L239)

Get a GPUTexture object by name.
---
根據名稱獲取 GPUTexture 物件。

#### Parameters

##### name

`string`

Texture name / 紋理名稱

#### Returns

`GPUTexture` \| `undefined`

***

### getTextureView()

> **getTextureView**(`name`, `type`): `GPUTextureView`

Defined in: [runtime/ResourceManager.ts:223](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L223)

Get a specific view of a texture by name and type.
---
獲取紋理的特定視圖（取樣用或儲存用）。

#### Parameters

##### name

`string`

Texture name / 紋理名稱

##### type

View type: 'sampled' (for texture()) or 'storage' (for imageStore())

`"sampled"` | `"storage"`

#### Returns

`GPUTextureView`

The requested GPUTextureView

***

### getUniform()

> **getUniform**(`name`): \{ `buffer`: `GPUBuffer`; `offset`: `number`; `size`: `number`; \} \| `undefined`

Defined in: [runtime/ResourceManager.ts:275](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L275)

Get uniform buffer segment info for a specific parameter.
---
獲取特定 Uniform 項目的緩衝區位址資訊。

#### Parameters

##### name

`string`

#### Returns

\{ `buffer`: `GPUBuffer`; `offset`: `number`; `size`: `number`; \} \| `undefined`

***

### getUniformBuffer()

> **getUniformBuffer**(): `GPUBuffer` \| `null`

Defined in: [runtime/ResourceManager.ts:322](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L322)

#### Returns

`GPUBuffer` \| `null`

## Update

### updateSceneBuffer()

> **updateSceneBuffer**(`width`, `height`, `outWidth`, `outHeight`): `void`

Defined in: [runtime/ResourceManager.ts:349](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L349)

Update scene buffer with current source dimensions.
---
以目前的影像尺寸更新場景資訊緩衝區。

#### Parameters

##### width

`number`

##### height

`number`

##### outWidth

`number` = `0`

##### outHeight

`number` = `0`

#### Returns

`void`

***

### updateTextureFromImage()

> **updateTextureFromImage**(`name`, `source`, `outWidth?`, `outHeight?`): `void`

Defined in: [runtime/ResourceManager.ts:288](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/runtime/ResourceManager.ts#L288)

Update a texture's content from an external image source.
---
從外部影像來源（如 Video 或 Canvas）更新紋理內容，並同步更新場景資訊。

#### Parameters

##### name

`string`

Target texture name (usually 'INPUT') / 目標紋理名稱

##### source

Source visual element / 來源影像元素

`ImageBitmap` | `HTMLVideoElement` | `HTMLCanvasElement` | `VideoFrame` | `OffscreenCanvas`

##### outWidth?

`number`

##### outHeight?

`number`

#### Returns

`void`
