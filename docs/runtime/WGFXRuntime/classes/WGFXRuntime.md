[**WGFX API Documentation v1.0.0**](../../../index.md)

***

[WGFX API Documentation](../../../modules.md) / [runtime/WGFXRuntime](../index.md) / WGFXRuntime

# Class: WGFXRuntime

Defined in: [runtime/WGFXRuntime.ts:19](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L19)

Core runtime for WGFX, managing resources, pipelines, and execution.
---
WGFX 的核心運行時環境，負責協調資源管理、管線編譯與指令調度執行。
它是連接高層 API 與底層 WebGPU 操作的橋樑。

## Constructors

### Constructor

> **new WGFXRuntime**(`device`): `WGFXRuntime`

Defined in: [runtime/WGFXRuntime.ts:43](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L43)

Initialize the WGFX runtime environment.
---
初始化 WGFX 運行時。

#### Parameters

##### device

`GPUDevice`

The active WebGPU device / 有效的 WebGPU 裝置

#### Returns

`WGFXRuntime`

#### Throws

如果 WebGPU 裝置無效則拋出錯誤

## Properties

### device

> **device**: `GPUDevice`

Defined in: [runtime/WGFXRuntime.ts:21](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L21)

#### Zh

WebGPU 裝置實例

***

### generatedModules

> **generatedModules**: `object`[] \| `null`

Defined in: [runtime/WGFXRuntime.ts:33](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L33)

#### Zh

生成的WGSL模組，用於除錯顯示

***

### pipelineManager

> **pipelineManager**: [`PipelineManager`](../../PipelineManager/classes/PipelineManager.md)

Defined in: [runtime/WGFXRuntime.ts:25](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L25)

#### Zh

管線管理員：負責計算與渲染管線的建立

***

### resourceManager

> **resourceManager**: [`ResourceManager`](../../ResourceManager/classes/ResourceManager.md)

Defined in: [runtime/WGFXRuntime.ts:23](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L23)

#### Zh

資源管理員：處理紋理、緩衝區與取樣器

***

### shaderInfo

> **shaderInfo**: [`WGFXShaderInfo`](../../../types/shader/interfaces/WGFXShaderInfo.md) \| `null`

Defined in: [runtime/WGFXRuntime.ts:31](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L31)

#### Zh

目前載入的特效元數據資訊，未編譯前為 null

***

### uniformBinder

> **uniformBinder**: [`UniformBinder`](../../UniformBinder/classes/UniformBinder.md)

Defined in: [runtime/WGFXRuntime.ts:29](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L29)

#### Zh

Uniform 更新繫結器：處理參數同步

***

### wgslCodeGenerator

> **wgslCodeGenerator**: `WGSLCodeGenerator`

Defined in: [runtime/WGFXRuntime.ts:27](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L27)

#### Zh

WGSL 代碼產生器：將解析後的資訊轉換為著色器代碼

## Execution

### dispatchPass()

> **dispatchPass**(`passName`, `commandEncoder`): `void`

Defined in: [runtime/WGFXRuntime.ts:107](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L107)

Dispatch a specific rendering pass to the GPU commands.
---
執行特定的渲染通道（Pass）。

#### Parameters

##### passName

`string`

Format "PASS_i" (e.g., "PASS_0") / 格式為 "PASS_i" 的通道名稱

##### commandEncoder

`GPUCommandEncoder`

The active GPU command encoder / 目前的 GPU 指令編碼器

#### Returns

`void`

#### Throws

如果特效尚未編譯或找不到指定通道則拋出錯誤

## Lifecycle

### compile()

> **compile**(`effectCode`, `externalResources`): `Promise`\<`void`\>

Defined in: [runtime/WGFXRuntime.ts:69](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L69)

Compile effect code and initialize all GPU pipelines.
---
編譯特效原始碼並初始化管線。這是 WGFX 啟動過程中最耗時的操作。

#### Parameters

##### effectCode

`string`

The source code of the effect / 特效原始碼

##### externalResources

`any` = `{}`

Optional external resource definitions / 選用的外部資源定義

#### Returns

`Promise`\<`void`\>

A promise that resolves when compilation is successful

#### Throws

當語法解析、代碼產生或管線編譯失敗時拋出異常

## Output

### getOutput()

> **getOutput**(): `GPUTextureView`

Defined in: [runtime/WGFXRuntime.ts:149](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L149)

Get the final output texture view for presentation or further processing.
---
獲取最終輸出的紋理視圖（Texture View）。

#### Returns

`GPUTextureView`

The GPUTextureView of the 'OUTPUT' texture

#### Throws

若找不到輸出紋理則拋出錯誤

## Query

### getResource()

> **getResource**(`name`): `GPUTexture` \| `GPUSampler` \| `GPUBuffer` \| `undefined`

Defined in: [runtime/WGFXRuntime.ts:166](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L166)

Retrieve a resource (Texture, Sampler, or Buffer) by its name.
---
根據名稱檢索資源實體。會依序從紋理、取樣器、緩衝區中尋找。

#### Parameters

##### name

`string`

The resource identifier / 資源標識符名稱

#### Returns

`GPUTexture` \| `GPUSampler` \| `GPUBuffer` \| `undefined`

The generic resource object or undefined if not found

## Update

### updateUniform()

> **updateUniform**(`name`, `value`): `void`

Defined in: [runtime/WGFXRuntime.ts:133](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/WGFXRuntime.ts#L133)

Update a uniform value by its identifier name.
---
根據名稱更新 Uniform 參數數值。

#### Parameters

##### name

`string`

The uniform name defined in shader / 著色器中定義的名稱

##### value

New value(s) to upload / 要上傳的新數值

`number` | `number`[]

#### Returns

`void`
