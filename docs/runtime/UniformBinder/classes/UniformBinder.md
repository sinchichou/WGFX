[**WGFX API Documentation v1.0.0**](../../../index.md)

***

[WGFX API Documentation](../../../modules.md) / [runtime/UniformBinder](../index.md) / UniformBinder

# Class: UniformBinder

Defined in: [runtime/UniformBinder.ts:13](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/UniformBinder.ts#L13)

Handles the binding and updating of WebGPU uniform buffers.
---
處理 WebGPU Uniform 緩衝區的繫結與數據更新。
負責將 CPU 端的數值同步至 GPU 端的緩衝區。

## Constructors

### Constructor

> **new UniformBinder**(`device`, `resourceManager`): `UniformBinder`

Defined in: [runtime/UniformBinder.ts:27](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/UniformBinder.ts#L27)

Initialize the uniform binder.
---
初始化 Uniform 繫結器。

#### Parameters

##### device

`GPUDevice`

The active WebGPU device / 有效的 WebGPU 裝置

##### resourceManager

[`ResourceManager`](../../ResourceManager/classes/ResourceManager.md)

The resource manager / 資源管理員

#### Returns

`UniformBinder`

## Properties

### device

> **device**: `GPUDevice`

Defined in: [runtime/UniformBinder.ts:15](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/UniformBinder.ts#L15)

#### Zh

WebGPU 裝置實例

***

### resourceManager

> **resourceManager**: [`ResourceManager`](../../ResourceManager/classes/ResourceManager.md)

Defined in: [runtime/UniformBinder.ts:17](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/UniformBinder.ts#L17)

#### Zh

資源管理員，用於檢索 Uniform 元數據

## Update

### updateUniform()

> **updateUniform**(`name`, `value`): `void`

Defined in: [runtime/UniformBinder.ts:48](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/runtime/UniformBinder.ts#L48)

Update a specific uniform value in the GPU buffer.
---
更新 GPU 緩衝區中特定 Uniform 的數值。
自動處理單一數值或陣列型數據的封裝與傳輸。

#### Parameters

##### name

`string`

The name of the uniform defined in shader / 著色器中定義的 Uniform 名稱

##### value

The new value(s) to upload / 要上傳的新數值（單一數值或數值陣列）

`number` | `number`[]

#### Returns

`void`

#### Example

```ts
binder.updateUniform('u_Brightness', 0.5);
binder.updateUniform('u_Color', [1.0, 0.0, 0.0]);
```
