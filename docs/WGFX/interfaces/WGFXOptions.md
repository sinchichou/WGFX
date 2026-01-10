[**WGFX API Documentation v1.0.0**](../../index.md)

***

[WGFX API Documentation](../../modules.md) / [WGFX](../index.md) / WGFXOptions

# Interface: WGFXOptions

Defined in: [WGFX.ts:12](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/WGFX.ts#L12)

Options for creating a WGFX instance.
---
建立 WGFX 實例的配置選項。

## Properties

### device

> **device**: `GPUDevice`

Defined in: [WGFX.ts:17](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/WGFX.ts#L17)

The WebGPU device to use.

#### Zh

使用的 WebGPU 裝置

***

### effectCode

> **effectCode**: `string`

Defined in: [WGFX.ts:22](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/WGFX.ts#L22)

The WGSL or custom effect code string.

#### Zh

WGSL 或自定義特效代碼字串

***

### externalResources?

> `optional` **externalResources**: `any`

Defined in: [WGFX.ts:28](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/WGFX.ts#L28)

#### Zh

選用的外部資源 (textures, samplers, etc.)

***

### height

> **height**: `number`

Defined in: [WGFX.ts:26](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/WGFX.ts#L26)

#### Zh

初始處理高度

***

### width

> **width**: `number`

Defined in: [WGFX.ts:24](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/WGFX.ts#L24)

#### Zh

初始處理寬度
