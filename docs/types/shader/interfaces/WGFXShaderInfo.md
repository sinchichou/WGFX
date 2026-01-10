[**WGFX API Documentation v1.0.0**](../../../index.md)

***

[WGFX API Documentation](../../../modules.md) / [types/shader](../index.md) / WGFXShaderInfo

# Interface: WGFXShaderInfo

Defined in: [types/shader.ts:73](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L73)

Structured information of a complete WGFX shader
完整 WGFX 著色器的結構化資訊

## Properties

### commonCode

> **commonCode**: `string`

Defined in: [types/shader.ts:83](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L83)

Common code shared across all passes / 所有通道共享的通用代碼

***

### parameters

> **parameters**: [`ParameterInfo`](ParameterInfo.md)[]

Defined in: [types/shader.ts:79](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L79)

List of all adjustable parameters / 所有可調整參數的列表

***

### passes

> **passes**: [`PassInfo`](PassInfo.md)[]

Defined in: [types/shader.ts:81](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L81)

List of all execution passes / 所有執行通道的列表

***

### samplers

> **samplers**: [`SamplerInfo`](SamplerInfo.md)[]

Defined in: [types/shader.ts:77](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L77)

List of all samplers / 所有取樣器的列表

***

### textures

> **textures**: [`TextureInfo`](TextureInfo.md)[]

Defined in: [types/shader.ts:75](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L75)

List of all textures / 所有紋理的列表
