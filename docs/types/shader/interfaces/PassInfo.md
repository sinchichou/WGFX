[**WGFX API Documentation v1.0.0**](../../../index.md)

***

[WGFX API Documentation](../../../modules.md) / [types/shader](../index.md) / PassInfo

# Interface: PassInfo

Defined in: [types/shader.ts:54](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L54)

Information about a rendering/compute pass
渲染/計算通道資訊

## Properties

### blockSize?

> `optional` **blockSize**: `number`[]

Defined in: [types/shader.ts:66](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L66)

Optional block size for coordinate calculation / 用於座標計算的選用區塊大小

***

### code

> **code**: `string`

Defined in: [types/shader.ts:62](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L62)

WGSL/HLSL code snippet for this pass / 此通道的 WGSL/HLSL 代碼片段

***

### in

> **in**: `string`[]

Defined in: [types/shader.ts:58](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L58)

Names of input textures / 輸入紋理名稱列表

***

### index

> **index**: `number`

Defined in: [types/shader.ts:56](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L56)

1-based index of the pass / 通道索引 (從 1 開始)

***

### numThreads

> **numThreads**: \[`number`, `number`, `number`\]

Defined in: [types/shader.ts:64](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L64)

Workgroup size [x, y, z] / 工作群組尺寸 [x, y, z]

***

### out

> **out**: `string`[]

Defined in: [types/shader.ts:60](https://github.com/sinchichou/WGFX/blob/e1d8323112badb471e6655823c5e8f540a5d8581/src/types/shader.ts#L60)

Names of output textures / 輸出紋理名稱列表
