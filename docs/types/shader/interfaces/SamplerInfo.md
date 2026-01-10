[**WGFX API Documentation v1.0.0**](../../../index.md)

***

[WGFX API Documentation](../../../modules.md) / [types/shader](../index.md) / SamplerInfo

# Interface: SamplerInfo

Defined in: [types/shader.ts:22](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L22)

Information about a sampler resource used in the shader
著色器中使用的取樣器資源資訊

## Properties

### address

> **address**: `string`

Defined in: [types/shader.ts:28](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L28)

Address mode: 'WRAP' or 'CLAMP' / 定址模式

***

### filter

> **filter**: `string`

Defined in: [types/shader.ts:26](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L26)

Filter mode: 'LINEAR' or 'NEAREST' / 過濾模式

***

### name

> **name**: `string`

Defined in: [types/shader.ts:24](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L24)

Sampler identifier used in code / 程式碼中使用的取樣器識別符
