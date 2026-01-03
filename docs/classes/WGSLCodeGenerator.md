[**WGFX API Documentation v1.0.0**](../README.md)

***

[WGFX API Documentation](../globals.md) / WGSLCodeGenerator

# Class: WGSLCodeGenerator

Defined
in: [runtime/WGSLCodeGenerator.ts:12](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/runtime/WGSLCodeGenerator.ts#L12)

Generator for WGSL shader code from parsed shader information.
---
從解析後的著色器資訊產生 WGSL 著色器代碼。
負責注入標準資源、修正語法相容性並封裝計算著色器進入點。

## Constructors

### Constructor

> **new WGSLCodeGenerator**(): `WGSLCodeGenerator`

Defined
in: [runtime/WGSLCodeGenerator.ts:13](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/runtime/WGSLCodeGenerator.ts#L13)

#### Returns

`WGSLCodeGenerator`

## Generation

### generate()

> **generate**(`shaderInfo`): `object`[]

Defined
in: [runtime/WGSLCodeGenerator.ts:26](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/runtime/WGSLCodeGenerator.ts#L26)

Generate complete WGSL modules for each pass in the shader info.
---
為每個渲染通道產生完整的 WGSL 模組。
自動處理資源綁定索引（Binding Index）並注入全域 Uniforms。

#### Parameters

##### shaderInfo

`WGFXShaderInfo`

The parsed shader metadata / 解析後的著色器元數據

#### Returns

`object`[]

Array of generated modules / 產生的模組列表
