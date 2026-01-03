[**WGFX API Documentation v1.0.0**](../README.md)

***

[WGFX API Documentation](../globals.md) / WGFX

# Class: WGFX

Defined
in: [WGFX.ts:73](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L73)

Main WGFX class for handling WebGPU graphics effects.
---
處理 WebGPU 圖形特效的主要控制器。

## Example

```ts
const wgfx = await WGFX.create({device, effectCode, width: 1280, height: 720});
const output = await wgfx.process(videoElement);
// 渲染完成後關閉
wgfx.dispose();
```

## Properties

### currentInputSource

> **currentInputSource**: `any`

Defined
in: [WGFX.ts:83](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L83)

#### Zh

目前輸入源物件

***

### height

> **height**: `number`

Defined
in: [WGFX.ts:81](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L81)

#### Zh

目前處理高度

***

### initialized

> **initialized**: `boolean`

Defined
in: [WGFX.ts:77](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L77)

#### Zh

實例是否已初始化完成

***

### width

> **width**: `number`

Defined
in: [WGFX.ts:79](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L79)

#### Zh

目前處理寬度

## Configuration

### setDebug()

> `static` **setDebug**(`enabled`): `void`

Defined
in: [WGFX.ts:106](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L106)

Enable or disable debug mode globally.
---
全域啟用或停用偵錯模式。

#### Parameters

##### enabled

`boolean`

Whether to enable debug mode / 是否啟用偵錯模式

#### Returns

`void`

## Lifecycle

### dispose()

> **dispose**(): `void`

Defined
in: [WGFX.ts:335](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L335)

Dispose all resources and clean up.
---
釋放所有 WebGPU 資源並清理內部管理器，防止記憶體洩漏。

#### Returns

`void`

***

### create()

> `static` **create**(`options`): `Promise`\<`WGFX`\>

Defined
in: [WGFX.ts:120](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L120)

Factory method to create and initialize a WGFX instance.
---
建立並初始化 WGFX 實例的工廠方法。

#### Parameters

##### options

`WGFXOptions`

Configuration options / 配置選項

#### Returns

`Promise`\<`WGFX`\>

A promise that resolves to a WGFX instance

#### Throws

如果 WebGPU 裝置、代碼無效或編譯失敗時拋出錯誤

## Metadata

### initialize()

> **initialize**(): `WGFXInfo`

Defined
in: [WGFX.ts:187](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L187)

Get information about the initialized effect.
---
獲取已初始化特效的相關資訊（如參數列表、寬高）。

#### Returns

`WGFXInfo`

Metadata about the current effect / 目前特效的元數據

#### Throws

若實例尚未初始化則拋出錯誤

## Rendering

### getOutputView()

> **getOutputView**(): `GPUTextureView`

Defined
in: [WGFX.ts:324](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L324)

Get the texture view of the final output.
---
獲取最終輸出的紋理視圖 (TextureView)。

#### Returns

`GPUTextureView`

The output GPUTextureView

***

### process()

> **process**(`inputSource`): `Promise`\<`GPUTexture`\>

Defined
in: [WGFX.ts:254](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L254)

Process an input source and return the output texture.
---
處理輸入源並回傳輸出紋理。支援多種影像來源。

#### Parameters

##### inputSource

The image/video source to process / 要處理的影像來源

`ImageBitmap` | `HTMLVideoElement` | `HTMLCanvasElement` | `VideoFrame`

#### Returns

`Promise`\<`GPUTexture`\>

A promise that resolves to the output GPUTexture

#### Throws

當輸入來源維度與初始化不符時拋出錯誤

***

### updateUniforms()

> **updateUniforms**(`uniforms`): `void`

Defined
in: [WGFX.ts:224](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/WGFX.ts#L224)

Update shader uniform values.
---
更新 Uniform 數值，可用於即時調整特效參數。

#### Parameters

##### uniforms

`Record`\<`string`, `number` \| `number`[]\>

Key-value pairs of uniform names and values / 名稱與數值的鍵值對

#### Returns

`void`
