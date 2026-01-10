[**WGFX API Documentation v1.0.0**](../../../index.md)

***

[WGFX API Documentation](../../../modules.md) / [types/shader](../index.md) / ParameterInfo

# Interface: ParameterInfo

Defined in: [types/shader.ts:35](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L35)

Information about an adjustable parameter (uniform)
可調整的參數 (Uniform) 資訊

## Properties

### default

> **default**: `number` \| `number`[]

Defined in: [types/shader.ts:41](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L41)

Default value or array of values / 預設值或數值陣列

***

### max?

> `optional` **max**: `number`

Defined in: [types/shader.ts:45](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L45)

Maximum value allowed / 允許的最大值

***

### min?

> `optional` **min**: `number`

Defined in: [types/shader.ts:43](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L43)

Minimum value allowed / 允許的最小值

***

### name

> **name**: `string`

Defined in: [types/shader.ts:37](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L37)

Parameter identifier / 參數識別符

***

### step?

> `optional` **step**: `number`

Defined in: [types/shader.ts:47](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L47)

Step value for UI adjustment / UI 調整的步進值

***

### type

> **type**: `string`

Defined in: [types/shader.ts:39](https://github.com/sinchichou/WGFX/blob/f7ebc77d9e083596c467ce00afea57991492f246/src/types/shader.ts#L39)

Data type: 'float' or 'int' / 資料類型
