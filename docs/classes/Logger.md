[**WGFX API Documentation v1.0.0**](../README.md)

***

[WGFX API Documentation](../globals.md) / Logger

# Class: Logger

Defined
in: [utils/Logger.ts:46](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/utils/Logger.ts#L46)

Logger class for unified error output and debug mode control.
---
用於統一錯誤輸出與偵錯模式控制的靜態類別。

## Example

```ts
// Enable debug mode to see all logs / 啟用偵錯模式以查看所有日誌
Logger.setDebug(true);
Logger.debug("Checking resources...");

// Set specific level / 設定特定層級
Logger.setLevel(LogLevel.ERROR);
```

## Constructors

### Constructor

> **new Logger**(): `Logger`

#### Returns

`Logger`

## Configuration

### setDebug()

> `static` **setDebug**(`enabled`): `void`

Defined
in: [utils/Logger.ts:74](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/utils/Logger.ts#L74)

Quickly enable or disable debug mode.
---
快速啟用或停用偵錯模式。
啟用後層級會設為 DEBUG，停用則恢復為 INFO。

#### Parameters

##### enabled

`boolean`

True to see all logs / 是否啟用偵錯模式

#### Returns

`void`

***

### setLevel()

> `static` **setLevel**(`level`): `void`

Defined
in: [utils/Logger.ts:61](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/utils/Logger.ts#L61)

Set the current log level.
---
設定目前的日誌過濾層級。低於此層級的日誌將不會被輸出。

#### Parameters

##### level

[`LogLevel`](../enumerations/LogLevel.md)

The target [LogLevel](../enumerations/LogLevel.md) / 目標日誌層級

#### Returns

`void`

## Logging

### debug()

> `static` **debug**(`message`, ...`args`): `void`

Defined
in: [utils/Logger.ts:87](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/utils/Logger.ts#L87)

Log a debug message.
---
記錄偵錯訊息。僅在層級為 DEBUG 時顯示。

#### Parameters

##### message

`string`

The debug message / 要記錄的偵錯訊息

##### args

...`any`[]

Additional contextual data / 額外的上下文參數

#### Returns

`void`

***

### error()

> `static` **error**(`message`, ...`args`): `void`

Defined
in: [utils/Logger.ts:132](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/utils/Logger.ts#L132)

Log an error message.
---
記錄錯誤訊息。用於記錄導致功能失效或執行中斷的嚴重問題。

#### Parameters

##### message

`string`

The error message / 要記錄的錯誤訊息

##### args

...`any`[]

Additional data or Error object / 額外參數或 Error 物件

#### Returns

`void`

***

### info()

> `static` **info**(`message`, ...`args`): `void`

Defined
in: [utils/Logger.ts:102](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/utils/Logger.ts#L102)

Log an info message.
---
記錄一般資訊。適用於系統狀態更新。

#### Parameters

##### message

`string`

The info message / 要記錄的一般訊息

##### args

...`any`[]

Additional data / 額外參數

#### Returns

`void`

***

### warn()

> `static` **warn**(`message`, ...`args`): `void`

Defined
in: [utils/Logger.ts:117](https://github.com/sinchichou/WGFX/blob/d5d57c73d3c7e8023844a7ca17c4e832db074ccb/src/utils/Logger.ts#L117)

Log a warning message.
---
記錄警告訊息。用於提醒可能存在但不會中斷執行的問題。

#### Parameters

##### message

`string`

The warning message / 要記錄的警告訊息

##### args

...`any`[]

Additional data / 額外參數

#### Returns

`void`
