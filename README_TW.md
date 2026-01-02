### [English Version](README.md)

# WGFX 專案：完整技術規格與開發指南

## 開發狀態

**狀態：** `測試中`

本專案的所有模組（包括運行時和 CLI 工具）均已根據本文件中概述的技術規格實現。每個檔案都已添加註釋，以闡明其功能和職責。該專案現在功能齊全，可供使用。

本文件旨在提供 WGFX 專案的完整架構、`.wgsl` 特效檔案格式的詳細規格、Runtime 執行流程，以及從既有 HLSL (Magpie) 遷移至 WGSL
的核心轉換指南。

## 1. 專案架構與檔案職責

專案遵循單一職責原則，確保模組化與可維護性。以下為核心檔案及其職責：

```
WGFX/
├─ package.json              # Node.js 專案設定、依賴管理
├─ package-lock.json         # Node.js 依賴鎖定檔案
├─ README.md                 # 專案說明、使用指南
├─ src/
│   ├─ index.ts              # 專案對外的主要接口 (API Entry Point)
│   ├─ WGFX.ts               # 高階 API 類別
│   │
│   ├─ runtime/              # Runtime 核心邏輯
│   │   ├─ WGFXRuntime.ts    # Runtime 核心流程控制器
│   │   ├─ ShaderParser.js   # shader 解析成 AST (Peggy Generated)
│   │   ├─ ShaderParser.d.ts # Parser 類型定義
│   │   ├─ ResourceManager.ts  # GPU 資源 (Texture, Buffer, Sampler) 管理器
│   │   ├─ PipelineManager.ts  # Render/Compute Pipeline 與 Bind Group 管理器
│   │   ├─ WGSLCodeGenerator.ts # 從中間表示 (IR) 生成 WGSL Shader Code
│   │   └─ UniformBinder.ts    # 提供更新 Uniform Buffer 的接口
│   │
│   ├─ types/                # TypeScript 類型定義
│   │   └─ shader.ts         # Shader 相關介面
│   │
│   └─ utils/                # 共用工具函式
│       └─ Logger.ts         # 統一日誌與偵錯模式控制
│
├─ test/                     # 測試檔案
└─ examples/                 # 範例
```

### 1.1. 核心模組詳解

- **`index.ts` (Runtime 對外接口)**

  - **職責**: 統一管理並封裝 Runtime API，作為外部調用的唯一入口。
  - **提供 API**:
    - `compile()`: 編譯特效檔案。
    - `dispatchPass()`: 執行一個渲染通道 (Pass)。
    - `updateUniform()`: 更新一個 Uniform 參數。
    - `getOutput()`: 獲取最終渲染結果。

- **`WGFXRuntime.ts` (Runtime 核心流程)**

  - **職責**: 控制整個 Runtime 的生命週期。
  - **流程**:
    1. 調用 `ShaderParser.js` 解析 `.wgsl` 檔案，生成中間表示 (IR)。
    2. 調用 `ResourceManager.ts` 根據 IR 初始化 GPU 資源 (Texture, Sampler, Uniform Buffer)。
    3. 調用 `PipelineManager.ts` 建立 Compute/Render Pipeline 與 Bind Group。
    4. 根據 Pass 之間的依賴關係進行排序。
    5. 管理 Workgroup 配置並執行 Pass (Dispatch)。

- **`ShaderParser.js` (核心解析器)**

  - **職責**: Runtime 與 CLI 共享的核心模組，負責讀取 `.wgsl` 檔案並轉換為結構化的中間表示 (IR)。
  - **解析內容**: `Header`, `Parameter`, `Texture`, `Sampler`, `Common`, `Pass` 等區塊。
  - **資訊收集**: 收集函數多載、宏定義 (`MP_*`, `MF`, `MULADD`)。
  - **輸出 (IR)**:
    ```json
    {
      "commonCode": "...",
      "passes": [...],
      "textures": [...],
      "samplers": [...],
      "parameters": [...]
    }
    ```

- **`ResourceManager.ts` (GPU 資源管理)**

  - **職責**: 處理所有與 GPU 資源創建和維護相關的任務。
  - **功能**: 創建 Texture, Sampler, Uniform Buffer，並維護一個名稱對應表 (e.g., `"MyTex" -> GPUTexture Object`)。

- **`PipelineManager.ts` (Pipeline 管理)**

  - **職責**: 負責 Pass 的依賴分析、排序，並創建執行所需的 GPU 物件。
  - **功能**: 創建 Pipeline Layout, Bind Group, Compute Pipeline，並分派 Shader 執行。

- **`WGSLCodeGenerator.ts` (WGSL 程式碼生成)**

    - **職責**: 將 `ShaderParser.js` 產生的 IR 轉換為合法的 WGSL Shader 程式碼。
  - **功能**: 插入 `Common` 區塊、展開宏、處理函數重載、根據配置決定參數是 `inline` 還是 `uniform buffer`。

- **`UniformBinder.ts` (Uniform 更新)**

  - **職責**: 提供 `updateUniform(name, value)` 接口，用於動態更新 Uniform Buffer 的內容。

- **`utils/Logger.ts` (日誌工具)**
    - **職責**: 統一日誌系統，支援不同層級 (DEBUG, INFO, WARN, ERROR)。
    - **偵錯模式**: 透過 `WGFX.setDebug(true)` 切換詳細日誌輸出。

## 2. WGFX 特效檔案格式規格 (`.wgsl`)

`.wgsl` 檔案由一系列以 `//!` 開頭的指令區塊和 WGSL 程式碼片段組成。

### 2.1. 檔頭區塊 (Header Block)

檔頭區塊定義了檔案的元數據和全域設定。

| 指令                       | 必需性   | 說明                                                     | 範例                            |
| :------------------------- | :------- | :------------------------------------------------------- | :------------------------------ |
| `//! MAGPIE WebGPU EFFECT` | **必需** | 檔案的魔術字串，必須位於檔案第一行。                     | `//! Magpie WebGPU EFFECT`      |
| `//! VERSION <number>`     | **必需** | 特效格式版本，必須與解析器內建版本匹配。                 | `//! VERSION 4`                 |
| `//! SORT_NAME <string>`   | 可選     | 用於 UI 排序的名稱。                                     | `//! SORT_NAME "My Effect"`     |
| `//! USE <flags>`          | 可選     | 啟用特定功能。支援 `MULADD`, `_DYNAMIC` (大小寫不敏感)。 | `//! USE MULADD`                |
| `//! CAPABILITY <flags>`   | 可選     | 聲明所需硬體能力。支援 `FP16` (大小寫不敏感)。           | `//! CAPABILITY FP16`           |
| `#include <...>`           | 允許     | 解析器會識別並跳過 `#include` 指令行。                   | `#include "common_functions.h"` |

### 2.2. 參數區塊 (Parameter Block)

定義可在 UI 中調整的 Uniform 參數。

| 指令                   | 必需性   | 說明                                                              | 範例                       |
| :--------------------- | :------- | :---------------------------------------------------------------- | :------------------------- |
| `//! PARAMETER <name>` | **選擇** | 宣告一個參數區塊的開始及其在程式碼中的識別符，如無，則下列皆不須  | `//! PARAMETER Brightness` |
| `//! DEFAULT <value>`  | **必需** | 參數的預設值。                                                    | `//! DEFAULT 1.0`          |
| `//! MIN <value>`      | **必需** | 參數的最小值。                                                    | `//! MIN 0.0`              |
| `//! MAX <value>`      | **必需** | 參數的最大值。                                                    | `//! MAX 2.0`              |
| `//! STEP <value>`     | **必需** | 參數在 UI 中調整的步進值。                                        | `//! STEP 0.01`            |
| `//! LABEL <string>`   | 可選     | 在 UI 上顯示的標籤名稱。                                          | `//! LABEL "Brightness"`   |
| `type name;`           | **必需** | 區塊結尾必須是 HLSL 格式的變數宣告。類型僅支援 `float` 或 `int`。 | `float Brightness;`        |

- **驗證規則**:
  - `DEFAULT`, `MIN`, `MAX`, `STEP` 四個指令必須全部存在。
  - 數值必須滿足 `MIN <= DEFAULT <= MAX`。

### 2.3. 紋理區塊 (Texture Block)

定義效果中使用的紋理資源。

| 指令                  | 必需性       | 說明                                               | 範例                        |
| :-------------------- | :----------- | :------------------------------------------------- | :-------------------------- |
| `//! TEXTURE <name>`  | **必需**     | 宣告一個紋理區塊的開始及其識別符。                 | `//! TEXTURE MyTex`         |
| `//! SOURCE <string>` | 可選         | 指定紋理來源於檔案。若指定此項，則不能有其他選項。 | `//! SOURCE "noise.png"`    |
| `//! FORMAT <format>` | 多數情況必需 | 紋理格式。格式名稱需匹配預定義列表。               | `//! FORMAT R8G8B8A8_UNORM` |
| `//! WIDTH <expr>`    | 可選         | 紋理寬度，可為數字或表達式 (如 `INPUT_WIDTH`)。    | `//! WIDTH 1920`            |
| `//! HEIGHT <expr>`   | 可選         | 紋理高度。`WIDTH` 和 `HEIGHT` 必須成對出現。       | `//! HEIGHT INPUT_HEIGHT`   |
| `Texture2D name;`     | **必需**     | 區塊結尾必須是 HLSL 格式的紋理宣告。               | `Texture2D MyTex;`          |

- **特殊內建紋理**:
  - `INPUT`: 預設的輸入紋理 (index 0)。
  - `OUTPUT`: 預設的輸出紋理 (index 1)。
  - 解析器會對這兩個名稱進行特殊處理。

### 2.4. 採樣器區塊 (Sampler Block)

定義紋理採樣的行為。

| 指令                 | 必需性   | 說明                                   | 範例                      |
| :------------------- | :------- | :------------------------------------- | :------------------------ |
| `//! SAMPLER <name>` | **必需** | 宣告一個採樣器區塊的開始及其識別符。   | `//! SAMPLER MySampler`   |
| `//! FILTER <mode>`  | **必需** | 過濾模式。支援 `LINEAR` 或 `POINT`。   | `//! FILTER LINEAR`       |
| `//! ADDRESS <mode>` | 可選     | 尋址模式。支援 `CLAMP` 或 `WRAP`。     | `//! ADDRESS CLAMP`       |
| `SamplerState name;` | **必需** | 區塊結尾必須是 HLSL 格式的採樣器宣告。 | `SamplerState MySampler;` |

### 2.5. 通用程式碼區塊 (Common Block)

`//! COMMON` 區塊內的程式碼會被插入到每一個 `PASS` 的 Shader 之前，用於定義共用的函式、結構或常數。

```hlsl
//! COMMON
float PI = 3.1415926535;

float3 grayscale(float3 color) {
    float luminance = dot(color, float3(0.299, 0.587, 0.114));
    return float3(luminance, luminance, luminance);
}
```

### 2.6. 渲染通道區塊 (Pass Block)

定義單次的渲染/計算操作。一個特效檔案可以包含多個 Pass。

| 指令                      | 必需性    | 說明                                                                         | 範例                                  |
| :------------------------ | :-------- | :--------------------------------------------------------------------------- | :------------------------------------ |
| `//! PASS <index>`        | **必需**  | 宣告一個 Pass 及其索引。索引必須從 `1` 開始且連續。                          | `//! PASS 1`                          |
| `//! IN <tex-list>`       | **必需**  | 指定該 Pass 的輸入紋理列表，以逗號分隔。                                     | `//! IN INPUT, MyTex`                 |
| `//! OUT <tex-list>`      | **必需**  | 指定該 Pass 的輸出紋理列表，以逗號分隔。                                     | `//! OUT TempTex`                     |
| `//! BLOCK_SIZE <w,h>`    | `CS` 必需 | **Compute Shader** 專用，定義區塊大小。                                      | `//! BLOCK_SIZE 16,16`                |
| `//! NUM_THREADS <x,y,z>` | `CS` 必需 | **Compute Shader** 專用，定義每個 workgroup 的執行緒數量。                   | `//! NUM_THREADS 8,8,1`               |
| `//! STYLE <PS\|CS>`      | 可選      | 指定 Pass 類型為 Pixel Shader (`PS`) 或 Compute Shader (`CS`)。預設為 `CS`。 | `//! STYLE CS`                        |
| `//! DESC <string>`       | 可選      | Pass 的描述文字，可用於除錯或 UI 顯示。                                      | `//! DESC "First Gaussian Blur Pass"` |

- **驗證規則**:
  - `IN` 和 `OUT` 是每個 Pass 的必需指令。
  - 若 `STYLE` 為 `CS` (或未指定)，則 `BLOCK_SIZE` 和 `NUM_THREADS` 必需提供。
  - 最後一個 Pass 的 `OUT` 必須是 `OUTPUT`。
  - 中間 Pass 的 `OUT` 不能是 `INPUT` 或 `OUTPUT`。

### 2.7. 內建全域變數 (Built-in Global Variables)

為了支援動態解析度，Runtime 內建了全域的 Uniform 結構 `SceneInfo`，在所有 Pass 中皆可使用。

```wgsl
struct SceneInfo {
    inputSize: vec2<u32>, // 輸入來源的寬高 (e.g. 1920, 1080)
    inputPt: vec2<f32>,   // 像素大小倒數 (1.0/width, 1.0/height)
}
@group(0) @binding(4) var<uniform> scene: SceneInfo;
```

- **使用方式**: 直接在程式碼中存取 `scene.inputSize` 或 `scene.inputPt`。

## 3. Runtime 執行流程

系統從載入 `.wgsl` 檔案到最終渲染輸出的完整流程如下：

1. **解析與生成 IR**:

   - `WGFXRuntime` 調用 `ShaderParser.js` 讀取 `.wgsl` 檔案。
   - `ShaderParser.js` 逐一解析 `Header`, `Parameter`, `Texture`, `Sampler`, `Common`, `Pass` 等區塊。
   - 生成一份結構化的中間表示 (IR)，包含所有解析出的元數據和程式碼片段。

2. **GPU 資源創建**:

   - `ResourceManager.ts` 根據 IR 中的 `textures`, `samplers`, `parameters` 列表，創建對應的 `GPUTexture`, `GPUSampler`,
     `GPUBuffer` (for uniforms)。
   - 建立一個從資源名稱到 GPU 物件的映射表，供後續使用。

3. **Pipeline 建立**:

   - `PipelineManager.ts` 遍歷 IR 中的 `passes` 列表。
   - 對於每個 Pass，`WGSLCodeGenerator.ts` 將其 HLSL 片段與 `commonCode` 結合，並轉換成完整的 WGSL Compute Shader 程式碼。
   - `PipelineManager.ts` 根據 Pass 的 `IN` 和 `OUT` 資源，推導出 `GPUBindGroupLayout`。
   - 使用生成的 WGSL Shader 和 Layout 創建 `GPUComputePipeline`。

4. **執行與渲染**:

   - 當外部調用 `dispatchPass(index)` 時：
   - `PipelineManager.ts` 根據 Pass 所需的資源，從 `ResourceManager` 獲取對應的 GPU 物件，並創建 `GPUBindGroup`。
   - `WGFXRuntime` 發出 `setPipeline`, `setBindGroup`, `dispatchWorkgroups` 等 GPU 命令。
   - GPU 執行 Compute Shader，將運算結果寫入 `OUT` 指定的紋理。

5. **動態更新**:
   - 當外部調用 `updateUniform(name, value)` 時：
   - `UniformBinder.ts` 將新值寫入對應的 `GPUBuffer` 中，實現參數的動態更新。

## 4. HLSL/Magpie 至 WGSL/WebGPU 轉換指南

### 4.1. 資源類型映射

| HLSL / Magpie          | WGSL / WebGPU                              | 說明                                            |
| :--------------------- | :----------------------------------------- | :---------------------------------------------- |
| `Texture2D<T>` (SRV)   | `var tex: texture_2d<f32>;`                | 用於讀取的紋理，在 WGSL 中與 Sampler 分開綁定。 |
| `RWTexture2D<T>` (UAV) | `var tex: texture_storage_2d<fmt, write>;` | 用於寫入的存儲紋理 (Storage Texture)。          |
| `SamplerState`         | `var smp: sampler;`                        | 採樣器。                                        |
| `cbuffer` / `uniform`  | `var<uniform> uniforms: MyUniforms;`       | Uniform 常數緩衝區。                            |

### 4.2. 執行緒與 Workgroup

| HLSL / Magpie           | WGSL / WebGPU                 | 說明                                                      |
| :---------------------- | :---------------------------- | :-------------------------------------------------------- |
| `[numthreads(x, y, z)]` | `@workgroup_size(x, y, z)`    | 在 Compute Shader 入口函式上的屬性，定義 workgroup 大小。 |
| `Dispatch(X, Y, Z)`     | `dispatchWorkgroups(X, Y, Z)` | 分派的 Workgroup 網格數量。                               |

### 4.3. 內建宏與函式

- **`MF` 宏**: 需根據 `CAPABILITY` 中的 `FP16` 標旗，轉換為 WGSL 中的 `f32` 或 `f16`。
- **`MULADD`**: 若 `USE MULADD` 被啟用，需要將對應的 HLSL 函式實現轉換為 WGSL 函式，或直接使用 `a * b + c` 的形式。
- **`MP_*` 宏**: 如 `MP_BLOCK_WIDTH` 等，需要根據 `//! BLOCK_SIZE` 等指令的值，在生成 WGSL 程式碼時直接替換為常數。

### 4.4. 綁定模型

WebGPU 使用 Bind Group 模型，取代了 HLSL 的 `register(t0, u0, s0)`。

- **策略**: 解析器必須為每個 Pass 的所有資源 (`IN`, `OUT`, `Samplers`, `Uniforms`) 分配一組唯一的
  `@group(N) @binding(M)` 索引，並在生成 WGSL 時寫入。
- **範例綁定**:

| 資源            | WGSL 綁定                |
| :-------------- | :----------------------- |
| `sam` (Sampler) | `@group(0) @binding(0)`  |
| `uniforms`      | `@group(0) @binding(1)`  |
| `scene`         | `@group(0) @binding(4)`  |
| `TexN`          | `@group(0) @binding(6+)` |

### 4.5. 基礎型態與函式映射

| HLSL                  | WGSL                 |
| :-------------------- | :------------------- |
| `float`, `float1`     | `f32`                |
| `float2`              | `vec2<f32>`          |
| `float3`              | `vec3<f32>`          |
| `float4`              | `vec4<f32>`          |
| `float2x2`            | `mat2x2<f32>`        |
| `float3x3`            | `mat3x3<f32>`        |
| `float4x4`            | `mat4x4<f32>`        |
| `int`, `int1`         | `i32`                |
| `int2`                | `vec2<i32>`          |
| `uint`                | `u32`                |
| `mul(matrix, vector)` | `matrix * vector`    |
| `lerp(a, b, x)`       | `mix(a, b, x)`       |
| `saturate(x)`         | `clamp(x, 0.0, 1.0)` |
| `frac(x)`             | `fract(x)`           |
| `ddx(v)`              | `dpdx(v)`            |
| `ddy(v)`              | `dpdy(v)`            |
