# **WGFX 專案完整架構與檔案職責**

**專案已初始化，所有檔案已根據此 README 建立。**

```
WGFX/
├─ package.json
│   - Node.js 專案設定、依賴管理
├─ README.md
│   - 專案說明、使用說明、範例指令
├─ src/
│   ├─ index.js
│   │   - Runtime 對外接口
│   │   - 封裝 Runtime API：compile(), dispatchPass(), updateUniform(), getOutput()
│   │   - 統一管理 Runtime 模組與 CLI 模組的調用入口
│   │
│   ├─ runtime/
│   │   ├─ WGFXRuntime.js
│   │   │   - Runtime 核心流程控制
│   │   │   - 解析 FX 檔案
│   │   │   - 初始化 GPU 資源
│   │   │   - 建立 Pipeline 與 bind group
│   │   │   - Pass 排序與 dispatch
│   │   │   - Workgroup 配置
│   │   │
│   │   ├─ Parser.js
│   │   │   - 核心解析器（Runtime 與 CLI 共享）
│   │   │   - 解析 Header、Parameter、Texture、Sampler、Common、Pass
│   │   │   - 收集函數多載資訊
│   │   │   - 收集宏 / 巨集 / MF / MP_* / MULADD
│   │   │   - 產生中間結構：
│   │   │     {
│   │   │       commonCode: "...WGSL...",
│   │   │       passes: [...],
│   │   │       textures: [...],
│   │   │       samplers: [...],
│   │   │       parameters: [...] 
│   │   │     }
│   │   │
│   │   ├─ ResourceManager.js
│   │   │   - GPU 資源管理
│   │   │   - Texture / Sampler / Uniform buffer 建立
│   │   │   - 維護名稱對應表（WGFX FX 名稱 → GPU object）
│   │   │   - 支援動態更新 Uniform
│   │   │
│   │   ├─ PipelineManager.js
│   │   │   - Pass 排序 / Pass 依賴分析
│   │   │   - Pipeline 建立
│   │   │   - Bind group 建立與維護
│   │   │   - Dispatch Pass shader
│   │   │
│   │   ├─ WGSLCodeGenerator.js
│   │   │   - 將解析結果生成合法 WGSL shader
│   │   │   - 插入 Common 區塊
│   │   │   - 宏展開 / 函數重載替換
│   │   │   - 支援 inline parameters 或 uniform buffer
│   │   │
│   │   └─ UniformBinder.js
│   │       - 動態更新 uniform buffer
│   │       - 提供 API updateUniform(name, value)
│   │
│   ├─ cli/
│   │   ├─ wgfx-compile.js
│   │   │   - CLI 入口
│   │   │   - 解析命令列參數（輸入 FX、輸出 Zip、版本號等）
│   │   │   - 調用 StaticParser、WGSLCodeGeneratorCLI、PipelineMetadataGenerator、OutputPackager
│   │   │
│   │   ├─ StaticParser.js
│   │   │   - 靜態 FX 解析（只生成固定 Shader / Pipeline）
│   │   │   - 不支援動態 Uniform
│   │   │
│   │   ├─ WGSLCodeGeneratorCLI.js
│   │   │   - CLI 專用 Shader 生成
│   │   │   - 宏展開與函數重載替換
│   │   │
│   │   ├─ PipelineMetadataGenerator.js
│   │   │   - 生成 Pipeline JSON（Pass 序列、IN/OUT、bind group、workgroup size）
│   │   │   - 生成 Metadata JSON（版本、名稱、描述等）
│   │   │
│   │   └─ OutputPackager.js
│   │       - 將 Shader / Pipeline JSON / Metadata 打包成 Zip
│   │       - 供 Runtime 直接解壓使用
│   │
│   └─ utils/
│       ├─ FunctionOverloadResolver.js
│       │   - 解析函數重載，生成唯一函數名稱
│       │   - 確保 WGSL 不依賴重載
│       │
│       ├─ MacroExpander.js
│       │   - 展開 HLSL / WGFX FX 巨集與宏（MP_*, MF, MULADD 等）
│       │   - 生成合法 WGSL 常數 / 型別 / 函式
│       │
│       └─ FileUtils.js
│           - 檔案讀寫
│           - Zip 打包 / 解壓
│           - 檔案 / 路徑檢查
│
└─ examples/
    ├─ demo.fx
    │   - 範例 WGFX FX 檔案
    │   - 含多 Pass、Texture、Sampler、Parameter
    └─ demo_runtime.js
        - 範例 Runtime 使用示範
        - 呼叫 WGFX Runtime API
        - 動態更新 Uniform / Dispatch Pass / 取得 Output
```

## **解析器核心內容（Parser.js / StaticParser.js）**

### **1. 檔頭相關（Header 區塊）**

* `//! MAGPIE EFFECT`

    * 用途：檢測檔案為 MagpieFX 格式的開頭。必須存在（`CheckMagic`）。
    * 格式：`//! MAGPIE EFFECT <任意非空行>`（後面必須有內容行）
* `//! VERSION <number>` （必需）

    * 範例：`//! VERSION 4`。必須等於靜態常數 `MAGPIE_FX_VERSION`。
* `//! SORT_NAME <string>`（可選）

    * 範例：`//! SORT_NAME someName`。只有在 `noCompile` 為真時才會被寫入 `desc.sortName`（程式邏輯）。
* `//! USE <comma-separated flags-string>`（可選）

    * 支援值（大小寫不敏感）：`MULADD`、`_DYNAMIC`（程式中以 `UseMulAdd`、`UseDynamic` 處理；`_DYNAMIC` 為保留但仍被接受）
    * 若遇未知 flag，會記錄 warning。
* `//! CAPABILITY <comma-separated flags-string>`（可選）

    * 支援值（大小寫不敏感）：`FP16`（會將 `EffectFlags::SupportFP16`、並視 `noFP16` 決定是否同時設 `EffectFlags::FP16`）
    * 未知 flag 會 warning。
* 在 Header 區塊中，唯一允許的非 `//!` 指令是 `#include <...>`（程式只檢查並跳過整行 `#include`）。

### **2. 參數區塊（Parameter block）**

* 區塊起始：`//! PARAMETER`（必需）

    * 緊接一個名稱 token（作為參數在 code 中的識別符）
* 參數內子指令（必需/可選）：

    * `//! DEFAULT <value>`（必需）
    * `//! LABEL <string>`（可選） → 會賦予 `paramDesc.label`
    * `//! MIN <value>`（必需）
    * `//! MAX <value>`（必需）
    * `//! STEP <value>`（必需）
* 參數類型與結尾：

    * 在子指令群之後，程式會讀一個 token 作為類型，僅接受 `float` 或 `int`。
    * 接著讀取參數名稱 token（即實際變數名）。
    * 必須以 `;` 結束該行。
* 檢查：`DEFAULT`、`MIN`、`MAX`、`STEP` 四項必須都出現；數值檢查會驗證 `min <= default <= max`。

### **3. 紋理區塊（Texture block）**

* 區塊起始：`//! TEXTURE`（必需）

    * 緊接一個名稱 token（紋理識別符）
* 子選項（依情況互斥或必需）：

    * `//! SOURCE <string>`

        * 若指定 `SOURCE`，該 texture 不能和其他選項混用（程式有檢查）。
    * `//! FORMAT <format-string>`（對非 INPUT/OUTPUT 必需）

        * `FORMAT` 會轉查 `EffectHelper::FORMAT_DESCS` 清單以決定格式枚舉。若找不到則錯誤。
    * `//! WIDTH <expr>`（可選） // 與 HEIGHT 必須成對出現
    * `//! HEIGHT <expr>`（可選） // 與 WIDTH 必須成對出現
* 特殊內建 textures：

    * 程式會預先建立兩個 texture： `INPUT`（index 0）與 `OUTPUT`（index 1）。
    * 若解析到名字剛好等於 `INPUT` 或 `OUTPUT`，會有特殊行為（不能當作輸出/輸入視情況）。
* 代碼部分期待：接著會出現 `Texture2D` token，然後一個 token 為紋理名字，最後以 `;` 結尾（HLSL 片段）。解析器需比對名稱以判定是否為
  INPUT/OUTPUT 或新紋理。

### **4. 採樣器區塊（Sampler block）**

* 區塊起始：`//! SAMPLER`（必需）

    * 緊接一個名稱 token（採樣器識別符）
* 子選項：

    * `//! FILTER <LINEAR|POINT>`（必需）
    * `//! ADDRESS <CLAMP|WRAP>`（可選）
* 代碼部分期待：`SamplerState` token 之後為 sampler 名稱，最後 `;` 結尾。

### **5. COMMON 區塊**

* 區塊起始：`//! COMMON`（無選項）

    * 以 `//! COMMON` 開頭的區塊內不允許再有另一個 `//!`（程式中檢查）。
    * 代表一段可在每個 Pass 前加入的共用 HLSL 片段（函式、巨集等）。

### **6. PASS 區塊（多個）**

* 區塊起始：`//! PASS <index>`（必需）

    * `index` 為整數。所有 PASS 必須序號連續從 1 開始（程式會排序並檢查連續性）。
* PASS 區塊內支持的子指令（必需/可選）：

    * `//! IN <comma-separated texture-names>`（必需）

        * 指定該 pass 的輸入紋理，不能包含 `OUTPUT`。
    * `//! OUT <comma-separated texture-names>`（必需）

        * 最後一個 pass（index == 最後）只能輸出到 `OUTPUT`。其它 pass 的輸出不能是 `INPUT`、不能是 `OUTPUT`（除最後），且不能來自
          `SOURCE`（file-backed texture）。
        * 每個 pass 最多 8 個輸出。
    * `//! BLOCK_SIZE <w[,h]>`（必需，除非 STYLE==PS）

        * 若只給一個數字則視為寬高相同。數值須 >0。
    * `//! NUM_THREADS <x[,y[,z]]>`（必需，除非 STYLE==PS）

        * 允許 1~3 個數字，缺省填充 1。
    * `//! STYLE <PS|CS>`（可選）

        * 若 `STYLE == PS`，表示使用 PSStyle。PSStyle 下禁止使用 `BLOCK_SIZE` 或 `NUM_THREADS`，而 parser
          會用內建值（blockSize=16x16, numThreads={64,1,1}）。若 `STYLE != PS` 並非 `CS` 則錯誤。
    * `//! DESC <string>`（可選） → pass 的描述文字。
* 必要條件檢查：每個 pass 必須提供 `IN` 和 `OUT`。若非 PSStyle 則 `BLOCK_SIZE` 與 `NUM_THREADS` 都必需提供。

### **7. 其他可被解析但不是

//!
的 token（Header/通用）**

* `#include <...>`（Header 區塊中允許一行 `#include`，parser 只是跳過該行）
* 在 code 部分大量期望 HLSL tokens，例如 `Texture2D`、`RWTexture2D`、`SamplerState`、`cbuffer`、`[numthreads(...)]`、function
  名稱、`;` 等。解析器只檢驗這些關鍵 token 的存在/名稱一致性。

### **8. 支援的 flag 值（彙總）**

* USE flags：`MULADD`、`_DYNAMIC`
* CAPABILITY flags：`FP16`
* SAMPLER FILTER：`LINEAR`、`POINT`
* SAMPLER ADDRESS：`CLAMP`、`WRAP`
* STYLE：`PS`、`CS`
* PARAMETER 類型：`float`、`int`
* TEXTURE FORMAT：來自 `EffectHelper::FORMAT_DESCS`（程式用這個清單，比對格式名稱；實際格式名稱需讀取你的程式碼中
  `FORMAT_DESCS` 定義，常見例如 `R8G8B8A8_UNORM`、`R16G16B16A16_FLOAT` 等）

---

## **BNF-like 文法與檢查規則**

根級（檔案）：

```
file := headerBlock { paramBlock | textureBlock | samplerBlock | commonBlock | passBlock }+
```

headerBlock：

```
headerBlock := "//! MAGPIE EFFECT" NEWLINE
               { "//!" HEADER_ENTRY NEWLINE | "#include" LINE }
HEADER_ENTRY := "VERSION" NUMBER
              | "SORT_NAME" STRING
              | "USE" FLAG_LIST
              | "CAPABILITY" FLAG_LIST
```

paramBlock：

```
paramBlock := "//! PARAMETER" ID NEWLINE
              { "//!" PARAM_ENTRY NEWLINE }+
              TYPE ID ";"
PARAM_ENTRY := "DEFAULT" VALUE
             | "LABEL" STRING
             | "MIN" VALUE
             | "MAX" VALUE
             | "STEP" VALUE
TYPE := "float" | "int"
```

textureBlock：

```
textureBlock := "//! TEXTURE" ID NEWLINE
                { "//!" TEXTURE_ENTRY NEWLINE }+
                "Texture2D" ID ";"
TEXTURE_ENTRY := "SOURCE" STRING
               | "FORMAT" FORMAT_NAME
               | "WIDTH" EXPR
               | "HEIGHT" EXPR
```

samplerBlock：

```
samplerBlock := "//! SAMPLER" ID NEWLINE
                { "//!" SAMPLER_ENTRY NEWLINE }+
                "SamplerState" ID ";"
SAMPLER_ENTRY := "FILTER" ("LINEAR"|"POINT")
              | "ADDRESS" ("CLAMP"|"WRAP")
```

commonBlock：

```
commonBlock := "//! COMMON" NEWLINE
              <任意不含其他 //! 的內容直到下一區塊>
```

passBlock：

```
passBlock := "//! PASS" NUMBER NEWLINE
             { "//!" PASS_ENTRY NEWLINE }+
             <HLSL pass body>
PASS_ENTRY := "IN" ID_LIST
            | "OUT" ID_LIST
            | "BLOCK_SIZE" NUM_LIST (1 or 2 items)
            | "NUM_THREADS" NUM_LIST (1..3)
            | "STYLE" ("PS"|"CS")
            | "DESC" STRING
```

附註：ID_LIST 為以逗號分隔的識別符，STRING 為整行（trim 前後空白），EXPR 為單行表達式（去除空白）。

---

## **Resource / Uniform / Pipeline 管理**

* **ResourceManager.js**：建立 Texture / Sampler / Uniform buffer，維護名稱對應
* **PipelineManager.js**：建立 Pipeline、bind group、dispatch Pass
* **UniformBinder.js**：提供外部接口動態更新 uniform

---

## **MagpieFX 到 WebGPU/WGSL 轉換指南**

將 Magpie 指令「轉成 WebGPU/WGSL」時的實務對應建議（重點、坑、與必做項）：

### **1. 資源類型映射（Texture / UAV / SRV / Sampler）**

* HLSL `Texture2D<T>`（通常用於讀） → WGSL `texture_sampled_2d`（對應 `texture_2d<T>` 與 `sampler` 分開）
* HLSL `RWTexture2D<T>`（寫入） → WGSL `texture_storage_2d<format, write>`（寫入 storage texture，format 必須為 WGSL 支援的
  storage 格式）
* HLSL `SamplerState` → WGSL `sampler` 或 `sampler_comparison`，綁定組（binding group）與 index 要對應。
* 注意：WGSL 與 WebGPU 對可寫 storage texture 的格式支援有限，需將 `EffectHelper::FORMAT_DESCS` 映射為
  WGSL/GPUSupportedFormats。若來源格式不可存為 storage texture，需調整設計（例如使用替代格式或雙緩衝拷貝）。

### **2. 常數緩衝區（cbuffer）**

* HLSL `cbuffer` 的行為在 WebGPU 中常以 `uniform` buffer 或 `storage` buffer 實作。
* 若參數有 `InlineParams`（原程式會把參數 inline 成 static const），在 WGSL 可把它直接作為 `let` 常數或作為 module-level
  const。否則把它們放入 `uniform` buffer 結構，並在 pipeline bind group 中綁定。

### **3. Thread group / Workgroup 大小**

* HLSL `numthreads(x,y,z)` 對應 WGSL `@workgroup_size(x,y,z)`（在 compute shader entry function 上的 attribute）。
* PASS 中的 `BLOCK_SIZE`、`NUM_THREADS` 需轉譯成 WGSL 的 `@workgroup_size` 與 workgroup dispatch 行為。PSStyle
  的特殊展開（16x16，64 thread pack）在 WGSL 中要用類似的 tile loop 與 `workgroup` memory 或手動計算。

### **4. 內建宏與函式**

* Magpie 在 `GeneratePassSource` 插入大量宏（如 `MP_BLOCK_WIDTH`、`MP_FP16`、`MF` 等）與內建函式。轉 WGSL 時需：
    * 將 `MF` 宏轉為 WGSL 型別別名或直接替換：`float` / `f16`（若支援）
    * 若使用 FP16（半精度），WebGPU 需硬體支援與 feature flag（`shader-f16`），並以 `f16` 型別或 extension 處理。
    * `MulAdd` 等函式若被 `USE MULADD` 指令啟用，需把其 HLSL 實現翻成 WGSL function。

### **5. 命名、binding index 與 layout**

* 原程式以 `register(t0)`, `register(u0)`, `register(s0)` 指定 binding index。WebGPU 使用 bind group 和 binding index（例如
  `@group(0) @binding(0)`）。你的 parser 在生成 WGSL 時必須根據 pass 的 inputs/outputs/samplers 為每個資源產生一致的 bind
  group/binding 方案。
* 注意 `inputs` 與 `outputs` 的 index 欄位在現有 code 用於註冊 tN/uN/sN。遷移時確保 mapping 保持順序。

### **6. TEXTURE FORMAT 與 WGSL compatibility**

* HLSL 可用很多 texel 格式，但 WebGPU 對 storage texture 與 sampling
  的支援有表格限制。必須建立一個格式對應表（EffectHelper::FORMAT_DESCS -> WebGPU format string），並在不支援時報錯或採替代策略。

### **7. Shader entry function 與多 render target（MRT）**

* HLSL PSStyle 中若有多個 outputs，程式會把它們當作多渲染目標或多 RWTexture2D。WGSL 必須為每個 output 創建對應的
  `texture_storage_2d<...>` binding，並在 compute shader 中寫入對應 storage texture。

### **8. 錯誤處理與限制檢查**

* 保持原有 parser 的檢查規則：必須檢查 PASS 序號連續性、IN/OUT 必要性、TEXTURE WIDTH/HEIGHT 配對、PARAMETER
  必要欄位等。這些都是遷移後行為必須保留的語法驗證邏輯。

---

範例：將 `//!` 定義的範例片段轉成 WGSL 的最小對應（示意）

原 Magpie：

```hlsl
//! PARAMETER Brightness
//! DEFAULT 1.0
//! MIN 0.0
//! MAX 2.0
//! STEP 0.01
float Brightness;
```

WGSL（可能的結果）：

```wgsl
// inline param
const Brightness: f32 = 1.0; // 或從 uniform 讀取
```

原 Magpie Pass/資源：

```hlsl
//! TEXTURE MyTex
//! FORMAT R8G8B8A8_UNORM
Texture2D MyTex : register(t0);

// in pass
//! PASS 1
//! IN INPUT, MyTex
//! OUT TEMP1
[numthreads(8,8,1)]
void __M(...) { ... }
```

WGSL 要點：

* 綁定 `MyTex` 為 `@group(0) @binding(N) var MyTex: texture_2d<f32>;`
* 綁定輸出 `TEMP1` 為 `@group(0) @binding(M) var<storage, write> TEMP1: texture_storage_2d<rgba8unorm, write>;`
* entry function 使用 `@workgroup_size(8,8,1)`。

---

# **WGFX 專案架構整理與重點翻譯總結**

以下為整份架構說明的核心內容萃取版，只保留你後續要做解析器與 WebGPU HOOK 時會用到的資訊。省略 CLI 不相關的細節。

# **1. 專案目錄職責（重點版）**

保持單一責任原則，每一個檔案只做一件事，後續你才能安全 hook。

### **index.js**

統一入口。負責轉呼叫：

* compile()
* dispatchPass()
* updateUniform()
* getOutput()

### **runtime/WGFXRuntime.js**

Runtime 主控流程：

* 解析 FX（調 Parser）
* 建 GPU resource（resource manager）
* 建 pipeline（pipeline manager）
* 排序 Pass
* dispatch

### **runtime/Parser.js**

核心解析器，為 runtime 和 CLI 共用：

* 解析所有 //! metadata 指令
* 建立中間 IR（commonCode, passes, textures, samplers, params）

重點：所有 WebGPU mapping 都要建立在這份 IR 上。

### **runtime/ResourceManager.js**

* 建立 texture / sampler / uniform buffer
* 建立 name → GPU object mapping

### **runtime/PipelineManager.js**

* 分析 Pass 依賴
* 建 pipeline
* 建 bind group
* dispatch pass shader

### **runtime/WGSLCodeGenerator.js**

* 把 IR → WGSL
* 巨集展開
* 函數多載解決
* 參數 inline or uniform buffer

### **runtime/UniformBinder.js**

* updateUniform()
* 動態寫入 uniform buffer

---

# **2. Parser（解析器）支援的 //! 指令總表（翻譯總整理）**

以下為所有可被解析器讀取的 command（依區塊分類）。

---

## **Header 區塊**

| 指令                       | 功能          | 註解               |
|--------------------------|-------------|------------------|
| `//! MAGPIE EFFECT`      | 文件魔法字       | 必須存在             |
| `//! VERSION <num>`      | MagpieFX 版本 | 必須等於編譯器內定版本      |
| `//! SORT_NAME <string>` | UI 排序名稱     | 非必要              |
| `//! USE <flags>`        | 功能啟用標記      | MULADD, _DYNAMIC |
| `//! CAPABILITY <flags>` | 硬體能力標記      | FP16             |
| `#include <...>`         | 允許          | parser 直接跳過      |

---

## **PARAMETER 區塊**

| 指令                       | 必需性 | 翻譯重點     |
|--------------------------|-----|----------|
| `//! PARAMETER <name>`   | 必需  | 開始一個參數區塊 |
| `//! DEFAULT <value>`    | 必需  | 預設值      |
| `//! MIN <value>`        | 必需  |          |
| `//! MAX <value>`        | 必需  |          |
| `//! STEP <value>`       | 必需  |          |
| `//! LABEL <string>`     | 可選  | UI 顯示名   |
| 尾段：`float X;` 或 `int X;` | 必需  | 型別與參數名   |

---

## **TEXTURE 區塊**

| 指令                     | 必需性    | 翻譯重點        |
|------------------------|--------|-------------|
| `//! TEXTURE <name>`   | 必需     |             |
| `//! SOURCE <Pass>`    | 可選     | 與其他互斥       |
| `//! FORMAT <format>`  | 多數情況必需 | Magpie 格式名稱 |
| `//! WIDTH <expr>`     | 選配但需成對 |             |
| `//! HEIGHT <expr>`    | 同上     |             |
| 尾段：`Texture2D <name>;` | 必需     | 宣告與名字一致     |

---

## **SAMPLER 區塊**

| 指令                          | 必需性 |    |
|-----------------------------|-----|----|
| `//! SAMPLER <name>`        | 必需  |    |
| `//! FILTER <LINEAR POINT>` |     | 必需 |
| `//! ADDRESS <CLAMPWRAP>`   |     | 可選 |
| 尾段：`SamplerState <name>;`   | 必需  |    |

---

## **COMMON 區塊**

| 指令           | 說明                        |
|--------------|---------------------------|
| `//! COMMON` | 從這裡開始到下一個 //! 為共用 HLSL 片段 |

---

## **PASS 區塊**

| 指令                            | 必須      | 重點              |
|-------------------------------|---------|-----------------|
| `//! PASS <index>`            | 必需      | pass index 必須連續 |
| `//! IN <tex-list>`           | 必需      | 輸入紋理            |
| `//! OUT <tex-list>`          | 必需      | 輸出紋理            |
| `//! BLOCK_SIZE <n[,m]>`      | 必需（非PS） |                 |
| `//! NUM_THREADS <x[,y[,z]]>` | 必需（非PS） |                 |
| `//! STYLE <PS/CS>`           | 可選      |                 |
| `//! DESC <string>`           | 可選      |                 |

---

---

# **3. 你要做的事（解析器 hook 指引）**

你已經完成 shader HLSL → WGSL翻譯。
接下來解析器需要負責：

1. 讀取所有 //! metadata
2. 建 IR
3. 把 IR 轉成 WebGPU 資源建立資訊
4. UI 需要 metadata → 所以你必須提供參數、texture、sampler、pass 的完整表

---

### **Hook 流程：**

#### **入口點：Pass1（你提到的）**

這通常是「第一個 Pass」或「第一個輸入 texture」。

parser 做完 IR 後：

**1）ResourceManager**

* 解析 `TEXTURE`, `SAMPLER`, `PARAMETER`
* 建立 GPU 資源
* 建 mapping table

**2）PipelineManager**

* 根據 Pass 的輸入與輸出
* 產生 bindGroupLayout
* 產生 pipeline

**3）UI Metadata**
從 IR 整理出：

* 參數（名稱、預設、max/min/step、label）
* 紋理列表
* sampler 列表
* pass 描述與依賴

然後你的 UI 可以讀 metadata 並渲染 sliders、dropdown 等。

---

# **4. 範例：你的解析器實作模型（IR 例子）**

```js
{
    parameters: [
        {
            name: "Brightness",
            type: "float",
            default: 1.0,
            min: 0.0,
            max: 2.0,
            step: 0.01,
            label: "Brightness"
        }
    ],

        textures
:
    [
        {
            name: "MyTex",
            format: "R8G8B8A8_UNORM",
            width: "INPUT_WIDTH",
            height: "INPUT_HEIGHT"
        }
    ],

        samplers
:
    [
        {
            name: "DefaultSampler",
            filter: "LINEAR",
            address: "CLAMP"
        }
    ],

        passes
:
    [
        {
            index: 1,
            in: ["INPUT", "MyTex"],
            out: ["TEMP1"],
            blockSize: [8, 8],
            numThreads: [8, 8, 1],
            style: "CS",
            desc: "主運算"
        }
    ],

        commonCode
:
    " ... "
}
```
