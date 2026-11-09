# Magpie HLSL → WGFX WGSL 移植指南

本文件是 WGFX 移植 `effects_src/` 下 Magpie HLSL 特效的**作業規範**。
所有移植效果都必須遵守本指南，並在 `docs/porting-guide.md` 之外的個別檔頭
記錄該特效特有的偏差（porting notes）。

來源：`effects_src/`（Magpie HLSL 效果庫；OpenNR-ReShade 為 git submodule）。

---

## 1. 執行模型差異（最重要）

Magpie 有兩種 pass 風格，WGFX 統一為 **compute shader**。

| Magpie | WGFX |
|---|---|
| `//!STYLE PS` + `float4 Pass1(float2 pos)` | compute，`fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>)` |
| `//!STYLE CS` + `void Pass1(uint2 blockStart, uint3 threadId)` | compute，`fn Pass1(@builtin(workgroup_id) wg, @builtin(local_invocation_id) lid)` |
| `//!STYLE VS` + `float4 Pass1(float4 pos: SV_Position)` | compute（見 §7） |

### 1.1 `//!STYLE PS` 的 `pos` 語義

Magpie 的 PS pass 傳入的 `pos` 是**正規化 UV（0..1）**，不是 texel 座標。
證據：`Sharpen/LumaSharpen.hlsl` 直接以 `INPUT.SampleLevel(sam, pos, 0)` 取樣，
並以 `GetInputPt()`（= 1/inputSize）作為「一個 texel 的 UV 尺寸」來加偏移。

WGFX 對應寫法：

```wgsl
fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outSize = scene.outputSize;
    if (gid.x >= outSize.x || gid.y >= outSize.y) {
        return;
    }
    // Magpie PS 的 pos = 正規化 UV
    let pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.outputPt;
    let inputPt = scene.inputPt;          // 一個 input texel 的 UV 尺寸
    let color = textureSampleLevel(INPUT, sam, pos, 0.0).rgb;
    textureStore(OUTPUT, gid.xy, vec4<f32>(color, 1.0));
}
```

### 1.2 `//!STYLE CS` 的 blockStart / threadId 語義

Magpie 的 CS pass 收到 `blockStart`（block 的起點）與 `threadId`（block 內的執行緒索引）。
Magpie 慣例是 `BLOCK_SIZE 16`（16×16 block、64 threads，每個 thread 負責 2×2 quad）
或 `BLOCK_SIZE 8`（8×8 block、64 threads，每個 thread 負責 1 pixel）。

WGFX 對應寫法（見 `examples/Anime4K_Restore_S.wgsl`）：

```wgsl
//! NUM_THREADS 64   →  @workgroup_size(64)
fn Pass1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var gxy: vec2<u32> = (Rmp8x8(lid.x) << vec2<u32>(1u)) + wg.xy * vec2<u32>(16u);
    ...
}
```

`Rmp8x8(a) = uint2(a / 8u, a % 8u)`（Magpie `StubDefs.hlsli` 的 8×8 展開）。

> WGFX 的 `//! NUM_THREADS x,y,z` 決定 `@workgroup_size`。Magpie 的 `//! NUM_THREADS 64`
> 是「總執行緒數」，WGFX 要寫成 `//! NUM_THREADS 64,1,1`（單維），
> 或依需求寫成 `//! NUM_THREADS 8,8,1`（二維）。

---

## 2. 型別對應

Magpie `StubDefs.hlsli` 的別名 → WGSL：

| Magpie | WGSL |
|---|---|
| `MF` | `f32` |
| `MF2` / `float2` | `vec2<f32>` |
| `MF3` / `float3` | `vec3<f32>` |
| `MF4` / `float4` | `vec4<f32>` |
| `uint2` | `vec2<u32>` |
| `int2` | `vec2<i32>` |
| `MF3x4` | `mat3x4<f32>` |
| `MF4x3` | `mat4x3<f32>` |
| `MF4x4` | `mat4x4<f32>` |

### 2.1 矩陣乘法順序（HLSL vs WGSL）

HLSL 的 `MulAdd(v, MF RxC(W), acc)` 等價於 WGSL 的 `acc + M * v`：
HLSL 的列優先建構子與 WGSL 的欄優先建構子，在乘法順序反轉時互相抵消，
因此**權重數字可逐字照抄，不需轉置**。

```wgsl
// HLSL: result = MulAdd(src, MF3x4(...), result);
result = mat3x4<f32>(...) * src + result;
```

### 2.2 HLSL → WGSL 常用內建

| HLSL | WGSL |
|---|---|
| `saturate(x)` | `saturate(x)`（WGSL 內建） |
| `lerp(a,b,t)` | `mix(a,b,t)` |
| `frac(x)` | `fract(x)` |
| `step(edge,x)` | `step(edge,x)`（WGSL 語意相同：`x < edge ? 0 : 1`） |
| `mul(x,y)` | `x * y` |
| `dot`, `cross`, `length`, `normalize` | 同名 |
| `rcp(x)` | `1.0 / x`（**注意 NaN 語意差異**，見 §6） |
| `float3 r[4]; r[i]` | `var r: array<vec3<f32>, 4>; r[i]` |
| `Texture2D.SampleLevel(s, uv, 0)` | `textureSampleLevel(tex, s, uv, 0.0)` |
| `Texture2D.GatherRed(s, uv)` | `textureGather(0, tex, s, uv)` |
| `tex[pos]`（storage） | `textureStore(tex, pos, v)` |
| `tex.Load(int3(pos,0))` | `textureLoad(tex, vec2<i32>(pos), 0)` |
| `transpose(M)` | `transpose(M)` |

`textureGather` 的分量順序 WGSL 與 D3D **一致**：
`x=(0,1) y=(1,1) z=(1,0) w=(0,0)`（相對取樣點左上方的 texel）。
Magpie 的 `"w z / x y"` 填法可逐字照抄。

---

## 3. 資源與綁定

### 3.1 紋理

```wgsl
//! TEXTURE
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D tex1;
```

WGFX 會依 pass 的 `IN`/`OUT` 自動決定 sampled / storage：

* 名稱以 `_storaged` 結尾 → 該 pass 的 `OUT` 時為 `texture_storage_2d<fmt, write>`。
* 其餘 → `texture_2d<f32>`（可取樣）。

寫入用 `textureStore(tex, vec2<i32>(p), v)`；讀取用 `textureLoad` / `textureSampleLevel`。

> **`INPUT` 的預設格式是 `rgba8unorm`**，其他紋理預設 `rgba16float`。
> 需要 f16 精度的中間紋理必須顯式宣告 `//! FORMAT R16G16B16A16_FLOAT`。

### 3.2 取樣器

```wgsl
//! SAMPLER
//! FILTER LINEAR      // LINEAR | POINT | NEAREST
//! ADDRESS CLAMP      // CLAMP | WRAP
SamplerState sam;
```

* `FILTER LINEAR` → `filtering` sampler；`FILTER POINT`/`NEAREST` → `non-filtering`。
* `ADDRESS CLAMP` → `clamp-to-edge`；`ADDRESS WRAP` → `repeat`。
* WGFX 自動命名為 `var sam: sampler;`，取樣時直接用 `textureSampleLevel(INPUT, sam, uv, 0.0)`。

> **non-filtering sampler 的限制**：WGSL 規定 non-filtering sampler 只能用
> `textureSampleLevel`（或 `textureSample` 在 fragment），且**不能**搭配
> `textureGather`（gather 需要 filtering sampler）。
> 若原 HLSL 用 POINT sampler 做 Gather，WGFX 端改用 `textureLoad` 手動取鄰居。

### 3.3 Uniform 參數

Magpie 的 `float threshold;` 宣告 → WGFX 的 `//! PARAMETER` 區塊：

```wgsl
//! PARAMETER Threshold
//! TYPE float
//! LABEL "Threshold"
//! MIN 0
//! MAX 512
//! DEFAULT 64
//! STEP 4
```

pass 內讀取：`uniforms.Threshold`（名稱與 `//! PARAMETER` 的名稱一致）。

> Magpie 的參數名常是小寫（`threshold`），WGFX 保留原名即可
> （`uniforms.threshold`）。

---

## 4. 場景資訊（`scene`）

WGFX 自動注入 `scene`（`@group(0) @binding(4)`）：

| 欄位 | 型別 | 意義 |
|---|---|---|
| `scene.inputSize` | `vec2<u32>` | 輸入紋理尺寸（texel） |
| `scene.inputPt` | `vec2<f32>` | `1 / inputSize`（一個 input texel 的 UV 尺寸） |
| `scene.outputSize` | `vec2<u32>` | 輸出紋理尺寸（texel） |
| `scene.outputPt` | `vec2<f32>` | `1 / outputSize` |
| `scene.scale` | `vec2<f32>` | `outputSize / inputSize` |
| `scene.frame` | `u32` | 影格計數器（時間性特效用） |

Magpie 的 `GetInputSize()` / `GetInputPt()` / `GetOutputSize()` / `GetOutputPt()`
分別對應 `scene.inputSize` / `scene.inputPt` / `scene.outputSize` / `scene.outputPt`。

---

## 5. 檔案骨架（模板）

```wgsl
// <EffectName> — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/<path>.hlsl
// (原授權：<作者/授權>；本移植僅改寫 pass 結構與取樣語意，演算法常數逐字保留。)
//
// Porting notes:
//   - <逐條列出語意差異與處理方式>

//! VERSION 4
//! SORT_NAME <EffectName>

//! PARAMETER ...
//! TEXTURE ...
//! SAMPLER ...

//! PASS 1
//! DESC "..."
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    ...
}

//! END
```

> **進入點必須是 pass 區塊的第一個函式**——WGFX 產生器會把
> `@compute @workgroup_size(...)` 掛在 `fn Pass<N>(` 或 `fn main(` 上。
> helper 函式可放在進入點**之前**（在同一 pass 區塊內）。

---

## 6. 已知的語意陷阱

1. **`min`/`max` 的 NaN 傳播**：HLSL 的 `min`/`max` 以比較語意 heals NaN；
   WGSL 的 `min`/`max` 依規格傳播 NaN。若原碼會產生 NaN，必須顯式加保護
   （參考 `examples/FSR_x2.wgsl` 的 `hitMin`/`hitMax` 寫法）。
2. **`rcp(0)`**：HLSL 的 `rcp(0)` = `+Inf`；WGSL 的 `1.0/0.0` 也是 `+Inf`，
   但 `0 * Inf` = NaN。若原碼依賴 HLSL 的 NaN healing，改用 `select(...)` 明確保護。
3. **整數除法**：`a / b` 在 WGSL 中若兩者皆為整數是整數除法（與 HLSL 一致），
   但 `f32` 參與時是實數除法。移植時逐行核對型別。
4. **紋理邊界**：WGSL 的 `textureLoad` 越界回傳 0（與 D3D `Load` 一致）；
   `textureSampleLevel` 依 sampler 的 address mode。Magpie 的 POINT sampler
   預設為 CLAMP，移植時要對齊。
5. **`textureSampleLevel` 在 compute shader 中合法**（`textureSample` 不合法）。
6. **f16**：`//! CAPABILITY FP16` 僅為標記；WGFX 端請用 `rgba16float` 紋理
   與 `f32` 運算（與既有 Anime4K 移植一致）。

### 6.1 `f32 → f16` 轉換是**截斷**（不是四捨五入）

本專案使用的 node-webgpu（Dawn Node 綁定）在把 `f32` 寫入 `rgba16float` 紋理時，
**向零截斷** mantissa，而非 IEEE 的 round-to-nearest-even。實測（`test/f16_conversion.mjs`）：

| f32 值 | mantissa 小數部分 | 正確（四捨五入） | 實際（截斷） |
|---|---|---|---|
| `0.1019607843`（26/255） | 646.526 | `0x2e87` | `0x2e86` |
| `0.15` | 204.800 | `0x30cd` | `0x30cc` |
| `0.7777777778` | 568.889 | `0x3a39` | `0x3a38` |
| `0.8745098039`（223/255） | 766.996 | `0x3aff` | `0x3afe` |

**影響**：CPU 參考若直接與 GPU 輸出比對，會出現最大約 1 個 f16 ulp 的偏差
（在 1.0 附近 ≈ 9.8e-4，在 0.5 附近 ≈ 4.9e-4）。因此：

* 通用容差用 **1e-3**（涵蓋 1.0 以下的 1 ulp）；
* 需要位元精確比對時，CPU 端必須**模擬截斷**（`trunc_f16()`）；
* 在 WGSL 內**手動**做 f16 轉換（像 OpenNR 的 `f32tof16()`）則不受影響——
  那是純 WGSL 位元運算，與硬體轉換指令無關。

### 6.2 `copyTextureToBuffer` 的 `bytesPerRow` 必須是 256 的倍數

WebGPU 規範要求 `copyTextureToBuffer` 的 `bytesPerRow` 為 256 的倍數。
讀回 rgba16float 紋理時：

```js
const bytesPerRow = Math.ceil((w * 8) / 256) * 256;   // 不是 /16
```

（`/16` 只在 256 的倍數恰好成立時才會過，例如 w=64 → 512、w=256 → 2048。）


---

## 7. 驗收

每個移植效果都必須通過：

```bash
node test/port_check.mjs <path/to/Effect.wgsl>
```

檢查項目：

* 編譯無驗證錯誤；
* 所有 pass dispatch 無驗證錯誤；
* OUTPUT 內容非全黑、且與 INPUT 有實質差異（表示確實有作用）；
* 對可解析的 CPU 參考（若有）逐通道比對。

`test/port_check.mjs` 是**通用**驗收（編譯 + 執行 + 健全性）。
數值正確性的強驗收請另寫專屬測試（例如 `test/impulse_response.mjs`）。

> 本輪移植戰役（2026-11）新增的框架事實與實務規則見 §9–§15：`//! SOURCE` 外部紋理、
> 裝置上限宿主契約、HLSL→WGSL 語彙落差（含逐條實測錯誤訊息）、storage 格式限制、
> 驗收實務、workgroup 記憶體超限的三種處理、移植注記慣例。
> 家族級覆蓋率與已知限制見 [`docs/PORTING_STATUS.md`](PORTING_STATUS.md)，兩份文件互補不重複。

---

## 8. 產出位置

| 類型 | 位置 |
|---|---|
| 移植後的效果 | `effects/<Family>/<Name>.wgsl` |
| 既有示範效果 | `examples/` |
| 專屬驗收測試 | `test/<name>.mjs` |
| 移植筆記 | 效果檔頭的 `Porting notes:` |

`effects/` 下的目錄依來源家族分：`OpenNR/`、`Anime4K/`、`Sharpen/`、`RAVU/`…

---

## 9. `!SOURCE` 外部紋理 / External textures

LUT 系卷積（RAVU / SMAA / NIS）把權重放在外部 DDS，而不是內嵌常數。效果檔只記錄**檔名**，
像素由宿主餵入。LUT-family convolutions keep their weights in an external DDS; the effect
records only the **file name** and the host supplies the pixels.

### 9.1 宣告語法 / Declaration

```wgsl
//! TEXTURE
//! SOURCE ravu_3x_lut2_f16.dds      // 檔名（大小寫不敏感；不含目錄）
//! FORMAT rgba16float
Texture2D ravu_3x_lut2;

//! SAMPLER
//! FILTER LINEAR
SamplerState sam_ravu_3x_lut2;
```

* `//! SOURCE` 必須落在**它所屬的** `//! TEXTURE` 區塊內、且在 `Texture2D <name>;` **之前** ——
  `tools/source_textures.mjs` 的 `scanSourceDecls()` 以 TEXTURE 區塊為單位配對名稱與檔名，
  順序錯了就等同沒宣告。The directive must sit inside its own `//! TEXTURE` block, before
  the declaration line; matching is per TEXTURE block.
* **不要**宣告 `//! WIDTH` / `//! HEIGHT`：尺寸來自 DDS 檔頭（`ResourceManager` 對有 `source`
  卻找不到宿主紋理的情況會給出明確錯誤，見 §9.3）。
* 取樣寫法與一般紋理相同：`textureSampleLevel(<name>, sam_<name>, uv, 0.0)`。
* 真實範例：`effects/RAVU/RAVU_3x_R2.wgsl`（1 個 LUT）、`effects/RAVU/RAVU_Zoom_AR_R2.wgsl`
  （2 個：主 LUT + anti-ringing LUT）、`effects/SMAA/SMAA_Low.wgsl`（`AreaTex` + `SearchTex`）。

### 9.2 宿主如何餵入 / How the host supplies it

裝置與紋理都由宿主建立（見 §10）。`WGFXRuntime.compile(effectCode, externalResources)` 的
`externalResources.textures[<name>]` 就是外部紋理契約（實作在
`src/runtime/ResourceManager.ts`）：

| 欄位 Field | 說明 |
|---|---|
| `size: [w, h]` | texel 尺寸（必填） |
| `format` | WebGPU 格式名（必填；DDS 解碼器已解出） |
| `usage` | 至少 `TEXTURE_BINDING \| COPY_DST` |
| `data` | 解碼後的像素位元組；有值時立即 `queue.writeTexture` 上傳 |
| `bytesPerRow` | 可選；非 256 倍數時 `writeTextureData()` 會自動重排到 256 對齊 |

```js
import { readDDSFile } from './tools/dds.mjs';
// → { width, height, format, data, bytesPerPixel, bytesPerRow, mipCount }
const lut = readDDSFile('effects_src/RAVU/ravu_3x_lut2_f16.dds');
const U = GPUTextureUsage;
await runtime.compile(code, {
  defines: { INPUT_WIDTH: 256, INPUT_HEIGHT: 256 },
  textures: {
    INPUT:  { size: [256, 256], format: 'rgba8unorm',  usage: U.TEXTURE_BINDING | U.COPY_DST },
    OUTPUT: { size: [768, 768], format: 'rgba16float', usage: U.STORAGE_BINDING | U.COPY_SRC },
    ravu_3x_lut2: {
      size: [lut.width, lut.height], format: lut.format,
      usage: U.TEXTURE_BINDING | U.COPY_DST,
      data: lut.data, bytesPerRow: lut.bytesPerRow,
    },
  },
});
```

> 別名規則：`<name>_sampled` / `<name>_storaged` 是同一邏輯紋理的雙視圖；外部紋理請用
> **邏輯名**（去掉後綴）提供。

### 9.3 驗收端如何自動解析 / Automatic resolution

`test/port_check.mjs` 與 `test/test_runtime.js` **共用** `tools/source_textures.mjs`，
因此「手動驗收」與「框架套件」的行為一致：

| 匯出 Export | 用途 |
|---|---|
| `scanSourceDecls(code)` | 掃出 `Map<紋理名, SOURCE 檔名>` |
| `indexFilesByBasename(dir)` | 遞迴建立「小寫檔名 → 路徑」索引（大小寫不敏感，先到先得） |
| `loadDDSTexture(file)` | 以 `tools/dds.mjs` 解碼 → `{size, format, usage, data, bytesPerRow}` |
| `prepareExternalTextures(code, opts)` | 一次備妥全部外部紋理；`opts = { textureDir, explicitTextures, quiet, label }` |

```bash
node test/port_check.mjs effects/RAVU/RAVU_3x_R2.wgsl         # 未給 --texture 時自 effects_src/ 遞迴以檔名解析
node test/port_check.mjs <effect.wgsl> --texture NAME=<path>   # 顯式指定（可重複；NAME 大小寫不敏感）
node test/port_check.mjs <effect.wgsl> --texture-dir <dir>     # 換掉自動解析的根目錄（預設 effects_src/）
npm run test:external                                          # node test/test_external_textures.mjs（契約測試）
npm test                                                       # Test E 亦會解析 SOURCE 後真跑
```

解析順序：顯式 `--texture` / `explicitTextures` 優先，其次 `textureDir` 底下的檔名索引。
**LUT 系效果不需要進 `test/effects-suite.json` 的 `skip` 表**（`skip` 是留給需要 storage buffer
之類「宿主非 LUT 資源」的效果，如 OpenNR）。

找不到檔案時的兩種錯誤（皆為實際字串；這兩條就是本契約的可觀測行為）:

```
# A. 宿主沒餵 → src/runtime/ResourceManager.ts
Texture "ravu_lut2" declares //! SOURCE ravu_lut2_f16.dds but no texture was provided.
Supply it via externalResources.textures["ravu_lut2"] (port_check: --texture ravu_lut2=<path>).

# B. 驗收端也解析不到 → tools/source_textures.mjs
效果宣告 //! SOURCE ravu_lut2_f16.dds（紋理 ravu_lut2）但找不到檔案:請以 --texture ravu_lut2=<路徑> 提供或確認 effects_src/ 下存在該檔
```

實例：`effects_src/RAVU/` 有 **13** 個 DDS，被 26 個效果引用（`Zoom_AR` 系列各用 2 個；
`ravu_3x_lut2/3/4`、`ravu_lite_lut2/3/4`、`ravu_lut2/3/4`、`ravu_zoom_lut2`、`ravu_zoom_lut3`、
`ravu_zoom_lut2_ar`、`ravu_zoom_lut3_ar`）。家族級覆蓋率與「DDS 不內嵌」的限制見
[`docs/PORTING_STATUS.md`](PORTING_STATUS.md) §2 / §5。

---

## 10. 裝置上限契約 / Device limits (host contract)

WGFX **不建立** `GPUDevice`（`src/` 內沒有任何 `requestDevice` 呼叫），裝置一律由宿主建立。
所以「這個效果需要更高上限」是**宿主契約**，不是效果檔的缺陷 —— 驗收端沒索取就會得到假紅燈。
WGFX never creates the `GPUDevice`; requesting higher limits is the host's job, and a harness
that forgets it will report a false failure.

| 上限 Limit | WebGPU 預設 Default | 本機 adapter | 需要它的效果 Needed by |
|---|---:|---:|---|
| `maxStorageTexturesPerShaderStage` | 4 | 8 | 單一 pass 寫多張 storage texture 的多輸出 CNN。**實測：目前 repo 內的上限剛好是 4**（14 檔，例如 `effects/CuNNy2/CuNNy-4x24-NVL.wgsl`、`effects/CuNNy/CuNNy-16x16C-NVL.wgsl`、`effects/Anime4K/Anime4K_Restore_Soft_UL.wgsl`），**預設值 4 即可通過**；驗收端索取 8 屬於未來擴充的餘裕 |
| `maxComputeWorkgroupStorageSize` | 16,384 B | 32,768 B | `RAVU_R3_RGB`（19,264 B）、`RAVU_R4_RGB`（23,424 B） |

不索取時的**確切** Dawn 錯誤（皆為 `popErrorScope()` 取回的原始訊息）:

```
# maxComputeWorkgroupStorageSize
# 重現：node test/port_check.mjs effects/RAVU/RAVU_R4_RGB.wgsl（未索取上限的裝置）
The total use of workgroup storage (23424 bytes) is larger than the maximum allowed (16384 bytes).
This adapter supports a higher maxComputeWorkgroupStorageSize of 32768, which can be specified in
requiredLimits when calling requestDevice(). Limits differ by hardware, so always check the adapter
limits prior to requesting a higher limit.

# maxStorageTexturesPerShaderStage
# 重現：預設 limits 的裝置 + 單一 compute stage 綁 8 張 write-only storage texture
The number of storage textures (8) in the Compute stage exceeds the maximum per-stage limit (4).
This adapter supports a higher maxStorageTexturesPerShaderStage of 8, which can be specified in
requiredLimits when calling requestDevice(). Limits differ by hardware, so always check the adapter
limits prior to requesting a higher limit.
```

正解 / Fix（驗收端與真實宿主都該這樣寫）:

```js
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
    requiredLimits: {
        maxStorageTexturesPerShaderStage: 8,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
});
```

* 先在這台機器量一次：`node tools/_storage_probe.mjs`（逐格式 × 逐存取模式探測 storage 能力），
  或在程式裡直接讀 `adapter.limits`（上表「本機 adapter」欄就是這樣得到的）。
* 驗收端已比照索取：`test/port_check.mjs`、`test/test_runtime.js`
  （見 [`docs/PORTING_STATUS.md`](PORTING_STATUS.md) §3）。
* 宿主面向的說明：[`README.md`](../README.md) 的 **Device limits (host contract)** 章節、
  [`README_TW.md`](../README_TW.md) 的 **裝置上限（宿主契約）** 章節。
* 移植時若撞到上限，先讀 §14 的三種處理再決定要不要動宿主契約。

---

## 11. HLSL→WGSL 語彙落差 / Lexical gaps (measured)

本節每一條都是本輪移植實際踩到、並在真實裝置上驗證過的落差。錯誤訊息是此
Dawn/node-webgpu 版本（Tint）的**原始輸出**，不是轉述。Every rule below was hit for real;
the messages are verbatim Tint output from this build.

### 11.1 快速對照表 / Quick reference

| HLSL | WGSL | 照抄的後果 / If copied verbatim |
|---|---|---|
| `mul(a, b)` | `(a) * (b)` | `error: unresolved call target 'mul'` |
| `atan(y, x)`（prescalers 模板的 `#define atan atan2`） | `atan2(y, x)` | `no matching call to 'atan(f32, f32)'` |
| `mod(a, b)` | `a % b`（f32 亦有定義）或 COMMON helper | `no matching call to 'mod(f32, f32)'` |
| `mix(a, b, boolCond)` | `select(b, a, boolCond)` | 巢狀時產生無效的 4 引數 `select` |
| `vecN<f32> x = e;`（typed local decl） | `var x: vecN<f32> = e;` | `error: expected '=' for assignment` |
| module-scope `const N = scene.x;` | 移進函式，或 `var<private>` | `'var scene' cannot be referenced at module-scope` |
| 非方形矩陣按「列」使用（`m[i]` 是 vec3） | WGSL 矩陣以**欄**索引 → 需轉置表示 | 語意相反（不報錯，但數值錯） |
| 兩矩陣 `cg * cg`（HLSL `*` 是分量相乘） | 自備 `matrixCompMult()` | `no matching overload for 'operator * (mat4x3<f32>, mat4x3<f32>)'` |
| `mat4x3(0.0)`（純量 splat 初始化矩陣） | 12 個純量，或 4 個 `vec3` 欄 | `no matching constructor for 'mat4x3<f32>(abstract-float)'` |
| `clamp(v4, 0.0, 1.0)`（純量→向量隱式廣播） | `clamp(v4, vec4<f32>(0.0), vec4<f32>(1.0))` | `no matching call to 'clamp(vec4<f32>, abstract-float, abstract-float)'` |
| `ivec2(u32Var, u32Var)`、`(uint)id / intVar` | `vec2<i32>(i32(x), i32(y))`、`u32(id) / u32(v)` | `no matching constructor for 'vec2<i32>(u32, u32)'`／引數型別不符 |
| `if (c) return;` | `if (c) { return; }` | `error: expected '{' for if statement` |
| `tex.SampleLevel(s, uv, 0.0, offset)`（offset 可變） | offset 必須是 **const-expression** | `the offset argument must be a const-expression` |

### 11.2 矩陣語意：`mul`、轉置與「列索引」/ Matrix semantics

§2.1 已規範「HLSL `mul(M, v)` ↔ WGSL `M * v`，權重數字逐字照抄（HLSL 列優先建構子與 WGSL
欄優先建構子在乘法順序反轉時互相抵消）」。本輪補的是**非方形矩陣被當成「列」使用**的情況：

```hlsl
// prescalers.hlsli: #define mat4x3 float4x3    // HLSL float4x3 = 4 列 × 3 行
mat4x3 cg;                                       // 之後用 cg[0] / cg[2]，每個都是 vec3
float4x3 outerProduct(float3 l, float4 r) { return mul(float4x1(r), float1x3(l)); }
```

`float4x1` / `float1x3` 在 WGSL 沒有對應型別，而「`m[i]` 取第 i 列」在 WGSL 也不成立
（WGSL 的 `m[i]` 是第 i **欄**）。本輪採用的作法是**轉置慣例**：把 HLSL 的第 i 列放進 WGSL
`mat4x3<f32>` 的第 i 欄，於是 `m[i]`、`+`、`+=` 與 HLSL 完全同義；`outerProduct` 與
`matrixCompMult` 則寫成 COMMON 輔助：

```wgsl
// 由 effects/RAVU/_gen_ravu.mjs 產生；實例見 effects/RAVU/RAVU_Zoom_AR_R2_RGB.wgsl
fn outerProduct(l: vec3<f32>, r: vec4<f32>) -> mat4x3<f32> {
    return mat4x3<f32>(r.x * l, r.y * l, r.z * l, r.w * l);
}
fn matrixCompMult(a: mat4x3<f32>, b: mat4x3<f32>) -> mat4x3<f32> {
    return mat4x3<f32>(a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]);
}
```

* WGSL 的 `*` 對兩矩陣**不是**分量相乘；形狀不符會直接報錯（見上表第 8 列）。
* WGSL 的矩陣建構子**不接受單一純量 splat**：要嘛 12 個純量（欄優先填入），要嘛 4 個
  `vec3`。零矩陣寫 `mat4x3<f32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)`。
* 檔頭請把採用的慣例寫進 Porting notes（§15）—— 這是「不報錯但會算錯」的高風險區。

### 11.3 `mix` 的兩種身分 / Two faces of `mix`

Magpie 的 `prescalers.hlsli` 把 `mix` 重載成布林版 lerp：
`float mix(float a, float b, bool c) { return c ? b : a; }`，所以

```hlsl
theta = mix(mod(atan(L1 - a, b) + PI, PI), 0.0, abs(b) < 1.192092896e-7);   // 第三引數是 bool
```
```wgsl
var theta: f32 = fmodWGSL(atan2(L1 - a, b) + PI, PI);
theta = select(0.0, theta, abs(b) < 1.192092896e-7);
// select(false_value, true_value, cond) == HLSL 的 c ? b : a
```

但同一個模板裡的 `LUTPOS`：

```hlsl
#define LUTPOS(x, lut_size) mix(0.5 / (lut_size), 1.0 - 0.5 / (lut_size), (x))
subpix = LUTPOS(subpix, vec2(9.0, 9.0));    // 第三引數是 vec2 → 這是真正的 lerp
```
```wgsl
subpix = mix(0.5 / (vec2<f32>(9.0, 9.0)), 1.0 - 0.5 / (vec2<f32>(9.0, 9.0)), subpix);
```

判準：**第三個引數含比較運算子 → `select`；否則 → `mix`**。
實作這個判準要用括號感知的遞迴展開：`mix(mix(0.0,1.0,c1), mix(2.0,3.0,c2), c3)` 這種巢狀
若交給 naïve 的 `[^,]+` 正規式，會生出 4 引數的無效 `select`（本輪 `effects/RAVU/_gen_ravu.mjs` 因此
在 HLSL 端自行展開，並把純 lerp 的 `mix` 先改名 `__MIX__` 避開工具的正規式）。

### 11.4 module-scope `const` 不得引用 uniform

```wgsl
const N: u32 = scene.inputSize.x;   // ✗
// error: 'var scene' cannot be referenced at module-scope
```

Magpie 模板常見 `static const float2 INPUT_size = float2(GetInputSize());`。兩種寫法都可行：

1. 移進使用它的函式內（最單純）；
2. 交給框架自動處理 —— `src/runtime/WGSLCodeGenerator.ts` 的 `_promoteUniformConstsToPrivate()`
   會把它改成 `var<private>` 並在進入點開頭初始化（鏈式引用會依宣告順序處理）。
   產物實例：`effects/RAVU/RAVU_Lite_R2.wgsl`（來源是
   `static const float2 INPUT_size = float2(GetInputSize());`）。

純常數（不引用 uniform）不受影響，維持 `const` 即可。

### 11.5 型別與轉型要顯式 / Make types explicit

```wgsl
// (a) HLSL typed local declaration：框架相容層接受，但原生 WGSL 要 var
vec2<f32> UV = vec2<f32>(0.0);        // error: expected '=' for assignment
var UV: vec2<f32> = vec2<f32>(0.0);   // ✓

// (b) 純量→向量不做隱式廣播（函式引數層級）
let v = clamp(vec4<f32>(x), 0.0, 1.0);
// error: no matching call to 'clamp(vec4<f32>, abstract-float, abstract-float)'
let v = clamp(vec4<f32>(x), vec4<f32>(0.0), vec4<f32>(1.0));   // ✓

// (c) int 與 uint 不互相隱式提升（HLSL 會）
let v = vec2<i32>(x, y);              // x, y: u32
// error: no matching constructor for 'vec2<i32>(u32, u32)'
let v = vec2<i32>(i32(x), i32(y));    // ✓
```

> 本輪的 `(uint)id / rect.x`（int→uint）與 `ivec2(x, y)`（uint→int）都屬於 (c)。
> 框架相容層會處理 `(uint)x` 這種 C 式轉型，但**不會**替你推斷運算元型別；遇到混算請在
> 來源端就補上顯式轉換。

### 11.6 `textureSampleLevel` 的 `offset` 必須是 const-expression

```wgsl
var o: vec2<i32> = vec2<i32>(1, 0);
let c = textureSampleLevel(t, s, uv, 0.0, o);
// error: the offset argument must be a const-expression
```

需要**可變**位移時，改為把位移加進 uv（本 repo 既有做法，見 §2.2 與
`effects/Anime4K/Anime4K_Upscale_S.wgsl` 的取樣慣例），或改用 `textureLoad` 手動取鄰居。

### 11.7 來源檔行尾 / Source line endings

`effects_src/` 的 HLSL 是 **CRLF**。以 JS 正規式處理時，未加 `m` 的 `/…$/`、
`l.replace(/\/\/.*$/, '')` 這類寫法在行尾有 `\r` 時會**靜默失配**（本輪 `effects/RAVU/_gen_ravu.mjs`
在註解剝除上踩過，導致殘留掃描誤報）。轉換器請先 `src.replace(/\r\n?/g, '\n')`。

---

## 12. storage texture 格式限制 / Storage texture formats

WebGPU（本 Dawn 建置）支援 write-only storage 存取的格式是一個**有限集合**，窄通道格式
不在其中。The set of write-only storage formats is limited; narrow-channel formats are absent.

```
# 實測（預設 limits 的裝置）：把 rg8unorm 建為 storage texture
The texture usage (TextureUsage::(TextureBinding|StorageBinding)) includes
TextureUsage::StorageBinding, which is incompatible with the format (TextureFormat::RG8Unorm).
 - While validating [TextureDescriptor """"].
 - While calling [Device].CreateTexture([TextureDescriptor """"]).
```

| 想用 Want | 改用 Substitute | 備註 |
|---|---|---|
| `r8unorm` / `r8snorm` / `r8uint` / `r8sint` | `rgba8unorm` / `rgba8snorm` / `rgba8uint` / `rgba8sint` | 只用到 `.x` 時量化不變 |
| `rg8unorm`（HLSL `R8G8_UNORM`） | `rgba8unorm` | SMAA `edgesTex` 實例；r/g 的 8-bit 量化完全相同，b/a 未使用 |
| `r16float` / `rg16float` | `rgba16float` | RAVU `ravu_int11` 實例（來源 hlsl 是 `R16_FLOAT`） |
| `r16unorm` / `rg16unorm` / `rgba16unorm`、`rg11b10ufloat`、`rgb9e5ufloat`、`rgb10a2unorm` | `rgba16float` | |
| `bgra8unorm`、`*-srgb` | `rgba8unorm` | |

* **框架會自動映射**：`src/runtime/WGSLCodeGenerator.ts` 的 `storageCapableFormat()` 同時用於
  WGSL 宣告、BindGroupLayout 與 `GPUTexture` 建立，所以效果檔宣告 `r16float` 也能跑。
  不過**明示**可 storage 的格式（如 `//! FORMAT rgba16float`）可讀性更好，也是本輪 RAVU 產物的
  作法（`effects/RAVU/RAVU_R2.wgsl` 的 `ravu_int11`）。
* 單通道語意不會跑掉：寫入用 `textureStore(tex, pos, vec4<f32>(value, 0.0, 0.0, 0.0))`、
  讀取端只取 `.x`（見 `effects/RAVU/RAVU_R2.wgsl` 的 `imageStoreOverride`）。
* 自己量一次：`node tools/_storage_probe.mjs`（逐格式 × write-only / read-only / read-write）。
  完整的可/不可用集合也在 `src/runtime/WGSLCodeGenerator.ts` 的 `STORAGE_CAPABLE_FORMATS`
  與 `STORAGE_FORMAT_SUBSTITUTE`。家族級說明見
  [`docs/PORTING_STATUS.md`](PORTING_STATUS.md) §5（SMAA `edgesTex`）。

---

## 13. 驗收實務 / Verification practice

### 13.1 一檔一 process / One effect per process

Dawn 的 node 綁定在**同一 process 內交錯多個效果的 submit + mapAsync** 會不穩定
（無訊息失敗或原生崩潰）。批次驗收請用外部迴圈，每個效果一個 Node process；
`test/port_check.mjs` 本身也只對單一效果做一次 mapAsync。

```bash
# bash：逐檔一 process
for f in effects/RAVU/*.wgsl; do node test/port_check.mjs "$f" --quiet || echo "FAIL $f"; done
```

```powershell
# PowerShell：逐檔一 process
Get-ChildItem effects/RAVU -Filter *.wgsl | ForEach-Object {
  $out = (& node test/port_check.mjs $_.FullName --quiet 2>&1 | Out-String)
  if ($out -notmatch 'ALL PASS') { "FAIL $($_.Name)" }
}
```

`test/test_runtime.js` 的 Test E 也是「一檔一子行程」的設計，可以放心批次跑：

```bash
node test/effect_sweep.mjs effects        # 動態批次掃描（每檔一子行程，含 NaN/非零統計）
node test/port_check.mjs <effect.wgsl>    # 單檔：編譯 + 全 pass + 非全黑 + 無 NaN
npm test                                  # Test A–E（Test E 掃整個 effects/ 樹，會解析 SOURCE）
```

### 13.2 量測噪音：只用「同輪前後對照」/ Same-round A/B only

GPU 測試與 benchmark 互相競爭會讓數字漂移；跨輪比較（「昨天 1.2 ms，今天 0.9 ms」）沒有意義。
要主張效能改變，請在同一輪、同一 process 排程下做前後對照，並固定 `test/bench_runtime.mjs`
的輸入與幀數。GPU contention makes cross-round numbers meaningless — compare before/after
**within the same round**.

### 13.3 靜態與診斷輔助 / Static & diagnostic helpers

| 工具 Tool | 用途 |
|---|---|
| `tools/port/check_effect.mjs` | 靜態 lint（指令、區塊、常見筆誤） |
| `tools/port/wgsl_diag.mjs` | 編譯診斷（只驗 WGSL，不 dispatch） |
| `tools/port/weight-integrity.mjs` | 權重數字多重集比對（內嵌常數類效果） |
| `tools/port/analytic_check.mjs` | CPU 語意 vs GPU 實跑對照 |
| `tools/_storage_probe.mjs` | storage 格式能力探測（§12） |

需要宿主非 LUT 資源（例如 OpenNR 的 storage buffer）而無法由通用套件執行的效果，才列進
`test/effects-suite.json` 的 `skip` 表並附原因；`//! SOURCE` 類**不需要**列入（§9.3）。
另外別忘了驗收端要索取 §10 的裝置上限，否則會出現假的紅燈。

---

## 14. workgroup 記憶體超限的三種處理 / Workgroup storage over the limit

`var<workgroup>`（HLSL `groupshared`）很容易撞上預設 16,384 B。三條正解，按「先問數學、
再問宿主」的順序選：Three legitimate fixes, in order.

| 作法 | 何時適用 | repo 實例 |
|---|---|---|
| (1) 不落地、改為**重算** | 需要的資料是「已載入資料的子視窗」 | `effects/NIS/NIS.wgsl`：`shEdgeMap`（34×34 `float4` = 18,496 B）＋亮度磚（38×38 f32 = 5,776 B）＝ 27,344 B 超限 → 丟掉 edge map，**每個 thread 由已載入的 6×6 support 重算**它需要的 2×2 邊鄰域；相同輸入、相同算式 → 位元相同，用量降為 8,848 B（見 [`docs/PORTING_STATUS.md`](PORTING_STATUS.md) §5） |
| (2) **縮小 tile** | 演算法允許較小視窗或分批 | 需自行論證邊界正確性；RAVU 不適用（視窗由 LUT 模板固定） |
| (3) **請宿主提高上限** | 演算法本質需要這塊記憶體 | `RAVU_R3_RGB`（19,264 B）／`RAVU_R4_RGB`（23,424 B）→ §10 的 `maxComputeWorkgroupStorageSize`（D3D 宿主本來就開 32 KB） |

**「拆成 f32 陣列」不一定能解決** / splitting into `array<f32, …>` is not a guaranteed fix:

* WGSL 的 `array<vec3<f32>, N>` 每元素 stride 是 **16 B**（vec3 對齊 4×f32），而 HLSL 的
  `groupshared float3 v[N]` 在 D3D 只要 12 B。拆成 f32 陣列確實省 25%，但省下來不一定夠 ——
  以 RAVU 的兩塊 RGB tile + 兩塊 luma tile 實算（per-thread 需 4 塊，兩個 pass 共用同一份宣告）：
  * `RAVU_R3_RGB`：481 × (12+4) × 2 組 = **15,392 B ✓** 可壓進 16,384 B；
  * `RAVU_R4_RGB`：585 × (12+4) × 2 組 = **18,720 B ✗** 仍然超限。
* 決策順序：先算 (1) 能不能不落地 → 不行再算 (2) 最小可行用量 → 都不行才走 (3)，
  並在檔頭注記這是**宿主契約**（§15），讓下一個使用者知道要 `requestDevice` 帶上限。

---

## 15. 移植注記慣例 / Porting notes convention

每個 `.wgsl` 檔頭的 `Porting notes:` 是**該效果語意的權威來源**；`docs/PORTING_STATUS.md` §5
只記家族級的共同限制。至少寫這五項：Every ported effect should document, in its header:

| 項目 | 內容 | 可抄的範例 |
|---|---|---|
| 來源 | upstream URL／作者／授權，與 Magpie 路徑 | `effects/Anime4K/Anime4K_Upscale_S.wgsl` 檔頭 |
| 執行模型對映 | PS/CS → compute 的座標語意（`blockStart` / `threadId` / `pos`） | `effects/NIS/NIS.wgsl` note 1 |
| 語意偏離與**理由** | 每條偏離都要能回答「為什麼」 | `effects/SMAA/SMAA_Low.wgsl` note 8（`R8G8_UNORM` → `rgba8unorm`）；`effects/NIS/NIS.wgsl` note 2（以重算取代共享記憶體，含「裝置未索取上限」的前提） |
| 框架待補／採用的慣例 | 需要的能力，或不明顯的等價轉換 | `effects/RAVU/RAVU_Zoom_AR_R2_RGB.wgsl`（`mat4x3` 轉置慣例、`outerProduct` / `matrixCompMult` 進 COMMON） |
| 實測數字 | 超限／用量／tile 大小等**可驗算**的數字 | `effects/NIS/NIS.wgsl` note 2 的 27,344 B / 8,848 B；本指南 §14 的 RAVU 算式 |

* 寫「**為什麼**」而不只是「改了什麼」：下一個移植者需要的是判斷依據，不是 diff。
* 自動轉換產生的檔案，請把轉換器的缺口清單放進檔頭或轉換器本身。實例：
  `effects/RAVU/_gen_ravu.mjs` 檔頭列出 **9** 條模板缺口；實作另外還處理了
  `imageStoreOverride` 的 `float4` 寫入、`mat4x3`/`outerProduct`/`matrixCompMult`、
  `ivec2` 的 int/uint 顯式化、CRLF 正規化，以及 `__MIX__` 屏蔽 —— 這些寫在實作與
  產物檔頭的 `[RAVU]` 註記裡。Generated files should carry their converter's gap list.
* 驗收指令也要能一行跑完（§13.1），並記得 LUT 系效果會自動解析 `//! SOURCE`（§9.3）。
