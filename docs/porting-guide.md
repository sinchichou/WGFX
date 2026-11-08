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
**向零截斷** mantissa，而非 IEEE 的 round-to-nearest-even。實測（`test/_probe_f16.mjs`）：

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

---

## 8. 產出位置

| 類型 | 位置 |
|---|---|
| 移植後的效果 | `effects/<Family>/<Name>.wgsl` |
| 既有示範效果 | `examples/` |
| 專屬驗收測試 | `test/<name>.mjs` |
| 移植筆記 | 效果檔頭的 `Porting notes:` |

`effects/` 下的目錄依來源家族分：`OpenNR/`、`Anime4K/`、`Sharpen/`、`RAVU/`…
