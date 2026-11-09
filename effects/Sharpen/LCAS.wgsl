// LCAS — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/Sharpen/LCAS.hlsl
// 原授權 / 作者：Magpie 效果庫為 GPL-3.0（作者 Blinue）；
//   LCAS = linear interpolation + light version of CAS（源自 AMD CAS 的公開實作）。
// 本移植僅改寫 pass 結構與取樣語意；濾波公式與所有常數逐字保留。
//
// Porting notes:
//   - Magpie 的這支效果是 `//!STYLE CS`（`void Pass1(uint2 blockStart, uint3 threadId)`,
//     BLOCK_SIZE 16 / NUM_THREADS 64）。WGFX 以 compute 實作：
//     `gxy = Rmp8x8(local_id.x) + workgroup_id.xy * 16`（Magpie 的 `blockStart` 是
//     workgroup 區塊的起始像素 = workgroup_id × BLOCK_SIZE），
//     並以 `//! BLOCK_SIZE 16` 決定 dispatch 網格（ceil(out/16)² 個 workgroup）。
//   - 原碼每個執行緒負責 2×2 quad 的**四個**像素（相隔 8 像素）：
//     (x,y)、(x+8,y)、(x+8,y+8)、(x,y+8)。Magpie 寫入 render target 時
//     越界寫入會被丟棄；WGFX 端以顯式邊界檢查跳過越界寫入（語意相同）。
//   - `INPUT.SampleLevel(sam, uv, 0)` → `textureSampleLevel(INPUT, sam, uv, 0.0)`
//     （compute stage 不可用隱含 LOD 的 textureSample）。
//     Magpie 的 `SamplerState sam` 為 LINEAR / CLAMP，對應
//     `//! FILTER LINEAR` + `//! ADDRESS CLAMP`。
//   - HLSL `lerp` → WGSL `mix`；HLSL 的純量與向量運算（HLSL 允許廣播）在 WGSL
//     同樣合法（`1.0 - mxRGB`、`e * 5.0`、`/ 4.0` 等）。
//   - Magpie 的 OUTPUT 未宣告尺寸（預設 = INPUT）；WGFX 明確宣告
//     `//! WIDTH INPUT_WIDTH` / `//! HEIGHT INPUT_HEIGHT`。
//   - 註：原碼 `sqrt(min(mnRGB, 1.0 - mxRGB) / mxRGB)` 在 mxRGB = 0（純黑）時
//     為 0/0 → NaN，且 mxRGB > 1（HDR）時 sqrt(負數) → NaN；此為原演算法
//     行為，未加保護（與 HLSL 相同）。

//! VERSION 4
//! SORT_NAME LCAS

//! PARAMETER sharpness
//! TYPE float
//! LABEL "Sharpness"
//! MIN 0
//! MAX 1
//! DEFAULT 0.5
//! STEP 0.01
float sharpness;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! SAMPLER
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! DESC "LCAS: linear interpolation + light CAS sharpen"
//! IN INPUT
//! OUT OUTPUT
//! BLOCK_SIZE 16
//! NUM_THREADS 64

fn Rmp8x8(a: u32) -> vec2<u32> {
    return vec2<u32>(a / 8u, a % 8u);
}

fn LCAS(ip: vec2<u32>, peak: f32) -> vec3<f32> {
    let pos = (vec2<f32>(ip) + vec2<f32>(0.5)) * scene.outputPt;
    let inputPt = scene.inputPt;

    // fetch a 3x3 neighborhood around the pixel 'e',
    //	a b c
    //	d(e)f
    //	g h i
    let a = textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x, -inputPt.y), 0.0).rgb;
    let b = textureSampleLevel(INPUT, sam, pos + vec2<f32>(0.0, -inputPt.y), 0.0).rgb;
    let c = textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x, -inputPt.y), 0.0).rgb;
    let d = textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x, 0.0), 0.0).rgb;
    let f = textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x, 0.0), 0.0).rgb;
    let g = textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x, inputPt.y), 0.0).rgb;
    let h = textureSampleLevel(INPUT, sam, pos + vec2<f32>(0.0, inputPt.y), 0.0).rgb;
    let i = textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x, inputPt.y), 0.0).rgb;

    let x = a + c + g + i;
    let y = b + d + f + h;

    var e = textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x * 0.25, inputPt.y * 0.5), 0.0).rgb;
    e += textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x * 0.25, -inputPt.y * 0.5), 0.0).rgb;
    e += textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x * 0.5, -inputPt.y * 0.25), 0.0).rgb;
    e += textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x * 0.5, inputPt.y * 0.25), 0.0).rgb;
    e /= 4.0;

    // Soft min and max.
    //  a b c
    //  d e f
    //  g h i
    let mnRGB = min(min(min(min(d, e), min(f, b)), h), min(min(a, i), min(c, g)));

    let mxRGB = max(max(max(max(d, e), max(f, b)), h), max(max(a, i), max(c, g)));

    // Shaping amount of sharpening.
    let wRGB = sqrt(min(mnRGB, 1.0 - mxRGB) / mxRGB) * peak;

    // Filter shape.
    //  w w w
    //  w 1 w
    //  w w w
    let color = ((x + y) * wRGB + (e * 5.0 - (x + y * 2.0 + e * 4.0) / 4.0)) / (1.0 + 8.0 * wRGB);
    return (color + clamp(color, mnRGB, mxRGB) * 4.0) / 5.0;
}

fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Magpie 的 blockStart = workgroup 區塊的起始像素 = workgroup_id * BLOCK_SIZE(16)
    var gxy = workgroup_id.xy * 16u + Rmp8x8(local_id.x);

    let outputSize = scene.outputSize;
    if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
        return;
    }

    let peak = mix(0.0, -0.1111111111111111, uniforms.sharpness);

    if (gxy.x < outputSize.x && gxy.y < outputSize.y) {
        textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(LCAS(gxy, peak), 1.0));
    }

    let gxy1 = vec2<u32>(gxy.x + 8u, gxy.y);
    if (gxy1.x < outputSize.x && gxy1.y < outputSize.y) {
        textureStore(OUTPUT, vec2<i32>(gxy1), vec4<f32>(LCAS(gxy1, peak), 1.0));
    }

    let gxy2 = vec2<u32>(gxy.x + 8u, gxy.y + 8u);
    if (gxy2.x < outputSize.x && gxy2.y < outputSize.y) {
        textureStore(OUTPUT, vec2<i32>(gxy2), vec4<f32>(LCAS(gxy2, peak), 1.0));
    }

    let gxy3 = vec2<u32>(gxy.x, gxy.y + 8u);
    if (gxy3.x < outputSize.x && gxy3.y < outputSize.y) {
        textureStore(OUTPUT, vec2<i32>(gxy3), vec4<f32>(LCAS(gxy3, peak), 1.0));
    }
}

//! END
