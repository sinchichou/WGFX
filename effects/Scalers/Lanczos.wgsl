// Lanczos6 — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/Lanczos.hlsl
//   （Magpie 的 HLSL 移植自 libretro/common-shaders 的 windowed/shaders/lanczos6.cg）
// 原授權 / 作者：Magpie 效果庫為 GPL-3.0（作者 Blinue）；
//   Lanczos6 核函數出自 libretro/common-shaders（Team XBMC / Stefanos A.，
//   該檔為 GNU GPL v2 或更新版本）。
// 本移植僅改寫 pass 結構與取樣語意；權重公式與所有常數逐字保留。
//
// Porting notes:
//   - Magpie 的 `//!STYLE PS` pass 傳入的 `pos` 是**正規化 UV**（輸出 texel 中心
//     (x+0.5)/outputSize）。WGFX 端：`pos = (gid.xy + 0.5) * scene.outputPt`；
//     原碼 `pos *= GetInputSize()`（正規化 UV → input texel 座標）逐字保留。
//   - `INPUT.GatherRed/GatherGreen/GatherBlue(sam, tpos)` →
//     `textureGather(0|1|2, INPUT, sam, tpos)`。WGSL 對 `texture_2d<f32>` 的
//     textureGather 只有「component」多載載；D3D 與 WGSL 的 gather 分量順序一致
//     （相對於 base texel = floor(uv*size-0.5)：
//     x=(0,1) y=(1,1) z=(1,0) w=(0,0)），故原碼的 "w z / x y" 填法逐字保留。
//   - HLSL 的矩陣建構子（以向量建構）填的是**列**，WGSL 的建構子填的是**欄**。
//     已對照 libretro lanczos6.cg 驗證列優先解讀成立：
//     `mul(linetaps1, float3x3(src[0][i], src[2][i], src[4][i]))` 必須是
//     「3 個 tap 的加權和」（列優先），欄優先會變成 `dot(權重, 顏色)` 的無意義形式。
//     因此 HLSL `mul(v, M)`（列向量 × 矩陣）在 WGSL 寫成
//     `v * transpose(mat3x3<f32>(...))`，而 HLSL `mul(M, v)` 寫成
//     `transpose(mat3x3<f32>(...)) * v`（已用 GPU 探針實測確認兩式對應）。
//     建構子引數順序與數字皆不變。
//   - HLSL `rcp(s*s)` → `1.0 / (s*s)`；`FIX(c) = max(abs(c), 1e-5)` 以
//     `max(abs(c), vec3<f32>(1e-5))` 廣播實作（避免 0/0 → NaN）。
//   - Magpie 的 `SamplerState sam`（POINT / CLAMP）對應
//     `//! FILTER POINT` + `//! ADDRESS CLAMP`；gather 只取相鄰 2×2 紋素，
//     過濾模式不影響結果。
//   - Magpie 的 OUTPUT 未宣告尺寸（預設 = INPUT）；WGFX 明確宣告
//     `//! WIDTH INPUT_WIDTH` / `//! HEIGHT INPUT_HEIGHT`。

//! VERSION 4
//! SORT_NAME Lanczos

//! PARAMETER ARStrength
//! TYPE float
//! LABEL "Anti-ringing Strength"
//! MIN 0
//! MAX 1
//! DEFAULT 0.5
//! STEP 0.01
float ARStrength;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! SAMPLER
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! DESC "Lanczos6 resampling + anti-ringing"
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

const PI: f32 = 3.14159265359;

fn weight3(x: f32) -> vec3<f32> {
    const rcpRadius: f32 = 1.0 / 3.0;
    let s = max(abs(2.0 * PI * vec3<f32>(x - 1.5, x - 0.5, x + 0.5)), vec3<f32>(1e-5));
    // Lanczos3. Note: we normalize outside this function, so no point in multiplying by radius.
    return /*radius **/ sin(s) * sin(s * rcpRadius) * (1.0 / (s * s));
}

fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outSize = scene.outputSize;
    if (gid.x >= outSize.x || gid.y >= outSize.y) {
        return;
    }
    // Magpie PS 的 pos = 正規化 UV（輸出 texel 中心）
    let pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.outputPt;
    let inputPt = scene.inputPt;

    var p = pos * vec2<f32>(scene.inputSize);

    let f = fract(p + 0.5);
    var linetaps1 = weight3(0.5 - f.x * 0.5);
    var linetaps2 = weight3(1.0 - f.x * 0.5);
    var columntaps1 = weight3(0.5 - f.y * 0.5);
    var columntaps2 = weight3(1.0 - f.y * 0.5);

    // make sure all taps added together is exactly 1.0, otherwise some
    // (very small) distortion can occur
    let suml = dot(linetaps1, vec3<f32>(1.0)) + dot(linetaps2, vec3<f32>(1.0));
    let sumc = dot(columntaps1, vec3<f32>(1.0)) + dot(columntaps2, vec3<f32>(1.0));
    linetaps1 = linetaps1 / suml;
    linetaps2 = linetaps2 / suml;
    columntaps1 = columntaps1 / sumc;
    columntaps2 = columntaps2 / sumc;

    p -= f + 1.5;

    var src: array<array<vec3<f32>, 6>, 6>;

    for (var i = 0u; i <= 4u; i += 2u) {
        for (var j = 0u; j <= 4u; j += 2u) {
            let tpos = (p + vec2<f32>(f32(i), f32(j))) * inputPt;
            let sr = textureGather(0, INPUT, sam, tpos);
            let sg = textureGather(1, INPUT, sam, tpos);
            let sb = textureGather(2, INPUT, sam, tpos);

            // w z
            // x y
            src[i][j] = vec3<f32>(sr.w, sg.w, sb.w);
            src[i][j + 1u] = vec3<f32>(sr.x, sg.x, sb.x);
            src[i + 1u][j] = vec3<f32>(sr.z, sg.z, sb.z);
            src[i + 1u][j + 1u] = vec3<f32>(sr.y, sg.y, sb.y);
        }
    }

    // final sum and weight normalization
    var color = vec3<f32>(0.0);
    for (var i = 0u; i <= 4u; i += 2u) {
        color += (linetaps1 * transpose(mat3x3<f32>(src[0][i], src[2][i], src[4][i]))
                + linetaps2 * transpose(mat3x3<f32>(src[1][i], src[3][i], src[5][i]))) * columntaps1[i / 2u]
               + (linetaps1 * transpose(mat3x3<f32>(src[0][i + 1u], src[2][i + 1u], src[4][i + 1u]))
                + linetaps2 * transpose(mat3x3<f32>(src[1][i + 1u], src[3][i + 1u], src[5][i + 1u]))) * columntaps2[i / 2u];
    }

    // 抗振铃
    let min_sample = min(min(src[2][2], src[3][2]), min(src[2][3], src[3][3]));
    let max_sample = max(max(src[2][2], src[3][2]), max(src[2][3], src[3][3]));
    color = mix(color, clamp(color, min_sample, max_sample), uniforms.ARStrength);

    textureStore(OUTPUT, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}

//! END
