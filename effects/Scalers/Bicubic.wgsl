// Bicubic — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/Bicubic.hlsl
//   （Magpie 的 HLSL 移植自 ActualMandM/ceu_graphic_packs 的
//    Filters/Bicubic/output.glsl）
// 原授權 / 作者：Magpie 效果庫為 GPL-3.0（作者 Blinue）；
//   Bicubic 核函數出自 ActualMandM/ceu_graphic_packs（授權依該來源庫）。
// 本移植僅改寫 pass 結構與取樣語意；權重公式與所有常數逐字保留。
//
// Porting notes:
//   - Magpie 的 `//!STYLE PS` pass 傳入的 `pos` 是**正規化 UV**（輸出 texel 中心
//     (x+0.5)/outputSize），不是 texel 座標。WGFX 端以 compute 實作：
//     `pos = (gid.xy + 0.5) * scene.outputPt`；原碼 `pos *= inputSize`
//     （正規化 UV → input texel 座標）逐字保留，因此輸出尺寸 ≠ 輸入尺寸時
//     仍維持 Magpie 的重取樣行為。
//   - `INPUT.Load(int3(c, 0))` → `textureLoad(INPUT, vec2<i32>(c), 0)`；
//     `INPUT.SampleLevel(sam, uv, 0)` → `textureSampleLevel(INPUT, sam, uv, 0.0)`
//     （compute stage 不可用隱含 LOD 的 textureSample）。
//   - HLSL `max(vec2, 0.5)` / `min(vec2, vec2)` 的純量廣播，WGSL 需顯式寫成
//     `vec2<f32>(0.5)`；HLSL `int2(float2)` 為向零截斷，WGSL `vec2<i32>(f32vec2)`
//     同樣向零截斷，語意一致。
//   - HLSL `const float B = paramB;` 讀的是 uniform（執行期值），WGSL 只能
//     用 `let B = uniforms.paramB;`（`const` 要求 const-expression）。
//   - Magpie 的 `SamplerState sam`（LINEAR / CLAMP）對應
//     `//! FILTER LINEAR` + `//! ADDRESS CLAMP`。
//   - Magpie 的 OUTPUT 未宣告尺寸（預設 = INPUT）；WGFX 明確宣告
//     `//! WIDTH INPUT_WIDTH` / `//! HEIGHT INPUT_HEIGHT`（宿主提供的 OUTPUT 尺寸優先）。

//! VERSION 4
//! SORT_NAME Bicubic

//! PARAMETER paramB
//! TYPE float
//! LABEL "B"
//! MIN 0
//! MAX 1
//! DEFAULT 0.33
//! STEP 0.01
float paramB;

//! PARAMETER paramC
//! TYPE float
//! LABEL "C"
//! MIN 0
//! MAX 1
//! DEFAULT 0.33
//! STEP 0.01
float paramC;

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
//! DESC "Bicubic (Catmull-Rom family) resampling"
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

fn weight(x: f32) -> f32 {
    let B = uniforms.paramB;
    let C = uniforms.paramC;

    let ax = abs(x);

    if (ax < 1.0) {
        return (x * x * ((12.0 - 9.0 * B - 6.0 * C) * ax + (-18.0 + 12.0 * B + 6.0 * C)) + (6.0 - 2.0 * B)) / 6.0;
    } else if (ax >= 1.0 && ax < 2.0) {
        return (x * x * ((-B - 6.0 * C) * ax + (6.0 * B + 30.0 * C)) + (-12.0 * B - 48.0 * C) * ax + (8.0 * B + 24.0 * C)) / 6.0;
    } else {
        return 0.0;
    }
}

fn weight4(x: f32) -> vec4<f32> {
    return vec4<f32>(
        weight(x - 2.0),
        weight(x - 1.0),
        weight(x),
        weight(x + 1.0)
    );
}

fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outSize = scene.outputSize;
    if (gid.x >= outSize.x || gid.y >= outSize.y) {
        return;
    }
    // Magpie PS 的 pos = 正規化 UV（輸出 texel 中心）
    let pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.outputPt;
    let inputPt = scene.inputPt;
    let inputSize = vec2<f32>(scene.inputSize);

    var p = pos * inputSize;
    let pos1 = floor(p - 0.5) + 0.5;
    let f = p - pos1;

    var rowtaps = weight4(1.0 - f.x);
    var coltaps = weight4(1.0 - f.y);

    // make sure all taps added together is exactly 1.0, otherwise some (very small) distortion can occur
    rowtaps = rowtaps / (rowtaps.r + rowtaps.g + rowtaps.b + rowtaps.a);
    coltaps = coltaps / (coltaps.r + coltaps.g + coltaps.b + coltaps.a);

    let uv1 = pos1 * inputPt;
    let uv0 = uv1 - inputPt;
    let uv2 = uv1 + inputPt;
    let uv3 = uv2 + inputPt;

    let u_weight_sum = rowtaps.y + rowtaps.z;
    let u_middle_offset = rowtaps.z * inputPt.x / u_weight_sum;
    let u_middle = uv1.x + u_middle_offset;

    let v_weight_sum = coltaps.y + coltaps.z;
    let v_middle_offset = coltaps.z * inputPt.y / v_weight_sum;
    let v_middle = uv1.y + v_middle_offset;

    let coord_top_left = vec2<i32>(max(uv0 * inputSize, vec2<f32>(0.5)));
    let coord_bottom_right = vec2<i32>(min(uv3 * inputSize, inputSize - vec2<f32>(0.5)));

    var top = textureLoad(INPUT, coord_top_left, 0).rgb * rowtaps.x;
    top += textureSampleLevel(INPUT, sam, vec2<f32>(u_middle, uv0.y), 0.0).rgb * u_weight_sum;
    top += textureLoad(INPUT, vec2<i32>(coord_bottom_right.x, coord_top_left.y), 0).rgb * rowtaps.w;
    var total = top * coltaps.x;

    var middle = textureSampleLevel(INPUT, sam, vec2<f32>(uv0.x, v_middle), 0.0).rgb * rowtaps.x;
    middle += textureSampleLevel(INPUT, sam, vec2<f32>(u_middle, v_middle), 0.0).rgb * u_weight_sum;
    middle += textureSampleLevel(INPUT, sam, vec2<f32>(uv3.x, v_middle), 0.0).rgb * rowtaps.w;
    total += middle * v_weight_sum;

    var bottom = textureLoad(INPUT, vec2<i32>(coord_top_left.x, coord_bottom_right.y), 0).rgb * rowtaps.x;
    bottom += textureSampleLevel(INPUT, sam, vec2<f32>(u_middle, uv3.y), 0.0).rgb * u_weight_sum;
    bottom += textureLoad(INPUT, coord_bottom_right, 0).rgb * rowtaps.w;
    total += bottom * coltaps.w;

    textureStore(OUTPUT, vec2<i32>(gid.xy), vec4<f32>(total, 1.0));
}

//! END
