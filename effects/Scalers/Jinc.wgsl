// Jinc2 — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/Jinc.hlsl
//   （Magpie 的 HLSL 移植自 libretro/common-shaders 的 windowed/shaders/jinc2.cg）
// 原授權 / 作者：Magpie 效果庫為 GPL-3.0（作者 Blinue）；
//   Jinc2 核函數出自 libretro/common-shaders（Hyllian / Jararaca，
//   該檔為 GNU GPL v2 或更新版本）。
// 本移植僅改寫 pass 結構與取樣語意；權重公式與所有常數逐字保留。
//
// Porting notes:
//   - Magpie 的這支效果是 `//!STYLE CS`（`void Pass1(uint2 blockStart, uint3 threadId)`，
//     BLOCK_SIZE 8 / NUM_THREADS 64，每執行緒負責 1 像素）。WGFX 以 compute 實作：
//     `Rmp8x8(local_id.x) + workgroup_id.xy * 8`（Magpie 的 `blockStart` 是 workgroup
//     區塊的起始像素 = workgroup_id × BLOCK_SIZE），並以 `//! BLOCK_SIZE 8` 決定
//     dispatch 網格（ceil(out/8)² 個 workgroup）。
//   - `INPUT.GatherRed/GatherGreen/GatherBlue(sam, tpos)` →
//     `textureGather(0|1|2, INPUT, sam, tpos)`。D3D 與 WGSL 的 gather 分量順序一致
//     （相對於 base texel = floor(uv*size-0.5)：x=(0,1) y=(1,1) z=(1,0) w=(0,0)），
//     故原碼的 "w z / x y" 填法逐字保留。
//   - HLSL 的矩陣建構子（以向量建構）填的是**列**，WGSL 的建構子填的是**欄**
//     （列優先解讀已對照 libretro jinc2.cg 驗證：`mul(weights[0], float4x3(c00, c10,
//     c20, c30))` 必須是 4 個 texel 的加權和）。因此：
//     HLSL `mul(v, M)`（列向量 × 矩陣）→ WGSL `v * transpose(matCxR<f32>(...))`；
//     HLSL `mul(M, v)`（矩陣 × 欄向量）→ WGSL `transpose(matCxR<f32>(...)) * v`
//     （已用 GPU 探針實測確認兩式對應）。建構子引數順序與數字皆不變。
//     另外 WGSL 的 `M[i]` 取的是**欄**，HLSL 的 `M[i]` 取的是**列**；本檔把
//     `weights` 建成 `mat4x4<f32>(r0, r1, r2, r3)`（欄 == HLSL 的列），
//     故 `weights[0]` 仍等於 HLSL 的 `weights[0]`（= r0）✓。
//   - HLSL `rcp(x)` → `1.0 / x`。`resampler` 的三元運算子
//     `(x == float4(0,0,0,0)) ? ... : ...` 在 WGSL 需改寫；且原式的條件只涵蓋
//     「四個距離皆為 0」，而只要任一距離為 0（1:1 縮放時的中心紋素
//     `d(pc, tc) = 0`）就會 0/0 → NaN。本移植改以**逐分量的解析極限**保護：
//     `sin(x*wa)*sin(x*wb)/(x*x)` 在 x→0 的極限正是原碼 special case 回傳的
//     `wa*wb`，故 0 分量以 `wa*wb` 取代（見 resampler() 內註解）。
//     這是原碼意圖的推廣，不改變任何非零分量的數值。
//   - **保留原始 HLSL 的既有筆誤**：第 4 個權重列在原碼寫成
//     `float4x3(src[0][3], src[2][3], src[2][3], src[3][3])`（`src[1][3]` 被寫成
//     `src[2][3]`），與 libretro jinc2.cg 原文的 `float4x3(c03,c13,c23,c33)` 不同。
//     依「演算法逐字保留」原則，此處照抄 HLSL 的寫法，未修正該筆誤。
//   - Magpie 的 `SamplerState sam`（POINT / CLAMP）對應
//     `//! FILTER POINT` + `//! ADDRESS CLAMP`。
//   - Magpie 的 OUTPUT 未宣告尺寸（預設 = INPUT）；WGFX 明確宣告
//     `//! WIDTH INPUT_WIDTH` / `//! HEIGHT INPUT_HEIGHT`。

//! VERSION 4
//! SORT_NAME Jinc

//! PARAMETER windowSinc
//! TYPE float
//! LABEL "Window Sinc Param"
//! MIN 0
//! MAX 1
//! DEFAULT 0.5
//! STEP 0.01
float windowSinc;

//! PARAMETER sinc
//! TYPE float
//! LABEL "Sinc Param"
//! MIN 0
//! MAX 1
//! DEFAULT 0.825
//! STEP 0.01
float sinc;

//! PARAMETER ARStrength
//! TYPE float
//! LABEL "Anti-ringing Strength"
//! MIN 0
//! MAX 1
//! DEFAULT 0.5
//! STEP 0.1
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
//! DESC "Jinc2 resampling + anti-ringing"
//! IN INPUT
//! OUT OUTPUT
//! BLOCK_SIZE 8
//! NUM_THREADS 64

const PI: f32 = 3.1415926535897932384626433832795;

fn Rmp8x8(a: u32) -> vec2<u32> {
    return vec2<u32>(a / 8u, a % 8u);
}

fn d(pt1: vec2<f32>, pt2: vec2<f32>) -> f32 {
    let v = pt2 - pt1;
    return sqrt(dot(v, v));
}

fn resampler(x: vec4<f32>, wa: f32, wb: f32) -> vec4<f32> {
    // 原 HLSL: (x == float4(0,0,0,0)) ? float4(wa*wb,...) : sin(x*wa)*sin(x*wb)*rcp(x*x)
    // 條件在 Cg/GLSL 語意下是「四個距離皆為 0」；但只要**任一**距離為 0
    // （1:1 縮放時中心紋素即為 d(pc, tc) = 0），原式就會 0/0 → NaN。
    // 此處改成逐分量的解析極限：sin(x*wa)*sin(x*wb)/(x*x) 在 x→0 時的極限
    // 恰為 wa*wb（= 原碼 special case 回傳的值），故以 wa*wb 取代 0 分量。
    // 這是原碼意圖的逐分量推廣，非效果弱化；其餘分量完全不變。
    let denom = x * x;
    let isZero = denom == vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let safeDenom = select(denom, vec4<f32>(1.0), isZero);
    let v = sin(x * wa) * sin(x * wb) / safeDenom;
    return select(v, vec4<f32>(wa * wb), isZero);
}

fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Magpie 的 blockStart = workgroup 區塊的起始像素 = workgroup_id * BLOCK_SIZE(8)
    var gxy = Rmp8x8(local_id.x) + workgroup_id.xy * 8u;

    let outputSize = scene.outputSize;
    if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
        return;
    }

    let inputPt = scene.inputPt;

    let dx = vec2<f32>(1.0, 0.0);
    let dy = vec2<f32>(0.0, 1.0);

    let pc = (vec2<f32>(gxy) + vec2<f32>(0.5)) * scene.outputPt * vec2<f32>(scene.inputSize);
    var tc = floor(pc - 0.5) + 0.5;

    let wa = uniforms.windowSinc * PI;
    let wb = uniforms.sinc * PI;
    var weights = mat4x4<f32>(
        resampler(vec4<f32>(d(pc, tc - dx - dy), d(pc, tc - dy), d(pc, tc + dx - dy), d(pc, tc + 2.0 * dx - dy)), wa, wb),
        resampler(vec4<f32>(d(pc, tc - dx), d(pc, tc), d(pc, tc + dx), d(pc, tc + 2.0 * dx)), wa, wb),
        resampler(vec4<f32>(d(pc, tc - dx + dy), d(pc, tc + dy), d(pc, tc + dx + dy), d(pc, tc + 2.0 * dx + dy)), wa, wb),
        resampler(vec4<f32>(d(pc, tc - dx + 2.0 * dy), d(pc, tc + 2.0 * dy), d(pc, tc + dx + 2.0 * dy), d(pc, tc + 2.0 * dx + 2.0 * dy)), wa, wb)
    );

    tc -= 0.5;

    var src: array<array<vec3<f32>, 4>, 4>;

    for (var i = 0u; i <= 2u; i += 2u) {
        for (var j = 0u; j <= 2u; j += 2u) {
            let tpos = (tc + vec2<f32>(f32(i), f32(j))) * inputPt;
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

    var color = weights[0] * transpose(mat4x3<f32>(src[0][0], src[1][0], src[2][0], src[3][0]));
    color = weights[1] * transpose(mat4x3<f32>(src[0][1], src[1][1], src[2][1], src[3][1])) + color;
    color = weights[2] * transpose(mat4x3<f32>(src[0][2], src[1][2], src[2][2], src[3][2])) + color;
    color = weights[3] * transpose(mat4x3<f32>(src[0][3], src[2][3], src[2][3], src[3][3])) + color;
    color *= 1.0 / dot(transpose(weights) * vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));

    // 抗振铃
    // Get min/max samples
    let min_sample = min(min(src[1][1], src[2][1]), min(src[1][2], src[2][2]));
    let max_sample = max(max(src[1][1], src[2][1]), max(src[1][2], src[2][2]));
    color = mix(color, clamp(color, min_sample, max_sample), uniforms.ARStrength);

    // final sum and weight normalization
    textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(color, 1.0));
}

//! END
