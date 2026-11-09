// ImageAdjustment — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/ImageAdjustment.hlsl
//   （Magpie 的 HLSL 移植自 libretro/slang-shaders 的 misc/image-adjustment.slang）
// 原授權 / 作者：Magpie 效果庫為 GPL-3.0（作者 Blinue）；
//   image-adjustment.slang 出自 libretro/slang-shaders（授權依該來源庫）。
// 本移植僅改寫 pass 結構與取樣語意；色彩轉換公式與所有常數逐字保留。
//
// Porting notes:
//   - Magpie 的 `//!STYLE PS` pass 傳入的 `pos` 是**正規化 UV**（輸出 texel 中心
//     (x+0.5)/outputSize）。WGFX 端：`pos = (gid.xy + 0.5) * scene.outputPt`。
//   - `INPUT.SampleLevel(sam, pos, 0)` → `textureSampleLevel(INPUT, sam, pos, 0.0)`
//     （compute stage 不可用隱含 LOD 的 textureSample）。
//   - HLSL `lerp` → WGSL `mix`；`saturate` / `frac` / `abs` 兩邊同名同義。
//     WGSL 的 `pow` / `step` 要求兩參數同型別（HLSL 允許純量廣播），
//     故指數／門檻以 vec3 展開，數值不變。
//   - HLSL 的 swizzle（`c.bg`、`c.xxx`、`K.wz`、`p.xyw`、`p.yzx`）在 WGSL 皆合法，
//     逐字照抄。`HSVtoRGB` 內 HLSL 以 float4 計算後再截回 float3（多出的 .w
//     分量會被丟棄），WGSL 直接以 vec3 計算前三個分量，數值完全相同。
//   - HLSL 的 `float4 p = c.y < c.z ? float4(c.bg, K.wz) : float4(c.gb, K.xy);`
//     為三元運算子，WGSL 以 `select()` 改寫（`select(falseValue, trueValue, cond)`），
//     條件方向與原碼一致（cond 為 `c.y < c.z` 時取 `float4(c.bg, K.wz)`）。
//   - Magpie 的 `SamplerState sam` 為 POINT / CLAMP，對應
//     `//! FILTER POINT` + `//! ADDRESS CLAMP`。
//   - Magpie 的 OUTPUT 尺寸 = INPUT；WGFX 明確宣告
//     `//! WIDTH INPUT_WIDTH` / `//! HEIGHT INPUT_HEIGHT`。
//   - 註：原碼 `color = saturate(color / (1 - blackLevel));` 在 blackLevel = 1 時
//     除以 0（Inf/NaN）。此為原演算法行為，未加保護（預設值 0 不會觸發）。

//! VERSION 4
//! SORT_NAME ImageAdjustment

//! PARAMETER targetGamma
//! TYPE float
//! LABEL "Target Gamma"
//! MIN 0.01
//! MAX 5
//! DEFAULT 2.2
//! STEP 0.01
float targetGamma;

//! PARAMETER monitorGamma
//! TYPE float
//! LABEL "Monitor Gamma"
//! MIN 0.01
//! MAX 5
//! DEFAULT 2.2
//! STEP 0.01
float monitorGamma;

//! PARAMETER saturation
//! TYPE float
//! LABEL "Saturation"
//! MIN 0
//! MAX 5
//! DEFAULT 1
//! STEP 0.01
float saturation;

//! PARAMETER luminance
//! TYPE float
//! LABEL "Luminance"
//! MIN 0
//! MAX 2
//! DEFAULT 1
//! STEP 0.01
float luminance;

//! PARAMETER contrast
//! TYPE float
//! LABEL "Contrast"
//! MIN 0
//! MAX 10
//! DEFAULT 1
//! STEP 0.1
float contrast;

//! PARAMETER brightBoost
//! TYPE float
//! LABEL "Brightness Boost"
//! MIN -1
//! MAX 1
//! DEFAULT 0
//! STEP 0.01
float brightBoost;

//! PARAMETER blackLevel
//! TYPE float
//! LABEL "Black Level"
//! MIN -1
//! MAX 1
//! DEFAULT 0
//! STEP 0.01
float blackLevel;

//! PARAMETER r
//! TYPE float
//! LABEL "Red Channel"
//! MIN 0
//! MAX 2
//! DEFAULT 1
//! STEP 0.01
float r;

//! PARAMETER g
//! TYPE float
//! LABEL "Green Channel"
//! MIN 0
//! MAX 2
//! DEFAULT 1
//! STEP 0.01
float g;

//! PARAMETER b
//! TYPE float
//! LABEL "Blue Channel"
//! MIN 0
//! MAX 2
//! DEFAULT 1
//! STEP 0.01
float b;

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
//! DESC "HSV saturation/luminance + contrast/brightness/black level/gamma + channel gain"
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

fn RGBtoHSV(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = select(vec4<f32>(c.gb, K.xy), vec4<f32>(c.bg, K.wz), c.y < c.z);
    let q = select(vec4<f32>(c.x, p.yzx), vec4<f32>(p.xyw, c.x), c.x < p.x);

    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn HSVtoRGB(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    // HLSL: abs(frac(c.xxx + K.xyz) * 6.0 - K.www) 為 float4，再與 float3 的
    // K.xxx 相減後整體截回 float3；此處直接以 vec3 運算，只保留會被回傳的
    // 前三個分量（每個分量的減項皆為 K.w）。
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.w);
    return c.z * mix(K.xyz, saturate(p - K.xyz), c.y);
}

fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outSize = scene.outputSize;
    if (gid.x >= outSize.x || gid.y >= outSize.y) {
        return;
    }
    // Magpie PS 的 pos = 正規化 UV（輸出 texel 中心）
    let pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.outputPt;

    var color = textureSampleLevel(INPUT, sam, pos, 0.0).rgb;

    // saturation and luminance
    color = saturate(HSVtoRGB(RGBtoHSV(color) * vec3<f32>(1.0, uniforms.saturation, uniforms.luminance)));

    // contrast and brightness
    color = saturate((color - 0.5) * uniforms.contrast + 0.5 + uniforms.brightBoost);

    // black level
    color -= uniforms.blackLevel;
    color = saturate(color / (1.0 - uniforms.blackLevel));

    // gamma correction
    // WGSL 的 pow 兩參數須同型別（HLSL 允許純量指數），故把指數展開成 vec3。
    color = pow(color, vec3<f32>(uniforms.targetGamma / uniforms.monitorGamma));

    color *= vec3<f32>(uniforms.r, uniforms.g, uniforms.b);

    textureStore(OUTPUT, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}

//! END
