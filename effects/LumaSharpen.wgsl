// LumaSharpen 1.5 — 移植自 effects_src/Sharpen/LumaSharpen.hlsl（Magpie v4）
// 上游：SweetFX LumaSharpen.fx（CeeJay.dk）— luma 域 unsharp mask，可限幅防光暈
//
// 移植偏差（記錄）：0.5/sharpClamp 加 max(sharpClamp, 1e-4) 防護——sharpClamp=0
// 時上游為 ±Inf/NaN（saturate 後趨近中性），防護後同為中性輸出；sharpClamp > 1e-4
// 時與上游逐位一致。
//! VERSION 4

//! PARAMETER sharpStrength
//! LABEL "Sharpening Strength"
//! MIN 0
//! MAX 3
//! DEFAULT 0.65
//! STEP 0.01
float sharpStrength;

//! PARAMETER sharpClamp
//! LABEL "Sharpening Limit"
//! MIN 0
//! MAX 1
//! DEFAULT 0.035
//! STEP 0.005
float sharpClamp;

//! PARAMETER pattern
//! LABEL "Sample Pattern"
//! MIN 0
//! MAX 3
//! DEFAULT 1
//! STEP 1
int pattern;

//! PARAMETER offsetBias
//! LABEL "Offset Bias"
//! MIN 0
//! MAX 6
//! DEFAULT 1
//! STEP 0.01
float offsetBias;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! SAMPLER sam
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "Luma 域 unsharp mask（4 種取樣圖樣）"

const COEF_LUMA = vec3<f32>(0.2126, 0.7152, 0.0722); // BT.709 / sRGB luma 係數

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let pos = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) * scene.outputPt;
    let BUFFER_PIXEL_SIZE = scene.inputPt;

    // 原始像素
    let ori = textureSampleLevel(INPUT, sam, pos, 0.0).rgb;

    // 強度 × luma 乘數
    var sharp_strength_luma = COEF_LUMA * uniforms.sharpStrength;
    var blur_ori = vec3<f32>(0.0);
    let ob = uniforms.offsetBias;

    if (uniforms.pattern == 0) {
        // 圖樣 0（快）：2+1 fetch 高斯
        blur_ori = textureSampleLevel(INPUT, sam, pos + (BUFFER_PIXEL_SIZE / 3.0) * ob, 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos + (-BUFFER_PIXEL_SIZE / 3.0) * ob, 0.0).rgb;
        blur_ori /= 2.0;
        sharp_strength_luma *= 1.5;
    }

    if (uniforms.pattern == 1) {
        // 圖樣 1（標準）：4+1 fetch 9-tap 高斯
        blur_ori = textureSampleLevel(INPUT, sam, pos + vec2<f32>(BUFFER_PIXEL_SIZE.x, -BUFFER_PIXEL_SIZE.y) * 0.5 * ob, 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos - BUFFER_PIXEL_SIZE * 0.5 * ob, 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos + BUFFER_PIXEL_SIZE * 0.5 * ob, 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos - vec2<f32>(BUFFER_PIXEL_SIZE.x, -BUFFER_PIXEL_SIZE.y) * 0.5 * ob, 0.0).rgb;
        blur_ori *= 0.25;
    }

    if (uniforms.pattern == 2) {
        // 圖樣 2（寬）：4+1 fetch 17-tap 高斯（實驗性）
        blur_ori = textureSampleLevel(INPUT, sam, pos + BUFFER_PIXEL_SIZE * vec2<f32>(0.4, -1.2) * ob, 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos - BUFFER_PIXEL_SIZE * vec2<f32>(1.2, 0.4) * ob, 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos + BUFFER_PIXEL_SIZE * vec2<f32>(1.2, 0.4) * ob, 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos - BUFFER_PIXEL_SIZE * vec2<f32>(0.4, -1.2) * ob, 0.0).rgb;
        blur_ori *= 0.25;
        sharp_strength_luma *= 0.51;
    }

    if (uniforms.pattern == 3) {
        // 圖樣 3（金字塔）：4+1 fetch 9-tap high-pass
        blur_ori = textureSampleLevel(INPUT, sam, pos + vec2<f32>(0.5 * BUFFER_PIXEL_SIZE.x, -BUFFER_PIXEL_SIZE.y * ob), 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos + vec2<f32>(ob * -BUFFER_PIXEL_SIZE.x, 0.5 * -BUFFER_PIXEL_SIZE.y), 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos + vec2<f32>(ob * BUFFER_PIXEL_SIZE.x, 0.5 * BUFFER_PIXEL_SIZE.y), 0.0).rgb;
        blur_ori += textureSampleLevel(INPUT, sam, pos + vec2<f32>(0.5 * -BUFFER_PIXEL_SIZE.x, BUFFER_PIXEL_SIZE.y * ob), 0.0).rgb;
        blur_ori /= 4.0;
        sharp_strength_luma *= 0.666;
    }

    // 銳化量 = 原圖 - 模糊圖
    let sharp = ori - blur_ori;

    // 強度調整 + 限幅（限幅部分折入 dot）
    let sharp_strength_luma_clamp = vec4<f32>(sharp_strength_luma * (0.5 / max(uniforms.sharpClamp, 1e-4)), 0.5);
    var sharp_luma = saturate(dot(vec4<f32>(sharp, 1.0), sharp_strength_luma_clamp));
    sharp_luma = (uniforms.sharpClamp * 2.0) * sharp_luma - uniforms.sharpClamp;

    // 加回原圖
    let outputcolor = ori + vec3<f32>(sharp_luma);

    textureStore(OUTPUT, vec2<i32>(global_id.xy), vec4<f32>(saturate(outputcolor), 1.0));
}
