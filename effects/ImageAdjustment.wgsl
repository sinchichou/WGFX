// ImageAdjustment — 移植自 effects_src/ImageAdjustment.hlsl（Magpie v4）
// 上游：libretro slang-shaders image-adjustment.slang
//! VERSION 4

//! PARAMETER targetGamma
//! LABEL "Target Gamma"
//! MIN 0.01
//! MAX 5
//! DEFAULT 2.2
//! STEP 0.01
float targetGamma;

//! PARAMETER monitorGamma
//! LABEL "Monitor Gamma"
//! MIN 0.01
//! MAX 5
//! DEFAULT 2.2
//! STEP 0.01
float monitorGamma;

//! PARAMETER saturation
//! LABEL "Saturation"
//! MIN 0
//! MAX 5
//! DEFAULT 1
//! STEP 0.01
float saturation;

//! PARAMETER luminance
//! LABEL "Luminance"
//! MIN 0
//! MAX 2
//! DEFAULT 1
//! STEP 0.01
float luminance;

//! PARAMETER contrast
//! LABEL "Contrast"
//! MIN 0
//! MAX 10
//! DEFAULT 1
//! STEP 0.1
float contrast;

//! PARAMETER brightBoost
//! LABEL "Brightness Boost"
//! MIN -1
//! MAX 1
//! DEFAULT 0
//! STEP 0.01
float brightBoost;

//! PARAMETER blackLevel
//! LABEL "Black Level"
//! MIN -1
//! MAX 1
//! DEFAULT 0
//! STEP 0.01
float blackLevel;

//! PARAMETER r
//! LABEL "Red Channel"
//! MIN 0
//! MAX 2
//! DEFAULT 1
//! STEP 0.01
float r;

//! PARAMETER g
//! LABEL "Green Channel"
//! MIN 0
//! MAX 2
//! DEFAULT 1
//! STEP 0.01
float g;

//! PARAMETER b
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

//! SAMPLER sam
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "HSV 飽和度/亮度、對比/亮度、黑階、gamma 修正、通道增益"

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
    let p = abs(fract(vec3<f32>(c.x) + K.xyz) * 6.0 - K.www);
    return c.z * mix(vec3<f32>(K.x), saturate(p - K.xxx), vec3<f32>(c.y));
}

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let pos = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) * scene.outputPt;
    var color = textureSampleLevel(INPUT, sam, pos, 0.0).rgb;

    // 飽和度與亮度
    color = saturate(HSVtoRGB(RGBtoHSV(color) * vec3<f32>(1.0, uniforms.saturation, uniforms.luminance)));

    // 對比與亮度
    color = saturate((color - 0.5) * uniforms.contrast + 0.5 + uniforms.brightBoost);

    // 黑階
    color -= vec3<f32>(uniforms.blackLevel);
    color = saturate(color / (1.0 - uniforms.blackLevel)); // checker:ok — blackLevel ∈ [-1,1)，分母 ≥ 2

    // gamma 修正
    color = pow(color, vec3<f32>(uniforms.targetGamma / uniforms.monitorGamma));

    color *= vec3<f32>(uniforms.r, uniforms.g, uniforms.b);

    textureStore(OUTPUT, vec2<i32>(global_id.xy), vec4<f32>(color, 1.0));
}
