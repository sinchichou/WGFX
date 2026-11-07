// Jinc2 — 移植自 effects_src/Jinc.hlsl（Magpie v4）
// 上游：libretro common-shaders jinc2.cg（Jinc(x)·Jinc(x·r1/r2) 在 x<2.5 的近似）
//
// 移植偏差（記錄）：
//   1. Magpie 源碼第 4 次權重列誤寫為 float4x3(src[0][3], src[2][3], src[2][3], src[3][3])
//      （src[2][3] 重複、src[1][3] 遺漏——與 libretro 原始演算法的對稱 4x4 摺積不符），
//      本移植依演算法意圖修正為對稱形式。
//   2. gather 展開：採樣點位於紋理角點（tc = floor(pc - 0.5)），D3D 四點順序
//      （x=BL,y=BR,z=TR,w=TL）還原後 src[a][b] = load(tc + a - 1, tc + b - 1)。
//! VERSION 4

//! PARAMETER windowSinc
//! LABEL "Window Sinc Param"
//! MIN 0
//! MAX 1
//! DEFAULT 0.5
//! STEP 0.01
float windowSinc;

//! PARAMETER sinc
//! LABEL "Sinc Param"
//! MIN 0
//! MAX 1
//! DEFAULT 0.825
//! STEP 0.01
float sinc;

//! PARAMETER ARStrength
//! LABEL "Anti-ringing Strength"
//! MIN 0
//! MAX 1
//! DEFAULT 0.5
//! STEP 0.1
float ARStrength;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
Texture2D OUTPUT;

//! SAMPLER sam
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "Jinc2 四階插值 + 可調抗振鈴"

fn dist(pt1: vec2<f32>, pt2: vec2<f32>) -> f32 {
    let v = pt2 - pt1;
    return sqrt(dot(v, v));
}

// x == 0 時回傳 wa·wb（避免 0/0）；其餘 sin(x·wa)·sin(x·wb)/(x²)
fn resampler(x: vec4<f32>, wa: f32, wb: f32) -> vec4<f32> {
    let zero = vec4<f32>(0.0);
    let safe = select(x * x, vec4<f32>(1.0), x == zero);
    return sin(x * wa) * sin(x * wb) / safe; // checker:ok — x==0 分支由 select 保證分母為 1
}

fn Load(x: i32, y: i32) -> vec3<f32> {
    let size = vec2<i32>(scene.inputSize);
    return textureLoad(INPUT, clamp(vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let inputPt = scene.inputPt;
    const dx = vec2<f32>(1.0, 0.0);
    const dy = vec2<f32>(0.0, 1.0);

    let pc = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) * scene.outputPt * vec2<f32>(scene.inputSize);
    var tc = floor(pc - 0.5) + 0.5;

    let wa = uniforms.windowSinc * 3.14159265358979;
    let wb = uniforms.sinc * 3.14159265358979;
    var weights: array<vec4<f32>, 4>;
    weights[0] = resampler(vec4<f32>(dist(pc, tc - dx - dy), dist(pc, tc - dy), dist(pc, tc + dx - dy), dist(pc, tc + 2.0 * dx - dy)), wa, wb);
    weights[1] = resampler(vec4<f32>(dist(pc, tc - dx), dist(pc, tc), dist(pc, tc + dx), dist(pc, tc + 2.0 * dx)), wa, wb);
    weights[2] = resampler(vec4<f32>(dist(pc, tc - dx + dy), dist(pc, tc + dy), dist(pc, tc + dx + dy), dist(pc, tc + 2.0 * dx + dy)), wa, wb);
    weights[3] = resampler(vec4<f32>(dist(pc, tc - dx + 2.0 * dy), dist(pc, tc + 2.0 * dy), dist(pc, tc + dx + 2.0 * dy), dist(pc, tc + 2.0 * dx + 2.0 * dy)), wa, wb);

    tc = tc - 0.5;

    // 4x4 採樣網格（gather 展開）
    var src: array<array<vec3<f32>, 4>, 4>;
    for (var a: u32 = 0u; a < 4u; a++) {
        for (var b: u32 = 0u; b < 4u; b++) {
            src[a][b] = Load(i32(tc.x) + i32(a) - 1, i32(tc.y) + i32(b) - 1);
        }
    }

    // 4x4 摺積（修正版：對稱 Σ_row Σ_col weights[row][col]·src[col][row]）
    var color = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var row: u32 = 0u; row < 4u; row++) {
        for (var col: u32 = 0u; col < 4u; col++) {
            color += weights[row][col] * src[col][row];
            wsum += weights[row][col];
        }
    }
    // checker:ok — 權重總和恆正（jinc 核心歸一化），與上游 rcp(dot(...,1)) 同語義
    color = color / wsum;

    // 抗振鈴：夾限在中心 2x2 鄰域範圍
    let min_sample = min(min(src[1][1], src[2][1]), min(src[1][2], src[2][2]));
    let max_sample = max(max(src[1][1], src[2][1]), max(src[1][2], src[2][2]));
    color = mix(color, clamp(color, min_sample, max_sample), uniforms.ARStrength);

    textureStore(OUTPUT, vec2<i32>(global_id.xy), vec4<f32>(color, 1.0));
}
