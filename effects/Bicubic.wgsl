// Bicubic（B/C 參數化 bicubic 插值）— 移植自 effects_src/Bicubic.hlsl（Magpie v4）
// 上游：cemu_graphic_packs Bicubic filter
//! VERSION 4

//! PARAMETER paramB
//! LABEL "B"
//! MIN 0
//! MAX 1
//! DEFAULT 0.33
//! STEP 0.01
float paramB;

//! PARAMETER paramC
//! LABEL "C"
//! MIN 0
//! MAX 1
//! DEFAULT 0.33
//! STEP 0.01
float paramC;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
Texture2D OUTPUT;

//! SAMPLER sam
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "Bicubic（Mitchell-Netravali B/C）9-tap 兩段式取樣"

fn weight(x: f32) -> f32 {
    let B = uniforms.paramB;
    let C = uniforms.paramC;
    let ax = abs(x);

    if (ax < 1.0) {
        return (x * x * ((12.0 - 9.0 * B - 6.0 * C) * ax + (-18.0 + 12.0 * B + 6.0 * C)) + (6.0 - 2.0 * B)) / 6.0;
    } else if (ax < 2.0) {
        return (x * x * ((-B - 6.0 * C) * ax + (6.0 * B + 30.0 * C)) + (-12.0 * B - 48.0 * C) * ax + (8.0 * B + 24.0 * C)) / 6.0;
    } else {
        return 0.0;
    }
}

fn weight4(x: f32) -> vec4<f32> {
    return vec4<f32>(weight(x - 2.0), weight(x - 1.0), weight(x), weight(x + 1.0));
}

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let inputPt = scene.inputPt;
    let inputSize = vec2<f32>(scene.inputSize);
    var pos = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) * scene.outputPt;

    pos = pos * inputSize;
    let pos1 = floor(pos - 0.5) + 0.5;
    let fr = pos - pos1;

    var rowtaps = weight4(1.0 - fr.x);
    var coltaps = weight4(1.0 - fr.y);

    // 讓所有 tap 總和恰為 1.0，避免極小失真
    rowtaps = rowtaps / (rowtaps.x + rowtaps.y + rowtaps.z + rowtaps.w);
    coltaps = coltaps / (coltaps.x + coltaps.y + coltaps.z + coltaps.w);

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

    // HLSL int2(float2) 截斷語義；0.5/size-0.5 夾限對應 CLAMP 邊界
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

    textureStore(OUTPUT, vec2<i32>(global_id.xy), vec4<f32>(total, 1.0));
}
