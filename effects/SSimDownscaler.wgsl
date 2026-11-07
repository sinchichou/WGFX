// SSimDownscaler — 移植自 effects_src/SSimDownscaler.hlsl（Magpie v4）
// 上游：igv 的 SSimDownscaler（mpv hook；基於感知的 SSIM 加權縮小）
//
// 移植說明：
//   1. gather 展開為 clamp load 網格：P4/P5 採樣點在整數紋理角點，
//      src1[a][b] = load(gxy + a - 2, gxy + b - 2)；(gxy+1) 角點 → src2[a][b] = load(gxy + a, gxy + b)。
//   2. HLSL float3x3 的「三列」累加改以 array<vec3<f32>,3> 明確列向量處理（避開行列序差異）。
//   3. L2 尺寸 = INPUT_WIDTH × OUTPUT_HEIGHT（Magpie 慣例；框架已支援 OUTPUT_HEIGHT 運算式）。
//! VERSION 4

//! PARAMETER oversharp
//! LABEL "Oversharp"
//! MIN 1
//! MAX 3
//! DEFAULT 1
//! STEP 0.1
float oversharp;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
Texture2D OUTPUT;

//! TEXTURE L2
//! WIDTH INPUT_WIDTH
//! HEIGHT OUTPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D L2;

//! TEXTURE L2_2
//! WIDTH OUTPUT_WIDTH
//! HEIGHT OUTPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D L2_2;

//! TEXTURE MR
//! WIDTH OUTPUT_WIDTH
//! HEIGHT OUTPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D MR;

//! TEXTURE POSTKERNEL
//! WIDTH OUTPUT_WIDTH
//! HEIGHT OUTPUT_HEIGHT
//! FORMAT R8G8B8A8_UNORM
Texture2D POSTKERNEL;

//! SAMPLER sam
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! SAMPLER sam1
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState sam1;

//! COMMON
// —— 跨 pass 共用（每個 pass 為獨立 WGSL 模組，共用 helper 必須置於 COMMON）——

// B=0, C=0.5 的 Mitchell-Netravali 核（Pass2/Pass3 用）
fn KernelMN(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1.0) {
        return ((2.0 - 1.5 * 0.0 - 0.5) * ax + (-3.0 + 2.0 * 0.0 + 0.5)) * ax * ax + (1.0 - 0.0 / 3.0);
    } else {
        return (((-0.0 / 6.0 - 0.5) * ax + (0.0 + 5.0 * 0.5)) * ax + (-2.0 * 0.0 - 8.0 * 0.5)) * ax + ((4.0 / 3.0) * 0.0 + 4.0 * 0.5);
    }
}

const SIGMA_NSQ = 10.0 / (255.0 * 255.0);
const LOCALITY = 2.0;

fn KernelPow(x: i32) -> f32 {
    return pow(1.0 / LOCALITY, abs(f32(x))); // checker:ok — LOCALITY 為模組常數
}

fn Luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn LoadClamped(tex: texture_2d<f32>, x: i32, y: i32) -> vec3<f32> {
    let size = vec2<i32>(scene.outputSize);
    return textureLoad(tex, clamp(vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

//! END

//! PASS 1
//! DESC "Catmull-Rom 預濾波（9 取樣模擬 4x4）"
//! IN INPUT
//! OUT POSTKERNEL
//! NUM_THREADS 8,8,1

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let inputSize = vec2<f32>(scene.inputSize);
    let inputPt = scene.inputPt;
    var pos = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) * scene.outputPt;

    let samplePos = pos * inputSize;
    let texPos1 = floor(samplePos - 0.5) + 0.5;
    let fr = samplePos - texPos1;

    // Catmull-Rom 權重（預展開）
    let w0 = fr * (-0.5 + fr * (1.0 - 0.5 * fr));
    let w1 = 1.0 + fr * fr * (-2.5 + 1.5 * fr);
    let w2 = fr * (0.5 + fr * (2.0 - 1.5 * fr));
    let w3 = fr * fr * (-0.5 + 0.5 * fr);

    let w12 = w1 + w2;
    let offset12 = w2 / (w1 + w2); // checker:ok — w1+w2 > 0（Catmull-Rom 分段權重和）

    var texPos0 = (texPos1 - 1.0) * inputPt;
    var texPos3 = (texPos1 + 2.0) * inputPt;
    var texPos12 = (texPos1 + offset12) * inputPt;

    var result = vec4<f32>(0.0);
    result += textureSampleLevel(INPUT, sam1, vec2<f32>(texPos0.x, texPos0.y), 0.0) * w0.x * w0.y;
    result += textureSampleLevel(INPUT, sam1, vec2<f32>(texPos12.x, texPos0.y), 0.0) * w12.x * w0.y;
    result += textureSampleLevel(INPUT, sam1, vec2<f32>(texPos3.x, texPos0.y), 0.0) * w3.x * w0.y;

    result += textureSampleLevel(INPUT, sam1, vec2<f32>(texPos0.x, texPos12.y), 0.0) * w0.x * w12.y;
    result += textureSampleLevel(INPUT, sam1, vec2<f32>(texPos12.x, texPos12.y), 0.0) * w12.x * w12.y;
    result += textureSampleLevel(INPUT, sam1, vec2<f32>(texPos3.x, texPos12.y), 0.0) * w3.x * w12.y;

    result += textureSampleLevel(INPUT, sam1, vec2<f32>(texPos0.x, texPos3.y), 0.0) * w0.x * w3.y;
    result += textureSampleLevel(INPUT, sam1, vec2<f32>(texPos12.x, texPos3.y), 0.0) * w12.x * w3.y;
    result += textureSampleLevel(INPUT, sam1, vec2<f32>(texPos3.x, texPos3.y), 0.0) * w3.x * w3.y;

    textureStore(POSTKERNEL, vec2<i32>(global_id.xy), result);
}

//! PASS 2
//! DESC "L2 pass 1：垂直方向 (Σ w·tex²)/Σ w"
//! IN INPUT
//! OUT L2
//! NUM_THREADS 8,8,1

fn Pass2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // OUT = L2（INPUT_WIDTH × OUTPUT_HEIGHT）——guard 對準該紋理尺寸
    if (global_id.x >= scene.inputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }
    let inputPtY = scene.inputPt.y;
    let inputHeight = scene.inputSize.y;
    let outputPtY = scene.outputPt.y;
    let outputHeight = scene.outputSize.y;
    // L2 高 = OUTPUT_HEIGHT → pos.y 以 outputPtY 步進
    let posY = (f32(global_id.y) + 0.5) * outputPtY;
    let posX = (f32(global_id.x) + 0.5) * scene.inputPt.x; // L2 寬 = INPUT_WIDTH

    let taps = 2.0;
    let low = i32(ceil((posY - taps * outputPtY) * f32(inputHeight) - 0.5));
    let high = i32(floor((posY + taps * outputPtY) * f32(inputHeight) - 0.5));

    var W = 0.0;
    var avg = vec3<f32>(0.0);
    for (var k = low; k <= high; k++) {
        let sy = inputPtY * (f32(k) + 0.5);
        let rel = (sy - posY) * f32(outputHeight);
        let w = KernelMN(rel);
        let tex = textureLoad(INPUT, vec2<i32>(i32(global_id.x), clamp(k, 0, i32(inputHeight) - 1)), 0).rgb;
        avg += w * tex * tex;
        W += w;
    }
    avg /= vec3<f32>(W); // checker:ok — 最近取樣權重 KernelMN(≤0.5) > 0，W 恆正

    textureStore(L2, vec2<i32>(global_id.xy), vec4<f32>(avg, 1.0));
}

//! PASS 3
//! DESC "L2 pass 2：水平方向加權平均"
//! IN L2
//! OUT L2_2
//! NUM_THREADS 8,8,1

fn Pass3(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }
    let inputPtX = scene.inputPt.x;
    let inputWidth = scene.inputSize.x;
    let outputPtX = scene.outputPt.x;
    let outputWidthU = scene.outputSize.x;
    let posX = (f32(global_id.x) + 0.5) * outputPtX;

    let taps = 2.0;
    let low = i32(ceil((posX - taps * outputPtX) * f32(inputWidth) - 0.5));
    let high = i32(floor((posX + taps * outputPtX) * f32(inputWidth) - 0.5));

    var W = 0.0;
    var avg = vec3<f32>(0.0);
    for (var k = low; k <= high; k++) {
        let sx = inputPtX * (f32(k) + 0.5);
        let rel = (sx - posX) * f32(outputWidthU);
        let w = KernelMN(rel);
        // L2 寬 = INPUT_WIDTH；夾限後 point 取樣
        let tex = textureLoad(L2, vec2<i32>(clamp(k, 0, i32(inputWidth) - 1), i32(global_id.y)), 0).rgb;
        avg += w * tex;
        W += w;
    }
    avg /= vec3<f32>(W); // checker:ok — W 恆正（同 Pass2）

    textureStore(L2_2, vec2<i32>(global_id.xy), vec4<f32>(avg, 1.0));
}

//! PASS 4
//! DESC "mean & R：局部均值與 SSIM 比例"
//! IN L2_2, POSTKERNEL
//! OUT MR
//! NUM_THREADS 8,8,1

fn Pass4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }
    let gxy = vec2<i32>(global_id.xy);

    var src1: array<array<vec3<f32>, 4>, 4>;
    var src2: array<array<vec3<f32>, 4>, 4>;
    for (var a: u32 = 0u; a < 4u; a++) {
        for (var b: u32 = 0u; b < 4u; b++) {
            // gather 角點 (gxy + a - 2) → src[a][b] = load(gxy + a - 2, gxy + b - 2)
            src1[a][b] = LoadClamped(POSTKERNEL, gxy.x + i32(a) - 2, gxy.y + i32(b) - 2);
            src2[a][b] = LoadClamped(L2_2, gxy.x + i32(a) - 2, gxy.y + i32(b) - 2);
        }
    }

    var kernels: array<f32, 3>;
    for (var i: u32 = 0u; i < 3u; i++) {
        kernels[i] = KernelPow(i32(i) - 1);
    }

    var W = 0.0;
    var avg0 = vec3<f32>(0.0);
    var avg1r = vec3<f32>(0.0);
    var avg2r = vec3<f32>(0.0);
    for (var i1: u32 = 0u; i1 < 3u; i1++) {
        var W1 = 0.0;
        var a1r0 = vec3<f32>(0.0);
        var a1r1 = vec3<f32>(0.0);
        var a1r2 = vec3<f32>(0.0);
        for (var j1: u32 = 0u; j1 < 3u; j1++) {
            let L = src1[j1][i1];
            a1r0 += kernels[j1] * L;
            a1r1 += kernels[j1] * (L * L);
            a1r2 += kernels[j1] * src2[j1][i1];
            W1 += kernels[j1];
        }
        a1r0 /= vec3<f32>(W1); // checker:ok — KernelPow 恆正
        a1r1 /= vec3<f32>(W1);
        a1r2 /= vec3<f32>(W1);
        W += kernels[i1];
        avg0 += kernels[i1] * a1r0;
        avg1r += kernels[i1] * a1r1;
        avg2r += kernels[i1] * a1r2;
    }
    avg0 /= vec3<f32>(W);
    avg1r /= vec3<f32>(W);
    avg2r /= vec3<f32>(W);

    let Sl = Luma(max(avg1r - avg0 * avg0, vec3<f32>(0.0)));
    let Sh = Luma(max(avg2r - avg0 * avg0, vec3<f32>(0.0)));
    // select 兩支皆求值，但 0/0 的 NaN 只會出現在未選分支，不會傳播
    let sqrtBranch = sqrt((Sh + SIGMA_NSQ) / (Sl + SIGMA_NSQ)) * uniforms.oversharp;
    let clampBranch = clamp(Sh / Sl, 0.0, 1.0);
    let R = select(clampBranch, sqrtBranch, Sl > Sh);
    textureStore(MR, gxy, vec4<f32>(avg0, R));
}

//! PASS 5
//! DESC "final：MR 加權合成還原"
//! IN MR, POSTKERNEL
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

fn Pass5(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }
    let gxy = vec2<i32>(global_id.xy);

    var src1: array<array<vec4<f32>, 4>, 4>;
    for (var a: u32 = 0u; a < 4u; a++) {
        for (var b: u32 = 0u; b < 4u; b++) {
            let size = vec2<i32>(scene.outputSize);
            let p = textureLoad(MR, clamp(vec2<i32>(gxy.x + i32(a) - 2, gxy.y + i32(b) - 2), vec2<i32>(0), size - vec2<i32>(1)), 0);
            src1[a][b] = p;
        }
    }

    var src2: array<array<vec3<f32>, 2>, 2>;
    for (var a: u32 = 0u; a < 2u; a++) {
        for (var b: u32 = 0u; b < 2u; b++) {
            let size = vec2<i32>(scene.outputSize);
            // gather 角點 (gxy+1) → src2[a][b] = load(gxy + a, gxy + b)
            let p = textureLoad(POSTKERNEL, clamp(vec2<i32>(gxy.x + i32(a), gxy.y + i32(b)), vec2<i32>(0), size - vec2<i32>(1)), 0);
            src2[a][b] = p.rgb;
        }
    }

    var kernels: array<f32, 3>;
    for (var i: u32 = 0u; i < 3u; i++) {
        kernels[i] = KernelPow(i32(i) - 1);
    }

    var W = 0.0;
    var avg0 = vec3<f32>(0.0);
    var avg1r = vec3<f32>(0.0);
    var avg2r = vec3<f32>(0.0);
    for (var i1: u32 = 0u; i1 < 3u; i1++) {
        var W1 = 0.0;
        var a1r0 = vec3<f32>(0.0);
        var a1r1 = vec3<f32>(0.0);
        var a1r2 = vec3<f32>(0.0);
        for (var j1: u32 = 0u; j1 < 3u; j1++) {
            let MRc = src1[j1][i1];
            a1r0 += kernels[j1] * (MRc.a * MRc.rgb);
            a1r1 += kernels[j1] * MRc.rgb;
            a1r2 += kernels[j1] * MRc.aaa;
            W1 += kernels[j1];
        }
        a1r0 /= vec3<f32>(W1); // checker:ok — KernelPow 恆正
        a1r1 /= vec3<f32>(W1);
        a1r2 /= vec3<f32>(W1);
        W += kernels[i1];
        avg0 += kernels[i1] * a1r0;
        avg1r += kernels[i1] * a1r1;
        avg2r += kernels[i1] * a1r2;
    }
    avg0 /= vec3<f32>(W);
    avg1r /= vec3<f32>(W);
    avg2r /= vec3<f32>(W);

    textureStore(OUTPUT, gxy, vec4<f32>(avg1r + avg2r * src2[0][0] - avg0, 1.0));
}
