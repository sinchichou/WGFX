// Anime4K_Denoise_Bilateral_Median
// 移植自 https://github.com/Blinue/Magpie/blob/main/src/Effects/Anime4K/Anime4K_Denoise_Bilateral_Median.hlsl
// （原始 GLSL：https://github.com/bloc97/Anime4K/blob/master/glsl/Denoise/Anime4K_Denoise_Bilateral_Median.glsl）

// MAGPIE WebGPU EFFECT
//!VERSION 4
//!SORT_NAME Anime4K Denoise Bilateral Median
//!CAPABILITY FP16

//!PARAMETER
//!LABEL Strength
//!DEFAULT 0.1
//!MIN 0.01
//!MAX 5
//!STEP 0.01
// 參數由產生器注入的 uniforms 結構（@group(0) @binding(1)）提供，以 uniforms.intensitySigma 存取
float intensitySigma;

//! COMMON
// --- 類型別名 (Type Aliases) ---
alias MF = f32;
alias MF2 = vec2<f32>;
alias MF3 = vec3<f32>;
alias MF4 = vec4<f32>;
alias uint2 = vec2<u32>;
alias int2 = vec2<i32>;

// --- 常數（由 SPATIAL_SIGMA = 1.0 推導，與原始 #define 一致）---
// KERNELSIZE = max(uint(SPATIAL_SIGMA), 1) * 2 + 1 = 3
const KERNELSIZE: u32 = 3u;
const KERNELHALFSIZE: i32 = 1; // int(KERNELSIZE / 2)
const KERNELLEN: u32 = 9u; // KERNELSIZE * KERNELSIZE
const INTENSITY_POWER_CURVE: f32 = 1.0;
// 原始 #define HISTOGRAM_REGULARIZATION 0.0：加權正規化分支為死碼，故省略
// （若日後要啟用，請參照原始 HLSL 的 histogram_wn 分支移植）

// --- 輔助函數 (Helper Functions) ---
// Magpie 8x8 Block Remapping Helper
fn Rmp8x8(a: u32) -> uint2 {
    let x = a % 8u;
    let y = a / 8u;
    return uint2(x, y);
}

fn get_luma(rgba: MF3) -> f32 {
    return dot(MF3(0.299, 0.587, 0.114), rgba);
}

fn gaussian(x: f32, s: f32, m: f32) -> f32 {
    let scaled = (x - m) / s;
    return exp(-0.5 * scaled * scaled);
}

fn GETOFFSET(i: u32) -> int2 {
    return int2(i32(i % KERNELSIZE) - KERNELHALFSIZE, i32(i / KERNELSIZE) - KERNELHALFSIZE);
}

// 加權中位數：依核心掃描順序回傳第一個滿足條件的值（與原始 HLSL 一致）
// 以指標傳遞函式區域陣列，確保動態索引在所有實作上皆合法
fn getMedian(v: ptr<function, array<MF3, 9>>, w: ptr<function, array<f32, 9>>, n: f32) -> MF3 {
    var result = MF3(0.0);

    for (var i: u32 = 0u; i < KERNELLEN; i++) {
        var w_above: f32 = 0.0;
        var w_below: f32 = 0.0;
        for (var j: u32 = 0u; j < KERNELLEN; j++) {
            if (*v)[j].x > (*v)[i].x {
                w_above += (*w)[j];
            } else if ((*v)[j].x < (*v)[i].x) {
                w_below += (*w)[j];
            }
        }

        if ((n - w_above) / n >= 0.5 && w_below / n <= 0.5) {
            result = (*v)[i];
            break;
        }
    }

    return result;
}
//!END COMMON

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
Texture2D INPUT;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//!SAMPLER
//!FILTER POINT
SamplerState sam;


//!PASS 1
//!IN INPUT
//!OUT OUTPUT
//!BLOCK_SIZE 8
//!NUM_THREADS 64


@compute @workgroup_size(64, 1, 1)
fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Magpie remapping logic: gxy = Rmp8x8(threadId.x) + blockStart
    // BLOCK_SIZE = 8 → blockStart = workgroup_id.xy * 8，每執行緒處理 1 個像素
    let gxy: uint2 = Rmp8x8(local_id.x) + (workgroup_id.xy * 8u);

    let outputSize = scene.outputSize;
    // Early exit if the base coordinate is out of bounds
    if gxy.x >= outputSize.x || gxy.y >= outputSize.y {
        return;
    }

    let inputPt = scene.inputPt;
    let pos = (vec2<f32>(gxy) + 0.5) * inputPt;

    var histogram_v: array<MF3, 9>; // KERNELLEN = KERNELSIZE * KERNELSIZE
    var histogram_l: array<f32, 9>;
    var histogram_w: array<f32, 9>;
    var n: f32 = 0.0;

    // 中心像素亮度（直接取樣，與原始 SampleLevel 一致）
    let vc = get_luma(textureSampleLevel(INPUT, sam, pos, 0.0).rgb);

    // INTENSITY_SIGMA 即 uniforms.intensitySigma（binding 1）
    let is_val = pow(vc + 0.0001, INTENSITY_POWER_CURVE) * uniforms.intensitySigma;
    let ss = 1.0; // SPATIAL_SIGMA

    for (var i: u32 = 0u; i < KERNELLEN; i++) {
        let ipos = GETOFFSET(i);
        histogram_v[i] = textureSampleLevel(INPUT, sam, pos + vec2<f32>(ipos) * inputPt, 0.0).rgb;
        histogram_l[i] = get_luma(histogram_v[i]);
        histogram_w[i] = gaussian(histogram_l[i], is_val, vc) * gaussian(length(vec2<f32>(ipos)), ss, 0.0);
        n += histogram_w[i];
    }

    // 原始 HLSL 的 HISTOGRAM_REGULARIZATION = 0.0，正規化分支為死碼，予以省略
    textureStore(OUTPUT, vec2<i32>(gxy), MF4(getMedian(&histogram_v, &histogram_w, n), 1.0));
}
