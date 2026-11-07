// Anime4K_Denoise_Bilateral_Mean
// 移植自 https://github.com/Blinue/Magpie/blob/main/src/Effects/Anime4K/Anime4K_Denoise_Bilateral_Mean.hlsl
// （原始 GLSL：https://github.com/bloc97/Anime4K/blob/master/glsl/Denoise/Anime4K_Denoise_Bilateral_Mean.glsl）

// MAGPIE WebGPU EFFECT
//!VERSION 4
//!SORT_NAME Anime4K Denoise Bilateral Mean
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
// KERNELSIZE = max(uint(ceil(SPATIAL_SIGMA * 2.0)), 1) * 2 + 1 = 3
const KERNELSIZE: u32 = 3u;
// 原始為 uint(KERNELSIZE / 2) = 1；此處以 i32 儲存便於位移算術，值相同
const KERNELHALFSIZE: i32 = 1;
const INTENSITY_POWER_CURVE: f32 = 1.0;

// --- 輔助函數 (Helper Functions) ---
// Magpie 8x8 Block Remapping Helper
fn Rmp8x8(a: u32) -> uint2 {
    let x = a % 8u;
    let y = a / 8u;
    return uint2(x, y);
}

// HLSL rcp(x) = 1 / x（WGSL 不允許使用者函式多載，故標量與向量版分開命名）
fn rcp(x: f32) -> f32 {
    return 1.0 / x;
}

fn rcp3(x: MF3) -> MF3 {
    return 1.0 / x;
}

// 原始 gaussian 以「倒數」為參數：scaled = (x - m) * rcpS
fn gaussian(x: f32, rcpS: f32, m: f32) -> f32 {
    let scaled = (x - m) * rcpS;
    return exp(-0.5 * scaled * scaled);
}

fn gaussian_vec(x: MF3, rcpS: MF3, m: MF3) -> MF3 {
    let scaled = (x - m) * rcpS;
    return exp(-0.5 * scaled * scaled);
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
//!BLOCK_SIZE 16
//!NUM_THREADS 64


@compute @workgroup_size(64, 1, 1)
fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Magpie remapping logic: (Rmp8x8(threadId.x) << 1) + blockStart
    // BLOCK_SIZE = 16 → blockStart = workgroup_id.xy * 16，每執行緒處理 2x2 像素
    let gxy: uint2 = (Rmp8x8(local_id.x) * 2u) + (workgroup_id.xy * 16u);

    let outputSize = scene.outputSize;
    // Early exit if the base coordinate is out of bounds
    if gxy.x >= outputSize.x || gxy.y >= outputSize.y {
        return;
    }

    let inputPt = scene.inputPt;

    // src[KERNELSIZE + 1][KERNELSIZE + 1]：以 Gather 收集 4x4 鄰域（每個 Gather 取回 2x2）
    var src: array<array<MF3, 4>, 4>;

    for (var i: u32 = 0u; i <= KERNELSIZE - 1u; i += 2u) {
        for (var j: u32 = 0u; j <= KERNELSIZE - 1u; j += 2u) {
            // 與原始 HLSL 相同的取樣點：tpos = (gxy + (i,j) - KERNELHALFSIZE + 1) * inputPt
            let tpos = vec2<f32>(vec2<i32>(gxy) + int2(i32(i), i32(j)) - KERNELHALFSIZE + 1) * inputPt;

            // Gather 四點布局（與 HLSL 一致）：
            //   w z
            //   x y
            let sr = textureGather(0, INPUT, sam, tpos);
            let sg = textureGather(1, INPUT, sam, tpos);
            let sb = textureGather(2, INPUT, sam, tpos);

            src[i][j] = MF3(sr.w, sg.w, sb.w);
            src[i][j + 1u] = MF3(sr.x, sg.x, sb.x);
            src[i + 1u][j] = MF3(sr.z, sg.z, sb.z);
            src[i + 1u][j + 1u] = MF3(sr.y, sg.y, sb.y);
        }
    }

    // len[k][m]：核內各點到中心的空间距離（對應原始 len[KERNELSIZE][KERNELSIZE]）
    var len: array<array<f32, 3>, 3>;
    for (var k: u32 = 0u; k < KERNELSIZE; k++) {
        for (var m: u32 = 0u; m < KERNELSIZE; m++) {
            let off = vec2<i32>(i32(k) - KERNELHALFSIZE, i32(m) - KERNELHALFSIZE);
            len[k][m] = length(vec2<f32>(off));
        }
    }

    // Process the 2x2 pixels handled by this thread
    for (var i: u32 = 0u; i <= 1u; i++) {
        for (var j: u32 = 0u; j <= 1u; j++) {
            let destPos = gxy + uint2(i, j);

            // Boundary check for write
            if destPos.x >= outputSize.x || destPos.y >= outputSize.y {
                continue;
            }

            var sum = MF3(0.0);
            var n = MF3(0.0);

            // 中心像素（彩色三通道，非亮度——原始 Mean 版為向量雙邊濾波）
            let vc = src[u32(KERNELHALFSIZE) + i][u32(KERNELHALFSIZE) + j];

            // rcpIs = rcp(pow(vc + 0.0001, INTENSITY_POWER_CURVE) * INTENSITY_SIGMA)
            // INTENSITY_SIGMA 即 uniforms.intensitySigma（binding 1）
            let rcpIs = rcp3(pow(vc + 0.0001, MF3(INTENSITY_POWER_CURVE)) * uniforms.intensitySigma);
            let rcpSs = rcp(1.0); // rcp(SPATIAL_SIGMA)，SPATIAL_SIGMA = 1.0

            for (var k: u32 = 0u; k < KERNELSIZE; k++) {
                for (var m: u32 = 0u; m < KERNELSIZE; m++) {
                    let v = src[k + i][m + j];
                    let d = gaussian_vec(v, rcpIs, vc) * gaussian(len[k][m], rcpSs, 0.0);
                    sum += d * v;
                    n += d;
                }
            }

            textureStore(OUTPUT, vec2<i32>(destPos), MF4(sum / n, 1.0));
        }
    }
}
