// Anime4K_Denoise_Bilateral_Mode
// 移植自 https://github.com/bloc97/Anime4K/blob/master/glsl/Denoise/Anime4K_Denoise_Bilateral_Mode.glsl

// MAGPIE WebGPU EFFECT
//!VERSION 4
//!SORT_NAME Anime4K Denoise Bilateral
//!CAPABILITY FP16

//!PARAMETER
//!LABEL Strength
//!DEFAULT 0.1
//!MIN 0.01
//!MAX 5
//!STEP 0.01
//!VAR intensitySigma
// float intensitySigma; // WGSL 不使用這行

//! COMMON
// --- 類型別名 (Type Aliases) ---
alias MF = f32;
alias MF2 = vec2<f32>;
alias MF3 = vec3<f32>;
alias MF4 = vec4<f32>;
alias uint2 = vec2<u32>;
alias int2 = vec2<i32>;

// --- 結構定義 (Struct Definitions) ---
struct SceneInfo {
    inputSize: uint2,
    inputPt: MF2,
}

// --- Uniform 綁定 (必須在 COMMON 內) ---
// 參數由產生器注入的 uniforms 結構（@binding(1)）提供，COMMON 不自訂參數 uniform。
@group(0) @binding(4) var<uniform> scene: SceneInfo;

// --- 輔助函數 (Helper Functions) ---
fn GetInputPt() -> MF2 {
    return scene.inputPt;
}

fn GetOutputSize() -> uint2 {
    return scene.inputSize;
}

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

// Constants derived from SPATIAL_SIGMA = 1.0
// KERNELSIZE = max(1, 1) * 2 + 1 = 3
const KERNELSIZE: u32 = 3u; 
const KERNELHALFSIZE: i32 = 1;
const KERNELLEN: u32 = 9u; // 3 * 3

const HISTOGRAM_REGULARIZATION: f32 = 0.2;
const INTENSITY_POWER_CURVE: f32 = 1.0;
//!END COMMON

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
@group(0) @binding(0) var INPUT: texture_2d<f32>;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
@group(0) @binding(1) var OUTPUT: texture_storage_2d<rgba16float, write>;

//!SAMPLER
//!FILTER POINT
@group(0) @binding(3) var sam: sampler;


//!PASS 1
//!IN INPUT
//!OUT OUTPUT
//!BLOCK_SIZE 16
//!NUM_THREADS 64


@compute @workgroup_size(64, 1, 1)
fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Magpie remapping logic: (Rmp8x8(threadId.x) << 1) + blockStart
    // blockStart is effectively workgroup_id.xy * 16 (matches BLOCK_SIZE)
    // Rmp8x8 returns 0..7 coordinates, *2 spreads them out for processing 2x2 pixels per thread
    let gxy: uint2 = (Rmp8x8(local_id.x) * 2u) + (workgroup_id.xy * 16u);

    let outputSize = GetOutputSize();
    // Early exit if the base coordinate is out of bounds
    if gxy.x >= outputSize.x || gxy.y >= outputSize.y {
        return;
    }

    let inputPt = GetInputPt();
    
    // src array to hold gathered neighborhood. 
    // Size covers KERNELSIZE+1 to handle interpolation/offsets
    var src: array<array<MF4, 4>, 4>;

    // Load data using Gather (fetching 2x2 blocks)
    for (var i: u32 = 0u; i <= KERNELSIZE - 1u; i += 2u) {
        for (var j: u32 = 0u; j <= KERNELSIZE - 1u; j += 2u) {
            // 與 Magpie 原版一致：以象限「角點座標」取樣 Gather。
            // Gather 的 base = floor(u * size - 0.5)，以角點座標（整數紋素座標）取樣時
            // base = 該角點左上鄰素，返回的四象限正好涵蓋 [gxy+i-1, gxy+i] 的 2x2 鄰域，
            // 使 src[HALF+dx][HALF+dy] = texel(gxy+dx, gxy+j+dy)（核心中心對齊目標像素）。
            // 原先的 (pos + 0.5) * pt 中心取樣會把整個 gather 視窗平移 (+1,+1)，
            // 且在非 2 的冪次尺寸下有浮點脆弱性（u * size - 0.5 落在整數附近）。
            let tpos = MF2(gxy + uint2(i, j)) * inputPt;

            // Gather returns 4 values from a 2x2 quad.
            // Component 0=R, 1=G, 2=B.
            // 實測（Dawn）與 D3D GatherRed 相同的分量佈局：x=左下、y=右下、z=右上、w=左上，
            // 對應原版 HLSL 註解的「w z / x y」排列。
            let sr = textureGather(0, INPUT, sam, tpos);
            let sg = textureGather(1, INPUT, sam, tpos);
            let sb = textureGather(2, INPUT, sam, tpos);

            // w=TL -> src[i][j]、x=BL -> src[i][j+1]、z=TR -> src[i+1][j]、y=BR -> src[i+1][j+1]
            src[i][j] = MF4(sr.w, sg.w, sb.w, get_luma(MF3(sr.w, sg.w, sb.w)));
            src[i][j + 1u] = MF4(sr.x, sg.x, sb.x, get_luma(MF3(sr.x, sg.x, sb.x)));
            src[i + 1u][j] = MF4(sr.z, sg.z, sb.z, get_luma(MF3(sr.z, sg.z, sb.z)));
            src[i + 1u][j + 1u] = MF4(sr.y, sg.y, sb.y, get_luma(MF3(sr.y, sg.y, sb.y)));
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

            var histogram_v: array<MF3, 9>;
            var histogram_l: array<f32, 9>;
            var histogram_w: array<f32, 9>;
            var histogram_wn: array<f32, 9>;

            // Center pixel luma
            let vc = src[u32(KERNELHALFSIZE) + i][u32(KERNELHALFSIZE) + j].a;

            // 參數 intensitySigma 由產生器注入的 uniforms 結構提供（binding 1）
            let is_val = pow(vc + 0.0001, INTENSITY_POWER_CURVE) * uniforms.intensitySigma;
            let ss = 1.0; // SPATIAL_SIGMA

            // Calculate weights based on spatial and intensity distance
            for (var k: u32 = 0u; k < KERNELLEN; k++) {
                let ipos = GETOFFSET(k);
                // Index calculation: uint2(i, j) + ipos.yx + KERNELHALFSIZE
                // Note: ipos.yx swaps x and y offset.
                let idx_x = u32(i32(i) + ipos.y + KERNELHALFSIZE);
                let idx_y = u32(i32(j) + ipos.x + KERNELHALFSIZE);

                histogram_v[k] = src[idx_x][idx_y].rgb;
                histogram_l[k] = src[idx_x][idx_y].a;

                let dist = length(vec2<f32>(f32(ipos.x), f32(ipos.y)));
                histogram_w[k] = gaussian(histogram_l[k], is_val, vc) * gaussian(dist, ss, 0.0);
                histogram_wn[k] = 0.0;
            }

            // Histogram regularization
            for (var k: u32 = 0u; k < KERNELLEN; k++) {
                histogram_wn[k] += gaussian(0.0, HISTOGRAM_REGULARIZATION, 0.0) * histogram_w[k];
                for (var m: u32 = (k + 1u); m < KERNELLEN; m++) {
                    let d = gaussian(histogram_l[m], HISTOGRAM_REGULARIZATION, histogram_l[k]);
                    histogram_wn[m] += d * histogram_w[k];
                    histogram_wn[k] += d * histogram_w[m];
                }
            }

            // Find max weight
            var maxv: MF3 = MF3(0.0);
            var maxw: f32 = 0.0;

            for (var k: u32 = 0u; k < KERNELLEN; k++) {
                if histogram_wn[k] >= maxw {
                    maxw = histogram_wn[k];
                    maxv = histogram_v[k];
                }
            }

            textureStore(OUTPUT, vec2<i32>(destPos), MF4(maxv, 1.0));
        }
    }
}