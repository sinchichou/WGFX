// AdaptiveSharpen — 移植自 effects_src/Sharpen/AdaptiveSharpen.hlsl（Magpie v4）
// 上游：libretro slang-shaders adaptive-sharpen.slang（2015-05-15 版，需 gamma 光域）
//
// 移植說明：
//   1. 8x8 gather 網格展開為 clamp load（採樣點在紋理角點，src[a][b] = load(gxy + a - 3, gxy + b - 3)）；
//     四角 texel（{0,7}×{0,7}）在演算法中從不被讀，仍載入（夾限值，無害）。
//   2. 1 invocation 1 像素（Magpie 的 2x2 sub-pixel 佈局為效能細節）。
//! VERSION 4

//! PARAMETER curveHeight
//! LABEL "Sharpness"
//! MIN 0
//! MAX 2
//! DEFAULT 0.8
//! STEP 0.01
float curveHeight;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "自適應銳化（邊緣感知 + tanh 軟限幅）"

// 最佳化常數——非必要請勿更動
const CURVESLOPE_SCALE = 1.5;   // 邊緣區銳化曲線斜率乘數
const D_OVERSHOOT = 0.016;      // 暗部過衝上限
const D_COMP_RATIO = 0.250;     // 暗部壓縮比 (1/0.25 = 4x)
const L_OVERSHOOT = 0.004;      // 亮部過衝上限
const L_COMP_RATIO = 0.167;     // 亮部壓縮比 (1/0.167 = 6x)
const MAX_SCALE_LIM = 10.0;     // 最大壓縮前的絕對變化量 (1/10 = ±10%)

// 彩色 → 灰階（快速近似 gamma）
fn CtG(RGB: vec3<f32>) -> f32 {
    let sq = RGB * RGB;
    return sqrt((1.0 / 3.0) * (sq.r + sq.g + sq.b));
}

fn Load(x: i32, y: i32) -> vec4<f32> {
    let size = vec2<i32>(scene.inputSize);
    let rgb = textureLoad(INPUT, clamp(vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
    return vec4<f32>(rgb, CtG(rgb));
}

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let gxy = vec2<i32>(global_id.xy);

    // 8x8 網格（rgb + CtG 灰階 w）
    var src: array<array<vec4<f32>, 8>, 8>;
    for (var a: u32 = 0u; a < 8u; a++) {
        for (var b: u32 = 0u; b < 8u; b++) {
            src[a][b] = Load(gxy.x + i32(a) - 3, gxy.y + i32(b) - 3);
        }
    }

    let curveslope = uniforms.curveHeight * CURVESLOPE_SCALE;

    // 高斯 3x3 模糊（中心 = src[3][3]）
    let blur = (2.0 * (src[3][2].rgb + src[2][3].rgb + src[4][3].rgb + src[3][4].rgb)
        + (src[2][2].rgb + src[4][2].rgb + src[2][4].rgb + src[4][4].rgb)
        + 4.0 * src[3][3].rgb) / 16.0;
    let blur_Y = (blur.r * (1.0 / 3.0) + blur.g * (1.0 / 3.0) + blur.b * (1.0 / 3.0));

    // 邊緣偵測（十字權重 4、對角 1）
    var edge = length(
        abs(blur - src[3][3].rgb) + abs(blur - src[2][2].rgb) + abs(blur - src[3][2].rgb) + abs(blur - src[4][2].rgb)
        + abs(blur - src[2][3].rgb) + abs(blur - src[4][3].rgb) + abs(blur - src[2][4].rgb) + abs(blur - src[3][4].rgb) + abs(blur - src[4][4].rgb)
        + 0.25 * (abs(blur - src[3][1].rgb) + abs(blur - src[1][3].rgb) + abs(blur - src[5][3].rgb) + abs(blur - src[3][5].rgb))
    ) * (1.0 / 3.0);

    // 邊緣對比壓縮，中心 = 0.5
    edge *= min((0.8 + 2.7 * pow(2.0, (-7.4 * blur_Y))), 3.2);

    let c0_Y = src[3][3].w;

    // 25 點 kernel（次序對應上游）
    var kernel: array<f32, 25>;
    kernel[0] = c0_Y;
    kernel[1] = src[2][2].w; kernel[2] = src[3][2].w; kernel[3] = src[4][2].w;
    kernel[4] = src[2][3].w; kernel[5] = src[4][3].w; kernel[6] = src[2][4].w;
    kernel[7] = src[3][4].w; kernel[8] = src[4][4].w;
    kernel[9] = src[3][1].w; kernel[10] = src[1][3].w; kernel[11] = src[5][3].w; kernel[12] = src[3][5].w;
    kernel[13] = src[3][6].w; kernel[14] = src[4][5].w; kernel[15] = src[2][5].w; kernel[16] = src[6][3].w;
    kernel[17] = src[5][4].w; kernel[18] = src[5][2].w; kernel[19] = src[0][3].w; kernel[20] = src[1][4].w;
    kernel[21] = src[1][2].w; kernel[22] = src[3][0].w; kernel[23] = src[4][1].w; kernel[24] = src[2][1].w;

    // 部分拉普拉斯外像素加權
    let mdiff_c0 = 0.03 + 4.0 * (abs(kernel[0] - kernel[2]) + abs(kernel[0] - kernel[4])
        + abs(kernel[0] - kernel[5]) + abs(kernel[0] - kernel[7])
        + 0.25 * (abs(kernel[0] - kernel[1]) + abs(kernel[0] - kernel[3])
            + abs(kernel[0] - kernel[6]) + abs(kernel[0] - kernel[8])));

    let mdiff_c9 = (abs(kernel[9] - kernel[2]) + abs(kernel[9] - kernel[24])
        + abs(kernel[9] - kernel[23]) + abs(kernel[9] - kernel[22])
        + 0.5 * (abs(kernel[9] - kernel[1]) + abs(kernel[9] - kernel[3])));

    let mdiff_c10 = (abs(kernel[10] - kernel[20]) + abs(kernel[10] - kernel[19])
        + abs(kernel[10] - kernel[21]) + abs(kernel[10] - kernel[4])
        + 0.5 * (abs(kernel[10] - kernel[1]) + abs(kernel[10] - kernel[6])));

    let mdiff_c11 = (abs(kernel[11] - kernel[17]) + abs(kernel[11] - kernel[5])
        + abs(kernel[11] - kernel[18]) + abs(kernel[11] - kernel[16])
        + 0.5 * (abs(kernel[11] - kernel[3]) + abs(kernel[11] - kernel[8])));

    let mdiff_c12 = (abs(kernel[12] - kernel[13]) + abs(kernel[12] - kernel[15])
        + abs(kernel[12] - kernel[7]) + abs(kernel[12] - kernel[14])
        + 0.5 * (abs(kernel[12] - kernel[6]) + abs(kernel[12] - kernel[8])));

    let weights = vec4<f32>(
        min((mdiff_c0 / mdiff_c9), 2.0),   // checker:ok — mdiff_c0 ≥ 0.03，mdiff_c9=0 時 Inf → min 得 2.0
        min((mdiff_c0 / mdiff_c10), 2.0),
        min((mdiff_c0 / mdiff_c11), 2.0),
        min((mdiff_c0 / mdiff_c12), 2.0));

    // 負拉普拉斯矩陣
    let neg_laplace = (0.25 * (kernel[2] + kernel[4] + kernel[5] + kernel[7])
        + (kernel[1] + kernel[3] + kernel[6] + kernel[8])
        + ((kernel[9] * weights.x) + (kernel[10] * weights.y)
            + (kernel[11] * weights.z) + (kernel[12] * weights.w)))
        / (5.0 + weights.x + weights.y + weights.z + weights.w);

    // 銳化幅度函數（x = 邊緣量, y = 拉普拉斯量）
    let sharpen_val = 0.01 + (uniforms.curveHeight / (curveslope * pow(edge, 3.5) + 0.5))
        - (uniforms.curveHeight / (8192.0 * pow((edge * 2.2), 4.5) + 0.5));

    let sharpdiff0 = (c0_Y - neg_laplace) * (sharpen_val * 0.8);

    // 局部近端 min/max：cocktail 排序（兩輪往返，無分支）
    for (var k: u32 = 0u; k < 2u; k++) {
        for (var i1: u32 = 1u + k; i1 < 25u - k; i1++) {
            let temp = kernel[i1 - 1u];
            kernel[i1 - 1u] = min(kernel[i1 - 1u], kernel[i1]);
            kernel[i1] = max(temp, kernel[i1]);
        }
        var rev: u32 = 23u - k;
        while (rev > k) {
            let temp = kernel[rev - 1u];
            kernel[rev - 1u] = min(kernel[rev - 1u], kernel[rev]);
            kernel[rev] = max(temp, kernel[rev]);
            if (rev == 0u) { break; }
            rev -= 1u;
        }
    }

    let nmax = max(((kernel[23] + kernel[24]) / 2.0), c0_Y);
    let nmin = min(((kernel[0] + kernel[1]) / 2.0), c0_Y);

    // tanh 縮放（正/負）
    let nmax_scale = max((1.0 / ((nmax - c0_Y) + L_OVERSHOOT)), MAX_SCALE_LIM);
    let nmin_scale = max((1.0 / ((c0_Y - nmin) + D_OVERSHOOT)), MAX_SCALE_LIM);

    // tanh 軟限幅，混合以控制最大壓縮
    let sharpdiff = mix((tanh(max(sharpdiff0, 0.0) * nmax_scale) / nmax_scale), max(sharpdiff0, 0.0), L_COMP_RATIO)
        + mix((tanh(min(sharpdiff0, 0.0) * nmin_scale) / nmin_scale), min(sharpdiff0, 0.0), D_COMP_RATIO);

    textureStore(OUTPUT, gxy, vec4<f32>(src[3][3].rgb + vec3<f32>(sharpdiff), 1.0));
}
