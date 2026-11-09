// AdaptiveSharpen — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/Sharpen/AdaptiveSharpen.hlsl
//   （Magpie 的 HLSL 移植自 libretro/slang-shaders 的 sharpen/shaders/adaptive-sharpen.slang）
// 原授權 / 作者：Magpie 效果庫為 GPL-3.0（作者 Blinue）；
//   adaptive-sharpen.slang 出自 libretro/slang-shaders
//   （Adaptive sharpen - version 2015-05-15；授權依該來源庫）。
// 本移植僅改寫 pass 結構與取樣語意；演算法常數逐字保留。
//
// Porting notes:
//   - Magpie 的這支效果是 `//!STYLE CS`（`void Pass1(uint2 blockStart, uint3 threadId)`,
//     BLOCK_SIZE 16 / NUM_THREADS 64，每執行緒負責 2×2 像素）。
//     WGFX 以 compute 實作：
//     `gxy = (Rmp8x8(local_id.x) << 1) + workgroup_id.xy * 16`
//     （Magpie 的 `blockStart` 是 workgroup 區塊的起始像素 = workgroup_id × BLOCK_SIZE），
//     並以 `//! BLOCK_SIZE 16` 決定 dispatch 網格（ceil(out/16)² 個 workgroup）。
//   - `INPUT.GatherRed/GatherGreen/GatherBlue(sam, tpos)` →
//     `textureGather(0|1|2, INPUT, sam, tpos)`。D3D 與 WGSL 的 gather 分量順序一致
//     （相對於 base texel = floor(uv*size-0.5)：x=(0,1) y=(1,1) z=(1,0) w=(0,0)），
//     故原碼的 "w z / x y" 填法逐字保留。
//   - HLSL 可對 swizzle 賦值（`src[i][j].rgb = ...` / `src[i][j].w = ...`），
//     WGSL 不允許對 swizzle 賦值，故改為整個 vec4 一次賦值（數值相同）。
//   - HLSL `lerp` → WGSL `mix`；HLSL `pow(2, x)` → `pow(2.0, x)`；
//     HLSL 的 `1 / x` → `1.0 / x`（WGSL 無 int→float 隱式轉換）。
//   - WGSL 不支援**前綴** `++` / `--`（僅支援 `i++` / `i--` 陳述句），
//     故 HLSL 的 `++i` / `--i2` 改寫為 `i++` / `i2--`（迴圈行為相同）。
//   - 取樣位置原本寫成 `(vec2<f32>(vec2<i32>(gxy) + vec2<i32>(i,j) - vec2<i32>(2))) * inputPt`，
//     該嵌套樣式會觸發 Tint 解析器的 "expected ')'"（實測），故拆成兩行
//     （`off` 與 `tpos`），數值完全相同。
//   - `#define curveslope (curveHeight*1.5f)` 等巨集依賴 uniform，
//     在 WGSL 中改為區域 `let`（值相同）。
//   - Magpie 的 `SamplerState sam` 為 POINT / CLAMP，對應
//     `//! FILTER POINT` + `//! ADDRESS CLAMP`。
//   - Magpie 的 OUTPUT 尺寸 = INPUT；WGFX 明確宣告
//     `//! WIDTH INPUT_WIDTH` / `//! HEIGHT INPUT_HEIGHT`。
//   - 原碼在 gxy 為最後一列／行時會寫入越界像素（render target 會丟棄）；
//     WGFX 端以顯式邊界檢查跳過（語意相同）。

//! VERSION 4
//! SORT_NAME AdaptiveSharpen

//! PARAMETER curveHeight
//! TYPE float
//! LABEL "Sharpness"
//! MIN 0
//! MAX 2
//! DEFAULT 0.8
//! STEP 0.01
// Main control of sharpening strength [>0]
// 0.3 <-> 2.0 is a reasonable range of values
float curveHeight;

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
//! DESC "Adaptive sharpen (2015-05-15)"
//! IN INPUT
//! OUT OUTPUT
//! BLOCK_SIZE 16
//! NUM_THREADS 64

fn Rmp8x8(a: u32) -> vec2<u32> {
    return vec2<u32>(a / 8u, a % 8u);
}

// Colour to greyscale, fast approx gamma
fn CtG(RGB: vec3<f32>) -> f32 { return sqrt((1.0 / 3.0) * ((RGB * RGB).r + (RGB * RGB).g + (RGB * RGB).b)); }

fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Magpie 的 blockStart = workgroup 區塊的起始像素 = workgroup_id * BLOCK_SIZE(16)
    let gxy = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * 16u;

    let outputSize = scene.outputSize;
    if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
        return;
    }

    let inputPt = scene.inputPt;

    var src: array<array<vec4<f32>, 8>, 8>;
    for (var i = 0; i <= 6; i += 2) {
        for (var j = 0; j <= 6; j += 2) {
            // 四角共 16 个纹素无需采样
            if ((i == 0 && j == 0) || (i == 6 && j == 0) || (i == 0 && j == 6) || (i == 6 && j == 6)) {
                continue;
            }

            let off = vec2<i32>(gxy) + vec2<i32>(i, j) - vec2<i32>(2);
            let tpos = vec2<f32>(off) * inputPt;
            let sr = textureGather(0, INPUT, sam, tpos);
            let sg = textureGather(1, INPUT, sam, tpos);
            let sb = textureGather(2, INPUT, sam, tpos);

            // w z
            // x y
            let c00 = vec3<f32>(sr.w, sg.w, sb.w);
            src[i][j] = vec4<f32>(c00, CtG(c00));
            let c01 = vec3<f32>(sr.x, sg.x, sb.x);
            src[i][j + 1] = vec4<f32>(c01, CtG(c01));
            let c10 = vec3<f32>(sr.z, sg.z, sb.z);
            src[i + 1][j] = vec4<f32>(c10, CtG(c10));
            let c11 = vec3<f32>(sr.y, sg.y, sb.y);
            src[i + 1][j + 1] = vec4<f32>(c11, CtG(c11));
        }
    }

    for (var i = 0; i <= 1; i++) {
        for (var j = 0; j <= 1; j++) {
            let destPos = gxy + vec2<u32>(u32(i), u32(j));

            // Get points and saturate out of range values (BTB & WTW)
            // [                c22               ]
            // [           c24, c9,  c23          ]
            // [      c21, c1,  c2,  c3, c18      ]
            // [ c19, c10, c4,  c0,  c5, c11, c16 ]
            // [      c20, c6,  c7,  c8, c17      ]
            // [           c15, c12, c14          ]
            // [                c13               ]

            // Blur, gauss 3x3
            let blur = (2.0 * (src[i + 3][j + 2].rgb + src[i + 2][j + 3].rgb + src[i + 4][j + 3].rgb + src[i + 3][j + 4].rgb) + (src[i + 2][j + 2].rgb + src[i + 4][j + 2].rgb + src[i + 2][j + 4].rgb + src[i + 4][j + 4].rgb) + 4.0 * src[i + 3][j + 3].rgb) / 16.0;
            let blur_Y = (blur.r * (1.0 / 3.0) + blur.g * (1.0 / 3.0) + blur.b * (1.0 / 3.0));

            // Edge detection
            // Matrix, relative weights
            // [           1          ]
            // [       4,  4,  4      ]
            // [   1,  4,  4,  4,  1  ]
            // [       4,  4,  4      ]
            // [           1          ]
            var edge = length(abs(blur - src[i + 3][j + 3].rgb) + abs(blur - src[i + 2][j + 2].rgb) + abs(blur - src[i + 3][j + 2].rgb) + abs(blur - src[i + 4][j + 2].rgb)
                + abs(blur - src[i + 2][j + 3].rgb) + abs(blur - src[i + 4][j + 3].rgb) + abs(blur - src[i + 2][j + 4].rgb) + abs(blur - src[i + 3][j + 4].rgb) + abs(blur - src[i + 4][j + 4].rgb)
                + 0.25 * (abs(blur - src[i + 3][j + 1].rgb) + abs(blur - src[i + 1][j + 3].rgb) + abs(blur - src[i + 5][j + 3].rgb) + abs(blur - src[i + 3][j + 5].rgb))) * (1.0 / 3.0);

            // Edge detect contrast compression, center = 0.5
            edge *= min((0.8 + 2.7 * pow(2.0, (-7.4 * blur_Y))), 3.2);

            // RGB to greyscale
            let c0_Y = src[i + 3][j + 3].w;

            var kernel: array<f32, 25> = array<f32, 25>(
                c0_Y,  src[i + 2][j + 2].w, src[i + 3][j + 2].w, src[i + 4][j + 2].w, src[i + 2][j + 3].w, src[i + 4][j + 3].w, src[i + 2][j + 4].w, src[i + 3][j + 4].w, src[i + 4][j + 4].w,
                src[i + 3][j + 1].w, src[i + 1][j + 3].w, src[i + 5][j + 3].w, src[i + 3][j + 5].w, src[i + 3][j + 6].w, src[i + 4][j + 5].w, src[i + 2][j + 5].w, src[i + 6][j + 3].w,
                src[i + 5][j + 4].w, src[i + 5][j + 2].w, src[i][j + 3].w, src[i + 1][j + 4].w, src[i + 1][j + 2].w, src[i + 3][j].w, src[i + 4][j + 1].w, src[i + 2][j + 1].w
            );

            // Partial laplacian outer pixel weighting scheme
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

            let weights = vec4<f32>((min((mdiff_c0 / mdiff_c9), 2.0)), (min((mdiff_c0 / mdiff_c10), 2.0)),
                (min((mdiff_c0 / mdiff_c11), 2.0)), (min((mdiff_c0 / mdiff_c12), 2.0)));

            // Negative laplace matrix
            // Matrix, relative weights, *Varying 0<->8
            // [          8*         ]
            // [      4,  1,  4      ]
            // [  8*, 1,      1,  8* ]
            // [      4,  1,  4      ]
            // [          8*         ]
            let neg_laplace = (0.25 * (kernel[2] + kernel[4] + kernel[5] + kernel[7])
                + (kernel[1] + kernel[3] + kernel[6] + kernel[8])
                + ((kernel[9] * weights.x) + (kernel[10] * weights.y)
                    + (kernel[11] * weights.z) + (kernel[12] * weights.w)))
                / (5.0 + weights.x + weights.y + weights.z + weights.w);

            // Compute sharpening magnitude function, x = edge mag, y = laplace operator mag
            let curveslope = uniforms.curveHeight * 1.5;   // Sharpening curve slope, edge region
            const D_overshoot: f32 = 0.016;                 // Max dark overshoot before max compression
            const D_comp_ratio: f32 = 0.250;                // Max compression ratio, dark overshoot (1/0.25=4x)
            const L_overshoot: f32 = 0.004;                 // Max light overshoot before max compression
            const L_comp_ratio: f32 = 0.167;                // Max compression ratio, light overshoot (1/0.167=6x)
            const max_scale_lim: f32 = 10.0;                // Abs change before max compression (1/10=±10%)

            var sharpen_val = 0.01 + (uniforms.curveHeight / (curveslope * pow(edge, 3.5) + 0.5))
                - (uniforms.curveHeight / (8192.0 * pow((edge * 2.2), 4.5) + 0.5));

            // Calculate sharpening diff and scale
            var sharpdiff = (c0_Y - neg_laplace) * (sharpen_val * 0.8);

            // Calculate local near min & max, partial cocktail sort (No branching!)
            for (var k = 0; k < 2; k++) {
                for (var i1 = 1 + k; i1 < 25 - k; i1++) {
                    let temp = kernel[i1 - 1];
                    kernel[i1 - 1] = min(kernel[i1 - 1], kernel[i1]);
                    kernel[i1] = max(temp, kernel[i1]);
                }

                for (var i2 = 23 - k; i2 > k; i2--) {
                    let temp = kernel[i2 - 1];
                    kernel[i2 - 1] = min(kernel[i2 - 1], kernel[i2]);
                    kernel[i2] = max(temp, kernel[i2]);
                }
            }

            let nmax = max(((kernel[23] + kernel[24]) / 2.0), c0_Y);
            let nmin = min(((kernel[0] + kernel[1]) / 2.0), c0_Y);

            // Calculate tanh scale factor, pos/neg
            let nmax_scale = max((1.0 / ((nmax - c0_Y) + L_overshoot)), max_scale_lim);
            let nmin_scale = max((1.0 / ((c0_Y - nmin) + D_overshoot)), max_scale_lim);

            // Soft limit sharpening with tanh, mix to control maximum compression
            sharpdiff = mix((tanh((max(sharpdiff, 0.0)) * nmax_scale) / nmax_scale), (max(sharpdiff, 0.0)), L_comp_ratio)
                + mix((tanh((min(sharpdiff, 0.0)) * nmin_scale) / nmin_scale), (min(sharpdiff, 0.0)), D_comp_ratio);

            if (destPos.x < outputSize.x && destPos.y < outputSize.y) {
                textureStore(OUTPUT, vec2<i32>(destPos), vec4<f32>(src[i + 3][j + 3].rgb + sharpdiff, 1.0));
            }
        }
    }
}

//! END
