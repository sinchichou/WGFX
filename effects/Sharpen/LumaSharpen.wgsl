// LumaSharpen — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/Sharpen/LumaSharpen.hlsl
//   （Magpie 的 HLSL 移植自 SweetFX 的 Shaders/LumaSharpen.fx）
// 原授權 / 作者：LumaSharpen v1.5.0 by Christian Cann Schuldt Jensen ~ CeeJay.dk
//   （SweetFX License，見 https://github.com/CeeJayDK/SweetFX）；
//   Magpie 效果庫為 GPL-3.0（作者 Blinue）。
// 本移植僅改寫 pass 結構與取樣語意；取樣樣式與所有常數逐字保留。
//
// Porting notes:
//   - Magpie 的 `//!STYLE PS` pass 傳入的 `pos` 是**正規化 UV**（輸出 texel 中心
//     (x+0.5)/outputSize）。WGFX 端：`pos = (gid.xy + 0.5) * scene.outputPt`；
//     原碼 `GetInputPt()` → `scene.inputPt`（一個 input texel 的 UV 尺寸），
//     偏移量完全照抄。
//   - `INPUT.SampleLevel(sam, uv, 0)` → `textureSampleLevel(INPUT, sam, uv, 0.0)`
//     （compute stage 不可用隱含 LOD 的 textureSample）。
//   - `#define CoefLuma float3(0.2126, 0.7152, 0.0722)` → 直接寫成
//     `vec3<f32>(0.2126, 0.7152, 0.0722)`。
//   - 原碼四個取樣樣式是四個互斥的 `if (pattern == N)` 區塊（每個區塊都會
//     **賦值** blur_ori），WGSL 保留同樣的 `if` 結構（不是 else-if），
//     以完全對應原碼。
//   - `float3 + float` 的隱式廣播在 WGSL 合法；`saturate` / `dot` / `clamp`
//     兩邊同名同義。
//   - Magpie 的 `SamplerState sam` 為 LINEAR / CLAMP，對應
//     `//! FILTER LINEAR` + `//! ADDRESS CLAMP`。
//   - Magpie 的 OUTPUT 尺寸 = INPUT；WGFX 明確宣告
//     `//! WIDTH INPUT_WIDTH` / `//! HEIGHT INPUT_HEIGHT`。
//   - 註：原碼 `0.5 / sharpClamp` 在 sharpClamp = 0 時為 Inf（NaN 風險）；
//     此為原演算法行為，未加保護（預設值 0.035 不會觸發）。

//! VERSION 4
//! SORT_NAME LumaSharpen

//! PARAMETER sharpStrength
//! TYPE float
//! LABEL "Sharpening Strength"
//! MIN 0
//! MAX 3
//! DEFAULT 0.65
//! STEP 0.01
float sharpStrength;

//! PARAMETER sharpClamp
//! TYPE float
//! LABEL "Sharpening Limit"
//! MIN 0
//! MAX 1
//! DEFAULT 0.035
//! STEP 0.005
// Limits maximum amount of sharpening a pixel receives
// This helps avoid "haloing" artifacts which would otherwise occur when you raised the strength too much.
float sharpClamp;

//! PARAMETER pattern
//! TYPE int
//! LABEL "Sample Pattern"
//! MIN 0
//! MAX 3
//! DEFAULT 1
//! STEP 1
// 0 : Fast
// 1 : Normal
// 2 : Wider
// 3 : Pyramid shaped
int pattern;

//! PARAMETER offsetBias
//! TYPE float
//! LABEL "Offset Bias"
//! MIN 0
//! MAX 6
//! DEFAULT 1
//! STEP 0.01
// Offset bias adjusts the radius of the sampling pattern.I designed the pattern for an offset bias of 1.0, but feel free to experiment.
float offsetBias;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! SAMPLER
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! DESC "Luma sharpen (4 sampling patterns, luma-only, halo clamp)"
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outSize = scene.outputSize;
    if (gid.x >= outSize.x || gid.y >= outSize.y) {
        return;
    }
    // Magpie PS 的 pos = 正規化 UV（輸出 texel 中心）
    let pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.outputPt;

    let BUFFER_PIXEL_SIZE = scene.inputPt;

    // -- Get the original pixel --
    let ori = textureSampleLevel(INPUT, sam, pos, 0.0).rgb; // ori = original pixel

    // -- Combining the strength and luma multipliers --
    // CoefLuma: BT.709 & sRGB luma coefficient (Monitors and HD Television)
    var sharp_strength_luma = vec3<f32>(0.2126, 0.7152, 0.0722) * uniforms.sharpStrength;

    /*-----------------------------------------------------------.
    /                       Sampling patterns                     /
    '-----------------------------------------------------------*/
    var blur_ori = vec3<f32>(0.0);

    //   [ NW,   , NE ] Each texture lookup (except ori)
    //   [   ,ori,    ] samples 4 pixels
    //   [ SW,   , SE ]

    // -- Pattern 1 -- A (fast) 7 tap gaussian using only 2+1 texture fetches.
    if (uniforms.pattern == 0) {
        // -- Gaussian filter --
        //   [ 1/9, 2/9,    ]     [ 1 , 2 ,   ]
        //   [ 2/9, 8/9, 2/9]  =  [ 2 , 8 , 2 ]
        //   [    , 2/9, 1/9]     [   , 2 , 1 ]

        blur_ori = textureSampleLevel(INPUT, sam, pos + (BUFFER_PIXEL_SIZE / 3.0) * uniforms.offsetBias, 0.0).rgb;  // North West
        blur_ori += textureSampleLevel(INPUT, sam, pos + (-BUFFER_PIXEL_SIZE / 3.0) * uniforms.offsetBias, 0.0).rgb; // South East

        //blur_ori += tex2D(ReShade::BackBuffer, tex + (BUFFER_PIXEL_SIZE / 3.0) * offsetBias); // North East
        //blur_ori += tex2D(ReShade::BackBuffer, tex + (-BUFFER_PIXEL_SIZE / 3.0) * offsetBias); // South West

        blur_ori /= 2;  //Divide by the number of texture fetches

        sharp_strength_luma *= 1.5; // Adjust strength to aproximate the strength of pattern 2
    }

    // -- Pattern 2 -- A 9 tap gaussian using 4+1 texture fetches.
    if (uniforms.pattern == 1) {
        // -- Gaussian filter --
        //   [ .25, .50, .25]     [ 1 , 2 , 1 ]
        //   [ .50,   1, .50]  =  [ 2 , 4 , 2 ]
        //   [ .25, .50, .25]     [ 1 , 2 , 1 ]

        blur_ori = textureSampleLevel(INPUT, sam, pos + vec2<f32>(BUFFER_PIXEL_SIZE.x, -BUFFER_PIXEL_SIZE.y) * 0.5 * uniforms.offsetBias, 0.0).rgb; // South East
        blur_ori += textureSampleLevel(INPUT, sam, pos - BUFFER_PIXEL_SIZE * 0.5 * uniforms.offsetBias, 0.0).rgb;  // South West
        blur_ori += textureSampleLevel(INPUT, sam, pos + BUFFER_PIXEL_SIZE * 0.5 * uniforms.offsetBias, 0.0).rgb; // North East
        blur_ori += textureSampleLevel(INPUT, sam, pos - vec2<f32>(BUFFER_PIXEL_SIZE.x, -BUFFER_PIXEL_SIZE.y) * 0.5 * uniforms.offsetBias, 0.0).rgb; // North West

        blur_ori *= 0.25;  // ( /= 4) Divide by the number of texture fetches
    }

    // -- Pattern 3 -- An experimental 17 tap gaussian using 4+1 texture fetches.
    if (uniforms.pattern == 2) {
        // -- Gaussian filter --
        //   [   , 4 , 6 ,   ,   ]
        //   [   ,16 ,24 ,16 , 4 ]
        //   [ 6 ,24 ,   ,24 , 6 ]
        //   [ 4 ,16 ,24 ,16 ,   ]
        //   [   ,   , 6 , 4 ,   ]

        blur_ori = textureSampleLevel(INPUT, sam, pos + BUFFER_PIXEL_SIZE * vec2<f32>(0.4, -1.2) * uniforms.offsetBias, 0.0).rgb;  // South South East
        blur_ori += textureSampleLevel(INPUT, sam, pos - BUFFER_PIXEL_SIZE * vec2<f32>(1.2, 0.4) * uniforms.offsetBias, 0.0).rgb; // West South West
        blur_ori += textureSampleLevel(INPUT, sam, pos + BUFFER_PIXEL_SIZE * vec2<f32>(1.2, 0.4) * uniforms.offsetBias, 0.0).rgb; // East North East
        blur_ori += textureSampleLevel(INPUT, sam, pos - BUFFER_PIXEL_SIZE * vec2<f32>(0.4, -1.2) * uniforms.offsetBias, 0.0).rgb; // North North West

        blur_ori *= 0.25;  // ( /= 4) Divide by the number of texture fetches

        sharp_strength_luma *= 0.51;
    }

    // -- Pattern 4 -- A 9 tap high pass (pyramid filter) using 4+1 texture fetches.
    if (uniforms.pattern == 3) {
        // -- Gaussian filter --
        //   [ .50, .50, .50]     [ 1 , 1 , 1 ]
        //   [ .50,    , .50]  =  [ 1 ,   , 1 ]
        //   [ .50, .50, .50]     [ 1 , 1 , 1 ]

        blur_ori = textureSampleLevel(INPUT, sam, pos + vec2<f32>(0.5 * BUFFER_PIXEL_SIZE.x, -BUFFER_PIXEL_SIZE.y * uniforms.offsetBias), 0.0).rgb;  // South South East
        blur_ori += textureSampleLevel(INPUT, sam, pos + vec2<f32>(uniforms.offsetBias * -BUFFER_PIXEL_SIZE.x, 0.5 * -BUFFER_PIXEL_SIZE.y), 0.0).rgb; // West South West
        blur_ori += textureSampleLevel(INPUT, sam, pos + vec2<f32>(uniforms.offsetBias * BUFFER_PIXEL_SIZE.x, 0.5 * BUFFER_PIXEL_SIZE.y), 0.0).rgb; // East North East
        blur_ori += textureSampleLevel(INPUT, sam, pos + vec2<f32>(0.5 * -BUFFER_PIXEL_SIZE.x, BUFFER_PIXEL_SIZE.y * uniforms.offsetBias), 0.0).rgb; // North North West

        //blur_ori += (2 * ori); // Probably not needed. Only serves to lessen the effect.

        blur_ori /= 4.0;  //Divide by the number of texture fetches

        sharp_strength_luma *= 0.666; // Adjust strength to aproximate the strength of pattern 2
    }

    /*-----------------------------------------------------------.
   /                            Sharpen                          /
   '-----------------------------------------------------------*/

    // -- Calculate the sharpening --
    let sharp = ori - blur_ori;  //Subtracting the blurred image from the original image

    // -- Adjust strength of the sharpening and clamp it--
    let sharp_strength_luma_clamp = vec4<f32>(sharp_strength_luma * (0.5 / uniforms.sharpClamp), 0.5); //Roll part of the clamp into the dot

    //sharp_luma = saturate((0.5 / sharpClamp) * sharp_luma + 0.5); //scale up and clamp
    var sharp_luma = saturate(dot(vec4<f32>(sharp, 1.0), sharp_strength_luma_clamp)); //Calculate the luma, adjust the strength, scale up and clamp
    sharp_luma = (uniforms.sharpClamp * 2.0) * sharp_luma - uniforms.sharpClamp; //scale down

    // -- Combining the values to get the final sharpened pixel	--
    let outputcolor = ori + sharp_luma;    // Add the sharpening to the the original.

    /*-----------------------------------------------------------.
   /                     Returning the output                    /
   '-----------------------------------------------------------*/

    textureStore(OUTPUT, vec2<i32>(gid.xy), vec4<f32>(saturate(outputcolor), 1.0));
}

//! END
