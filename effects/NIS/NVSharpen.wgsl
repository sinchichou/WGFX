// NVSharpen (NVIDIA Image Scaling — adaptive directional sharpening) — Magpie v4 → WGFX WGSL port
//
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/NIS/NVSharpen.hlsl
//   which includes NIS_Scaler.hlsli (NVIDIA Image Scaling SDK v1.0.3), NIS_SCALER 0 branch:
//     https://github.com/NVIDIAGameWorks/NVIDIAImageScaling/blob/35e13ba316c98eeecf16f37eae70ce88019911f6/NIS/NIS_Scaler.h
//
// Original license / author:
//   NVIDIA Image Scaling SDK — Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. (MIT)
//   Magpie HLSL port — Copyright (c) Blinue and Magpie contributors. (MIT)
//   This WGFX WGSL port changes only the pass structure and the sampling idioms;
//   every NIS constant and every line of the filter math is copied verbatim.
//
// Porting notes (語意偏離與處理方式):
//   1. Magpie's `void Pass1(uint2 blockStart, uint3 threadId)` is a CS-style pass
//      (`//! BLOCK_SIZE 32,32` + `//! NUM_THREADS 256`); `blockStart` =
//      workgroup_id.xy * BLOCK_SIZE and `threadId.x` = local_invocation_id.x. The
//      framework derives the dispatch grid from BLOCK_SIZE, so no viewport guard is
//      needed (Magpie only has one behind NIS_VIEWPORT_SUPPORT, which is off).
//   2. `NVSHARED NVF shPixelsY[38][38]` → `var<workgroup> shPixelsY:
//      array<array<f32, 38>, 38>` (5,776 B, under WebGPU's 16,384 B default
//      `maxComputeWorkgroupStorageSize`). `GroupMemoryBarrierWithGroupSync()` →
//      `workgroupBarrier()`.
//   3. `NVH`/`NVH4` (min16float) → `f32`, per docs/porting-guide.md §6.6
//      (`//! CAPABILITY FP16` is only a marker). This shader uses the f32 NVF path
//      anyway: NIS_USE_HALF_PRECISION only affects the NVScaler coefficient banks.
//   4. Compile-time switches are Magpie's own defaults: NIS_SCALER 0,
//      NIS_USE_HALF_PRECISION 0 (f32), NIS_HDR_MODE 0, NIS_NV12_SUPPORT 0,
//      NIS_VIEWPORT_SUPPORT 0, NIS_CLAMP_OUTPUT 0.
//   5. `NVTEX_SAMPLE(in_texture, samplerLinearClamp, pos)` →
//      `textureSampleLevel(INPUT, samplerLinearClamp, pos, 0.0)` (the HLSL uses
//      SampleLevel with a LINEAR+CLAMP sampler; NVSharpen does not use gather).
//   6. `static const float sharpen_slider = sharpness - 0.5f;` and the eight
//      constants derived from it depend on the `sharpness` uniform. WGSL forbids a
//      module-scope `const` that references a uniform, so they are computed at the
//      top of `Pass1` and passed to the helpers in a `SharpConsts` struct. Values are
//      unchanged; only the declaration site moved.
//   7. `mul`/`mad` → `a * b (+ c)`; `lerp` → `mix`; `saturate`/`min`/`max`/`abs`
//      keep their names. HLSL's integer division in `kSupportSize / 2` and
//      `kSupportSize / 2 - 1` is integer division here too (`kShift`, the GetEdgeMap
//      window offset 1).
//   8. HLSL's `#define kNumPixelsX (NIS_BLOCK_WIDTH + kSupportSize + 1)` = 38 also
//      fixes the tile pitch; the load loop fills the whole tile for any scale, and
//      this effect is 1:1 (OUTPUT is declared INPUT_WIDTH x INPUT_HEIGHT).

//! VERSION 4
//! SORT_NAME NVSharpen
//! CAPABILITY FP16

//! PARAMETER sharpness
//! TYPE float
//! LABEL "Sharpness"
//! MIN 0
//! MAX 1
//! DEFAULT 0.5
//! STEP 0.01
float sharpness;

//! TEXTURE
Texture2D INPUT;

//! TEXTURE
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! SAMPLER
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState samplerLinearClamp;

//! PASS 1
//! DESC "NVIDIA Image Scaling — NVSharpen (adaptive directional sharpening, 1:1)"
//! IN INPUT
//! OUT OUTPUT
//! BLOCK_SIZE 32,32
//! NUM_THREADS 256

// ---- NIS shader constants (verbatim; none depend on a uniform) ----
const kDetectRatio: f32 = 2.0 * 1127.0 / 1024.0;
const kDetectThres: f32 = 64.0 / 1024.0;
const kEps: f32 = 1.0 / 255.0;
const kMinContrastRatio: f32 = 2.0;
const kMaxContrastRatio: f32 = 10.0;
const kRatioNorm: f32 = 1.0 / (kMaxContrastRatio - kMinContrastRatio);
const kContrastBoost: f32 = 1.0;
const kSharpStartY: f32 = 0.45;
const kSharpEndY: f32 = 0.9;
const kSharpScaleY: f32 = 1.0 / (kSharpEndY - kSharpStartY);

// `#define kSupportSize 5`, `#define kNumPixelsX (NIS_BLOCK_WIDTH + kSupportSize + 1)` = 38.
const kSupportSize: i32 = 5;
const kNumPixels: i32 = 38;
const kShift: f32 = -1.5; // 0.5 - kSupportSize / 2 = 0.5 - 5/2 = 0.5 - 2 (整數除法)

var<workgroup> shPixelsY: array<array<f32, 38>, 38>;

// `sharpness`-derived constants (see porting note 6).
struct SharpConsts {
    strength_min: f32,
    strength_scale: f32,
    limit_min: f32,
    limit_scale: f32,
};

// NVF getY(NVF3 rgba) — NIS_HDR_MODE_NONE.
fn getY(rgba: vec3<f32>) -> f32 {
    return 0.2126 * rgba.x + 0.7152 * rgba.y + 0.0722 * rgba.z;
}

// NVF4 GetEdgeMap(NVF p[5][5], NVI i, NVI j) — evaluated over the 3x3 window at
// (oi, oj). Verbatim apart from the window offset.
fn GetEdgeMapAt(p: ptr<function, array<array<f32, 5>, 5>>, oi: u32, oj: u32) -> vec4<f32> {
    let a = *p;

    let g_0 = abs(a[oi][oj] + a[oi][oj + 1u] + a[oi][oj + 2u] - a[oi + 2u][oj] - a[oi + 2u][oj + 1u] - a[oi + 2u][oj + 2u]);
    let g_45 = abs(a[oi + 1u][oj] + a[oi][oj] + a[oi][oj + 1u] - a[oi + 2u][oj + 1u] - a[oi + 2u][oj + 2u] - a[oi + 1u][oj + 2u]);
    let g_90 = abs(a[oi][oj] + a[oi + 1u][oj] + a[oi + 2u][oj] - a[oi][oj + 2u] - a[oi + 1u][oj + 2u] - a[oi + 2u][oj + 2u]);
    let g_135 = abs(a[oi + 1u][oj] + a[oi + 2u][oj] + a[oi + 2u][oj + 1u] - a[oi][oj + 1u] - a[oi][oj + 2u] - a[oi + 1u][oj + 2u]);

    let g_0_90_max = max(g_0, g_90);
    let g_0_90_min = min(g_0, g_90);
    let g_45_135_max = max(g_45, g_135);
    let g_45_135_min = min(g_45, g_135);

    var e_0_90: f32 = 0.0;
    var e_45_135: f32 = 0.0;

    if (g_0_90_max + g_45_135_max == 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    e_0_90 = min(g_0_90_max / (g_0_90_max + g_45_135_max), 1.0);
    e_45_135 = 1.0 - e_0_90;

    let c_0_90 = (g_0_90_max > (g_0_90_min * kDetectRatio)) && (g_0_90_max > kDetectThres) && (g_0_90_max > g_45_135_min);
    let c_45_135 = (g_45_135_max > (g_45_135_min * kDetectRatio)) && (g_45_135_max > kDetectThres) && (g_45_135_max > g_0_90_min);
    let c_g_0_90 = g_0_90_max == g_0;
    let c_g_45_135 = g_45_135_max == g_45;

    let f_e_0_90: f32 = select(1.0, e_0_90, c_0_90 && c_45_135);
    let f_e_45_135: f32 = select(1.0, e_45_135, c_0_90 && c_45_135);

    let weight_0: f32 = select(0.0, f_e_0_90, c_0_90 && c_g_0_90);
    let weight_90: f32 = select(0.0, f_e_0_90, c_0_90 && !c_g_0_90);
    let weight_45: f32 = select(0.0, f_e_45_135, c_45_135 && c_g_45_135);
    let weight_135: f32 = select(0.0, f_e_45_135, c_45_135 && !c_g_45_135);

    return vec4<f32>(weight_0, weight_90, weight_45, weight_135);
}

// NVF CalcLTIFast(const NVF y[5])
fn CalcLTIFast(y0: f32, y1: f32, y2: f32, y3: f32, y4: f32) -> f32 {
    let a_min: f32 = min(min(y0, y1), y2);
    let a_max: f32 = max(max(y0, y1), y2);

    let b_min: f32 = min(min(y2, y3), y4);
    let b_max: f32 = max(max(y2, y3), y4);

    let a_cont: f32 = a_max - a_min;
    let b_cont: f32 = b_max - b_min;

    let cont_ratio: f32 = max(a_cont, b_cont) / (min(a_cont, b_cont) + kEps);
    return (1.0 - saturate((cont_ratio - kMinContrastRatio) * kRatioNorm)) * kContrastBoost;
}

// NVF EvalUSM(const NVF pxl[5], const NVF sharpnessStrength, const NVF sharpnessLimit)
fn EvalUSM(pxl: ptr<function, array<f32, 5>>, sharpnessStrength: f32, sharpnessLimit: f32) -> f32 {
    let x = *pxl;
    // USM profile
    var y_usm: f32 = -0.6001 * x[1] + 1.2002 * x[2] - 0.6001 * x[3];
    // boost USM profile
    y_usm *= sharpnessStrength;
    // clamp to the limit
    y_usm = min(sharpnessLimit, max(-sharpnessLimit, y_usm));
    // reduce ringing
    y_usm *= CalcLTIFast(x[0], x[1], x[2], x[3], x[4]);

    return y_usm;
}

// NVF4 GetDirUSM(const NVF p[5][5])
fn GetDirUSM(p: ptr<function, array<array<f32, 5>, 5>>, sc: SharpConsts) -> vec4<f32> {
    let a = *p;
    // sharpness boost & limit are the same for all directions
    let scaleY: f32 = 1.0 - saturate((a[2u][2u] - kSharpStartY) * kSharpScaleY);
    // scale the ramp to sharpen as a function of luma
    let sharpnessStrength: f32 = scaleY * sc.strength_scale + sc.strength_min;
    // scale the ramp to limit USM as a function of luma
    let sharpnessLimit: f32 = (scaleY * sc.limit_scale + sc.limit_min) * a[2u][2u];

    var rval: vec4<f32>;

    // 0 deg filter
    var interp0Deg: array<f32, 5>;
    for (var i: i32 = 0; i < 5; i++) {
        interp0Deg[u32(i)] = a[u32(i)][2u];
    }
    rval.x = EvalUSM(&interp0Deg, sharpnessStrength, sharpnessLimit);

    // 90 deg filter
    var interp90Deg: array<f32, 5>;
    for (var i: i32 = 0; i < 5; i++) {
        interp90Deg[u32(i)] = a[2u][u32(i)];
    }
    rval.y = EvalUSM(&interp90Deg, sharpnessStrength, sharpnessLimit);

    // 45 deg filter
    var interp45Deg: array<f32, 5>;
    interp45Deg[0u] = a[1u][1u];
    interp45Deg[1u] = mix(a[2u][1u], a[1u][2u], 0.5);
    interp45Deg[2u] = a[2u][2u];
    interp45Deg[3u] = mix(a[3u][2u], a[2u][3u], 0.5);
    interp45Deg[4u] = a[3u][3u];
    rval.z = EvalUSM(&interp45Deg, sharpnessStrength, sharpnessLimit);

    // 135 deg filter
    var interp135Deg: array<f32, 5>;
    interp135Deg[0u] = a[3u][1u];
    interp135Deg[1u] = mix(a[3u][2u], a[2u][1u], 0.5);
    interp135Deg[2u] = a[2u][2u];
    interp135Deg[3u] = mix(a[2u][3u], a[1u][2u], 0.5);
    interp135Deg[4u] = a[1u][3u];
    rval.w = EvalUSM(&interp135Deg, sharpnessStrength, sharpnessLimit);

    return rval;
}

// void NVSharpen(NVU2 blockIdx, NVU threadIdx)  →  compute entry point
fn Pass1(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let threadIdx: u32 = lid.x;
    let blockIdx: vec2<u32> = wg.xy;

    // static const float sharpen_slider = sharpness - 0.5f; (see porting note 6)
    let sharpen_slider: f32 = uniforms.sharpness - 0.5;
    let MinScale: f32 = select(1.0, 1.25, sharpen_slider >= 0.0);
    let MaxScale: f32 = select(1.75, 1.25, sharpen_slider >= 0.0);
    let kSharpStrengthMin: f32 = max(0.0, 0.4 + sharpen_slider * MinScale * 1.2);
    let kSharpStrengthMax: f32 = 1.6 + sharpen_slider * MaxScale * 1.8;
    let kSharpStrengthScale: f32 = kSharpStrengthMax - kSharpStrengthMin;
    let LimitScale: f32 = select(1.0, 1.25, sharpen_slider >= 0.0);
    let kSharpLimitMin: f32 = max(0.1, 0.14 + sharpen_slider * LimitScale * 0.32);
    let kSharpLimitMax: f32 = 0.5 + sharpen_slider * LimitScale * 0.6;
    let kSharpLimitScale: f32 = kSharpLimitMax - kSharpLimitMin;
    let sc = SharpConsts(kSharpStrengthMin, kSharpStrengthScale, kSharpLimitMin, kSharpLimitScale);

    // static const float kSrcNormX = GetInputPt().x; ...
    let kSrcNormX: f32 = scene.inputPt.x;
    let kSrcNormY: f32 = scene.inputPt.y;

    let dstBlockX: i32 = i32(32u * blockIdx.x);
    let dstBlockY: i32 = i32(32u * blockIdx.y);

    // fill in input luma tile in batches of 2x2 pixels
    for (var i: i32 = i32(threadIdx) * 2; i < kNumPixels * kNumPixels / 2; i += 512) {
        let posx: i32 = i % kNumPixels;
        let posy: i32 = i / kNumPixels * 2;
        for (var dy: i32 = 0; dy < 2; dy++) {
            for (var dx: i32 = 0; dx < 2; dx++) {
                let tx: f32 = (f32(dstBlockX + posx + dx) + kShift) * kSrcNormX;
                let ty: f32 = (f32(dstBlockY + posy + dy) + kShift) * kSrcNormY;
                let px: vec4<f32> = textureSampleLevel(INPUT, samplerLinearClamp, vec2<f32>(tx, ty), 0.0);
                shPixelsY[u32(posy + dy)][u32(posx + dx)] = getY(vec3<f32>(px.x, px.y, px.z));
            }
        }
    }

    workgroupBarrier();

    // NIS_BLOCK_WIDTH * NIS_BLOCK_HEIGHT / NIS_THREAD_GROUP_SIZE = 32*32/256 = 4
    for (var k: i32 = i32(threadIdx); k < 1024; k += 256) {
        let posx: i32 = k % 32;
        let posy: i32 = k / 32;

        // load 5x5 support to regs
        var p: array<array<f32, 5>, 5>;
        for (var i: i32 = 0; i < 5; i++) {
            for (var j: i32 = 0; j < 5; j++) {
                p[u32(i)][u32(j)] = shPixelsY[u32(posy + i)][u32(posx + j)];
            }
        }

        // get directional filter bank output
        let dirUSM: vec4<f32> = GetDirUSM(&p, sc);

        // generate weights for directional filters
        // (kSupportSize / 2 - 1 = 1; integer division)
        let w: vec4<f32> = GetEdgeMapAt(&p, 1u, 1u);

        // final USM is a weighted sum filter outputs
        let usmY: f32 = dirUSM.x * w.x + dirUSM.y * w.y + dirUSM.z * w.z + dirUSM.w * w.w;

        // do bilinear tap and correct rgb texel so it produces new sharpened luma
        let dstX: i32 = dstBlockX + posx;
        let dstY: i32 = dstBlockY + posy;

        let coord: vec2<f32> = vec2<f32>((f32(dstX) + 0.5) * kSrcNormX, (f32(dstY) + 0.5) * kSrcNormY);
        var op: vec4<f32> = textureSampleLevel(INPUT, samplerLinearClamp, coord, 0.0);
        op = vec4<f32>(op.x + usmY, op.y + usmY, op.z + usmY, op.w);

        // NVTEX_STORE(out_texture, dstCoord, MF4(NVCLAMP(op)))
        // NIS_CLAMP_OUTPUT = 0 => NVCLAMP(op) = op
        textureStore(OUTPUT, vec2<i32>(dstX, dstY), op);
    }
}
//! END
