// NIS (NVIDIA Image Scaling — NVScaler) — Magpie v4 → WGFX WGSL port
//
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/NIS/NIS.hlsl
//   which includes NIS_Scaler.hlsli (NVIDIA Image Scaling SDK v1.0.3):
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
//      (`//! BLOCK_SIZE 32,32` + `//! NUM_THREADS 256`). It maps onto a WGFX compute
//      shader with `blockStart` = workgroup_id.xy * BLOCK_SIZE and `threadId.x` =
//      local_invocation_id.x; the framework also derives the dispatch grid from
//      BLOCK_SIZE (ceil(outSize / 32) per axis), so no viewport guard is needed.
//   2. **The groupshared edge map (`shEdgeMap`, 34x34 float4 = 18,496 B) is NOT
//      materialised.** Together with the groupshared luma tile (38x38 float =
//      5,776 B) it would need 27,344 B of workgroup storage, above WebGPU's default
//      `maxComputeWorkgroupStorageSize` (16,384 B, and the device is created without
//      raised limits). The shared coefficient banks (2 x 64 x 6 float = 3,072 B) are
//      kept, so workgroup usage is 8,848 B.
//      Instead each thread recomputes the 2x2 edge neighbourhood it needs directly
//      from the shared luma tile: `GetEdgeMap` uses the 3x3 window at (oi..oi+2,
//      oj..oj+2), and all four windows required by an output pixel are sub-windows
//      of the 6x6 support the thread already loads. Same luma inputs, same
//      expressions => bit-identical edge values; only the storage location differs.
//      The second thread-parallel loop and its barrier disappear with it.
//   3. `NVH`/`NVH4` (min16float with `NIS_USE_HALF_PRECISION 1`) → `f32`, per
//      docs/porting-guide.md §6.6 (`//! CAPABILITY FP16` is only a marker; WGFX uses
//      rgba16float textures with f32 math). The coefficient LUTs are still read from
//      their r16g16b16a16float textures, so the LUT values keep their f16 precision.
//   4. Compile-time switches are Magpie's own defaults: NIS_SCALER 1,
//      NIS_TEXTURE_GATHER 1, NIS_USE_HALF_PRECISION 0 (f32), NIS_HDR_MODE 0,
//      NIS_NV12_SUPPORT 0, NIS_VIEWPORT_SUPPORT 0, NIS_CLAMP_OUTPUT 0.
//   5. `NVTEX_LOAD(coef_scaler, int2(vIdx, phase))` → `textureLoad(coef_scaler,
//      vec2<i32>(vIdx, phase), 0)`; `NVTEX_SAMPLE_RED/GREEN/BLUE` →
//      `textureGather(0|1|2, INPUT, samplerLinearClamp, uv)`. WebGPU's gather uses
//      the same component order as D3D (w = top-left, z = top-right, x = bottom-left,
//      y = bottom-right), so Magpie's `p[0][0]=(sr.w,..) p[0][1]=(sr.z,..)
//      p[1][0]=(sr.x,..) p[1][1]=(sr.y,..)` fill is copied verbatim.
//   6. `GroupMemoryBarrierWithGroupSync()` → `workgroupBarrier()`.
//   7. `static const float sharpen_slider = sharpness - 0.5f;` and the eight
//      constants derived from it depend on the `sharpness` uniform. WGSL forbids a
//      module-scope `const` that references a uniform, so they are computed at the
//      top of `Pass1` and passed to the helpers in a `SharpConsts` struct. Values are
//      unchanged; only the declaration site moved.
//   8. `mul`/`mad` → `a * b (+ c)`; `lerp` → `mix`; `saturate`/`min`/`max`/`abs`/
//      `floor`/`ceil`/`round` keep their names. HLSL's integer division in
//      `(kSupportSize - 1) / 2` is integer division here too (`kTileShift`).
//   9. `OUTPUT` declares no WIDTH/HEIGHT — deliberately mirroring NIS.hlsl (Magpie's
//      upscaler writes into a host-sized, scaled render target; the scale is read
//      from `scene.scale`). The host must supply the OUTPUT texture.
//  10. The luma tile pitch 38 (= NIS_BLOCK_WIDTH 32 + kSupportSize 6) is Magpie's own
//      compile-time bound and is valid for scale >= 1 (the NIS upscaling use case).
//      The HLSL source has exactly the same fixed-size bound.

//! VERSION 4
//! SORT_NAME NIS
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
Texture2D OUTPUT;

//! TEXTURE
//! SOURCE Coef_Scale.dds
//! FORMAT R16G16B16A16_FLOAT
Texture2D coef_scaler;

//! TEXTURE
//! SOURCE Coef_USM.dds
//! FORMAT R16G16B16A16_FLOAT
Texture2D coef_usm;

//! SAMPLER
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState samplerLinearClamp;

//! PASS 1
//! DESC "NVIDIA Image Scaling — NVScaler (6-tap + directional adaptive sharpening)"
//! IN INPUT, coef_scaler, coef_usm
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

// `#define kPhaseCount 64`, `#define kSupportSize 6`,
// `#define kTilePitch (NIS_BLOCK_WIDTH + kPadSize)` = 32 + 6,
// `#define kEdgeMapPitch (NIS_BLOCK_WIDTH + 2)` = 34.
const kPhaseCount: i32 = 64;
const kSupportSize: i32 = 6;
const kTilePitch: i32 = 38;
const kTileShift: f32 = -1.5; // 0.5 - (kSupportSize - 1) / 2 = 0.5 - 5/2 = 0.5 - 2 (整數除法)

// Groupshared storage (var<workgroup>); see porting note 2.
var<workgroup> shPixelsY: array<f32, 1444>; // kTilePitch * kTilePitch
var<workgroup> shCoefScaler: array<array<f32, 6>, 64>;
var<workgroup> shCoefUSM: array<array<f32, 6>, 64>;

// `sharpness`-derived constants (see porting note 7).
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

// NVF4 GetEdgeMap(NVF p[4][4], NVI i, NVI j) — evaluated over the 3x3 window at
// (oi, oj) of the 6x6 support. Verbatim apart from the window offset.
fn GetEdgeMapAt(p: ptr<function, array<array<f32, 6>, 6>>, oi: u32, oj: u32) -> vec4<f32> {
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

// NVF CalcLTI(NVF p0..p5, NVI phase_index)
fn CalcLTI(p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, phase_index: i32) -> f32 {
    let selector: bool = phase_index <= kPhaseCount / 2;
    var sel: f32 = select(p3, p0, selector);
    let a_min: f32 = min(min(p1, p2), sel);
    let a_max: f32 = max(max(p1, p2), sel);
    sel = select(p5, p2, selector);
    let b_min: f32 = min(min(p3, p4), sel);
    let b_max: f32 = max(max(p3, p4), sel);

    let a_cont: f32 = a_max - a_min;
    let b_cont: f32 = b_max - b_min;

    let cont_ratio: f32 = max(a_cont, b_cont) / (min(a_cont, b_cont) + kEps);
    return (1.0 - saturate((cont_ratio - kMinContrastRatio) * kRatioNorm)) * kContrastBoost;
}

// NVF4 GetInterpEdgeMap(const NVF4 edge[2][2], NVF phase_frac_x, NVF phase_frac_y)
fn GetInterpEdgeMap(
    edge: ptr<function, array<array<vec4<f32>, 2>, 2>>,
    phase_frac_x: f32,
    phase_frac_y: f32,
) -> vec4<f32> {
    let e = *edge;
    let h0: vec4<f32> = mix(e[0][0], e[0][1], phase_frac_x);
    let h1: vec4<f32> = mix(e[1][0], e[1][1], phase_frac_x);
    return mix(h0, h1, phase_frac_y);
}

// NVF EvalPoly6(const NVF pxl[6], NVI phase_int)
fn EvalPoly6(pxl: ptr<function, array<f32, 6>>, phase_int: i32, sc: SharpConsts) -> f32 {
    let x = *pxl;
    var y: f32 = 0.0;
    for (var i: i32 = 0; i < 6; i++) {
        y += shCoefScaler[u32(phase_int)][u32(i)] * x[u32(i)];
    }
    var y_usm: f32 = 0.0;
    for (var i: i32 = 0; i < 6; i++) {
        y_usm += shCoefUSM[u32(phase_int)][u32(i)] * x[u32(i)];
    }

    // let's compute a piece-wise ramp based on luma
    let y_scale: f32 = 1.0 - saturate((y * (1.0 / 1.0) - kSharpStartY) * kSharpScaleY);

    // scale the ramp to sharpen as a function of luma
    let y_sharpness: f32 = y_scale * sc.strength_scale + sc.strength_min;

    y_usm *= y_sharpness;

    // scale the ramp to limit USM as a function of luma
    let y_sharpness_limit: f32 = (y_scale * sc.limit_scale + sc.limit_min) * y;

    y_usm = min(y_sharpness_limit, max(-y_sharpness_limit, y_usm));
    // reduce ringing
    y_usm *= CalcLTI(x[0], x[1], x[2], x[3], x[4], x[5], phase_int);

    return y + y_usm;
}

// NVF FilterNormal(const NVF p[6][6], NVI phase_x_frac_int, NVI phase_y_frac_int)
fn FilterNormal(
    p: ptr<function, array<array<f32, 6>, 6>>,
    phase_x_frac_int: i32,
    phase_y_frac_int: i32,
) -> f32 {
    let a = *p;
    var h_acc: f32 = 0.0;
    for (var j: i32 = 0; j < 6; j++) {
        var v_acc: f32 = 0.0;
        for (var i: i32 = 0; i < 6; i++) {
            v_acc += a[u32(i)][u32(j)] * shCoefScaler[u32(phase_y_frac_int)][u32(i)];
        }
        h_acc += v_acc * shCoefScaler[u32(phase_x_frac_int)][u32(j)];
    }

    // let's return the sum unpacked -> we can accumulate it later
    return h_acc;
}

// NVF AddDirFilters(NVF p[6][6], ...)
fn AddDirFilters(
    p: ptr<function, array<array<f32, 6>, 6>>,
    phase_x_frac: f32,
    phase_y_frac: f32,
    phase_x_frac_int: i32,
    phase_y_frac_int: i32,
    w: vec4<f32>,
    sc: SharpConsts,
) -> f32 {
    let a = *p;
    var f: f32 = 0.0;
    if (w.x > 0.0) {
        // 0 deg filter
        var interp0Deg: array<f32, 6>;
        for (var i: i32 = 0; i < 6; i++) {
            interp0Deg[u32(i)] = mix(a[u32(i)][2u], a[u32(i)][3u], phase_x_frac);
        }
        f += EvalPoly6(&interp0Deg, phase_y_frac_int, sc) * w.x;
    }
    if (w.y > 0.0) {
        // 90 deg filter
        var interp90Deg: array<f32, 6>;
        for (var i: i32 = 0; i < 6; i++) {
            interp90Deg[u32(i)] = mix(a[2u][u32(i)], a[3u][u32(i)], phase_y_frac);
        }
        f += EvalPoly6(&interp90Deg, phase_x_frac_int, sc) * w.y;
    }
    if (w.z > 0.0) {
        // 45 deg filter
        var pphase_b45: f32 = 0.5 + 0.5 * (phase_x_frac - phase_y_frac);

        var temp45: array<f32, 7>;
        temp45[1u] = mix(a[2u][1u], a[1u][2u], pphase_b45);
        temp45[3u] = mix(a[3u][2u], a[2u][3u], pphase_b45);
        temp45[5u] = mix(a[4u][3u], a[3u][4u], pphase_b45);
        {
            pphase_b45 = pphase_b45 - 0.5;
            let aa: f32 = select(a[2u][0u], a[0u][2u], pphase_b45 >= 0.0);
            let bb: f32 = select(a[3u][1u], a[1u][3u], pphase_b45 >= 0.0);
            let cc: f32 = select(a[4u][2u], a[2u][4u], pphase_b45 >= 0.0);
            let dd: f32 = select(a[5u][3u], a[3u][5u], pphase_b45 >= 0.0);
            temp45[0u] = mix(a[1u][1u], aa, abs(pphase_b45));
            temp45[2u] = mix(a[2u][2u], bb, abs(pphase_b45));
            temp45[4u] = mix(a[3u][3u], cc, abs(pphase_b45));
            temp45[6u] = mix(a[4u][4u], dd, abs(pphase_b45));
        }

        var interp45Deg: array<f32, 6>;
        var pphase_p45: f32 = phase_x_frac + phase_y_frac;
        if (pphase_p45 >= 1.0) {
            for (var i: i32 = 0; i < 6; i++) {
                interp45Deg[u32(i)] = temp45[u32(i + 1)];
            }
            pphase_p45 = pphase_p45 - 1.0;
        } else {
            for (var i: i32 = 0; i < 6; i++) {
                interp45Deg[u32(i)] = temp45[u32(i)];
            }
        }

        f += EvalPoly6(&interp45Deg, i32(pphase_p45 * 64.0), sc) * w.z;
    }
    if (w.w > 0.0) {
        // 135 deg filter
        var pphase_b135: f32 = 0.5 * (phase_x_frac + phase_y_frac);

        var temp135: array<f32, 7>;
        temp135[1u] = mix(a[3u][1u], a[4u][2u], pphase_b135);
        temp135[3u] = mix(a[2u][2u], a[3u][3u], pphase_b135);
        temp135[5u] = mix(a[1u][3u], a[2u][4u], pphase_b135);
        {
            pphase_b135 = pphase_b135 - 0.5;
            let aa: f32 = select(a[3u][0u], a[5u][2u], pphase_b135 >= 0.0);
            let bb: f32 = select(a[2u][1u], a[4u][3u], pphase_b135 >= 0.0);
            let cc: f32 = select(a[1u][2u], a[3u][4u], pphase_b135 >= 0.0);
            let dd: f32 = select(a[0u][3u], a[2u][5u], pphase_b135 >= 0.0);
            temp135[0u] = mix(a[4u][1u], aa, abs(pphase_b135));
            temp135[2u] = mix(a[3u][2u], bb, abs(pphase_b135));
            temp135[4u] = mix(a[2u][3u], cc, abs(pphase_b135));
            temp135[6u] = mix(a[1u][4u], dd, abs(pphase_b135));
        }

        var interp135Deg: array<f32, 6>;
        var pphase_p135: f32 = 1.0 + (phase_x_frac - phase_y_frac);
        if (pphase_p135 >= 1.0) {
            for (var i: i32 = 0; i < 6; i++) {
                interp135Deg[u32(i)] = temp135[u32(i + 1)];
            }
            pphase_p135 = pphase_p135 - 1.0;
        } else {
            for (var i: i32 = 0; i < 6; i++) {
                interp135Deg[u32(i)] = temp135[u32(i)];
            }
        }

        f += EvalPoly6(&interp135Deg, i32(pphase_p135 * 64.0), sc) * w.w;
    }
    return f;
}

// void NVScaler(NVU2 blockIdx, NVU threadIdx)  →  compute entry point
fn Pass1(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let threadIdx: u32 = lid.x;
    let blockIdx: vec2<u32> = wg.xy;

    // static const float sharpen_slider = sharpness - 0.5f; (see porting note 7)
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

    // static const float kScaleX = 1 / GetScale().x; ...
    let kScaleX: f32 = 1.0 / scene.scale.x;
    let kScaleY: f32 = 1.0 / scene.scale.y;
    let kSrcNormX: f32 = scene.inputPt.x;
    let kSrcNormY: f32 = scene.inputPt.y;

    // Figure out the range of pixels from input image that would be needed to be
    // loaded for this thread-block
    let dstBlockX: i32 = i32(32u * blockIdx.x);
    let dstBlockY: i32 = i32(32u * blockIdx.y);

    let srcBlockStartX: i32 = i32(floor((f32(dstBlockX) + 0.5) * kScaleX - 0.5));
    let srcBlockStartY: i32 = i32(floor((f32(dstBlockY) + 0.5) * kScaleY - 0.5));
    let srcBlockEndX: i32 = i32(ceil((f32(dstBlockX) + 32.0 + 0.5) * kScaleX - 0.5));
    let srcBlockEndY: i32 = i32(ceil((f32(dstBlockY) + 32.0 + 0.5) * kScaleY - 0.5));

    let numTilePixelsX0: i32 = srcBlockEndX - srcBlockStartX + kSupportSize - 1;
    let numTilePixelsY0: i32 = srcBlockEndY - srcBlockStartY + kSupportSize - 1;

    // round-up load region to even size since we're loading in 2x2 batches
    let numTilePixelsX: i32 = numTilePixelsX0 + (numTilePixelsX0 & 1);
    let numTilePixelsY: i32 = numTilePixelsY0 + (numTilePixelsY0 & 1);
    let numTilePixels: i32 = numTilePixelsX * numTilePixelsY;

    // fill in input luma tile (shPixelsY) in batches of 2x2 pixels
    // we use texture gather to get extra support necessary
    // to compute 2x2 edge map outputs too
    for (var i: u32 = threadIdx * 2u; i < (u32(numTilePixels) >> 1u); i += 512u) {
        let py: u32 = (i / u32(numTilePixelsX)) * 2u;
        let px: u32 = i % u32(numTilePixelsX);

        // 0.5 to be in the center of texel
        // - (kSupportSize - 1) / 2 to shift by the kernel support size
        let tx: f32 = (f32(srcBlockStartX + i32(px)) + kTileShift) * kSrcNormX;
        let ty: f32 = (f32(srcBlockStartY + i32(py)) + kTileShift) * kSrcNormY;

        let sr: vec4<f32> = textureGather(0, INPUT, samplerLinearClamp, vec2<f32>(tx, ty));
        let sg: vec4<f32> = textureGather(1, INPUT, samplerLinearClamp, vec2<f32>(tx, ty));
        let sb: vec4<f32> = textureGather(2, INPUT, samplerLinearClamp, vec2<f32>(tx, ty));

        let p00: f32 = getY(vec3<f32>(sr.w, sg.w, sb.w));
        let p01: f32 = getY(vec3<f32>(sr.z, sg.z, sb.z));
        let p10: f32 = getY(vec3<f32>(sr.x, sg.x, sb.x));
        let p11: f32 = getY(vec3<f32>(sr.y, sg.y, sb.y));

        let idx: u32 = py * u32(kTilePitch) + px;
        shPixelsY[idx] = p00;
        shPixelsY[idx + 1u] = p01;
        shPixelsY[idx + u32(kTilePitch)] = p10;
        shPixelsY[idx + u32(kTilePitch) + 1u] = p11;
    }
    workgroupBarrier();

    // void LoadFilterBanksSh(NVI i0) — kPhaseCount * 2 = 128 < 256 threads, so the
    // HLSL takes its `if (i < kPhaseCount * 2)` branch.
    if (i32(threadIdx) < kPhaseCount * 2) {
        let phase: i32 = i32(threadIdx) >> 1;
        let vIdx: i32 = i32(threadIdx) & 1;

        var v: vec4<f32> = textureLoad(coef_scaler, vec2<i32>(vIdx, phase), 0);
        let filterOffset: i32 = vIdx * 4;
        shCoefScaler[u32(phase)][u32(filterOffset)] = v.x;
        shCoefScaler[u32(phase)][u32(filterOffset + 1)] = v.y;
        if (vIdx == 0) {
            shCoefScaler[u32(phase)][2u] = v.z;
            shCoefScaler[u32(phase)][3u] = v.w;
        }

        v = textureLoad(coef_usm, vec2<i32>(vIdx, phase), 0);
        shCoefUSM[u32(phase)][u32(filterOffset)] = v.x;
        shCoefUSM[u32(phase)][u32(filterOffset + 1)] = v.y;
        if (vIdx == 0) {
            shCoefUSM[u32(phase)][2u] = v.z;
            shCoefUSM[u32(phase)][3u] = v.w;
        }
    }
    workgroupBarrier();

    // output coord within a tile
    let posX: i32 = i32(threadIdx) % 32;
    let posY: i32 = i32(threadIdx) / 32;
    // x coord inside the output image
    let dstX: i32 = dstBlockX + posX;
    // x coord inside the input image
    let srcX: f32 = (0.5 + f32(dstX)) * kScaleX - 0.5;
    // nearest integer part
    let px: i32 = i32(floor(srcX)) - srcBlockStartX;
    // fractional part
    let fx: f32 = srcX - floor(srcX);
    // discretized phase
    let fx_int: i32 = i32(fx * 64.0);

    // NIS_BLOCK_WIDTH * NIS_BLOCK_HEIGHT / NIS_THREAD_GROUP_SIZE = 32*32/256 = 4
    for (var k: i32 = 0; k < 4; k++) {
        // y coord inside the output image
        let dstY: i32 = dstBlockY + posY + k * 8;
        // y coord inside the input image
        let srcY: f32 = (0.5 + f32(dstY)) * kScaleY - 0.5;
        // nearest integer part
        let py: i32 = i32(floor(srcY)) - srcBlockStartY;
        // fractional part
        let fy: f32 = srcY - floor(srcY);
        // discretized phase
        let fy_int: i32 = i32(fy * 64.0);

        // load 6x6 support to regs
        var p: array<array<f32, 6>, 6>;
        let startTileIdx: i32 = py * kTilePitch + px;
        for (var i: i32 = 0; i < 6; i++) {
            for (var j: i32 = 0; j < 6; j++) {
                p[u32(i)][u32(j)] = shPixelsY[u32(startTileIdx + i * kTilePitch + j)];
            }
        }

        // generate weights for directional filters
        // (needs to shift edge map sampling since it's a 2x2 centered inside 6x6 grid)
        var edge: array<array<vec4<f32>, 2>, 2>;
        for (var i: i32 = 0; i < 2; i++) {
            for (var j: i32 = 0; j < 2; j++) {
                edge[u32(i)][u32(j)] = GetEdgeMapAt(&p, u32(i + 1), u32(j + 1));
            }
        }
        let w: vec4<f32> = GetInterpEdgeMap(&edge, fx, fy) * 1.0;

        // weigth for luma
        let baseWeight: f32 = 1.0 - w.x - w.y - w.z - w.w;

        // final luma is a weighted product of directional & normal filters
        var opY: f32 = 0.0;

        // get traditional scaler filter output
        opY += FilterNormal(&p, fx_int, fy_int) * baseWeight;

        // get directional filter bank output
        opY += AddDirFilters(&p, fx, fy, fx_int, fy_int, w, sc);

        let coord: vec2<f32> = vec2<f32>((srcX + 0.5) * kSrcNormX, (srcY + 0.5) * kSrcNormY);

        // do bilinear tap for chroma upscaling
        var op: vec4<f32> = textureSampleLevel(INPUT, samplerLinearClamp, coord, 0.0);
        let y: f32 = getY(vec3<f32>(op.x, op.y, op.z));

        // const NVF corr = opY * (1.0f / NIS_SCALE_FLOAT) - y;   (NIS_SCALE_FLOAT = 1)
        let corr: f32 = opY * (1.0 / 1.0) - y;
        op = vec4<f32>(op.x + corr, op.y + corr, op.z + corr, op.w);

        // NVTEX_STORE(out_texture, dstCoord, MF4(NVCLAMP(op)))
        // NIS_CLAMP_OUTPUT = 0 => NVCLAMP(op) = op
        textureStore(OUTPUT, vec2<i32>(dstX, dstY), op);
    }
}
//! END
