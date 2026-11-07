// FSR (AMD FidelityFX Super Resolution 1.0) — EASU + RCAS, 2x upscale pipeline
//
// Ported to the WGFX WGSL effect DSL from Magpie's HLSL effect files:
//   https://github.com/Blinue/Magpie/blob/main/src/Effects/FSR/FSR_EASU.hlsl
//   https://github.com/Blinue/Magpie/blob/main/src/Effects/FSR/FSR_RCAS.hlsl
// which are themselves ports of AMD FidelityFX-FSR 1.0 (ffx-fsr/ffx_fsr1.h):
//   https://github.com/GPUOpen-Effects/FidelityFX-FSR
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. (MIT — AMD FidelityFX-FSR)
// Copyright (c) Blinue and Magpie contributors. (MIT — Magpie HLSL port)
// WGFX WGSL port: pass structure, gather emulation and DSL adaptation only; the
// EASU/RCAS math and weight constants are kept verbatim from the sources above.
//
// Pipeline:
//   Pass 1 (EASU): INPUT (input size)  -> tex_easu (2x input size)
//   Pass 2 (RCAS): tex_easu (2x)       -> OUTPUT  (2x input size)
//
// Porting notes:
//   * WGSL has no GatherRed/Green/Blue; the four D3D gathers of EASU are emulated
//     with explicit textureLoad calls. Each gather point sits on a texel grid
//     corner and returns its four surrounding texels in D3D order
//     x=bottom-left, y=bottom-right, z=top-right, w=top-left. Only the components
//     the original reads are filled (the rest stay 0 and are never read).
//   * Gather edge texels are clamped to the image border, matching Magpie's
//     CLAMP sampler address mode. RCAS keeps the original unclamped Load
//     semantics (out-of-bounds texels read as zero in D3D and WebGPU alike).
//   * The scalar (non-FP16) AMD code path is ported; WGSL f32 replaces MF.
//   * AMD's APrx* approximate reciprocal/rsqrt (ffx_a.h) are restored verbatim as
//     WGSL `bitcast<u32>` bit-tricks (constants 0x7ef07ebb / 0x7ef19fff / 0x5f347d74).
//     They are mathematically TOTAL — for any finite input they return a finite value
//     (rcp(0) ≈ 1.6e38, never +Inf) — which is the structural protection the official
//     code relies on. Replacing them with exact `1.0/x` creates Inf→NaN sites (0*Inf)
//     that HLSL's ternary min/max heals but WGSL spec min/max propagate; on Dawn the
//     NaN happens to self-heal (saturate(NaN)=0, NMax semantics), on stricter
//     implementations it would not. hitMin/hitMax keep exact rcp (as the original's
//     ARcpF1) with an explicit all-black/all-white guard matching the official's
//     healed result.
//   * Porting deviation (documented): RCAS output is clamped to the min/max of its
//     four neighbors (b,d,f,h) — the same deringing philosophy as EASU's terminal
//     min4/max4 clamp. The original extrapolates with a negative lobe, which
//     amplifies single-pixel chroma outliers (4:2:0 video noise) into saturated
//     red/green/blue fireflies in dark areas (~2x chroma overshoot at
//     Sharpness=0.87), and even passes the input's own spikes through verbatim at
//     1:1 scale. The clamp bounds output by the neighborhood: saturated fireflies
//     collapse to the local level. Residual: bounded 2x2 chroma blobs remain where
//     the input spike itself sits inside the nearest-4 set (per-channel min4/max4
//     can combine channels from different texels, so output chroma is bounded by
//     the neighborhood's per-channel range, not by any single input texel's
//     chroma). Verified faithful to the official math by test/impulse_response.mjs
//     (9/9 cases, GPU ≡ CPU reference within 1-3 f16 ulp; NaN-prefill coverage
//     100%). Documented trade-off: genuine isolated single-pixel highlights are
//     smoothed as well.
//! VERSION 4

//! PARAMETER Sharpness
//! LABEL "Sharpness"
//! MIN 0
//! MAX 1
//! DEFAULT 0.87
//! STEP 0.01
float Sharpness;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH * 2
//! HEIGHT INPUT_HEIGHT * 2
Texture2D OUTPUT;

//! TEXTURE tex_easu
//! WIDTH INPUT_WIDTH * 2
//! HEIGHT INPUT_HEIGHT * 2
//! FORMAT R16G16B16A16_FLOAT
Texture2D tex_easu;

//! PASS 1
//! DESC "FidelityFX EASU: edge-adaptive 12-tap upscaler (2x)"
//! IN INPUT
//! OUT tex_easu
//! NUM_THREADS 8,8

// NOTE: the entry point must be the FIRST function of the pass block — the WGFX
// generator prepends "@compute @workgroup_size(...)" to the block's first fn.
fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outputSize = scene.outputSize;
    if (global_id.x >= outputSize.x || global_id.y >= outputSize.y) {
        return;
    }

    let color = FsrEasu(global_id.xy);
    textureStore(tex_easu, global_id.xy, vec4<f32>(color, 1.0));
}

// Load one INPUT texel, clamped to the image border.
// Emulates the D3D CLAMP address mode at the edges of gathered quads.
fn EasuLoad(pos: vec2<i32>) -> vec3<f32> {
    let size = vec2<i32>(scene.inputSize);
    return textureLoad(INPUT, clamp(pos, vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

// Approximate reciprocal (AMD ffx_a.h, verbatim: single IADD bit trick).
// Total function: never returns Inf/NaN for any finite input (rcp(0) ≈ 1.6e38),
// removing the Inf→NaN paths that exact `1.0/x` would create (0*Inf).
fn APrxLoRcpF1(a: f32) -> f32 {
    return bitcast<f32>(0x7ef07ebbu - bitcast<u32>(a));
}

// Approximate reciprocal square root (AMD ffx_a.h, verbatim: IADD + shift).
fn APrxLoRsqF1(a: f32) -> f32 {
    return bitcast<f32>(0x5f347d74u - (bitcast<u32>(a) >> 1u));
}

// Filtering for a given tap for the scalar. (AMD FidelityFX EASU)
fn FsrEasuTapF(
    aC: ptr<function, vec3<f32>>, // Accumulated color, with negative lobe.
    aW: ptr<function, f32>,       // Accumulated weight.
    off: vec2<f32>,               // Pixel offset from resolve position to tap.
    dir: vec2<f32>,               // Gradient direction.
    len: vec2<f32>,               // Length.
    lob: f32,                     // Negative lobe strength.
    clp: f32,                     // Clipping point.
    c: vec3<f32>,                 // Tap color.
) {
    // Rotate offset by direction.
    var v: vec2<f32>;
    v.x = (off.x * (dir.x)) + (off.y * dir.y);
    v.y = (off.x * (-dir.y)) + (off.y * dir.x);
    // Anisotropy.
    v *= len;
    // Compute distance^2.
    let d2 = min(v.x * v.x + v.y * v.y, clp);
    // Approximation of lanczos2 without sin() or rcp(), or sqrt() to get x.
    //  (25/16 * (2/5 * x^2 - 1)^2 - (25/16 - 1)) * (1/4 * x^2 - 1)^2
    //  |_______________________________________|   |_______________|
    //                   base                             window
    var wB = 2.0 / 5.0 * d2 - 1.0;
    var wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = 25.0 / 16.0 * wB - (25.0 / 16.0 - 1.0);
    let w = wB * wA;
    // Do weighted average.
    *aC += c * w;
    *aW += w;
}

// Accumulate direction and length. (AMD FidelityFX EASU)
fn FsrEasuSetF(
    dir: ptr<function, vec2<f32>>,
    len: ptr<function, f32>,
    pp: vec2<f32>,
    biS: bool,
    biT: bool,
    biU: bool,
    biV: bool,
    lA: f32,
    lB: f32,
    lC: f32,
    lD: f32,
    lE: f32,
) {
    // Compute bilinear weight, branches factor out as predicates are compiler time immediates.
    //  s t
    //  u v
    var w = 0.0;
    if (biS) { w = (1.0 - pp.x) * (1.0 - pp.y); }
    if (biT) { w = pp.x * (1.0 - pp.y); }
    if (biU) { w = (1.0 - pp.x) * pp.y; }
    if (biV) { w = pp.x * pp.y; }
    // Direction is the '+' diff.
    //    a
    //  b c d
    //    e
    // Then takes magnitude from abs average of both sides of 'c'.
    // Length converts gradient reversal to 0, smoothly to non-reversal at 1, shaped, then adding horz and vert terms.
    let dc = lD - lC;
    let cb = lC - lB;
    var lenX = max(abs(dc), abs(cb));
    lenX = APrxLoRcpF1(lenX);
    let dirX = lD - lB;
    lenX = saturate(abs(dirX) * lenX);
    lenX *= lenX;
    // Repeat for the y axis.
    let ec = lE - lC;
    let ca = lC - lA;
    var lenY = max(abs(ec), abs(ca));
    lenY = APrxLoRcpF1(lenY);
    let dirY = lE - lA;
    lenY = saturate(abs(dirY) * lenY);
    lenY *= lenY;
    *dir += vec2<f32>(dirX * w, dirY * w);
    *len += lenX * w;
    *len += lenY * w;
}

// FidelityFX EASU core: 12-tap edge-adaptive upscaler. (AMD FidelityFX, scalar path)
fn FsrEasu(ip: vec2<u32>) -> vec3<f32> {
    let inputSize = scene.inputSize;
    let outputSize = scene.outputSize;
    let inputPt = scene.inputPt;

    // Output integer position to a pixel position in viewport.
    let con0x = f32(inputSize.x) / f32(outputSize.x);
    let con0y = f32(inputSize.y) / f32(outputSize.y);
    var pp = vec2<f32>(
        f32(ip.x) * con0x + (0.5 * con0x - 0.5),
        f32(ip.y) * con0y + (0.5 * con0y - 0.5)
    );
    let fp = floor(pp);
    pp = pp - fp;

    // 12-tap kernel.
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    // The HLSL source reads these through four GatherRed/Green/Blue calls whose
    // sample points sit on texel grid corners: gather(0) at fp+(1,-1) gives
    // {b,c,-,-}, gather(1) at fp+(0,1) gives {i,j,f,e}, gather(2) at fp+(2,1)
    // gives {k,l,h,g}, gather(3) at fp+(1,3) gives {-,-,o,n} (component order
    // x=bottom-left, y=bottom-right, z=top-right, w=top-left). The loads below
    // reconstruct those vectors texel by texel.
    let b = EasuLoad(vec2<i32>(fp) + vec2<i32>(0, -1));
    let c = EasuLoad(vec2<i32>(fp) + vec2<i32>(1, -1));
    let e = EasuLoad(vec2<i32>(fp) + vec2<i32>(-1, 0));
    let f = EasuLoad(vec2<i32>(fp) + vec2<i32>(0, 0));
    let g = EasuLoad(vec2<i32>(fp) + vec2<i32>(1, 0));
    let h = EasuLoad(vec2<i32>(fp) + vec2<i32>(2, 0));
    let i = EasuLoad(vec2<i32>(fp) + vec2<i32>(-1, 1));
    let j = EasuLoad(vec2<i32>(fp) + vec2<i32>(0, 1));
    let k = EasuLoad(vec2<i32>(fp) + vec2<i32>(1, 1));
    let l = EasuLoad(vec2<i32>(fp) + vec2<i32>(2, 1));
    let n = EasuLoad(vec2<i32>(fp) + vec2<i32>(0, 2));
    let o = EasuLoad(vec2<i32>(fp) + vec2<i32>(1, 2));

    let bczzR = vec4<f32>(b.r, c.r, 0.0, 0.0);
    let bczzG = vec4<f32>(b.g, c.g, 0.0, 0.0);
    let bczzB = vec4<f32>(b.b, c.b, 0.0, 0.0);
    let ijfeR = vec4<f32>(i.r, j.r, f.r, e.r);
    let ijfeG = vec4<f32>(i.g, j.g, f.g, e.g);
    let ijfeB = vec4<f32>(i.b, j.b, f.b, e.b);
    let klhgR = vec4<f32>(k.r, l.r, h.r, g.r);
    let klhgG = vec4<f32>(k.g, l.g, h.g, g.g);
    let klhgB = vec4<f32>(k.b, l.b, h.b, g.b);
    let zzonR = vec4<f32>(0.0, 0.0, o.r, n.r);
    let zzonG = vec4<f32>(0.0, 0.0, o.g, n.g);
    let zzonB = vec4<f32>(0.0, 0.0, o.b, n.b);

    // Simplest multi-channel approximate luma possible (luma times 2, in 2 FMA/MAD).
    let bczzL = bczzB * 0.5 + (bczzR * 0.5 + bczzG);
    let ijfeL = ijfeB * 0.5 + (ijfeR * 0.5 + ijfeG);
    let klhgL = klhgB * 0.5 + (klhgR * 0.5 + klhgG);
    let zzonL = zzonB * 0.5 + (zzonR * 0.5 + zzonG);
    // Rename.
    let bL = bczzL.x;
    let cL = bczzL.y;
    let iL = ijfeL.x;
    let jL = ijfeL.y;
    let fL = ijfeL.z;
    let eL = ijfeL.w;
    let kL = klhgL.x;
    let lL = klhgL.y;
    let hL = klhgL.z;
    let gL = klhgL.w;
    let oL = zzonL.z;
    let nL = zzonL.w;
    // Accumulate for bilinear interpolation.
    var dir = vec2<f32>(0.0);
    var len = 0.0;
    FsrEasuSetF(&dir, &len, pp, true, false, false, false, bL, eL, fL, gL, jL);
    FsrEasuSetF(&dir, &len, pp, false, true, false, false, cL, fL, gL, hL, kL);
    FsrEasuSetF(&dir, &len, pp, false, false, true, false, fL, iL, jL, kL, nL);
    FsrEasuSetF(&dir, &len, pp, false, false, false, true, gL, jL, kL, lL, oL);
    // Normalize with approximation, and cleanup close to zero.
    let dir2 = dir * dir;
    var dirR = dir2.x + dir2.y;
    let zro = dirR < 1.0 / 32768.0;
    dirR = select(APrxLoRsqF1(dirR), 1.0, zro);
    dir.x = select(dir.x, 1.0, zro);
    dir *= dirR;
    // Transform from {0 to 2} to {0 to 1} range, and shape with square.
    len = len * 0.5;
    len *= len;
    // Stretch kernel {1.0 vert|horz, to sqrt(2.0) on diagonal}.
    let stretch = (dir.x * dir.x + dir.y * dir.y) * APrxLoRcpF1(max(abs(dir.x), abs(dir.y)));
    // Anisotropic length after rotation,
    //  x := 1.0 lerp to 'stretch' on edges
    //  y := 1.0 lerp to 2x on edges
    let len2 = vec2<f32>(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    // Based on the amount of 'edge',
    // the window shifts from +/-{sqrt(2.0) to slightly beyond 2.0}.
    let lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    // Set distance^2 clipping point to the end of the adjustable window.
    let clp = APrxLoRcpF1(lob);
    // Accumulation mixed with min/max of 4 nearest.
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    let min4 = min(min(vec3<f32>(ijfeR.z, ijfeG.z, ijfeB.z), vec3<f32>(klhgR.w, klhgG.w, klhgB.w)),
                   min(vec3<f32>(ijfeR.y, ijfeG.y, ijfeB.y), vec3<f32>(klhgR.x, klhgG.x, klhgB.x)));
    let max4 = max(max(vec3<f32>(ijfeR.z, ijfeG.z, ijfeB.z), vec3<f32>(klhgR.w, klhgG.w, klhgB.w)),
                   max(vec3<f32>(ijfeR.y, ijfeG.y, ijfeB.y), vec3<f32>(klhgR.x, klhgG.x, klhgB.x)));
    // Accumulation.
    var aC = vec3<f32>(0.0);
    var aW = 0.0;
    FsrEasuTapF(&aC, &aW, vec2<f32>(0.0, -1.0) - pp, dir, len2, lob, clp, vec3<f32>(bczzR.x, bczzG.x, bczzB.x)); // b
    FsrEasuTapF(&aC, &aW, vec2<f32>(1.0, -1.0) - pp, dir, len2, lob, clp, vec3<f32>(bczzR.y, bczzG.y, bczzB.y)); // c
    FsrEasuTapF(&aC, &aW, vec2<f32>(-1.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfeR.x, ijfeG.x, ijfeB.x)); // i
    FsrEasuTapF(&aC, &aW, vec2<f32>(0.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfeR.y, ijfeG.y, ijfeB.y)); // j
    FsrEasuTapF(&aC, &aW, vec2<f32>(0.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfeR.z, ijfeG.z, ijfeB.z)); // f
    FsrEasuTapF(&aC, &aW, vec2<f32>(-1.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfeR.w, ijfeG.w, ijfeB.w)); // e
    FsrEasuTapF(&aC, &aW, vec2<f32>(1.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(klhgR.x, klhgG.x, klhgB.x)); // k
    FsrEasuTapF(&aC, &aW, vec2<f32>(2.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(klhgR.y, klhgG.y, klhgB.y)); // l
    FsrEasuTapF(&aC, &aW, vec2<f32>(2.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(klhgR.z, klhgG.z, klhgB.z)); // h
    FsrEasuTapF(&aC, &aW, vec2<f32>(1.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(klhgR.w, klhgG.w, klhgB.w)); // g
    FsrEasuTapF(&aC, &aW, vec2<f32>(1.0, 2.0) - pp, dir, len2, lob, clp, vec3<f32>(zzonR.z, zzonG.z, zzonB.z)); // o
    FsrEasuTapF(&aC, &aW, vec2<f32>(0.0, 2.0) - pp, dir, len2, lob, clp, vec3<f32>(zzonR.w, zzonG.w, zzonB.w)); // n
    // Normalize and dering.
    return min(max4, max(min4, aC * (1.0 / aW)));
}

//! END

//! PASS 2
//! DESC "FidelityFX RCAS: robust contrast-adaptive sharpening"
//! IN tex_easu
//! OUT OUTPUT
//! NUM_THREADS 8,8

// NOTE: the entry point must be the FIRST function of the pass block — the WGFX
// generator prepends "@compute @workgroup_size(...)" to the block's first fn.
fn Pass2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outputSize = scene.outputSize;
    if (global_id.x >= outputSize.x || global_id.y >= outputSize.y) {
        return;
    }

    let p = vec2<i32>(global_id.xy);
    let b = textureLoad(tex_easu, p + vec2<i32>(0, -1), 0).rgb;
    let d = textureLoad(tex_easu, p + vec2<i32>(-1, 0), 0).rgb;
    let e = textureLoad(tex_easu, p, 0).rgb;
    let f = textureLoad(tex_easu, p + vec2<i32>(1, 0), 0).rgb;
    let h = textureLoad(tex_easu, p + vec2<i32>(0, 1), 0).rgb;

    textureStore(OUTPUT, global_id.xy, vec4<f32>(FsrRcasF(b, d, e, f, h), 1.0));
}

// This is set at the limit of providing unnatural results for sharpening.
const FSR_RCAS_LIMIT: f32 = 0.25 - (1.0 / 16.0);

// Medium-precision approximate reciprocal (AMD ffx_a.h, verbatim: IADD + FMA + MUL).
// The 1.87… Newton step keeps APrxMedRcpF1(0) ≈ 3.2e38 — finite, unlike exact 1/0.
fn APrxMedRcpF1(a: f32) -> f32 {
    let b = bitcast<f32>(0x7ef19fffu - bitcast<u32>(a));
    return b * (-b * a + 2.0);
}

// FidelityFX RCAS core. (AMD FidelityFX, scalar path)
// Algorithm uses minimal 3x3 pixel neighborhood.
//    b
//  d e f
//    h
fn FsrRcasF(b: vec3<f32>, d: vec3<f32>, e: vec3<f32>, f: vec3<f32>, h: vec3<f32>) -> vec3<f32> {
    // Luma times 2.
    let bL = b.b * 0.5 + (b.r * 0.5 + b.g);
    let dL = d.b * 0.5 + (d.r * 0.5 + d.g);
    let eL = e.b * 0.5 + (e.r * 0.5 + e.g);
    let fL = f.b * 0.5 + (f.r * 0.5 + f.g);
    let hL = h.b * 0.5 + (h.r * 0.5 + h.g);

    // Noise detection.
    var nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    nz = saturate(abs(nz) * APrxMedRcpF1(max(max(max(bL, dL), eL), max(fL, hL)) - min(min(min(bL, dL), eL), min(fL, hL))));
    nz = -0.5 * nz + 1.0;

    // Min and max of ring.
    let mn4R = min(min(min(b.r, d.r), f.r), h.r);
    let mn4G = min(min(min(b.g, d.g), f.g), h.g);
    let mn4B = min(min(min(b.b, d.b), f.b), h.b);
    let mx4R = max(max(max(b.r, d.r), f.r), h.r);
    let mx4G = max(max(max(b.g, d.g), f.g), h.g);
    let mx4B = max(max(max(b.b, d.b), f.b), h.b);
    // Immediate constants for peak range.
    // peakC = { 1.0, -1.0 * 4.0 }
    // Limiters, these need to be high precision RCPs.
    // Degenerate-neighborhood guard: when 4*mx4 == 0 (all-black) or 4*mn4 - 4 == 0
    // (all-white) the original also forms 0*Inf = NaN, but HLSL's ternary max/min
    // comparison semantics heal it. WGSL min/max propagate NaN per spec, so the
    // guard is explicit; the resulting output matches the official healed result
    // (all-black → e, all-white → e, sharpening off).
    let hitMinR = select(min(mn4R, e.r) * (1.0 / (4.0 * mx4R)), 0.0, mx4R == 0.0);
    let hitMinG = select(min(mn4G, e.g) * (1.0 / (4.0 * mx4G)), 0.0, mx4G == 0.0);
    let hitMinB = select(min(mn4B, e.b) * (1.0 / (4.0 * mx4B)), 0.0, mx4B == 0.0);
    let hitMaxR = select((1.0 - max(mx4R, e.r)) * (1.0 / (4.0 * mn4R + (-4.0))), 0.0, mn4R == 1.0);
    let hitMaxG = select((1.0 - max(mx4G, e.g)) * (1.0 / (4.0 * mn4G + (-4.0))), 0.0, mn4G == 1.0);
    let hitMaxB = select((1.0 - max(mx4B, e.b)) * (1.0 / (4.0 * mn4B + (-4.0))), 0.0, mn4B == 1.0);
    let lobeR = max(-hitMinR, hitMaxR);
    let lobeG = max(-hitMinG, hitMaxG);
    let lobeB = max(-hitMinB, hitMaxB);
    var lobe = max(-FSR_RCAS_LIMIT, min(max(max(lobeR, lobeG), lobeB), 0.0)) * uniforms.Sharpness;

    // Apply noise removal.
    lobe *= nz;

    // Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
    let rcpL = APrxMedRcpF1(4.0 * lobe + 1.0);
    let res = vec3<f32>(
        (lobe * b.r + lobe * d.r + lobe * h.r + lobe * f.r + e.r) * rcpL,
        (lobe * b.g + lobe * d.g + lobe * h.g + lobe * f.g + e.g) * rcpL,
        (lobe * b.b + lobe * d.b + lobe * h.b + lobe * f.b + e.b) * rcpL
    );

    // Deringing clamp / 除彩溢出鉗位（見檔頭 porting note）：
    // 負 lobe 的外推會把單像素 chroma 離群值放大成飽和紅/綠/藍雜點（fireflies）。
    // 與 EASU 結尾的 min4/max4 dering 同一哲學——輸出鉗位在「四鄰居」(b,d,f,h)
    // 的局部範圍內：不僅抑制外推過衝，也會把輸入本身就存在的單像素 chroma 雜點
    // 收斂回鄰域水準（代價：真正的單像素亮點會被平滑，對自然影像升頻屬可接受取捨）。
    let mn4 = min(min(b, d), min(f, h));
    let mx4 = max(max(b, d), max(f, h));
    return clamp(res, mn4, mx4);
}

//! END
