// CAS (AMD FidelityFX Contrast Adaptive Sharpening) — Magpie v4 → WGFX WGSL port
//
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/CAS/CAS.hlsl
//   which is itself a port of AMD FidelityFX-CAS:
//     https://github.com/GPUOpen-Effects/FidelityFX-CAS/blob/master/ffx-cas/ffx_cas.h
//
// Original license / author:
//   AMD FidelityFX-CAS — Copyright (c) 2021 Advanced Micro Devices, Inc. All rights
//     reserved. (MIT)
//   Magpie HLSL port — Copyright (c) Blinue and Magpie contributors. (MIT)
//   This WGFX WGSL port changes only the pass structure and the sampling idioms;
//   every CAS constant and every line of the filter math is copied verbatim.
//
// Porting notes:
//   1. Magpie's `void Pass1(uint2 blockStart, uint3 threadId)` is a CS-style pass.
//      It maps onto a WGFX compute shader: `blockStart` = workgroup_id.xy * BLOCK_SIZE
//      and `threadId.x` = local_invocation_id.x. Magpie's `//! BLOCK_SIZE 16` +
//      `//! NUM_THREADS 64` means "one 16x16 pixel tile per group, 64 threads, each
//      thread owns a 2x2 quad" — exactly the `Rmp8x8(lid.x) << 1` mapping used below.
//   2. The HLSL contains two variants selected by `#ifdef MP_FP16`: a packed
//      two-pixel variant (`CasFilterH`) and the scalar variant (`CasFilter`). They are
//      numerically identical — the packed variant's `.x` lane reproduces the scalar
//      variant at pos=(1,1) and its `.y` lane reproduces the scalar variant at
//      pos=(2,1) (verified tap by tap against the HLSL source) — so the scalar
//      variant is ported. Same for `CAS_BETTER_DIAGONALS`, which is commented out in
//      the HLSL source, so the non-diagonal branch is the one that is ported.
//   3. `INPUT.GatherRed/GatherGreen/GatherBlue` → WGSL
//      `textureGather(0|1|2, INPUT, sam, tpos)`. WebGPU's gather uses the same
//      convention as D3D (quad base texel = floor(uv * size - 0.5); the returned
//      vector is w = top-left, z = top-right, x = bottom-left, y = bottom-right),
//      so Magpie's "w z / x y" fill is copied verbatim. The WGFX runtime creates
//      every sampler as a *filtering* sampler (nearest min/mag), which is what
//      `textureGather` requires; the gather itself is not filtered.
//   4. `SamplerState sam` (POINT / CLAMP) → `//! FILTER POINT` + `//! ADDRESS CLAMP`.
//      The gather clamps at the border exactly like the D3D POINT+CLAMP sampler.
//   5. `rcp(x)` → `1.0 / x`. HLSL's `rcp(0)` is +Inf, and the following
//      `0 * Inf` forms a NaN that D3D's `saturate()` heals to 0 (D3D min/max use
//      comparison semantics). WGSL's min/max propagate NaN per spec, so the
//      reciprocal is guarded by an explicit `select` that reproduces the *healed*
//      HLSL result bit-for-bit (an all-black neighbourhood gives amp = 0, hence the
//      identity filter). `rcpWeight`'s denominator is provably in [0.2, 1] and can
//      never be zero, so the guard never changes it.
//   6. `lerp` → `mix`; `saturate`, `min`, `max`, `sqrt` keep their names (WGSL
//      built-ins with the same semantics).
//   7. `MF3 src[4][4]` (column-major: `src[x][y]`) becomes
//      `array<array<vec3<f32>, 4>, 4>`; the helper receives it through a
//      `ptr<function, …>` because WGSL has no `inout` array parameter.
//   8. `OUTPUT[gxy] = MF4(color, 1)` → `textureStore(OUTPUT, vec2<i32>(gxy), …)`.
//      WebGPU silently discards stores whose coordinates are outside the texture,
//      which reproduces D3D's behaviour for the 2x2 quad at the right/bottom edge.

//! VERSION 4
//! SORT_NAME CAS

//! PARAMETER sharpness
//! TYPE float
//! LABEL "Sharpness"
//! MIN 0
//! MAX 1
//! DEFAULT 0.4
//! STEP 0.01
float sharpness;

//! TEXTURE
Texture2D INPUT;

//! TEXTURE
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! SAMPLER
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! DESC "AMD FidelityFX CAS — contrast adaptive sharpening (1:1)"
//! IN INPUT
//! OUT OUTPUT
//! BLOCK_SIZE 16
//! NUM_THREADS 64

// Magpie's `Rmp8x8`: linear thread index → 8x8 position inside the group.
fn Rmp8x8(a: u32) -> vec2<u32> {
    return vec2<u32>(a / 8u, a % 8u);
}

// #define min3(x, y, z) (min(x, min(y, z)))
fn min3(x: f32, y: f32, z: f32) -> f32 {
    return min(x, min(y, z));
}

// #define max3(x, y, z) (max(x, max(y, z)))
fn max3(x: f32, y: f32, z: f32) -> f32 {
    return max(x, max(y, z));
}

// HLSL `rcp` with D3D's NaN-healing semantics — see porting note 5.
fn rcpH(x: f32) -> f32 {
    return select(0.0, 1.0 / x, x != 0.0);
}

// AMD FidelityFX CAS — scalar filter, verbatim from CAS.hlsl (`CasFilter`).
// `src` is the 4x4 tap window; the HLSL indexes it as src[x][y] and so does this.
fn CasFilter(
    src: ptr<function, array<array<vec3<f32>, 4>, 4>>,
    pos: vec2<u32>,
    peak: f32,
) -> vec3<f32> {
    let s = *src;

    // a b c
    // d e f
    // g h i
    let a = s[pos.x - 1u][pos.y - 1u];
    let b = s[pos.x][pos.y - 1u];
    let c = s[pos.x + 1u][pos.y - 1u];
    let d = s[pos.x - 1u][pos.y];
    let e = s[pos.x][pos.y];
    let f = s[pos.x + 1u][pos.y];
    let g = s[pos.x - 1u][pos.y + 1u];
    let h = s[pos.x][pos.y + 1u];
    let i = s[pos.x + 1u][pos.y + 1u];

    // Soft min and max.
    //  a b c             b
    //  d e f * 0.5  +  d e f * 0.5
    //  g h i             h
    // These are 2.0x bigger (factored out the extra multiply).
    let mnR = min3(min3(d.r, e.r, f.r), b.r, h.r);
    let mnG = min3(min3(d.g, e.g, f.g), b.g, h.g);
    let mnB = min3(min3(d.b, e.b, f.b), b.b, h.b);
    let mxR = max3(max3(d.r, e.r, f.r), b.r, h.r);
    let mxG = max3(max3(d.g, e.g, f.g), b.g, h.g);
    let mxB = max3(max3(d.b, e.b, f.b), b.b, h.b);
    // Smooth minimum distance to signal limit divided by smooth max.
    let rcpMR = rcpH(mxR);
    let rcpMG = rcpH(mxG);
    let rcpMB = rcpH(mxB);
    let ampR = saturate(min(mnR, 1.0 - mxR) * rcpMR);
    let ampG = saturate(min(mnG, 1.0 - mxG) * rcpMG);
    let ampB = saturate(min(mnB, 1.0 - mxB) * rcpMB);
    // Shaping amount of sharpening.
    let ampRs = sqrt(ampR);
    let ampGs = sqrt(ampG);
    let ampBs = sqrt(ampB);
    // Filter shape.
    //  0 w 0
    //  w 1 w
    //  0 w 0
    let wR = ampRs * peak;
    let wG = ampGs * peak;
    let wB = ampBs * peak;
    // Filter.
    // Using green coef only, depending on dead code removal to strip out the extra overhead.
    let rcpWeight = rcpH(1.0 + 4.0 * wG);

    return vec3<f32>(
        saturate((b.r * wG + d.r * wG + f.r * wG + h.r * wG + e.r) * rcpWeight),
        saturate((b.g * wG + d.g * wG + f.g * wG + h.g * wG + e.g) * rcpWeight),
        saturate((b.b * wG + d.b * wG + f.b * wG + h.b * wG + e.b) * rcpWeight)
    );
}

fn Pass1(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let blockStart = workgroup_id.xy * vec2<u32>(16u, 16u);
    var gxy = blockStart + (Rmp8x8(local_id.x) << vec2<u32>(1u, 1u));

    let outputSize = scene.outputSize;
    if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
        return;
    }

    let inputPt = scene.inputPt;

    var src: array<array<vec3<f32>, 4>, 4>;
    for (var i: u32 = 0u; i <= 2u; i += 2u) {
        for (var j: u32 = 0u; j <= 2u; j += 2u) {
            let tpos = vec2<f32>(gxy + vec2<u32>(i, j)) * inputPt;
            let sr = textureGather(0, INPUT, sam, tpos);
            let sg = textureGather(1, INPUT, sam, tpos);
            let sb = textureGather(2, INPUT, sam, tpos);

            // w z
            // x y
            src[i][j] = vec3<f32>(sr.w, sg.w, sb.w);
            src[i][j + 1u] = vec3<f32>(sr.x, sg.x, sb.x);
            src[i + 1u][j] = vec3<f32>(sr.z, sg.z, sb.z);
            src[i + 1u][j + 1u] = vec3<f32>(sr.y, sg.y, sb.y);
        }
    }

    // const MF peak = -rcp(lerp(8.0, 5.0, (MF)sharpness));
    let peak = -1.0 / mix(8.0, 5.0, uniforms.sharpness);

    textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(CasFilter(&src, vec2<u32>(1u, 1u), peak), 1.0));

    gxy.x += 1u;
    textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(CasFilter(&src, vec2<u32>(2u, 1u), peak), 1.0));

    gxy.y += 1u;
    textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(CasFilter(&src, vec2<u32>(2u, 2u), peak), 1.0));

    gxy.x -= 1u;
    textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(CasFilter(&src, vec2<u32>(1u, 2u), peak), 1.0));
}
