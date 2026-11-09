// CAS_Scaling (AMD FidelityFX Contrast Adaptive Sharpening, scaling variant)
// — Magpie v4 → WGFX WGSL port
//
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/CAS/CAS_Scaling.hlsl
//   which is itself a port of AMD FidelityFX-CAS:
//     https://github.com/GPUOpen-Effects/FidelityFX-CAS/blob/9fabcc9a2c45f958aff55ddfda337e74ef894b7f/ffx-cas/ffx_cas.h
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
//      and `threadId.x` = local_invocation_id.x. `//! BLOCK_SIZE 16` + `//! NUM_THREADS 64`
//      means "one 16x16 pixel tile per group, 64 threads"; the 8x8 thread grid covers
//      the 16x16 tile because each thread writes the four pixels
//      (x,y), (x+8,y), (x+8,y+8), (x,y+8).
//   2. The HLSL contains two variants selected by `#ifdef MP_FP16`: a packed
//      two-horizontal-tile variant (`CasFilterH`) and the scalar variant
//      (`CasFilter`). They are numerically identical: `CasFilterH`'s second tile is
//      evaluated at `pp.x + offset` with `offset = 8 / GetScale().x`, and because
//      `const0.x = inputSize.x / outputSize.x = 1 / scale.x` this is exactly the
//      `pp.x` that the scalar `CasFilter` computes for the pixel 8 columns to the
//      right. The scalar variant is ported.
//      `CAS_BETTER_DIAGONALS` is commented out in the HLSL source, so the
//      non-diagonal branch is the one that is ported.
//   3. `INPUT.Load(int3(sp + int2(dx, dy), 0))` → WGSL
//      `textureLoad(INPUT, sp + vec2<i32>(dx, dy), 0)`. D3D's `Load` returns 0 for
//      out-of-range coordinates and WebGPU's `textureLoad` does the same, so the
//      un-clamped border behaviour of the original is preserved exactly (this
//      effect deliberately does *not* use a sampler).
//   4. `rcp(x)` → `1.0 / x`. HLSL's `rcp(0)` is +Inf and the following `0 * Inf`
//      forms a NaN that D3D's `saturate()` heals to 0 (D3D min/max use comparison
//      semantics). WGSL's min/max propagate NaN per spec, so the reciprocal is
//      guarded by an explicit `select` that reproduces the *healed* HLSL result
//      (all-black neighbourhood → amp = 0 → identity filter).
//   5. `lerp` → `mix`; `saturate`, `min`, `max`, `sqrt` keep their names.
//   6. Magpie declares `Texture2D OUTPUT;` without a size, which means "same size as
//      the input". The WGFX equivalent is spelled out explicitly as
//      `//! WIDTH INPUT_WIDTH` / `//! HEIGHT INPUT_HEIGHT`. A host that wants to use
//      this effect as an actual scaler must pass `outWidth`/`outHeight` to
//      `WGFX.create()`, and the host-provided size then overrides the declaration
//      (the effect reads the real output size through `scene.outputSize` /
//      `scene.outputPt`, so it adapts to any output resolution).
//   7. `GetInputSize()`, `GetInputPt()`, `GetOutputPt()` → `scene.inputSize`,
//      `scene.inputPt`, `scene.outputPt`.
//   8. `OUTPUT[gxy] = float4(color, 1)` → `textureStore(OUTPUT, vec2<i32>(gxy), …)`;
//      WebGPU silently discards stores outside the texture, matching D3D.

//! VERSION 4
//! SORT_NAME CAS_Scaling

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

//! PASS 1
//! DESC "AMD FidelityFX CAS — contrast adaptive sharpening with scaling"
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

// HLSL `rcp` with D3D's NaN-healing semantics — see porting note 4.
fn rcpH(x: f32) -> f32 {
    return select(0.0, 1.0 / x, x != 0.0);
}

// AMD FidelityFX CAS — scaling filter, verbatim from CAS_Scaling.hlsl (`CasFilter`).
fn CasFilter(ip: vec2<u32>, const0: vec4<f32>, peak: f32) -> vec3<f32> {
    // Scaling algorithm adaptively interpolates between nearest 4 results of the
    // non-scaling algorithm.
    //  a b c d
    //  e f g h
    //  i j k l
    //  m n o p
    // Working these 4 results.
    //  +-----+-----+
    //  |     |     |
    //  |  f..|..g  |
    //  |  .  |  .  |
    //  +-----+-----+
    //  |  .  |  .  |
    //  |  j..|..k  |
    //  |     |     |
    //  +-----+-----+
    var pp = vec2<f32>(ip) * const0.xy + const0.zw;
    let fp = floor(pp);
    pp -= fp;
    let sp = vec2<i32>(fp);

    let a = textureLoad(INPUT, sp + vec2<i32>(-1, -1), 0).rgb;
    let b = textureLoad(INPUT, sp + vec2<i32>(0, -1), 0).rgb;
    let e = textureLoad(INPUT, sp + vec2<i32>(-1, 0), 0).rgb;
    let f = textureLoad(INPUT, sp + vec2<i32>(0, 0), 0).rgb;
    let c = textureLoad(INPUT, sp + vec2<i32>(1, -1), 0).rgb;
    let d = textureLoad(INPUT, sp + vec2<i32>(2, -1), 0).rgb;
    let g = textureLoad(INPUT, sp + vec2<i32>(1, 0), 0).rgb;
    let h = textureLoad(INPUT, sp + vec2<i32>(2, 0), 0).rgb;
    let i = textureLoad(INPUT, sp + vec2<i32>(-1, 1), 0).rgb;
    let j = textureLoad(INPUT, sp + vec2<i32>(0, 1), 0).rgb;
    let m = textureLoad(INPUT, sp + vec2<i32>(-1, 2), 0).rgb;
    let n = textureLoad(INPUT, sp + vec2<i32>(0, 2), 0).rgb;
    let k = textureLoad(INPUT, sp + vec2<i32>(1, 1), 0).rgb;
    let l = textureLoad(INPUT, sp + vec2<i32>(2, 1), 0).rgb;
    let o = textureLoad(INPUT, sp + vec2<i32>(1, 2), 0).rgb;
    let p = textureLoad(INPUT, sp + vec2<i32>(2, 2), 0).rgb;

    // Soft min and max.
    // These are 2.0x bigger (factored out the extra multiply).
    //  a b c             b
    //  e f g * 0.5  +  e f g * 0.5  [F]
    //  i j k             j
    let mnfR = min3(min3(b.r, e.r, f.r), g.r, j.r);
    let mnfG = min3(min3(b.g, e.g, f.g), g.g, j.g);
    let mnfB = min3(min3(b.b, e.b, f.b), g.b, j.b);
    let mxfR = max3(max3(b.r, e.r, f.r), g.r, j.r);
    let mxfG = max3(max3(b.g, e.g, f.g), g.g, j.g);
    let mxfB = max3(max3(b.b, e.b, f.b), g.b, j.b);
    //  b c d             c
    //  f g h * 0.5  +  f g h * 0.5  [G]
    //  j k l             k
    let mngR = min3(min3(c.r, f.r, g.r), h.r, k.r);
    let mngG = min3(min3(c.g, f.g, g.g), h.g, k.g);
    let mngB = min3(min3(c.b, f.b, g.b), h.b, k.b);
    let mxgR = max3(max3(c.r, f.r, g.r), h.r, k.r);
    let mxgG = max3(max3(c.g, f.g, g.g), h.g, k.g);
    let mxgB = max3(max3(c.b, f.b, g.b), h.b, k.b);
    //  e f g             f
    //  i j k * 0.5  +  i j k * 0.5  [J]
    //  m n o             n
    let mnjR = min3(min3(f.r, i.r, j.r), k.r, n.r);
    let mnjG = min3(min3(f.g, i.g, j.g), k.g, n.g);
    let mnjB = min3(min3(f.b, i.b, j.b), k.b, n.b);
    let mxjR = max3(max3(f.r, i.r, j.r), k.r, n.r);
    let mxjG = max3(max3(f.g, i.g, j.g), k.g, n.g);
    let mxjB = max3(max3(f.b, i.b, j.b), k.b, n.b);
    //  f g h             g
    //  j k l * 0.5  +  j k l * 0.5  [K]
    //  n o p             o
    let mnkR = min3(min3(g.r, j.r, k.r), l.r, o.r);
    let mnkG = min3(min3(g.g, j.g, k.g), l.g, o.g);
    let mnkB = min3(min3(g.b, j.b, k.b), l.b, o.b);
    let mxkR = max3(max3(g.r, j.r, k.r), l.r, o.r);
    let mxkG = max3(max3(g.g, j.g, k.g), l.g, o.g);
    let mxkB = max3(max3(g.b, j.b, k.b), l.b, o.b);
    // Smooth minimum distance to signal limit divided by smooth max.
    let rcpMfR = rcpH(mxfR);
    let rcpMfG = rcpH(mxfG);
    let rcpMfB = rcpH(mxfB);
    let rcpMgR = rcpH(mxgR);
    let rcpMgG = rcpH(mxgG);
    let rcpMgB = rcpH(mxgB);
    let rcpMjR = rcpH(mxjR);
    let rcpMjG = rcpH(mxjG);
    let rcpMjB = rcpH(mxjB);
    let rcpMkR = rcpH(mxkR);
    let rcpMkG = rcpH(mxkG);
    let rcpMkB = rcpH(mxkB);
    let ampfR = saturate(min(mnfR, 1.0 - mxfR) * rcpMfR);
    let ampfG = saturate(min(mnfG, 1.0 - mxfG) * rcpMfG);
    let ampfB = saturate(min(mnfB, 1.0 - mxfB) * rcpMfB);
    let ampgR = saturate(min(mngR, 1.0 - mxgR) * rcpMgR);
    let ampgG = saturate(min(mngG, 1.0 - mxgG) * rcpMgG);
    let ampgB = saturate(min(mngB, 1.0 - mxgB) * rcpMgB);
    let ampjR = saturate(min(mnjR, 1.0 - mxjR) * rcpMjR);
    let ampjG = saturate(min(mnjG, 1.0 - mxjG) * rcpMjG);
    let ampjB = saturate(min(mnjB, 1.0 - mxjB) * rcpMjB);
    let ampkR = saturate(min(mnkR, 1.0 - mxkR) * rcpMkR);
    let ampkG = saturate(min(mnkG, 1.0 - mxkG) * rcpMkG);
    let ampkB = saturate(min(mnkB, 1.0 - mxkB) * rcpMkB);
    // Shaping amount of sharpening.
    let ampfRs = sqrt(ampfR);
    let ampfGs = sqrt(ampfG);
    let ampfBs = sqrt(ampfB);
    let ampgRs = sqrt(ampgR);
    let ampgGs = sqrt(ampgG);
    let ampgBs = sqrt(ampgB);
    let ampjRs = sqrt(ampjR);
    let ampjGs = sqrt(ampjG);
    let ampjBs = sqrt(ampjB);
    let ampkRs = sqrt(ampkR);
    let ampkGs = sqrt(ampkG);
    let ampkBs = sqrt(ampkB);
    // Filter shape.
    //  0 w 0
    //  w 1 w
    //  0 w 0
    let wfR = ampfRs * peak;
    let wfG = ampfGs * peak;
    let wfB = ampfBs * peak;
    let wgR = ampgRs * peak;
    let wgG = ampgGs * peak;
    let wgB = ampgBs * peak;
    let wjR = ampjRs * peak;
    let wjG = ampjGs * peak;
    let wjB = ampjBs * peak;
    let wkR = ampkRs * peak;
    let wkG = ampkGs * peak;
    let wkB = ampkBs * peak;
    // Blend between 4 results.
    //  s t
    //  u v
    var s = (1.0 - pp.x) * (1.0 - pp.y);
    var t = pp.x * (1.0 - pp.y);
    var u = (1.0 - pp.x) * pp.y;
    var v = pp.x * pp.y;
    // Thin edges to hide bilinear interpolation (helps diagonals).
    // NOTE: the HLSL multiplies s/t/u/v **in place** here and the later `qf/qg/qj/qk`
    // weights reuse the already-thinned values, so `var` + `*=` is kept verbatim.
    let thinB = 1.0 / 32.0;
    s *= rcpH(thinB + (mxfG - mnfG));
    t *= rcpH(thinB + (mxgG - mngG));
    u *= rcpH(thinB + (mxjG - mnjG));
    v *= rcpH(thinB + (mxkG - mnkG));
    // Final weighting.
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    let qbeR = wfR * s;
    let qbeG = wfG * s;
    let qbeB = wfB * s;
    let qchR = wgR * t;
    let qchG = wgG * t;
    let qchB = wgB * t;
    let qfR = wgR * t + wjR * u + s;
    let qfG = wgG * t + wjG * u + s;
    let qfB = wgB * t + wjB * u + s;
    let qgR = wfR * s + wkR * v + t;
    let qgG = wfG * s + wkG * v + t;
    let qgB = wfB * s + wkB * v + t;
    let qjR = wfR * s + wkR * v + u;
    let qjG = wfG * s + wkG * v + u;
    let qjB = wfB * s + wkB * v + u;
    let qkR = wgR * t + wjR * u + v;
    let qkG = wgG * t + wjG * u + v;
    let qkB = wgB * t + wjB * u + v;
    let qinR = wjR * u;
    let qinG = wjG * u;
    let qinB = wjB * u;
    let qloR = wkR * v;
    let qloG = wkG * v;
    let qloB = wkB * v;
    // Filter.
    // Using green coef only, depending on dead code removal to strip out the extra overhead.
    let rcpWG = rcpH(2.0 * qbeG + 2.0 * qchG + 2.0 * qinG + 2.0 * qloG + qfG + qgG + qjG + qkG);

    return vec3<f32>(
        saturate((b.r * qbeG + e.r * qbeG + c.r * qchG + h.r * qchG + i.r * qinG + n.r * qinG + l.r * qloG + o.r * qloG + f.r * qfG + g.r * qgG + j.r * qjG + k.r * qkG) * rcpWG),
        saturate((b.g * qbeG + e.g * qbeG + c.g * qchG + h.g * qchG + i.g * qinG + n.g * qinG + l.g * qloG + o.g * qloG + f.g * qfG + g.g * qgG + j.g * qjG + k.g * qkG) * rcpWG),
        saturate((b.b * qbeG + e.b * qbeG + c.b * qchG + h.b * qchG + i.b * qinG + n.b * qinG + l.b * qloG + o.b * qloG + f.b * qfG + g.b * qgG + j.b * qjG + k.b * qkG) * rcpWG)
    );
}

fn Pass1(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let blockStart = workgroup_id.xy * vec2<u32>(16u, 16u);
    var gxy = blockStart + Rmp8x8(local_id.x);

    let outputSize = scene.outputSize;
    if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
        return;
    }

    // float4 const0;
    // const0.xy = GetInputSize() * GetOutputPt();
    // const0.zw = 0.5 * const0.xy - 0.5;
    let const0xy = vec2<f32>(scene.inputSize) * scene.outputPt;
    let const0 = vec4<f32>(const0xy, 0.5 * const0xy - 0.5);

    // const MF peak = MF(-rcp(lerp(8.0, 5.0, sharpness)));
    let peak = -1.0 / mix(8.0, 5.0, uniforms.sharpness);

    textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(CasFilter(gxy, const0, peak), 1.0));

    gxy.x += 8u;
    textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(CasFilter(gxy, const0, peak), 1.0));

    gxy.y += 8u;
    textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(CasFilter(gxy, const0, peak), 1.0));

    gxy.x -= 8u;
    textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(CasFilter(gxy, const0, peak), 1.0));
}
