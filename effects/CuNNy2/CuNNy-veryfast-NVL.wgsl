// CuNNy-veryfast-NVL — Magpie v4 → WGFX WGSL port
//
// Ported from effects_src/CuNNy2/CuNNy-veryfast-NVL.hlsl
//   (upstream: funnyplanter/CuNNy — GNU GPL v3; Magpie HLSL port by the Magpie project).
// Every trained weight constant is copied verbatim; only the pass structure and
// sampling idioms are translated.
//
// Weights are kept **inline** as literals, following the passed FSRCNNX /
// Anime4K_GAN ports (no storage buffers, no host-side upload).
//
// Porting notes:
//   1. Magpie CS passes (`void PassN(uint2 blockStart, uint3 tid)`) become WGFX compute
//      entry points: blockStart = workgroup_id.xy * BLOCK_SIZE, tid = local_invocation_id.
//      `//!NUM_THREADS 64` → @workgroup_size(64,1,1); the Rmp8x8 8x8 thread mapping is kept.
//   2. The `O(...)` / `L0..` / `l0..` sampling macros are expanded inline at each call site
//      (they reference the pass-local pos/pt, so inlining is semantics-preserving).
//   3. `mad(s, V4(w), acc)` → `acc + s * vec4<f32>(w)` (scalar broadcasts; component-wise,
//      identical to HLSL mad).  `MulAdd(v, M4(w), acc)` → `acc + mat4x4<f32>(w) * v` and
//      `MulAdd(v, M3x4(w), acc)` → `acc + mat3x4<f32>(w) * v`: the HLSL row-major constructor
//      and the WGSL column-major constructor cancel in the flipped product order, so every
//      weight number is copied verbatim without transposition (porting-guide §2.1).
//   4. The YUV colour matrices (`static const MF3x3`) are applied as mul(M, v) → M * v with
//      WGSL columns = HLSL rows; the flat number sequence is unchanged.
//   5. Intermediate textures keep the Magpie format (R8G8B8A8_UNORM → rgba8unorm),
//      declared as `_sampled`/`_storaged` alias pairs (same logical texture, two views).
//      Explicit //!WIDTH/HEIGHT were added — the HLSL relies on Magpie's implicit
//      input-size default, WGFX requires the declaration.
//   6. ReLU `max(x, 0.0)` → `max(x, vec4<f32>(0.0))`.  NaN note: WGSL max propagates NaN
//      while HLSL heals it; every input here comes from a UNORM/SNORM sampler (finite by
//      construction) and the weights are finite, so NaN healing is unreachable — no guard needed.
//   7. `tanh` is a WGSL builtin; used verbatim.
//   8. The last pass is the out-shuffle: BLOCK_SIZE 16, one 2x2 output quad per thread,
//      result added directly to INPUT rgb (saturate) and stored to the 2x OUTPUT.
//      gxy/pos are mutated in place there, so they are declared `var`.
//   9. //!USE MulAdd / //!CAPABILITY FP16 are capability markers, kept as-is; arithmetic
//      runs in f32 with rgba8unorm/rgba8snorm intermediates (same route as the passed
//      FSRCNNX / Anime4K ports; node-webgpu f32→f16 texture truncation does not apply).
//  10. Float literal suffixes (`0.5f`) are dropped; values unchanged.
//  11. Passes writing more than 4 storage textures (Dawn's default per-stage limit) are
//      split into consecutive sub-passes with identical math: each output vec4 is an
//      independent function of the same 9 samples, so only the textureStore lines are
//      partitioned; sub-passes re-read the same taps (marked "(part k/n)" in //!DESC).

//!VERSION 4
//!SORT_NAME CuNNy-0001
//!USE MulAdd
//!CAPABILITY FP16

//!TEXTURE
Texture2D INPUT;

//!TEXTURE
//!WIDTH INPUT_WIDTH * 2
//!HEIGHT INPUT_HEIGHT * 2
Texture2D OUTPUT;

//!SAMPLER
//!FILTER POINT
SamplerState SP;

//!SAMPLER
//!FILTER LINEAR
SamplerState SL;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R8G8B8A8_UNORM
var T0_sampled: texture_2d<f32>;
var T0_storaged: texture_storage_2d<rgba8unorm, write>;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R8G8B8A8_UNORM
var T1_sampled: texture_2d<f32>;
var T1_storaged: texture_storage_2d<rgba8unorm, write>;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R8G8B8A8_UNORM
var T2_sampled: texture_2d<f32>;
var T2_storaged: texture_storage_2d<rgba8unorm, write>;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R8G8B8A8_UNORM
var T3_sampled: texture_2d<f32>;
var T3_storaged: texture_storage_2d<rgba8unorm, write>;

//!COMMON
// --- Magpie StubDefs.hlsli helpers (verbatim WGSL equivalents) ---
alias uint2 = vec2<u32>;
alias uint3 = vec3<u32>;
alias int2 = vec2<i32>;
alias MF = f32;
alias MF2 = vec2<f32>;
alias MF3 = vec3<f32>;
alias MF4 = vec4<f32>;

fn GetInputSize() -> uint2 { return scene.inputSize; }
fn GetInputPt() -> MF2 { return scene.inputPt; }
fn GetOutputSize() -> uint2 { return scene.outputSize; }
fn GetOutputPt() -> MF2 { return scene.outputPt; }

// Magpie Rmp8x8: linear thread index → 8x8 position inside the group.
fn Rmp8x8(a: u32) -> uint2 { return uint2(a / 8u, a % 8u); }
//!END COMMON

//!PASS 1
//!DESC in (1x8)
//!BLOCK_SIZE 8
//!NUM_THREADS 64
//!IN INPUT
//!OUT T0_storaged, T1_storaged

fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) tid: vec3<u32>) {
    let blockStart = workgroup_id.xy * vec2<u32>(8u, 8u);
	let pt = GetInputPt();
	var gxy = Rmp8x8(tid.x) + blockStart;
	let sz = GetInputSize();
	if (gxy.x >= sz.x || gxy.y >= sz.y) {
		return;
	}
	var pos = (vec2<f32>(gxy) + vec2<f32>(0.5)) * pt;
	var s0_0_0: f32; var s0_0_1: f32; var s0_0_2: f32; var s0_1_0: f32; var s0_1_1: f32; var s0_1_2: f32; var s0_2_0: f32; var s0_2_1: f32; var s0_2_2: f32;
	var r0 = vec4<f32>(0.0); var r1 = vec4<f32>(0.0);
	r0 = vec4<f32>(1.026e-03, -2.981e-03, 2.268e-03, -1.057e-03);
	r1 = vec4<f32>(-1.665e-03, 3.286e-03, -3.161e-03, -9.035e-04);
	s0_0_0 = f32(dot(vec3<f32>(0.299, 0.587, 0.114), textureSampleLevel(INPUT, SP, pos + vec2<f32>((-1.0), (-1.0)) * pt, 0.0).rgb)); s0_0_1 = f32(dot(vec3<f32>(0.299, 0.587, 0.114), textureSampleLevel(INPUT, SP, pos + vec2<f32>((0.0), (-1.0)) * pt, 0.0).rgb)); s0_0_2 = f32(dot(vec3<f32>(0.299, 0.587, 0.114), textureSampleLevel(INPUT, SP, pos + vec2<f32>((1.0), (-1.0)) * pt, 0.0).rgb));
	s0_1_0 = f32(dot(vec3<f32>(0.299, 0.587, 0.114), textureSampleLevel(INPUT, SP, pos + vec2<f32>((-1.0), (0.0)) * pt, 0.0).rgb)); s0_1_1 = f32(dot(vec3<f32>(0.299, 0.587, 0.114), textureSampleLevel(INPUT, SP, pos + vec2<f32>((0.0), (0.0)) * pt, 0.0).rgb)); s0_1_2 = f32(dot(vec3<f32>(0.299, 0.587, 0.114), textureSampleLevel(INPUT, SP, pos + vec2<f32>((1.0), (0.0)) * pt, 0.0).rgb));
	s0_2_0 = f32(dot(vec3<f32>(0.299, 0.587, 0.114), textureSampleLevel(INPUT, SP, pos + vec2<f32>((-1.0), (1.0)) * pt, 0.0).rgb)); s0_2_1 = f32(dot(vec3<f32>(0.299, 0.587, 0.114), textureSampleLevel(INPUT, SP, pos + vec2<f32>((0.0), (1.0)) * pt, 0.0).rgb)); s0_2_2 = f32(dot(vec3<f32>(0.299, 0.587, 0.114), textureSampleLevel(INPUT, SP, pos + vec2<f32>((1.0), (1.0)) * pt, 0.0).rgb));
	r0 = ((s0_0_0) * (vec4<f32>(4.998e-03, -1.996e-02, 2.062e-02, -1.826e-02)) + (r0));
	r1 = ((s0_0_0) * (vec4<f32>(-5.265e-03, 2.075e-03, 2.429e-02, 3.332e-02)) + (r1));
	r0 = ((s0_0_1) * (vec4<f32>(2.804e-02, 4.874e-02, 3.034e-02, 7.068e-03)) + (r0));
	r1 = ((s0_0_1) * (vec4<f32>(2.430e-02, -1.450e-01, 1.032e-02, 4.446e-01)) + (r1));
	r0 = ((s0_0_2) * (vec4<f32>(1.752e-02, -4.398e-02, -1.954e-02, 1.824e-02)) + (r0));
	r1 = ((s0_0_2) * (vec4<f32>(-2.447e-02, 3.411e-02, -3.408e-02, -8.259e-02)) + (r1));
	r0 = ((s0_1_0) * (vec4<f32>(3.185e-02, -3.662e-01, -1.870e-02, 8.200e-01)) + (r0));
	r1 = ((s0_1_0) * (vec4<f32>(-7.897e-03, 1.151e-01, -2.607e-01, -3.053e-02)) + (r1));
	r0 = ((s0_1_1) * (vec4<f32>(-9.682e-02, 4.676e-01, -1.874e-01, -8.066e-01)) + (r0));
	r1 = ((s0_1_1) * (vec4<f32>(-8.105e-01, 4.792e-01, 8.066e-01, 9.627e-02)) + (r1));
	r0 = ((s0_1_2) * (vec4<f32>(4.775e-01, -8.455e-02, 8.943e-02, -2.106e-02)) + (r0));
	r1 = ((s0_1_2) * (vec4<f32>(8.912e-02, -9.258e-02, 3.846e-02, -7.281e-02)) + (r1));
	r0 = ((s0_2_0) * (vec4<f32>(-1.763e-02, -2.789e-01, 4.132e-01, -2.679e-02)) + (r0));
	r1 = ((s0_2_0) * (vec4<f32>(8.231e-03, 8.443e-02, -2.719e-01, 4.610e-04)) + (r1));
	r0 = ((s0_2_1) * (vec4<f32>(3.664e-03, 2.998e-01, -6.781e-02, 2.461e-02)) + (r0));
	r1 = ((s0_2_1) * (vec4<f32>(7.667e-01, -1.057e-02, -2.979e-01, 5.408e-02)) + (r1));
	r0 = ((s0_2_2) * (vec4<f32>(-6.392e-02, -1.812e-02, 1.094e-02, 2.662e-03)) + (r0));
	r1 = ((s0_2_2) * (vec4<f32>(-3.848e-02, 2.277e-02, -1.486e-02, -1.206e-02)) + (r1));
	r0 = max((r0), vec4<f32>(0.0));
	textureStore(T0_storaged, vec2<i32>(gxy), r0);
	r1 = max((r1), vec4<f32>(0.0));
	textureStore(T1_storaged, vec2<i32>(gxy), r1);
}

//!PASS 2
//!DESC conv1 (8x8)
//!BLOCK_SIZE 8
//!NUM_THREADS 64
//!IN T0_sampled, T1_sampled
//!OUT T2_storaged, T3_storaged

fn Pass2(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) tid: vec3<u32>) {
    let blockStart = workgroup_id.xy * vec2<u32>(8u, 8u);
	let pt = GetInputPt();
	var gxy = Rmp8x8(tid.x) + blockStart;
	let sz = GetInputSize();
	if (gxy.x >= sz.x || gxy.y >= sz.y) {
		return;
	}
	var pos = (vec2<f32>(gxy) + vec2<f32>(0.5)) * pt;
	var s0_0_0: vec4<f32>; var s0_0_1: vec4<f32>; var s0_0_2: vec4<f32>; var s0_1_0: vec4<f32>; var s0_1_1: vec4<f32>; var s0_1_2: vec4<f32>; var s0_2_0: vec4<f32>; var s0_2_1: vec4<f32>; var s0_2_2: vec4<f32>; var s1_0_0: vec4<f32>; var s1_0_1: vec4<f32>; var s1_0_2: vec4<f32>; var s1_1_0: vec4<f32>; var s1_1_1: vec4<f32>; var s1_1_2: vec4<f32>; var s1_2_0: vec4<f32>; var s1_2_1: vec4<f32>; var s1_2_2: vec4<f32>;
	var r0 = vec4<f32>(0.0); var r1 = vec4<f32>(0.0);
	s0_0_0 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((-1.0), (-1.0)) * pt, 0.0)); s0_0_1 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((0.0), (-1.0)) * pt, 0.0)); s0_0_2 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((1.0), (-1.0)) * pt, 0.0));
	s0_1_0 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((-1.0), (0.0)) * pt, 0.0)); s0_1_1 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((0.0), (0.0)) * pt, 0.0)); s0_1_2 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((1.0), (0.0)) * pt, 0.0));
	s0_2_0 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((-1.0), (1.0)) * pt, 0.0)); s0_2_1 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((0.0), (1.0)) * pt, 0.0)); s0_2_2 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((1.0), (1.0)) * pt, 0.0));
	s1_0_0 = vec4<f32>(textureSampleLevel(T1_sampled, SP, pos + vec2<f32>((-1.0), (-1.0)) * pt, 0.0)); s1_0_1 = vec4<f32>(textureSampleLevel(T1_sampled, SP, pos + vec2<f32>((0.0), (-1.0)) * pt, 0.0)); s1_0_2 = vec4<f32>(textureSampleLevel(T1_sampled, SP, pos + vec2<f32>((1.0), (-1.0)) * pt, 0.0));
	s1_1_0 = vec4<f32>(textureSampleLevel(T1_sampled, SP, pos + vec2<f32>((-1.0), (0.0)) * pt, 0.0)); s1_1_1 = vec4<f32>(textureSampleLevel(T1_sampled, SP, pos + vec2<f32>((0.0), (0.0)) * pt, 0.0)); s1_1_2 = vec4<f32>(textureSampleLevel(T1_sampled, SP, pos + vec2<f32>((1.0), (0.0)) * pt, 0.0));
	s1_2_0 = vec4<f32>(textureSampleLevel(T1_sampled, SP, pos + vec2<f32>((-1.0), (1.0)) * pt, 0.0)); s1_2_1 = vec4<f32>(textureSampleLevel(T1_sampled, SP, pos + vec2<f32>((0.0), (1.0)) * pt, 0.0)); s1_2_2 = vec4<f32>(textureSampleLevel(T1_sampled, SP, pos + vec2<f32>((1.0), (1.0)) * pt, 0.0));
	r0 = ((mat4x4<f32>(3.294e-01, 4.686e-02, -1.020e-01, -3.003e-01, -9.869e-02, 3.719e-02, -7.138e-03, 2.881e-01, 3.220e-01, -6.357e-02, -6.724e-02, 7.248e-02, -4.531e-02, 1.142e-02, -1.283e-02, 3.312e-03)) * (s0_0_0) + (r0));
	r1 = ((mat4x4<f32>(-1.897e-02, 1.714e-01, 7.384e-02, 5.059e-01, 1.216e-02, -7.572e-02, -1.294e-01, 1.965e-02, -2.641e-02, 3.440e-02, -3.206e-01, 7.578e-02, 4.822e-02, -5.309e-02, 4.980e-02, -2.297e-01)) * (s0_0_0) + (r1));
	r0 = ((mat4x4<f32>(-5.258e-02, -4.673e-02, -2.457e-01, 5.254e-01, -9.082e-01, -4.544e-01, -9.198e-02, 1.782e-01, -2.241e-01, -8.426e-02, -1.917e-01, 1.895e-01, 1.256e-01, 2.920e-01, -1.948e-01, 1.098e-02)) * (s0_0_1) + (r0));
	r1 = ((mat4x4<f32>(-1.061e-01, 3.819e-01, 9.685e-02, -5.909e+00, -8.986e-03, -5.059e-01, -6.348e-01, -4.566e-01, 1.974e-02, -3.641e-01, -6.473e-02, -7.168e-01, 8.116e-02, -9.474e-02, 6.230e-01, 1.415e-01)) * (s0_0_1) + (r1));
	r0 = ((mat4x4<f32>(8.010e-02, -4.801e-02, 6.959e-02, -1.714e-01, -1.210e-02, -3.453e-02, 7.624e-02, 1.421e-01, -8.770e-01, 2.378e-01, -8.130e-01, 5.840e-01, 2.632e-02, 1.018e-01, -1.675e-01, 2.058e-01)) * (s0_0_2) + (r0));
	r1 = ((mat4x4<f32>(1.740e-02, -2.677e-02, -6.515e-02, -1.401e+01, -4.665e-01, -4.795e-02, -1.572e-01, -2.678e+00, -5.329e-01, -4.122e-01, 2.206e-01, -1.176e+01, 9.385e-02, 1.746e-01, 1.694e-01, -2.664e+00)) * (s0_0_2) + (r1));
	r0 = ((mat4x4<f32>(-3.246e-02, 3.232e-01, -4.307e-01, 2.822e-01, 3.894e-03, -3.894e-02, 1.196e-01, -2.086e-01, 1.110e-01, -3.645e-03, 1.001e-02, -5.595e-02, -4.108e-02, 5.774e-02, -1.192e-02, -5.918e-02)) * (s0_1_0) + (r0));
	r1 = ((mat4x4<f32>(-1.642e-01, -1.149e+00, 2.587e-01, -7.493e-01, 3.742e-02, 2.881e-01, 1.076e-01, 3.382e-02, -7.062e-02, 6.909e-02, -8.023e-02, -3.361e-02, 6.265e-02, -4.848e-02, 1.150e-01, -5.995e-02)) * (s0_1_0) + (r1));
	r0 = ((mat4x4<f32>(2.838e-01, -5.586e-01, 6.230e-01, -4.002e-01, 3.255e-01, -1.077e-01, -5.773e-01, 1.763e-01, -2.401e-01, 1.249e-01, 3.205e-01, -7.300e-02, 6.839e-03, 3.467e-01, 7.362e-01, -7.519e-01)) * (s0_1_1) + (r0));
	r1 = ((mat4x4<f32>(-2.741e-01, 9.805e-01, -1.780e-01, 7.361e-01, -7.676e-01, 1.490e-01, 4.947e-01, 9.604e-02, -2.708e-01, 6.055e-01, -2.115e-01, 3.844e-01, 1.176e+00, 1.091e-01, -3.643e-01, -2.254e-01)) * (s0_1_1) + (r1));
	r0 = ((mat4x4<f32>(4.413e-02, 2.526e-02, 7.251e-02, 5.943e-02, 3.701e-01, -4.031e-02, -1.190e-01, -9.990e-02, -3.710e-02, 3.842e-02, -3.715e-02, -4.212e-01, 5.410e-01, 9.975e-03, 5.684e-01, 1.869e-01)) * (s0_1_2) + (r0));
	r1 = ((mat4x4<f32>(8.257e-02, 4.720e-02, 2.332e-02, -2.108e-01, 1.474e-02, 2.510e-01, 1.372e-01, 2.900e-01, 9.461e-03, 1.588e-01, 6.632e-02, 3.074e-01, -1.391e-01, 4.718e-01, -4.072e-02, -4.638e-01)) * (s0_1_2) + (r1));
	r0 = ((mat4x4<f32>(2.017e-02, -6.963e-02, -2.059e-01, 5.522e-02, 3.876e-03, -2.700e-02, -7.056e-02, -8.622e-02, -8.871e-02, -2.346e-02, 3.707e-02, -5.105e-02, -5.580e-02, 1.536e-02, -6.917e-02, 1.723e-01)) * (s0_2_0) + (r0));
	r1 = ((mat4x4<f32>(1.041e-01, -3.692e-01, -1.067e-01, -4.372e-01, 7.756e-02, -7.766e-02, -2.291e-02, 7.570e-02, 5.341e-02, -2.788e-02, -8.032e-02, 3.647e-02, 4.171e-02, -1.244e-01, -5.408e-02, 1.956e-02)) * (s0_2_0) + (r1));
	r0 = ((mat4x4<f32>(2.526e-01, -8.803e-02, -3.521e-02, -1.952e-01, 9.958e-02, 1.918e-02, 9.500e-02, -1.255e-01, 9.954e-02, 8.401e-02, 1.860e-01, -7.880e-02, -1.647e-01, -8.276e-04, 5.116e-02, 2.928e-01)) * (s0_2_1) + (r0));
	r1 = ((mat4x4<f32>(5.802e-01, 1.341e-01, 1.434e-02, -1.704e-01, 1.346e-01, 1.339e-01, 9.508e-02, -3.802e-02, -3.687e-02, 2.012e-01, 2.227e-01, -5.216e-02, 2.646e-01, -3.975e-01, -2.920e-01, -1.163e-01)) * (s0_2_1) + (r1));
	r0 = ((mat4x4<f32>(5.494e-04, 1.475e-03, 5.538e-03, 7.359e-02, 1.837e-01, 6.217e-02, 1.074e-01, 5.277e-03, 1.910e-02, 5.994e-02, 5.796e-02, -1.437e-01, -5.969e-02, 1.504e-02, 5.454e-03, 1.468e-01)) * (s0_2_2) + (r0));
	r1 = ((mat4x4<f32>(-6.347e-02, -6.738e-02, -2.679e-02, 6.836e-03, 2.155e-01, 2.283e-01, -1.834e-02, -2.290e-05, 1.763e-01, -2.847e-02, 3.850e-02, -1.163e-01, -7.949e-01, -2.105e-01, -2.335e-02, -8.468e-02)) * (s0_2_2) + (r1));
	r0 = ((mat4x4<f32>(-5.303e-02, 6.088e-02, 3.405e-02, -2.250e-01, -1.050e-01, -1.489e-02, 1.661e-01, 2.608e-01, 6.371e-02, -6.136e-02, 3.083e-02, -2.826e-01, -1.023e-01, -1.921e-02, 9.302e-02, 1.577e-02)) * (s1_0_0) + (r0));
	r1 = ((mat4x4<f32>(1.050e-01, -8.178e-02, 5.020e-01, -1.824e-01, -1.509e-01, -8.470e-02, -2.997e-01, -2.193e-01, -3.256e-02, -5.481e-02, 7.575e-02, -6.707e-02, -4.388e-02, -1.462e-02, -5.667e-02, 8.515e-02)) * (s1_0_0) + (r1));
	r0 = ((mat4x4<f32>(1.863e+00, 8.300e-01, 1.364e-01, -3.951e-01, 5.022e-01, 2.851e-02, 6.264e-01, -1.539e-01, 7.931e-01, 2.568e-01, 4.988e-01, -4.003e-01, 1.147e-01, 5.882e-02, 1.213e-01, 4.826e-02)) * (s1_0_1) + (r0));
	r1 = ((mat4x4<f32>(4.619e-01, 5.371e-01, 3.185e-01, 8.274e-02, 1.200e-01, -1.226e-01, -2.706e-01, -8.163e-01, 1.243e-01, 5.196e-01, 1.733e-01, 2.327e-01, 3.773e-03, -8.958e-02, -6.588e-02, -5.215e-01)) * (s1_0_1) + (r1));
	r0 = ((mat4x4<f32>(-5.725e-02, 9.622e-02, -1.639e-01, 4.223e-01, -7.787e-02, 1.686e-01, -1.608e-01, 2.978e-01, 4.661e-01, 1.002e-01, 2.627e-01, -1.418e-01, 8.262e-02, 5.574e-02, 3.143e-02, 2.579e-02)) * (s1_0_2) + (r0));
	r1 = ((mat4x4<f32>(-2.046e-01, 9.959e-02, 2.075e-01, -2.096e+00, -4.847e-01, -6.149e-01, 4.701e-02, -5.941e+00, 1.440e-01, 2.179e-01, 1.561e-01, -3.578e+00, -6.861e-02, -1.618e-01, 9.439e-03, -1.287e+01)) * (s1_0_2) + (r1));
	r0 = ((mat4x4<f32>(-9.579e-02, 6.189e-02, -1.159e-01, 2.150e-01, -4.549e-01, -2.409e-01, 3.925e-01, -4.460e-01, 1.360e-01, -4.751e-04, -3.381e-02, 2.064e-01, -1.230e-01, -4.066e-01, 5.518e-02, -5.011e-01)) * (s1_1_0) + (r0));
	r1 = ((mat4x4<f32>(1.352e-01, -2.319e-01, -1.343e-01, -3.817e-02, -3.801e-02, 4.277e-01, 4.445e-01, 4.374e-01, -2.427e-01, -1.002e-01, -1.179e-01, 1.841e-02, 1.593e-01, 4.485e-01, -2.762e-01, 5.878e-01)) * (s1_1_0) + (r1));
	r0 = ((mat4x4<f32>(7.918e-02, 3.248e-02, 5.526e-01, 3.818e-01, -5.114e-01, 3.427e-01, -5.737e-01, 4.692e-01, -1.458e-01, 1.323e-01, -5.878e-02, -1.936e-01, 8.602e-02, -1.909e-01, -2.575e-02, -3.231e-01)) * (s1_1_1) + (r0));
	r1 = ((mat4x4<f32>(1.096e-01, 8.324e-02, -1.157e+00, -1.651e-01, 1.479e-02, 3.467e-01, 1.509e-01, 8.535e-01, 2.386e-02, -7.060e-01, -1.040e-02, 1.156e-01, 4.657e-01, 8.064e-01, 3.275e-01, 5.316e-01)) * (s1_1_1) + (r1));
	r0 = ((mat4x4<f32>(-2.163e-01, 6.235e-03, -8.975e-02, 2.083e-01, 5.216e-01, 1.365e-01, 1.431e-01, -4.618e-02, 1.459e-01, 1.150e-01, -4.137e-02, 9.075e-03, -4.688e-01, -1.238e-01, -1.970e-01, -3.330e-01)) * (s1_1_2) + (r0));
	r1 = ((mat4x4<f32>(-7.615e-01, -1.597e-01, -4.504e-02, -3.390e-01, 1.909e-01, 2.195e-01, -5.178e-02, -6.446e-01, 3.213e-01, -3.822e-02, 9.009e-02, 1.444e-01, 6.707e-01, 2.809e-01, -1.399e-02, 6.922e-01)) * (s1_1_2) + (r1));
	r0 = ((mat4x4<f32>(-7.206e-02, -1.223e-02, -3.867e-02, 9.155e-02, 4.327e-01, 1.158e-01, -2.036e-01, -1.259e-01, 6.557e-02, -1.204e-03, 5.246e-02, -6.929e-02, 1.062e-01, -4.244e-02, -3.571e-01, 5.444e-01)) * (s1_2_0) + (r0));
	r1 = ((mat4x4<f32>(-1.416e-02, -6.707e-02, -1.206e-01, 2.679e-02, -3.135e-01, -8.026e-02, 3.923e-01, -2.604e-01, -1.131e-01, 5.136e-02, 6.091e-02, -6.370e-02, -2.769e-01, -6.045e-01, 2.380e-01, -1.016e-01)) * (s1_2_0) + (r1));
	r0 = ((mat4x4<f32>(-5.873e-02, -7.936e-02, -7.740e-02, 9.037e-02, 1.292e-02, -2.422e-01, -2.022e-01, -5.898e-01, -2.129e-02, 6.206e-02, 1.817e-02, -7.300e-02, -1.877e-02, 4.453e-01, 4.070e-01, 7.967e-01)) * (s1_2_1) + (r0));
	r1 = ((mat4x4<f32>(-1.115e-01, -3.104e-02, 2.083e-02, -9.055e-02, 4.538e-01, 1.284e-01, -2.979e-01, 1.177e-01, 1.194e-01, -3.457e-02, 5.920e-02, 1.283e-01, -1.695e-01, -3.700e-01, -1.649e-01, 2.046e-01)) * (s1_2_1) + (r1));
	r0 = ((mat4x4<f32>(-3.303e-02, 2.488e-04, -4.218e-02, 7.275e-03, -3.436e-02, -5.248e-02, 1.132e-01, 8.247e-02, -2.631e-02, 1.910e-02, 5.824e-02, 2.078e-02, -2.627e-01, 9.032e-02, 5.265e-02, 7.366e-02)) * (s1_2_2) + (r0));
	r1 = ((mat4x4<f32>(1.975e-02, -6.391e-02, -2.874e-02, -5.150e-02, 2.261e-01, -2.319e-01, -9.771e-02, 4.567e-01, 3.352e-04, -4.777e-02, -2.349e-02, -1.224e-03, -4.864e-01, -5.346e-01, 3.012e-02, -3.242e-01)) * (s1_2_2) + (r1));
	r0 = max((r0), vec4<f32>(0.0));
	textureStore(T2_storaged, vec2<i32>(gxy), r0);
	r1 = max((r1), vec4<f32>(0.0));
	textureStore(T3_storaged, vec2<i32>(gxy), r1);
}

//!PASS 3
//!DESC conv2 (8x4)
//!BLOCK_SIZE 8
//!NUM_THREADS 64
//!IN T2_sampled, T3_sampled
//!OUT T0_storaged

fn Pass3(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) tid: vec3<u32>) {
    let blockStart = workgroup_id.xy * vec2<u32>(8u, 8u);
	let pt = GetInputPt();
	var gxy = Rmp8x8(tid.x) + blockStart;
	let sz = GetInputSize();
	if (gxy.x >= sz.x || gxy.y >= sz.y) {
		return;
	}
	var pos = (vec2<f32>(gxy) + vec2<f32>(0.5)) * pt;
	var s0_0_0: vec4<f32>; var s0_0_1: vec4<f32>; var s0_0_2: vec4<f32>; var s0_1_0: vec4<f32>; var s0_1_1: vec4<f32>; var s0_1_2: vec4<f32>; var s0_2_0: vec4<f32>; var s0_2_1: vec4<f32>; var s0_2_2: vec4<f32>; var s1_0_0: vec4<f32>; var s1_0_1: vec4<f32>; var s1_0_2: vec4<f32>; var s1_1_0: vec4<f32>; var s1_1_1: vec4<f32>; var s1_1_2: vec4<f32>; var s1_2_0: vec4<f32>; var s1_2_1: vec4<f32>; var s1_2_2: vec4<f32>;
	var r0 = vec4<f32>(0.0);
	s0_0_0 = vec4<f32>(textureSampleLevel(T2_sampled, SP, pos + vec2<f32>((-1.0), (-1.0)) * pt, 0.0)); s0_0_1 = vec4<f32>(textureSampleLevel(T2_sampled, SP, pos + vec2<f32>((0.0), (-1.0)) * pt, 0.0)); s0_0_2 = vec4<f32>(textureSampleLevel(T2_sampled, SP, pos + vec2<f32>((1.0), (-1.0)) * pt, 0.0));
	s0_1_0 = vec4<f32>(textureSampleLevel(T2_sampled, SP, pos + vec2<f32>((-1.0), (0.0)) * pt, 0.0)); s0_1_1 = vec4<f32>(textureSampleLevel(T2_sampled, SP, pos + vec2<f32>((0.0), (0.0)) * pt, 0.0)); s0_1_2 = vec4<f32>(textureSampleLevel(T2_sampled, SP, pos + vec2<f32>((1.0), (0.0)) * pt, 0.0));
	s0_2_0 = vec4<f32>(textureSampleLevel(T2_sampled, SP, pos + vec2<f32>((-1.0), (1.0)) * pt, 0.0)); s0_2_1 = vec4<f32>(textureSampleLevel(T2_sampled, SP, pos + vec2<f32>((0.0), (1.0)) * pt, 0.0)); s0_2_2 = vec4<f32>(textureSampleLevel(T2_sampled, SP, pos + vec2<f32>((1.0), (1.0)) * pt, 0.0));
	s1_0_0 = vec4<f32>(textureSampleLevel(T3_sampled, SP, pos + vec2<f32>((-1.0), (-1.0)) * pt, 0.0)); s1_0_1 = vec4<f32>(textureSampleLevel(T3_sampled, SP, pos + vec2<f32>((0.0), (-1.0)) * pt, 0.0)); s1_0_2 = vec4<f32>(textureSampleLevel(T3_sampled, SP, pos + vec2<f32>((1.0), (-1.0)) * pt, 0.0));
	s1_1_0 = vec4<f32>(textureSampleLevel(T3_sampled, SP, pos + vec2<f32>((-1.0), (0.0)) * pt, 0.0)); s1_1_1 = vec4<f32>(textureSampleLevel(T3_sampled, SP, pos + vec2<f32>((0.0), (0.0)) * pt, 0.0)); s1_1_2 = vec4<f32>(textureSampleLevel(T3_sampled, SP, pos + vec2<f32>((1.0), (0.0)) * pt, 0.0));
	s1_2_0 = vec4<f32>(textureSampleLevel(T3_sampled, SP, pos + vec2<f32>((-1.0), (1.0)) * pt, 0.0)); s1_2_1 = vec4<f32>(textureSampleLevel(T3_sampled, SP, pos + vec2<f32>((0.0), (1.0)) * pt, 0.0)); s1_2_2 = vec4<f32>(textureSampleLevel(T3_sampled, SP, pos + vec2<f32>((1.0), (1.0)) * pt, 0.0));
	r0 = ((mat4x4<f32>(-8.148e-03, 2.568e-02, 4.651e-02, -7.485e-02, 1.790e-02, -8.190e-02, -1.489e-01, 1.323e-01, 3.400e-02, 6.812e-02, 3.208e-02, -2.434e-02, 6.154e-02, 8.815e-02, 6.566e-02, 5.507e-02)) * (s0_0_0) + (r0));
	r0 = ((mat4x4<f32>(3.119e-02, -4.280e-03, -6.519e-03, 1.538e-01, -2.105e-01, -1.431e-01, -1.406e-01, -4.139e-01, 8.038e-02, -1.392e-01, 9.856e-03, -9.555e-02, 1.765e-01, 2.941e-01, 9.466e-02, 3.756e-01)) * (s0_0_1) + (r0));
	r0 = ((mat4x4<f32>(-1.209e-01, 4.339e-02, -7.104e-02, -6.860e-02, 4.313e-02, -1.887e-01, 1.963e-02, -4.690e-02, -6.567e-02, -1.265e-01, -4.360e-02, -1.146e-01, -5.670e-02, 1.087e-01, -4.472e-02, 1.739e-02)) * (s0_0_2) + (r0));
	r0 = ((mat4x4<f32>(-4.968e-02, -1.421e-01, 3.657e-02, -2.672e-01, -7.897e-04, -3.623e-01, 3.450e-02, -2.245e-01, -4.632e-03, -1.128e-01, -1.792e-01, 5.298e-01, 1.125e-01, -1.141e-01, -8.822e-02, 6.274e-02)) * (s0_1_0) + (r0));
	r0 = ((mat4x4<f32>(-1.880e-01, -3.701e-01, -1.155e-01, 3.115e-01, 3.475e-01, 8.071e-01, 7.021e-01, -3.410e-01, -2.617e-01, 1.043e+00, -3.493e-01, -2.318e-01, -4.900e-01, -5.969e-01, -1.215e-01, -3.721e-01)) * (s0_1_1) + (r0));
	r0 = ((mat4x4<f32>(3.329e-01, 1.936e-01, 1.228e-01, -1.891e-02, -3.213e-01, -4.152e-01, -1.440e-01, 4.134e-02, 2.842e-01, -5.296e-02, 3.641e-01, 5.137e-01, 1.812e-01, 8.146e-02, 1.061e-01, 3.798e-02)) * (s0_1_2) + (r0));
	r0 = ((mat4x4<f32>(-6.376e-03, 2.285e-01, 5.671e-02, -8.081e-02, -9.302e-02, -1.174e-01, -1.714e-01, 2.654e-02, -1.334e-02, 1.460e-01, 5.519e-02, -1.432e-01, 3.235e-02, 5.162e-02, 3.121e-02, 6.723e-03)) * (s0_2_0) + (r0));
	r0 = ((mat4x4<f32>(1.175e-01, -3.771e-02, 5.432e-02, 3.030e-01, 1.248e-01, 3.087e-02, -5.464e-02, 9.374e-02, 1.291e-01, 2.582e-02, 2.026e-01, 3.218e-02, -3.019e-02, -6.113e-02, 1.022e-03, -1.526e-02)) * (s0_2_1) + (r0));
	r0 = ((mat4x4<f32>(-4.480e-02, 4.266e-02, -1.878e-02, -7.446e-02, 4.263e-02, -1.403e-01, -1.898e-01, -1.598e-01, -4.898e-02, -1.334e-01, -4.467e-03, 2.087e-02, 6.375e-03, 8.764e-02, 7.014e-02, 3.828e-02)) * (s0_2_2) + (r0));
	r0 = ((mat4x4<f32>(-1.823e-02, -5.078e-02, -4.285e-02, 4.404e-02, -9.971e-03, -3.043e-02, -1.849e-02, 1.066e-01, 1.313e-02, 2.819e-02, 6.397e-02, -4.005e-02, -2.264e-02, -4.141e-02, -6.211e-02, 2.856e-02)) * (s1_0_0) + (r0));
	r0 = ((mat4x4<f32>(-1.643e-01, -6.951e-02, -5.324e-02, -1.595e-01, 4.259e-02, 1.606e-01, 2.015e-02, -3.517e-04, 7.591e-02, 1.665e-01, 1.284e-01, 1.572e-01, -8.479e-02, -9.076e-02, -3.720e-02, -7.167e-02)) * (s1_0_1) + (r0));
	r0 = ((mat4x4<f32>(1.875e-01, 4.330e-02, 7.509e-02, 9.155e-02, 1.067e-01, -9.226e-03, 6.569e-02, 1.057e-01, 9.918e-02, 2.543e-03, 6.361e-02, 4.849e-02, -3.967e-02, 9.021e-02, -2.580e-02, -8.976e-03)) * (s1_0_2) + (r0));
	r0 = ((mat4x4<f32>(7.936e-03, 4.667e-02, 1.710e-01, -5.760e-01, 5.680e-03, 6.270e-01, 3.174e-01, 4.808e-02, 7.891e-03, -5.142e-03, 1.486e-02, -1.813e-02, -2.654e-02, -6.394e-01, -8.960e-02, -4.404e-01)) * (s1_1_0) + (r0));
	r0 = ((mat4x4<f32>(-9.929e-04, -5.820e-01, 1.195e-01, 4.442e-01, 9.473e-01, -7.623e-01, 3.154e-01, -6.255e-01, 4.396e-04, 7.951e-01, -1.909e-01, 1.098e+00, -2.184e-02, -4.709e-01, -1.576e-01, -5.169e-01)) * (s1_1_1) + (r0));
	r0 = ((mat4x4<f32>(3.076e-01, 2.549e-01, 2.183e-01, 2.803e-01, -6.310e-01, -3.174e-01, -4.287e-01, -4.186e-01, 1.036e-02, 1.632e-01, -9.137e-04, -2.596e-02, -2.581e-02, -9.876e-03, 6.714e-02, 3.123e-02)) * (s1_1_2) + (r0));
	r0 = ((mat4x4<f32>(7.675e-03, 5.052e-03, -3.337e-02, 9.983e-02, 1.332e-02, 1.577e-01, 1.304e-01, 1.257e-01, -6.344e-03, 1.044e-01, 5.069e-02, 2.343e-02, 8.552e-02, -9.318e-01, -1.662e-02, -5.734e-01)) * (s1_2_0) + (r0));
	r0 = ((mat4x4<f32>(-3.231e-02, 9.578e-02, -6.091e-02, -1.283e-01, 8.889e-02, -7.374e-02, 1.334e-01, 2.598e-02, 2.393e-01, 8.617e-02, 3.545e-01, 9.125e-02, 3.194e-01, 6.674e-01, -2.021e-02, -1.412e-01)) * (s1_2_1) + (r0));
	r0 = ((mat4x4<f32>(9.790e-02, 1.380e-02, 4.626e-02, 9.249e-02, 8.255e-03, 6.021e-02, 2.871e-04, 1.201e-01, 1.724e-01, 1.096e-01, 1.130e-01, 1.430e-01, -4.326e-01, -3.163e-01, -1.880e-01, -1.765e-01)) * (s1_2_2) + (r0));
	r0 = max((r0), vec4<f32>(0.0));
	textureStore(T0_storaged, vec2<i32>(gxy), r0);
}

//!PASS 4
//!DESC out-shuffle (4x4)
//!BLOCK_SIZE 16
//!NUM_THREADS 64
//!IN INPUT, T0_sampled
//!OUT OUTPUT

fn Pass4(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) tid: vec3<u32>) {
    let blockStart = workgroup_id.xy * vec2<u32>(16u, 16u);
	let pt = GetInputPt();
	var gxy = (Rmp8x8(tid.x) << vec2<u32>(1u)) + blockStart;
	let sz = GetOutputSize();
	if (gxy.x >= sz.x || gxy.y >= sz.y) {
		return;
	}
	var pos = (vec2<f32>(gxy >> vec2<u32>(1u)) + vec2<f32>(0.5)) * pt;
	var s0_0_0: vec4<f32>; var s0_0_1: vec4<f32>; var s0_0_2: vec4<f32>; var s0_1_0: vec4<f32>; var s0_1_1: vec4<f32>; var s0_1_2: vec4<f32>; var s0_2_0: vec4<f32>; var s0_2_1: vec4<f32>; var s0_2_2: vec4<f32>;
	var r0 = vec4<f32>(0.0);
	r0 = vec4<f32>(1.026e-04, 2.907e-04, -2.278e-04, -8.361e-05);
	s0_0_0 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((-1.0), (-1.0)) * pt, 0.0)); s0_0_1 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((0.0), (-1.0)) * pt, 0.0)); s0_0_2 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((1.0), (-1.0)) * pt, 0.0));
	s0_1_0 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((-1.0), (0.0)) * pt, 0.0)); s0_1_1 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((0.0), (0.0)) * pt, 0.0)); s0_1_2 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((1.0), (0.0)) * pt, 0.0));
	s0_2_0 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((-1.0), (1.0)) * pt, 0.0)); s0_2_1 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((0.0), (1.0)) * pt, 0.0)); s0_2_2 = vec4<f32>(textureSampleLevel(T0_sampled, SP, pos + vec2<f32>((1.0), (1.0)) * pt, 0.0));
	r0 = ((mat4x4<f32>(-2.078e-02, 2.658e-02, 5.912e-03, -8.754e-04, 3.267e-02, 6.010e-03, -8.934e-03, -7.844e-03, -3.699e-02, -2.191e-02, 1.024e-02, 1.211e-02, 2.276e-03, -2.706e-02, -1.511e-02, -6.482e-03)) * (s0_0_0) + (r0));
	r0 = ((mat4x4<f32>(4.173e-03, -2.643e-02, 4.702e-03, 8.715e-03, 1.479e-02, -1.440e-01, 3.235e-02, 3.772e-02, 1.870e-01, 1.372e-01, -6.517e-02, -4.455e-02, -1.782e-01, 9.614e-02, 5.090e-02, 1.126e-02)) * (s0_0_1) + (r0));
	r0 = ((mat4x4<f32>(9.667e-03, -4.513e-04, 6.245e-03, 1.301e-02, -5.169e-03, 1.161e-02, -1.311e-02, -1.338e-02, -2.069e-02, 2.227e-02, -7.714e-03, -3.542e-02, 1.850e-02, -5.652e-02, 1.112e-02, 4.749e-02)) * (s0_0_2) + (r0));
	r0 = ((mat4x4<f32>(-4.541e-01, 4.950e-02, -2.319e-01, 1.072e-01, 5.148e-02, -1.947e-02, 6.616e-02, 1.984e-02, -7.690e-02, 1.773e-02, -1.006e-01, -6.559e-02, 2.260e-03, -8.378e-03, -3.693e-02, -7.541e-02)) * (s0_1_0) + (r0));
	r0 = ((mat4x4<f32>(-8.618e-02, -6.200e-01, -3.466e-02, -3.779e-01, 5.723e-01, -2.387e-02, 6.342e-02, -4.658e-01, 1.304e-01, 1.130e-01, 7.051e-01, 5.566e-01, 2.580e-02, 3.877e-01, -5.566e-01, 1.724e-01)) * (s0_1_1) + (r0));
	r0 = ((mat4x4<f32>(-9.982e-03, 6.074e-03, 5.088e-03, -8.145e-03, -4.550e-02, 9.498e-02, -4.190e-02, -2.528e-02, 1.080e-02, -6.524e-02, -2.383e-02, 1.350e-01, -3.786e-03, 1.538e-01, 2.104e-02, -1.411e-01)) * (s0_1_2) + (r0));
	r0 = ((mat4x4<f32>(3.114e-02, -2.459e-02, -6.471e-02, 5.313e-02, -9.421e-03, 5.377e-03, 1.764e-02, 4.711e-03, 2.045e-02, 1.029e-02, -2.045e-02, 1.090e-02, 2.616e-02, -3.509e-03, 1.584e-02, 2.190e-02)) * (s0_2_0) + (r0));
	r0 = ((mat4x4<f32>(1.764e-02, 7.804e-02, -4.490e-02, -1.519e-01, -7.934e-02, 5.097e-03, 5.826e-02, 6.307e-03, -1.490e-02, -3.430e-02, 4.749e-02, -1.289e-02, -1.011e-02, -2.436e-02, 3.430e-02, 2.518e-02)) * (s0_2_1) + (r0));
	r0 = ((mat4x4<f32>(-3.636e-03, -1.442e-02, -1.291e-02, -2.985e-02, 1.788e-02, 5.287e-03, -1.129e-02, 1.125e-02, 1.328e-02, 3.210e-02, 1.753e-03, 1.867e-02, -4.918e-03, -3.528e-02, 2.455e-03, 5.595e-02)) * (s0_2_2) + (r0));
	let RY = mat3x3<f32>(0.299, 0.587, 0.114, -0.169, -0.331, 0.5, 0.5, -0.419, -0.081);
	let YR = mat3x3<f32>(1, -0.00093, 1.401687, 1, -0.3437, -0.71417, 1, 1.77216, 0.00099);
	let opt = GetOutputPt();
	let fpos = (vec2<f32>(gxy) + vec2<f32>(0.5)) * opt;
	var yuv: vec3<f32>;
	yuv = ((RY) * (textureSampleLevel(INPUT, SL, fpos + vec2<f32>(0.0, 0.0) * opt, 0.0).rgb));
	textureStore(OUTPUT, vec2<i32>(gxy) + vec2<i32>(0, 0), vec4<f32>(((YR) * (vec3<f32>(saturate(yuv.r + r0.x), yuv.yz))), 1.0));
	yuv = ((RY) * (textureSampleLevel(INPUT, SL, fpos + vec2<f32>(1.0, 0.0) * opt, 0.0).rgb));
	textureStore(OUTPUT, vec2<i32>(gxy) + vec2<i32>(1, 0), vec4<f32>(((YR) * (vec3<f32>(saturate(yuv.r + r0.y), yuv.yz))), 1.0));
	yuv = ((RY) * (textureSampleLevel(INPUT, SL, fpos + vec2<f32>(0.0, 1.0) * opt, 0.0).rgb));
	textureStore(OUTPUT, vec2<i32>(gxy) + vec2<i32>(0, 1), vec4<f32>(((YR) * (vec3<f32>(saturate(yuv.r + r0.z), yuv.yz))), 1.0));
	yuv = ((RY) * (textureSampleLevel(INPUT, SL, fpos + vec2<f32>(1.0, 1.0) * opt, 0.0).rgb));
	textureStore(OUTPUT, vec2<i32>(gxy) + vec2<i32>(1, 1), vec4<f32>(((YR) * (vec3<f32>(saturate(yuv.r + r0.w), yuv.yz))), 1.0));
}

