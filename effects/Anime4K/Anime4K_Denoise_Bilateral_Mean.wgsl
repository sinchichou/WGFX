// Anime4K_Denoise_Bilateral_Mean — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/bloc97/Anime4K/blob/master/glsl/Denoise/Anime4K_Denoise_Bilateral_Mean.glsl
// (Upstream: bloc97/Anime4K — MIT License; Magpie HLSL port by the Magpie project.)
// This port keeps every algorithmic constant, pass order and texture layout
// verbatim; only the pass structure and the sampling idioms are translated.
//
// Porting notes:
//   - HLSL `MulAdd(v, MF RxC(W), acc)` == WGSL `acc + matRxC<f32>(W) * v`: the HLSL row-major
//   -    matrix constructor and the WGSL column-major constructor cancel out when the product
//   -    order is flipped (vec*mat -> mat*vec), so every weight number below is VERBATIM from
//   -    the HLSL source (no transposition).
//   - Magpie CS passes (`uint2 blockStart, uint3 threadId`) become compute shaders using
//   -    `@builtin(workgroup_id)` + `@builtin(local_invocation_id)`; the 16x16 (2x2 quad per
//   -    thread) / 8x8 (1 pixel per thread) block mapping and the `Rmp8x8` expansion are kept
//   -    verbatim, and the NUM_THREADS directive is written as 64,1,1 (64 total threads, as in Magpie).
//   - WGSL textureGather returns the quad as x=(0,1) y=(1,1) z=(1,0) w=(0,0) relative to the
//   -    texel left-above the sample point — identical to D3D Gather — so Magpie's "w z / x y"
//   -    fill mapping is copied verbatim.
//   - `scene` (inputSize/inputPt/outputSize/outputPt/scale/frame) is auto-injected by WGFX at
//   -    @group(0) @binding(4); texture/sampler binding slots are owned by the code generator.
//   - Magpie `GetInputSize()/GetInputPt()/GetOutputSize()/GetOutputPt()` read the `scene`
//   -    uniform; intermediate feature maps are RGBA16F textures sampled with the POINT sampler
//   -    exactly as the HLSL does (Magpie's HLSL sampler is CLAMP/POINT, WGFX defaults to
//   -    clamp-to-edge).
//   - New pattern for this family: a PARAMETER block (`intensitySigma`, TYPE/LABEL/MIN/MAX/
//   -    DEFAULT/STEP copied from the HLSL) read as `uniforms.intensitySigma`; the HLSL
//   -    `#define INTENSITY_SIGMA intensitySigma` has no WGSL macro counterpart and is inlined
//   -    at every use site.
//   - The remaining `#define`s become WGSL consts with the formulas verbatim: SPATIAL_SIGMA,
//   -    INTENSITY_POWER_CURVE, KERNELSIZE (= 5), KERNELHALFSIZE and KERNELLEN.
//   - HLSL mixed int/uint arithmetic promotes to uint: in the gather loop
//   -    `(gxy + int2(i,j) - KERNELHALFSIZE + 1)` wraps for the two left/top quad offsets, and
//   -    in the spatial-length table `(int)i - KERNELHALFSIZE` wraps for i < KERNELHALFSIZE,
//   -    which zeroes the gaussian spatial weight of those taps. The port reproduces both wrap
//   -    semantics verbatim (u32 arithmetic); CLAMP-to-edge sampling keeps border reads finite
//   -    and the center kernel tap always has weight 1, so n > 0 — no divide by zero / NaN.
//   - The gather quad fill and the `Rmp8x8` block mapping are identical to the other Anime4K
//   -    ports (see the shared notes above).

//!VERSION 4
//!SORT_NAME Anime4K_Denoise_Bilateral_Mean

//!PARAMETER intensitySigma
//!TYPE float
//!LABEL "Strength"
//!DEFAULT 0.1
//!MIN 0.01
//!MAX 5
//!STEP 0.01
float intensitySigma;

//!TEXTURE
Texture2D INPUT;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//!SAMPLER
//!FILTER POINT
SamplerState sam;

//!PASS 1
//!IN INPUT
//!OUT OUTPUT
//!BLOCK_SIZE 16
//!NUM_THREADS 64,1,1

// HLSL macros, verbatim (SPATIAL_SIGMA = 1.0):
//   SPATIAL_SIGMA  = 1.0                                              // spatial window size, higher is stronger denoise
//   INTENSITY_POWER_CURVE = 1.0                                       // intensity window power curve; 0 treats all intensities equally,
//                                                                     // higher narrows the window in darker intensities
//   KERNELSIZE     = max(uint(ceil(SPATIAL_SIGMA * 2.0)), 1) * 2 + 1  // must be a positive odd integer
//   KERNELHALFSIZE = uint(KERNELSIZE/2)                               // trunc(KERNELSIZE/2)
//   KERNELLEN      = KERNELSIZE * KERNELSIZE                          // total area of kernel (unused by Mean, kept verbatim)
const SPATIAL_SIGMA: f32 = 1.0;
const INTENSITY_POWER_CURVE: f32 = 1.0;
const KERNELSIZE: u32 = max(u32(ceil(SPATIAL_SIGMA * 2.0)), 1u) * 2u + 1u;
const KERNELHALFSIZE: u32 = KERNELSIZE / 2u;
const KERNELLEN: u32 = KERNELSIZE * KERNELSIZE;
fn gaussian_vec(x: vec3<f32>, rcpS: vec3<f32>, m: vec3<f32>) -> vec3<f32> {
	var scaled: vec3<f32> = (x - m) * rcpS;
	return exp(-0.5 * scaled * scaled);
}

fn gaussian(x: f32, rcpS: f32, m: f32) -> f32 {
	var scaled: f32 = (x - m) * rcpS;
	return exp(-0.5 * scaled * scaled);
}


fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
	var gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * vec2<u32>(16u);

	let outputSize: vec2<u32> = GetOutputSize();
	if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
		return;
	}

	var inputPt: vec2<f32> = GetInputPt();
	var i: u32 = 0u;
	var j: u32 = 0u;

	var src: array<array<vec3<f32>, KERNELSIZE + 1>, KERNELSIZE + 1>;
	for (var i: u32 = 0; i <= KERNELSIZE - 1; i += 2u) {
		for (var j: u32 = 0; j <= KERNELSIZE - 1; j += 2u) {
			// HLSL: (gxy + int2(i, j) - KERNELHALFSIZE + 1) * inputPt — int2-op-uint promotes to uint,
			// so the two left/top quad offsets wrap; CLAMP-to-edge sampling of the wrapped UV is
			// reproduced verbatim (i32 arithmetic then u32 conversion, added mod 2^32).
			var tpos: vec2<f32> = vec2<f32>(gxy + vec2<u32>(vec2<i32>(i32(i), i32(j)) - vec2<i32>(i32(KERNELHALFSIZE)) + vec2<i32>(1))) * inputPt;
			let sr: vec4<f32> = textureGather(0, INPUT, sam, tpos);
			let sg: vec4<f32> = textureGather(1, INPUT, sam, tpos);
			let sb: vec4<f32> = textureGather(2, INPUT, sam, tpos);

			// w z
			// x y
			src[i][j] = vec3<f32>(sr.w, sg.w, sb.w);
			src[i][j + 1] = vec3<f32>(sr.x, sg.x, sb.x);
			src[i + 1][j] = vec3<f32>(sr.z, sg.z, sb.z);
			src[i + 1][j + 1] = vec3<f32>(sr.y, sg.y, sb.y);
		}
	}

	var len: array<array<f32, KERNELSIZE>, KERNELSIZE>;
	for (var i: u32 = 0; i < KERNELSIZE; i += 1u) {
		for (var j: u32 = 0; j < KERNELSIZE; j += 1u) {
			// HLSL: (int)i - KERNELHALFSIZE — mixed int/uint promotes to uint (wraps for i < KERNELHALFSIZE),
			// zeroing the gaussian spatial weight of those taps; reproduced verbatim (u32 arithmetic).
			len[i][j] = length(vec2<f32>(f32(u32(i32(i)) - KERNELHALFSIZE), f32(u32(i32(j)) - KERNELHALFSIZE)));
		}
	}

	for (var i: u32 = 0; i <= 1; i += 1u) {
		for (var j: u32 = 0; j <= 1; j += 1u) {
			var destPos: vec2<u32> = gxy + vec2<u32>(i, j);

			var sum: vec3<f32> = vec3<f32>(0.0);
			var n: vec3<f32> = vec3<f32>(0.0);

			var vc: vec3<f32> = src[KERNELHALFSIZE + i][KERNELHALFSIZE + j].rgb;

			// WGSL pow() has no mixed scalar/vector overload — the HLSL scalar exponent 1.0 is
			// broadcast explicitly (identical component-wise semantics).
			var rcpIs: vec3<f32> = vec3<f32>(1.0) / (pow(vc + vec3<f32>(0.0001), vec3<f32>(INTENSITY_POWER_CURVE)) * uniforms.intensitySigma);
			var rcpSs: f32 = 1.0 / SPATIAL_SIGMA;

			for (var k: u32 = 0; k < KERNELSIZE; k += 1u) {
				for (var m: u32 = 0; m < KERNELSIZE; m += 1u) {
					var v: vec3<f32> = src[k + i][m + j];
					var d: vec3<f32> = gaussian_vec(v, rcpIs, vc) * gaussian(len[k][m], rcpSs, 0.0);
					sum += d * v;
					n += d;
				}
			}

			textureStore(OUTPUT, vec2<i32>(destPos), vec4<f32>(sum / n, 1.0));
		}
	}
}

//!COMMON
// --- Integer Types (Magpie StubDefs.hlsli) ---
alias uint2 = vec2<u32>;
alias uint3 = vec3<u32>;
alias int2 = vec2<i32>;

// --- Float vectors / matrices (Magpie MF* types) ---
alias MF = f32;
alias MF2 = vec2<f32>;
alias MF3 = vec3<f32>;
alias MF4 = vec4<f32>;
// HLSL MF RxC (R rows = input channels, C cols = output channels) -> WGSL matRxC
// (R columns, C rows), numbers verbatim, applied as `M * v` — see porting note 1.
alias MF3x4 = mat3x4<f32>;
alias MF4x3 = mat4x3<f32>;
alias MF4x4 = mat4x4<f32>;

fn GetInputSize() -> uint2 {
    return scene.inputSize;
}

fn GetInputPt() -> MF2 {
    return scene.inputPt;
}

fn GetOutputSize() -> uint2 {
    return scene.outputSize;
}

fn GetOutputPt() -> MF2 {
    return scene.outputPt;
}

// Magpie Rmp8x8: 8x8 thread expansion inside a 16x16 / 8x8 block.
fn Rmp8x8(a: u32) -> uint2 {
    return uint2(a / 8u, a % 8u);
}
//!END COMMON
