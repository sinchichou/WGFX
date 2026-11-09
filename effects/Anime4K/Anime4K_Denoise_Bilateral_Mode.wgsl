// Anime4K_Denoise_Bilateral_Mode — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/bloc97/Anime4K/blob/master/glsl/Denoise/Anime4K_Denoise_Bilateral_Mode.glsl
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
//   -    HISTOGRAM_REGULARIZATION (= 0.2), INTENSITY_POWER_CURVE, KERNELSIZE (= 3),
//   -    KERNELHALFSIZE and KERNELLEN; `GETOFFSET` becomes a WGSL function.
//   - The histogram regularization (HISTOGRAM_REGULARIZATION = 0.2) always runs, exactly as in
//   -    the HLSL.
//   - HLSL `uint2 idx = uint2(i, j) + ipos.yx + KERNELHALFSIZE` mixes uint2+int2 (promotes to
//   -    uint); the port does the small-offset arithmetic in i32 and converts once — values
//   -    stay in [0, 3], so the result is exact.
//   - `length(ipos)` over an int2 offset is written `length(vec2<f32>(ipos))`; the gather
//   -    quad fill and the `Rmp8x8` block mapping are identical to the other Anime4K ports.

//!VERSION 4
//!SORT_NAME Anime4K_Denoise_Bilateral_Mode

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

// INTENSITY_SIGMA == intensitySigma, read as uniforms.intensitySigma (see the PARAMETER block above)
const SPATIAL_SIGMA: f32 = 1.0; // Spatial window size, higher is stronger denoise, must be a positive real number.
// HISTOGRAM_REGULARIZATION: histogram regularization window size, higher values approximate a bilateral "closest-to-mean" filter.
const HISTOGRAM_REGULARIZATION: f32 = 0.2;
// INTENSITY_POWER_CURVE: intensity window power curve. Setting it to 0 will make the intensity window treat all intensities equally, while increasing it will make the window narrower in darker intensities and wider in brighter intensities.
const INTENSITY_POWER_CURVE: f32 = 1.0;

// HLSL macros, verbatim (SPATIAL_SIGMA = 1.0):
//   KERNELSIZE     = uint(max(uint(SPATIAL_SIGMA), 1) * 2 + 1)  // must be a positive odd integer
//   KERNELHALFSIZE = int(KERNELSIZE/2)                          // trunc(KERNELSIZE/2)
//   KERNELLEN      = KERNELSIZE * KERNELSIZE                    // total area of kernel
const KERNELSIZE: u32 = max(u32(SPATIAL_SIGMA), 1u) * 2u + 1u;
const KERNELHALFSIZE: i32 = i32(KERNELSIZE / 2u);
const KERNELLEN: u32 = KERNELSIZE * KERNELSIZE;

// GETOFFSET(i): int2(int(i % KERNELSIZE) - KERNELHALFSIZE, int(i / KERNELSIZE) - KERNELHALFSIZE)
fn GETOFFSET(i: u32) -> vec2<i32> {
	return vec2<i32>(i32(i % KERNELSIZE) - KERNELHALFSIZE, i32(i / KERNELSIZE) - KERNELHALFSIZE);
}
fn get_luma(rgba: vec3<f32>) -> f32 {
	return dot(vec3<f32>(0.299, 0.587, 0.114), rgba);
}

fn gaussian(x: f32, s: f32, m: f32) -> f32 {
	var scaled: f32 = (x - m) / s;
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
	var k: u32 = 0u;
	var m: u32 = 0u;

	var src: array<array<vec4<f32>, KERNELSIZE + 1>, KERNELSIZE + 1>;
	for (var i: u32 = 0; i <= KERNELSIZE - 1; i += 2u) {
		for (var j: u32 = 0; j <= KERNELSIZE - 1; j += 2u) {
			// HLSL: (gxy + int2(i, j) - KERNELHALFSIZE + 1) * inputPt (KERNELHALFSIZE is int here);
			// the in-range i32 result is converted to u32 and added mod 2^32 — identical values.
			var tpos: vec2<f32> = vec2<f32>(gxy + vec2<u32>(vec2<i32>(i32(i), i32(j)) - vec2<i32>(KERNELHALFSIZE) + vec2<i32>(1))) * inputPt;
			let sr: vec4<f32> = textureGather(0, INPUT, sam, tpos);
			let sg: vec4<f32> = textureGather(1, INPUT, sam, tpos);
			let sb: vec4<f32> = textureGather(2, INPUT, sam, tpos);

			// w z
			// x y
			src[i][j] = vec4<f32>(sr.w, sg.w, sb.w, get_luma(vec3<f32>(sr.w, sg.w, sb.w)));
			src[i][j + 1] = vec4<f32>(sr.x, sg.x, sb.x, get_luma(vec3<f32>(sr.x, sg.x, sb.x)));
			src[i + 1][j] = vec4<f32>(sr.z, sg.z, sb.z, get_luma(vec3<f32>(sr.z, sg.z, sb.z)));
			src[i + 1][j + 1] = vec4<f32>(sr.y, sg.y, sb.y, get_luma(vec3<f32>(sr.y, sg.y, sb.y)));
		}
	}

	for (var i: u32 = 0; i <= 1; i += 1u) {
		for (var j: u32 = 0; j <= 1; j += 1u) {
			let destPos: vec2<u32> = gxy + vec2<u32>(i, j);

			var histogram_v: array<vec3<f32>, KERNELLEN>;
			var histogram_l: array<f32, KERNELLEN>;
			var histogram_w: array<f32, KERNELLEN>;
			var histogram_wn: array<f32, KERNELLEN>;

			var vc: f32 = src[u32(KERNELHALFSIZE) + i][u32(KERNELHALFSIZE) + j].a;

			var is: f32 = pow(vc + 0.0001, INTENSITY_POWER_CURVE) * uniforms.intensitySigma;
			var ss: f32 = SPATIAL_SIGMA;

			for (var k: u32 = 0; k < KERNELLEN; k += 1u) {
				let ipos: vec2<i32> = GETOFFSET(k);
				let idx: vec2<u32> = vec2<u32>(vec2<i32>(i32(i), i32(j)) + ipos.yx + vec2<i32>(KERNELHALFSIZE));
				histogram_v[k] = src[idx.x][idx.y].rgb;
				histogram_l[k] = src[idx.x][idx.y].a;
				histogram_w[k] = gaussian(histogram_l[k], is, vc) * gaussian(length(vec2<f32>(ipos)), ss, 0.0);
				histogram_wn[k] = 0.0;
			}

			for (var k: u32 = 0; k < KERNELLEN; k += 1u) {
				histogram_wn[k] += gaussian(0.0, HISTOGRAM_REGULARIZATION, 0.0) * histogram_w[k];
				for (var m: u32 = (k + 1); m < KERNELLEN; m += 1u) {
					var d: f32 = gaussian(histogram_l[m], HISTOGRAM_REGULARIZATION, histogram_l[k]);
					histogram_wn[m] += d * histogram_w[k];
					histogram_wn[k] += d * histogram_w[m];
				}
			}

			var maxv: vec3<f32> = vec3<f32>(0.0);
			var maxw: f32 = 0.0;

			for (var k: u32 = 0; k < KERNELLEN; k += 1u) {
				if (histogram_wn[k] >= maxw) {
					maxw = histogram_wn[k];
					maxv = histogram_v[k];
				}
			}

			textureStore(OUTPUT, vec2<i32>(destPos), vec4<f32>(maxv, 1.0));
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
