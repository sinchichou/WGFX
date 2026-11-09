// Anime4K_Thin_HQ — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/bloc97/Anime4K/blob/master/glsl/Experimental-Effects/Anime4K_Thin_HQ.glsl
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
//   -    verbatim, and `//! NUM_THREADS` is written as `64,1,1` (64 total threads, as in Magpie).
//   - WGSL textureGather returns the quad as x=(0,1) y=(1,1) z=(1,0) w=(0,0) relative to the
//   -    texel left-above the sample point — identical to D3D Gather — so Magpie's "w z / x y"
//   -    fill mapping is copied verbatim.
//   - `scene` (inputSize/inputPt/outputSize/outputPt/scale/frame) is auto-injected by WGFX at
//   -    @group(0) @binding(4); texture/sampler binding slots are owned by the code generator.
//   - Magpie `GetInputSize()/GetInputPt()/GetOutputSize()/GetOutputPt()` read the `scene`
//   -    uniform; intermediate feature maps are RGBA16F textures sampled with the POINT sampler
//   -    exactly as the HLSL does (Magpie's HLSL sampler is CLAMP/POINT, WGFX defaults to
//   -    clamp-to-edge).
//   - Magpie declares the two intermediate textures as R16G16_FLOAT; WebGPU has no
//   -    2-channel float texture format, so they are declared R16G16B16A16_FLOAT. The unused
//   -    blue/alpha channels carry 0.0/1.0 and are never read — the algorithm is unchanged.
//   - The HLSL parameters `strength` (float) and `iterations` (int) become WGFX
//   -    `//! PARAMETER` blocks (TYPE/LABEL/MIN/MAX/DEFAULT/STEP copied from the HLSL) and are
//   -    read as `uniforms.strength` / `uniforms.iterations`.
//   - HLSL mixes the u32 vector `destPos` with the float literal 0.5f; the port converts
//   -    explicitly (`vec2<f32>(destPos) + vec2<f32>(0.5, 0.5)`) — same value, WGSL has no
//   -    implicit u32→f32 vector promotion.
//   - The helpers `GetLuma` and `gaussian` are ported as WGSL functions inside the same pass
//   -    block (before the entry point), as allowed by the WGFX pass format.
//   - The HLSL `#ifdef MP_INLINE_PARAMS` guard only controlled HLSL inlining and has no WGSL
//   -    counterpart; it is dropped (no semantic effect).


//!VERSION 4
//!SORT_NAME Anime4K_Thin_HQ

//!PARAMETER strength
//!TYPE float
//!LABEL "Strength"
//!DEFAULT 0.6
//!MIN 0.1
//!MAX 10
//!STEP 0.1
float strength;

//!PARAMETER iterations
//!TYPE int
//!LABEL "Iterations"
//!DEFAULT 1
//!MIN 1
//!MAX 10
//!STEP 1
int iterations;

//!TEXTURE
Texture2D INPUT;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R16G16B16A16_FLOAT
var tex1_sampled: texture_2d<f32>;
var tex1_storaged: texture_storage_2d<rgba16float, write>;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R16G16B16A16_FLOAT
var tex2_sampled: texture_2d<f32>;
var tex2_storaged: texture_storage_2d<rgba16float, write>;

//!SAMPLER
//!FILTER POINT
SamplerState sam;

//!SAMPLER
//!FILTER LINEAR
SamplerState sam1;

//!PASS 1
//!DESC Sobel
//!IN INPUT
//!OUT tex2_storaged
//!BLOCK_SIZE 16
//!NUM_THREADS 64,1,1

fn GetLuma(rgb: vec3<f32>) -> f32 {
	return dot(vec3<f32>(0.299, 0.587, 0.114), rgb);
}

fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
	var gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * vec2<u32>(16u);
	var inputSize: vec2<u32> = GetInputSize();
	if (gxy.x >= inputSize.x || gxy.y >= inputSize.y) {
		return;
	}
	var inputPt: vec2<f32> = GetInputPt();

	var i: u32 = 0u;
	var j: u32 = 0u;

	var src: array<array<f32, 4>, 4>;
	for (var i: u32 = 0; i <= 2; i += 2u) {
		for (var j: u32 = 0; j <= 2; j += 2u) {
			var tpos: vec2<f32> = (vec2<f32>(gxy + vec2<u32>(i, j))) * inputPt;
			let sr: vec4<f32> = textureGather(0, INPUT, sam, tpos);
			let sg: vec4<f32> = textureGather(1, INPUT, sam, tpos);
			let sb: vec4<f32> = textureGather(2, INPUT, sam, tpos);

			// w z
			// x y
			src[i][j] = GetLuma(vec3<f32>(sr.w, sg.w, sb.w));
			src[i][j + 1] = GetLuma(vec3<f32>(sr.x, sg.x, sb.x));
			src[i + 1][j] = GetLuma(vec3<f32>(sr.z, sg.z, sb.z));
			src[i + 1][j + 1] = GetLuma(vec3<f32>(sr.y, sg.y, sb.y));
		}
	}

	for (var i: u32 = 1; i <= 2; i += 1u) {
		for (var j: u32 = 1; j <= 2; j += 1u) {
			var destPos: vec2<u32> = gxy + vec2<u32>(i - 1, j - 1);

			var xgrad: f32 = (-src[i - 1][j - 1] + src[i + 1][j - 1] - src[i - 1][j] + src[i + 1][j] - src[i - 1][j] + src[i + 1][j] - src[i - 1][j + 1] + src[i + 1][j + 1]) / 8.0;
			var ygrad: f32 = (-src[i - 1][j - 1] - src[i][j - 1] - src[i][j - 1] - src[i + 1][j - 1] + src[i - 1][j + 1] + src[i][j + 1] + src[i][j + 1] + src[i + 1][j + 1]) / 8.0;

			// Computes the luminance's gradient
			var norm: f32 = sqrt(xgrad * xgrad + ygrad * ygrad);
			textureStore(tex2_storaged, vec2<i32>(destPos), vec4<f32>(vec2<f32>(pow(norm, 0.7), 0), 0.0, 1.0));
		}
	}
}

//!PASS 2
//!DESC Gaussian-X
//!IN tex2_sampled
//!OUT tex1_storaged
//!BLOCK_SIZE 16
//!NUM_THREADS 64,1,1

fn gaussian(x: f32, s: f32, m: f32) -> f32 {
	var scaled: f32 = (x - m) / s;
	return exp(-0.5 * scaled * scaled);
}

fn Pass2(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
	let gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * vec2<u32>(16u);
	let inputSize: vec2<u32> = GetInputSize();
	if (gxy.x >= inputSize.x || gxy.y >= inputSize.y) {
		return;
	}
	let inputPt: vec2<f32> = GetInputPt();

	// Spatial window size, must be a positive real number.
	let SPATIAL_SIGMA: f32 = 2.0 * f32(GetInputSize().y) / 1080.0;
	// Kernel size, must be an positive odd integer.
	let KERNELSIZE: i32 = max(i32(ceil(SPATIAL_SIGMA * 2.0)), 1) * 2 + 1;
	//Half of the kernel size without remainder. Must be equal to trunc(KERNELSIZE/2).
	let KERNELHALFSIZE: i32 = KERNELSIZE >> 1;

	for (var i: u32 = 0; i <= 1; i += 1u) {
		for (var j: u32 = 0; j <= 1; j += 1u) {
			let destPos: vec2<u32> = gxy + vec2<u32>(i, j);

			if (i != 0 || j != 0) {
				if (destPos.x >= inputSize.x || destPos.y >= inputSize.y) {
					continue;
				}
			}

			let pos: vec2<f32> = (vec2<f32>(destPos) + vec2<f32>(0.5, 0.5)) * inputPt;

			var g: f32 = 0.0;
			var gn: f32 = 0.0;

			for (var k: i32 = 0; k < KERNELSIZE; k += 1) {
				var di: i32 = k - KERNELHALFSIZE;
				var gf: f32 = gaussian(f32(di), SPATIAL_SIGMA, 0.0);
				g = g + textureSampleLevel(tex2_sampled, sam, pos + vec2<f32>(f32(di) * inputPt.x, 0.0), 0.0).x * gf;
				gn = gn + gf;
			}

			textureStore(tex1_storaged, vec2<i32>(destPos), vec4<f32>(vec2<f32>(g / gn, 0), 0.0, 1.0));
		}
	}
}

//!PASS 3
//!DESC Gaussian-Y
//!IN tex1_sampled
//!OUT tex2_storaged
//!BLOCK_SIZE 16
//!NUM_THREADS 64,1,1

fn gaussian(x: f32, s: f32, m: f32) -> f32 {
	var scaled: f32 = (x - m) / s;
	return exp(-0.5 * scaled * scaled);
}

fn Pass3(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
	let gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * vec2<u32>(16u);
	let inputSize: vec2<u32> = GetInputSize();
	if (gxy.x >= inputSize.x || gxy.y >= inputSize.y) {
		return;
	}
	let inputPt: vec2<f32> = GetInputPt();

	// Spatial window size, must be a positive real number.
	let SPATIAL_SIGMA: f32 = 2.0 * f32(GetInputSize().y) / 1080.0;
	// Kernel size, must be an positive odd integer.
	let KERNELSIZE: i32 = max(i32(ceil(SPATIAL_SIGMA * 2.0)), 1) * 2 + 1;
	//Half of the kernel size without remainder. Must be equal to trunc(KERNELSIZE/2).
	let KERNELHALFSIZE: i32 = KERNELSIZE >> 1;

	for (var i: u32 = 0; i <= 1; i += 1u) {
		for (var j: u32 = 0; j <= 1; j += 1u) {
			let destPos: vec2<u32> = gxy + vec2<u32>(i, j);

			if (i != 0 || j != 0) {
				if (destPos.x >= inputSize.x || destPos.y >= inputSize.y) {
					continue;
				}
			}

			let pos: vec2<f32> = (vec2<f32>(destPos) + vec2<f32>(0.5, 0.5)) * inputPt;

			var g: f32 = 0.0;
			var gn: f32 = 0.0;

			for (var k: i32 = 0; k < KERNELSIZE; k += 1) {
				var di: i32 = k - KERNELHALFSIZE;
				var gf: f32 = gaussian(f32(di), SPATIAL_SIGMA, 0.0);

				g = g + textureSampleLevel(tex1_sampled, sam, pos + vec2<f32>(0.0, f32(di) * inputPt.y), 0.0).x * gf;
				gn = gn + gf;
			}

			textureStore(tex2_storaged, vec2<i32>(destPos), vec4<f32>(vec2<f32>(g / gn, 0), 0.0, 1.0));
		}
	}
}

//!PASS 4
//!DESC Kernel
//!IN tex2_sampled
//!OUT tex1_storaged
//!BLOCK_SIZE 16
//!NUM_THREADS 64,1,1

fn Pass4(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
	let gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * vec2<u32>(16u);
	let inputSize: vec2<u32> = GetInputSize();
	if (gxy.x >= inputSize.x || gxy.y >= inputSize.y) {
		return;
	}
	let inputPt: vec2<f32> = GetInputPt();

	var i: u32 = 0u;
	var j: u32 = 0u;

	var src: array<array<f32, 4>, 4>;
	for (var i: u32 = 0; i <= 2; i += 2u) {
		for (var j: u32 = 0; j <= 2; j += 2u) {
			var tpos: vec2<f32> = (vec2<f32>(gxy + vec2<u32>(i, j))) * inputPt;
			let sr: vec4<f32> = textureGather(0, tex2_sampled, sam, tpos);

			// w z
			// x y
			src[i][j] = sr.w;
			src[i][j + 1] = sr.x;
			src[i + 1][j] = sr.z;
			src[i + 1][j + 1] = sr.y;
		}
	}

	for (var i: u32 = 1; i <= 2; i += 1u) {
		for (var j: u32 = 1; j <= 2; j += 1u) {
			var destPos: vec2<u32> = gxy + vec2<u32>(i - 1, j - 1);

			var xgrad: f32 = -src[i - 1][j - 1] + src[i + 1][j - 1] - src[i - 1][j] + src[i + 1][j] - src[i - 1][j] + src[i + 1][j] - src[i - 1][j + 1] + src[i + 1][j + 1];
			var ygrad: f32 = -src[i - 1][j - 1] - src[i][j - 1] - src[i][j - 1] - src[i + 1][j - 1] + src[i - 1][j + 1] + src[i][j + 1] + src[i][j + 1] + src[i + 1][j + 1];

			// Computes the luminance's gradient
			textureStore(tex1_storaged, vec2<i32>(destPos), vec4<f32>(vec2<f32>(xgrad, ygrad) / 8.0, 0.0, 1.0));
		}
	}
}

//!PASS 5
//!DESC Warp
//!IN tex1_sampled, INPUT
//!OUT OUTPUT
//!BLOCK_SIZE 16
//!NUM_THREADS 64,1,1

fn Pass5(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
	let gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * vec2<u32>(16u);

	let outputSize: vec2<u32> = GetOutputSize();
	if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
		return;
	}

	let inputPt: vec2<f32> = GetInputPt();
	let relstr: f32 = f32(GetInputSize().y) / 1080.0 * uniforms.strength;

	for (var i: u32 = 0; i <= 1; i += 1u) {
		for (var j: u32 = 0; j <= 1; j += 1u) {
			let destPos: vec2<u32> = gxy + vec2<u32>(i, j);

			var pos: vec2<f32> = (vec2<f32>(destPos) + vec2<f32>(0.5, 0.5)) * inputPt;

			for (var i: i32 = 0; i < uniforms.iterations; i += 1) {
				var dn: vec2<f32> = textureSampleLevel(tex1_sampled, sam1, pos, 0.0).xy;
				var dd: vec2<f32> = (dn / (length(dn) + 0.01)) * inputPt * relstr; //Quasi-normalization for large vectors, avoids divide by zero
				pos -= dd;
			}

			textureStore(OUTPUT, vec2<i32>(destPos), textureSampleLevel(INPUT, sam1, pos, 0.0));
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
