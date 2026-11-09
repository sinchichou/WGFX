// xBRZ Freescale — Magpie v4 → WGFX WGSL port
//
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/xBRZ/xBRZ_Freescale.hlsl
//   which is itself a port of the libretro xBRZ free-scale multi-pass shader:
//     https://github.com/libretro/common-shaders/tree/master/xbrz/shaders/xbrz-freescale-multipass
//
// Original license / author:
//   xBRZ free-scale shader — Copyright (C) 2014-2016 DeSmuME team.
//     (GNU General Public License v3, as distributed with libretro/common-shaders)
//   xBR-vertex code and texel mapping — Copyright (C) 2011/2016 Hyllian (MIT).
//   Magpie HLSL port — Copyright (c) Blinue and Magpie contributors.
//   This WGFX WGSL port changes only the pass structure and the sampling idioms;
//   every xBRZ constant and every line of the algorithm is copied verbatim.
//
// Porting notes:
//   1. Magpie's `void PassN(uint2 blockStart, uint3 threadId)` are CS-style passes.
//      They map onto WGFX compute shaders: `blockStart` = workgroup_id.xy * BLOCK_SIZE
//      and `threadId.x` = local_invocation_id.x. Both passes use
//      `//! BLOCK_SIZE 8` + `//! NUM_THREADS 64`, i.e. one 8x8 output-pixel tile per
//      group with one pixel per thread.
//   2. The intermediate texture `tex1` (the per-pixel "blend info" produced by pass 1
//      and consumed by pass 2) is declared with the Magpie `_sampled`/`_storaged`
//      alias pair so that the WGFX runtime binds a sampled view in pass 2 and a
//      write-only storage view in pass 1 to the *same* logical texture. Its format is
//      declared as `//! FORMAT rgba8unorm` — Magpie's `R8G8B8A8_UNORM` is not a
//      WebGPU format name, so the equivalent `rgba8unorm` is used (the data layout
//      is identical: four unsigned 8-bit channels).
//   3. `INPUT.GatherRed/GatherGreen/GatherBlue` → WGSL
//      `textureGather(0|1|2, INPUT, sam, tpos)`. WebGPU's gather uses the same D3D
//      convention (quad base texel = floor(uv * size - 0.5); the returned vector is
//      w = top-left, z = top-right, x = bottom-left, y = bottom-right), so Magpie's
//      "w z / x y" fill is copied verbatim. The WGFX runtime creates every sampler as
//      a *filtering* sampler (nearest min/mag), which is what `textureGather`
//      requires; the gather itself is not filtered.
//   4. The `#define P(x,y) INPUT.SampleLevel(sam, pos + inputPt * float2(x, y), 0).rgb`
//      macro of pass 2 has no WGSL equivalent, so its five call sites are expanded
//      verbatim into `textureSampleLevel(INPUT, sam, pos + inputPt * vec2<f32>(x, y), 0.0).rgb`.
//   5. `OUTPUT[gxy] = float4(color, 1)` → `textureStore(OUTPUT, vec2<i32>(gxy), …)`;
//      WebGPU silently discards stores outside the texture, matching D3D.
//   6. `lerp` → `mix`; `frac` → `fract`; `step`, `sqrt`, `dot`, `min`, `max`,
//      `any`, `sign`, `length` and `smoothstep` keep their names (identical
//      semantics). HLSL's `f` float-literal suffix (`0.5f`) has no WGSL equivalent
//      and is dropped (the value is unchanged).
//   7. HLSL's `cond ? a : b` conditional expression is not implemented by the
//      Tint build used by node-webgpu, so it is rewritten as the equivalent WGSL
//      `select(b, a, cond)`.
//   8. In pass 2 the blend codes are compared against `BLEND_NONE`/`BLEND_DOMINANT`,
//      but there `blendResult` holds *floats* (the codes were divided by 255 in
//      pass 1). The comparisons therefore use the float literals 0.0 / 1.0 / 2.0
//      instead of the integer constants — the numeric values are identical.
//   9. The Tint build used by node-webgpu does not implement WGSL's `fmod`
//      builtin, so HLSL's `fmod(info, 4.0)` becomes the WGSL remainder operator
//      `info % vec4<f32>(4.0)` — the float remainder operator has exactly fmod
//      semantics (sign of the dividend). The vector second operand is required
//      because WGSL's `%` needs both operands to have the same type.
//  10. The intermediate `float3 src[5][5]` becomes
//      `array<array<vec3<f32>, 5>, 5>`; the HLSL's `int2(i - 2, 2)` index
//      arithmetic is done in `i32` (the HLSL computes it in the same signed type
//      after the `int2` cast, so the values are identical).
//
//! VERSION 4
//! SORT_NAME xBRZ_Freescale

//! TEXTURE
Texture2D INPUT;

//! TEXTURE
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! TEXTURE
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
//! FORMAT rgba8unorm
var tex1_sampled: texture_2d<f32>;
var tex1_storaged: texture_storage_2d<rgba8unorm, write>;

//! SAMPLER
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! COMMON
// Magpie's `Rmp8x8`: linear thread index → 8x8 position inside the group.
fn Rmp8x8(a: u32) -> vec2<u32> {
    return vec2<u32>(a / 8u, a % 8u);
}

const BLEND_NONE: i32 = 0;
const BLEND_NORMAL: i32 = 1;
const BLEND_DOMINANT: i32 = 2;
const LUMINANCE_WEIGHT: f32 = 1.0;
const EQUAL_COLOR_TOLERANCE: f32 = 30.0/255.0;
const STEEP_DIRECTION_THRESHOLD: f32 = 2.2;
const DOMINANT_DIRECTION_THRESHOLD: f32 = 3.6;

fn DistYCbCr(pixA: vec3<f32>, pixB: vec3<f32>) -> f32 {
	let w: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);
	let scaleB: f32 = 0.5 / (1.0 - w.b);
	let scaleR: f32 = 0.5 / (1.0 - w.r);
	var diff: vec3<f32> = pixA - pixB;
	var Y: f32 = dot(diff.rgb, w);
	var Cb: f32 = scaleB * (diff.b - Y);
	var Cr: f32 = scaleR * (diff.r - Y);

	return sqrt(((LUMINANCE_WEIGHT * Y) * (LUMINANCE_WEIGHT * Y)) + (Cb * Cb) + (Cr * Cr));
}

fn IsPixEqual(pixA: vec3<f32>, pixB: vec3<f32>) -> bool {
	return (DistYCbCr(pixA, pixB) < EQUAL_COLOR_TOLERANCE);
}

fn get_left_ratio(center: vec2<f32>, origin: vec2<f32>, direction: vec2<f32>, scale: vec2<f32>) -> f32 {
	var P0: vec2<f32> = center - origin;
	var proj: vec2<f32> = direction * (dot(P0, direction) / dot(direction, direction));
	var distv: vec2<f32> = P0 - proj;
	var orth: vec2<f32> = vec2<f32>(-direction.y, direction.x);
	var side: f32 = sign(dot(P0, orth));
	var v: f32 = side * length(distv * scale);

	//  return step(0, v);
	return smoothstep(-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, v);
}

//#define eq(a,b)  (a == b)
fn eq(a: vec3<f32>, b: vec3<f32>) -> bool {
	return ((a.x == b.x) && (a.y == b.y) && (a.z == b.z));
}
//#define neq(a,b) (a != b)
fn neq(a: vec3<f32>, b: vec3<f32>) -> bool {
	return !((a.x == b.x) && (a.y == b.y) && (a.z == b.z));
}
//! END COMMON

//! PASS 1
//! DESC "xBRZ free-scale — pass 1: per-pixel blend classification"
//! IN INPUT
//! OUT tex1_storaged
//! BLOCK_SIZE 8
//! NUM_THREADS 64

fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let blockStart: vec2<u32> = workgroup_id.xy * vec2<u32>(8u, 8u);
	var gxy: vec2<u32> = Rmp8x8(local_id.x) + blockStart;
	var inputSize: vec2<u32> = scene.inputSize;
	if (gxy.x >= inputSize.x || gxy.y >= inputSize.y) {
		return;
	}

	var inputPt: vec2<f32> = scene.inputPt;
	let pos: vec2<f32> = (vec2<f32>(gxy) + 0.5) * inputPt;
	var i: i32;
	var j: i32;

	var src: array<array<vec3<f32>, 5>, 5>;


	for (var i: i32 = 0; i <= 2; i += 2) {

		for (var j: i32 = 0; j <= 2; j += 2) {
			var tpos: vec2<f32> = vec2<f32>(vec2<i32>(gxy) + vec2<i32>(i, j) - vec2<i32>(1, 1)) * inputPt;
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


	for (var i: i32 = 0; i < 5; i += 1) {
		src[i][4] = textureSampleLevel(INPUT, sam, pos + vec2<f32>(vec2<i32>(i - 2, 2)) * inputPt, 0.0).rgb;
	}

	for (var j: i32 = 0; j < 4; j += 1) {
		src[4][j] = textureSampleLevel(INPUT, sam, pos + vec2<f32>(vec2<i32>(2, j - 2)) * inputPt, 0.0).rgb;
	}

	//---------------------------------------
	// Input Pixel Mapping:  -|x|x|x|-
	//                       x|A|B|C|x
	//                       x|D|E|F|x
	//                       x|G|H|I|x
	//                       -|x|x|x|-

	var A: vec3<f32> = src[1][1];
	var B: vec3<f32> = src[2][1];
	var C: vec3<f32> = src[3][1];
	var D: vec3<f32> = src[1][2];
	var E: vec3<f32> = src[2][2];
	var F: vec3<f32> = src[3][2];
	var G: vec3<f32> = src[1][3];
	var H: vec3<f32> = src[2][3];
	var I: vec3<f32> = src[3][3];

	// blendResult Mapping: x|y|
	//                      w|z|
	var blendResult: vec4<i32> = vec4<i32>(0, 0, 0, 0);

	// Preprocess corners
	// Pixel Tap Mapping: -|-|-|-|-
	//                    -|-|B|C|-
	//                    -|D|E|F|x
	//                    -|G|H|I|x
	//                    -|-|x|x|-
	if (!((eq(E, F) && eq(H, I)) || (eq(E, H) && eq(F, I)))) {
		var dist_H_F: f32 = DistYCbCr(G, E) + DistYCbCr(E, C) + DistYCbCr(src[2][4], I) + DistYCbCr(I, src[4][2]) + (4.0 * DistYCbCr(H, F));
		var dist_E_I: f32 = DistYCbCr(D, H) + DistYCbCr(H, src[3][4]) + DistYCbCr(B, F) + DistYCbCr(F, src[4][3]) + (4.0 * DistYCbCr(E, I));
		var dominantGradient: bool = (DOMINANT_DIRECTION_THRESHOLD * dist_H_F) < dist_E_I;
		blendResult.z = select( BLEND_NONE,  (select( BLEND_NORMAL,  BLEND_DOMINANT , (dominantGradient))) , ((dist_H_F < dist_E_I) && neq(E, F) && neq(E, H)));
	}


	// Pixel Tap Mapping: -|-|-|-|-
	//                    -|A|B|-|-
	//                    x|D|E|F|-
	//                    x|G|H|I|-
	//                    -|x|x|-|-
	if (!((eq(D, E) && eq(G, H)) || (eq(D, G) && eq(E, H)))) {
		var dist_G_E: f32 = DistYCbCr(src[0][3], D) + DistYCbCr(D, B) + DistYCbCr(src[1][4], H) + DistYCbCr(H, F) + (4.0 * DistYCbCr(G, E));
		var dist_D_H: f32 = DistYCbCr(src[0][2], G) + DistYCbCr(G, src[2][4]) + DistYCbCr(A, E) + DistYCbCr(E, I) + (4.0 * DistYCbCr(D, H));
		var dominantGradient: bool = (DOMINANT_DIRECTION_THRESHOLD * dist_D_H) < dist_G_E;
		blendResult.w = select( BLEND_NONE,  (select( BLEND_NORMAL,  BLEND_DOMINANT , (dominantGradient))) , ((dist_G_E > dist_D_H) && neq(E, D) && neq(E, H)));
	}

	// Pixel Tap Mapping: -|-|x|x|-
	//                    -|A|B|C|x
	//                    -|D|E|F|x
	//                    -|-|H|I|-
	//                    -|-|-|-|-
	if (!((eq(B, C) && eq(E, F)) || (eq(B, E) && eq(C, F)))) {
		var dist_E_C: f32 = DistYCbCr(D, B) + DistYCbCr(B, src[3][0]) + DistYCbCr(H, F) + DistYCbCr(F, src[4][1]) + (4.0 * DistYCbCr(E, C));
		var dist_B_F: f32 = DistYCbCr(A, E) + DistYCbCr(E, I) + DistYCbCr(src[2][0], C) + DistYCbCr(C, src[4][2]) + (4.0 * DistYCbCr(B, F));
		var dominantGradient: bool = (DOMINANT_DIRECTION_THRESHOLD * dist_B_F) < dist_E_C;
		blendResult.y = select( BLEND_NONE,  (select( BLEND_NORMAL,  BLEND_DOMINANT , (dominantGradient))) , ((dist_E_C > dist_B_F) && neq(E, B) && neq(E, F)));
	}

	// Pixel Tap Mapping: -|x|x|-|-
	//                    x|A|B|C|-
	//                    x|D|E|F|-
	//                    -|G|H|-|-
	//                    -|-|-|-|-
	if (!((eq(A, B) && eq(D, E)) || (eq(A, D) && eq(B, E)))) {
		var dist_D_B: f32 = DistYCbCr(src[0][2], A) + DistYCbCr(A, src[2][0]) + DistYCbCr(G, E) + DistYCbCr(E, C) + (4.0 * DistYCbCr(D, B));
		var dist_A_E: f32 = DistYCbCr(src[0][1], D) + DistYCbCr(D, H) + DistYCbCr(src[1][0], B) + DistYCbCr(B, F) + (4.0 * DistYCbCr(A, E));
		var dominantGradient: bool = (DOMINANT_DIRECTION_THRESHOLD * dist_D_B) < dist_A_E;
		blendResult.x = select( BLEND_NONE,  (select( BLEND_NORMAL,  BLEND_DOMINANT , (dominantGradient))) , ((dist_D_B < dist_A_E) && neq(E, D) && neq(E, B)));
	}

	var FragColor: vec4<f32> = vec4<f32>(blendResult);

	// Pixel Tap Mapping: -|-|-|-|-
	//                    -|-|B|C|-
	//                    -|D|E|F|x
	//                    -|G|H|I|x
	//                    -|-|x|x|-
	if (blendResult.z == BLEND_DOMINANT || (blendResult.z == BLEND_NORMAL &&
		!((blendResult.y != BLEND_NONE && !IsPixEqual(E, G)) || (blendResult.w != BLEND_NONE && !IsPixEqual(E, C)) ||
			(IsPixEqual(G, H) && IsPixEqual(H, I) && IsPixEqual(I, F) && IsPixEqual(F, C) && !IsPixEqual(E, I))))) {
		FragColor.z += 4.0;

		var dist_F_G: f32 = DistYCbCr(F, G);
		var dist_H_C: f32 = DistYCbCr(H, C);

		if ((STEEP_DIRECTION_THRESHOLD * dist_F_G <= dist_H_C) && neq(E, G) && neq(D, G)) {
			FragColor.z += 16.0;
		}

		if ((STEEP_DIRECTION_THRESHOLD * dist_H_C <= dist_F_G) && neq(E, C) && neq(B, C)) {
			FragColor.z += 64.0;
		}
	}

	// Pixel Tap Mapping: -|-|-|-|-
	//                    -|A|B|-|-
	//                    x|D|E|F|-
	//                    x|G|H|I|-
	//                    -|x|x|-|-
	if (blendResult.w == BLEND_DOMINANT || (blendResult.w == BLEND_NORMAL &&
		!((blendResult.z != BLEND_NONE && !IsPixEqual(E, A)) || (blendResult.x != BLEND_NONE && !IsPixEqual(E, I)) ||
			(IsPixEqual(A, D) && IsPixEqual(D, G) && IsPixEqual(G, H) && IsPixEqual(H, I) && !IsPixEqual(E, G))))) {
		FragColor.w += 4.0;

		var dist_H_A: f32 = DistYCbCr(H, A);
		var dist_D_I: f32 = DistYCbCr(D, I);

		if ((STEEP_DIRECTION_THRESHOLD * dist_H_A <= dist_D_I) && neq(E, A) && neq(B, A)) {
			FragColor.w += 16.0;
		}

		if ((STEEP_DIRECTION_THRESHOLD * dist_D_I <= dist_H_A) && neq(E, I) && neq(F, I)) {
			FragColor.w += 64.0;
		}
	}

	// Pixel Tap Mapping: -|-|x|x|-
	//                    -|A|B|C|x
	//                    -|D|E|F|x
	//                    -|-|H|I|-
	//                    -|-|-|-|-
	if (blendResult.y == BLEND_DOMINANT || (blendResult.y == BLEND_NORMAL &&
		!((blendResult.x != BLEND_NONE && !IsPixEqual(E, I)) || (blendResult.z != BLEND_NONE && !IsPixEqual(E, A)) ||
			(IsPixEqual(I, F) && IsPixEqual(F, C) && IsPixEqual(C, B) && IsPixEqual(B, A) && !IsPixEqual(E, C))))) {
		FragColor.y += 4.0;

		var dist_B_I: f32 = DistYCbCr(B, I);
		var dist_F_A: f32 = DistYCbCr(F, A);

		if ((STEEP_DIRECTION_THRESHOLD * dist_B_I <= dist_F_A) && neq(E, I) && neq(H, I)) {
			FragColor.y += 16.0;
		}

		if ((STEEP_DIRECTION_THRESHOLD * dist_F_A <= dist_B_I) && neq(E, A) && neq(D, A)) {
			FragColor.y += 64.0;
		}
	}

	// Pixel Tap Mapping: -|x|x|-|-
	//                    x|A|B|C|-
	//                    x|D|E|F|-
	//                    -|G|H|-|-
	//                    -|-|-|-|-
	if (blendResult.x == BLEND_DOMINANT || (blendResult.x == BLEND_NORMAL &&
		!((blendResult.w != BLEND_NONE && !IsPixEqual(E, C)) || (blendResult.y != BLEND_NONE && !IsPixEqual(E, G)) ||
			(IsPixEqual(C, B) && IsPixEqual(B, A) && IsPixEqual(A, D) && IsPixEqual(D, G) && !IsPixEqual(E, A))))) {
		FragColor.x += 4.0;

		var dist_D_C: f32 = DistYCbCr(D, C);
		var dist_B_G: f32 = DistYCbCr(B, G);

		if ((STEEP_DIRECTION_THRESHOLD * dist_D_C <= dist_B_G) && neq(E, C) && neq(F, C)) {
			FragColor.x += 16.0;
		}

		if ((STEEP_DIRECTION_THRESHOLD * dist_B_G <= dist_D_C) && neq(E, G) && neq(H, G)) {
			FragColor.x += 64.0;
		}
	}
	
	textureStore(tex1_storaged, vec2<i32>(gxy), FragColor / 255.0);
}

//! PASS 2
//! DESC "xBRZ free-scale — pass 2: edge-aware scale blending"
//! IN INPUT, tex1_sampled
//! OUT OUTPUT
//! BLOCK_SIZE 8
//! NUM_THREADS 64

fn Pass2(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let blockStart: vec2<u32> = workgroup_id.xy * vec2<u32>(8u, 8u);
	var gxy: vec2<u32> = Rmp8x8(local_id.x) + blockStart;

	let outputSize: vec2<u32> = scene.outputSize;
	if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
		return;
	}

	var inputPt: vec2<f32> = scene.inputPt;
	var outputPt: vec2<f32> = scene.outputPt;
	let pos: vec2<f32> = (vec2<f32>(gxy) + 0.5) * outputPt;

	//---------------------------------------
	// Input Pixel Mapping: -|B|-
	//                      D|E|F
	//                      -|H|-

	var scale: vec2<f32> = scene.scale;
	var f: vec2<f32> = fract(pos * vec2<f32>(scene.inputSize)) - 0.5;

	var B: vec3<f32> = textureSampleLevel(INPUT, sam, pos + inputPt * vec2<f32>(0, -1), 0.0).rgb;
	var D: vec3<f32> = textureSampleLevel(INPUT, sam, pos + inputPt * vec2<f32>(-1, 0), 0.0).rgb;
	var E: vec3<f32> = textureSampleLevel(INPUT, sam, pos + inputPt * vec2<f32>(0, 0), 0.0).rgb;
	var F: vec3<f32> = textureSampleLevel(INPUT, sam, pos + inputPt * vec2<f32>(1, 0), 0.0).rgb;
	var H: vec3<f32> = textureSampleLevel(INPUT, sam, pos + inputPt * vec2<f32>(0, 1), 0.0).rgb;

	var info: vec4<f32> = floor(textureSampleLevel(tex1_sampled, sam, pos, 0.0) * 255 + 0.5);

	// info Mapping: x|y|
	//               w|z|

	var blendResult: vec4<f32> = floor((info % vec4<f32>(4.0)));
	var doLineBlend: vec4<f32> = floor((info / 4.0 % vec4<f32>(4.0)));
	var haveShallowLine: vec4<f32> = floor((info / 16.0 % vec4<f32>(4.0)));
	var haveSteepLine: vec4<f32> = floor((info / 64.0 % vec4<f32>(4.0)));

	var res: vec3<f32> = E;

	// Pixel Tap Mapping: -|-|-
	//                    -|E|F
	//                    -|H|-

	if (blendResult.z > 0.0) {
		var origin: vec2<f32> = vec2<f32>(0.0, 1.0 / sqrt(2.0));
		var direction: vec2<f32> = vec2<f32>(1.0, -1.0);
		if (doLineBlend.z > 0.0) {
			origin = select(vec2<f32>(0.0, 0.5), vec2<f32>(0.0, 0.25), haveShallowLine.z > 0.0);
			direction.x += haveShallowLine.z;
			direction.y -= haveSteepLine.z;
		}

		var blendPix: vec3<f32> = mix(H, F, step(DistYCbCr(E, F), DistYCbCr(E, H)));
		res = mix(res, blendPix, get_left_ratio(f, origin, direction, scale));
	}

	// Pixel Tap Mapping: -|-|-
	//                    D|E|-
	//                    -|H|-
	if (blendResult.w > 0.0) {
		var origin: vec2<f32> = vec2<f32>(-1.0 / sqrt(2.0), 0.0);
		var direction: vec2<f32> = vec2<f32>(1.0, 1.0);
		if (doLineBlend.w > 0.0) {
			origin = select(vec2<f32>(-0.5, 0.0), vec2<f32>(-0.25, 0.0), haveShallowLine.w > 0.0);
			direction.y += haveShallowLine.w;
			direction.x += haveSteepLine.w;
		}

		var blendPix: vec3<f32> = mix(H, D, step(DistYCbCr(E, D), DistYCbCr(E, H)));
		res = mix(res, blendPix, get_left_ratio(f, origin, direction, scale));
	}

	// Pixel Tap Mapping: -|B|-
	//                    -|E|F
	//                    -|-|-
	if (blendResult.y > 0.0) {
		var origin: vec2<f32> = vec2<f32>(1.0 / sqrt(2.0), 0.0);
		var direction: vec2<f32> = vec2<f32>(-1.0, -1.0);

		if (doLineBlend.y > 0.0) {
			origin = select(vec2<f32>(0.5, 0.0), vec2<f32>(0.25, 0.0), haveShallowLine.y > 0.0);
			direction.y -= haveShallowLine.y;
			direction.x -= haveSteepLine.y;
		}

		var blendPix: vec3<f32> = mix(F, B, step(DistYCbCr(E, B), DistYCbCr(E, F)));
		res = mix(res, blendPix, get_left_ratio(f, origin, direction, scale));
	}

	// Pixel Tap Mapping: -|B|-
	//                    D|E|-
	//                    -|-|-
	if (blendResult.x > 0.0) {
		var origin: vec2<f32> = vec2<f32>(0.0, -1.0 / sqrt(2.0));
		var direction: vec2<f32> = vec2<f32>(-1.0, 1.0);
		if (doLineBlend.x > 0.0) {
			origin = select(vec2<f32>(0.0, -0.5), vec2<f32>(0.0, -0.25), haveShallowLine.x > 0.0);
			direction.x -= haveShallowLine.x;
			direction.y += haveSteepLine.x;
		}

		var blendPix: vec3<f32> = mix(D, B, step(DistYCbCr(E, B), DistYCbCr(E, D)));
		res = mix(res, blendPix, get_left_ratio(f, origin, direction, scale));
	}

	textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(res, 1.0));
}
