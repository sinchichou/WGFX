// xBRZ 2x — Magpie v4 → WGFX WGSL port
//
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/xBRZ/xBRZ_2x.hlsl
//   which is itself a port of the libretro xBRZ shader:
//     https://github.com/libretro/common-shaders/blob/master/xbrz/shaders/2xbrz.cg
//
// Original license / author:
//   2xBRZ shader — Copyright (C) 2014-2016 DeSmuME team.
//     (GNU General Public License v3, as distributed with libretro/common-shaders)
//   xBR-vertex code and texel mapping — Copyright (C) 2011/2016 Hyllian (MIT).
//   Magpie HLSL port — Copyright (c) Blinue and Magpie contributors.
//   This WGFX WGSL port changes only the pass structure and the sampling idioms;
//   every xBRZ constant and every line of the algorithm is copied verbatim.
//
// Porting notes:
//   1. Magpie's `void Pass1(uint2 blockStart, uint3 threadId)` is a CS-style pass.
//      It maps onto a WGFX compute shader: `blockStart` = workgroup_id.xy * BLOCK_SIZE
//      and `threadId.x` = local_invocation_id.x. Magpie's `//! BLOCK_SIZE 16`
//      + `//! NUM_THREADS 64` means "one 16x16 output-pixel tile per
//      group, 64 threads"; the 8x8 thread grid (Rmp8x8) times the 2x scale factor
//      covers the whole tile.
//   2. The output texture is declared `//! WIDTH INPUT_WIDTH * 2` /
//      `//! HEIGHT INPUT_HEIGHT * 2` so that the host allocates the 2x
//      scaled target; the pass derives the input coordinate with `gxy >> 1`
//      (or `gxy / 2` for the odd variants) exactly like the HLSL.
//   3. `INPUT.SampleLevel(sam, uv, 0).rgb` → WGSL
//      `textureSampleLevel(INPUT, sam, uv, 0.0).rgb`. The HLSL samples the texel
//      *centres* (uv = (i + 0.5) * inputPt) with a POINT sampler; WGSL's
//      `textureSampleLevel` with the WGFX nearest sampler picks the same texel, and
//      the CLAMP address mode reproduces D3D's border clamping.
//   4. `OUTPUT[gxy] = float4(color, 1)` → `textureStore(OUTPUT, vec2<i32>(gxy), …)`.
//      WebGPU silently discards stores outside the texture, matching D3D.
//   5. `lerp` → `mix`; `frac` → `fract`; `step`, `sqrt`, `dot`, `min`, `max`,
//      `any`, `sign` and `smoothstep` keep their names (identical semantics).
//      HLSL's `f` float-literal suffix (`0.5f`) has no WGSL equivalent and is dropped
//      (the value is unchanged).
//   6. HLSL arrays become WGSL `array<T, N>`. `void ScalePixel(const int4 blend,
//      const float3 k[9], inout float3 dst[N])` has no WGSL equivalent, so the
//      function takes `k` by value and `dst` as a `ptr<function, …>`, which has
//      exactly the same in/out semantics.
//   7. `int4 blendResult` becomes `vec4<i32>`; the HLSL's `blendResult[2] = …`
//      becomes the equivalent component assignment `blendResult.z = …`.
//   8. `const static uint destIdx[N][N]` becomes a function-local
//      `var destIdx = array<array<u32, N>, N>(…)` so it can be indexed with the
//      runtime loop variables (WGSL forbids dynamic indexing of a module-scope
//      `const` array). The values are copied verbatim.
//   9. Locals that the source re-assigns (e.g. `haveShallowLine` in the 4x/5x/6x
//      variants) are declared with `var`; WGSL forbids assigning to a `let`.
//  10. All HLSL local declarations become `var`/`const` declarations with an
//      explicit type (`var name: T = …`) because WGSL has no type-first local
//      declaration form.
//
//! VERSION 4
//! SORT_NAME xBRZ_2x

//! TEXTURE
Texture2D INPUT;

//! TEXTURE
//! WIDTH INPUT_WIDTH * 2
//! HEIGHT INPUT_HEIGHT * 2
Texture2D OUTPUT;

//! SAMPLER
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! DESC "xBRZ 2x — high quality pixel art upscaler"
//! IN INPUT
//! OUT OUTPUT
//! BLOCK_SIZE 16
//! NUM_THREADS 64

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
const M_PI: f32 = 3.1415926535897932384626433832795;

fn reduce(color: vec3<f32>) -> f32 {
	return dot(color, vec3<f32>(65536.0, 256.0, 1.0));
}

fn DistYCbCr(pixA: vec3<f32>, pixB: vec3<f32>) -> f32 {
	let w: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);
	let scaleB: f32 = 0.5 / (1.0 - w.b);
	let scaleR: f32 = 0.5 / (1.0 - w.r);
	var diff: vec3<f32> = pixA - pixB;
	var Y: f32 = dot(diff, w);
	var Cb: f32 = scaleB * (diff.b - Y);
	var Cr: f32 = scaleR * (diff.r - Y);

	return sqrt(((LUMINANCE_WEIGHT * Y) * (LUMINANCE_WEIGHT * Y)) + (Cb * Cb) + (Cr * Cr));
}

fn IsPixEqual(pixA: vec3<f32>, pixB: vec3<f32>) -> bool {
	return (DistYCbCr(pixA, pixB) < EQUAL_COLOR_TOLERANCE);
}

fn IsBlendingNeeded(blend: vec4<i32>) -> bool {
	return any(!(blend == vec4<i32>(0, 0, 0, 0)));
}

fn ScalePixel(blend: vec4<i32>, k: array<vec3<f32>, 9>, dst: ptr<function, array<vec3<f32>, 4>>) {
	var v0: f32 = reduce(k[0]);
	var v4: f32 = reduce(k[4]);
	var v5: f32 = reduce(k[5]);
	var v7: f32 = reduce(k[7]);
	var v8: f32 = reduce(k[8]);

	var dist_01_04: f32 = DistYCbCr(k[1], k[4]);
	var dist_03_08: f32 = DistYCbCr(k[3], k[8]);
	var haveShallowLine: bool = (STEEP_DIRECTION_THRESHOLD * dist_01_04 <= dist_03_08) && (v0 != v4) && (v5 != v4);
	var haveSteepLine: bool = (STEEP_DIRECTION_THRESHOLD * dist_03_08 <= dist_01_04) && (v0 != v8) && (v7 != v8);
	var needBlend: bool = (blend[2] != BLEND_NONE);
	var doLineBlend: bool = (blend[2] >= BLEND_DOMINANT ||
		!((blend[1] != BLEND_NONE && !IsPixEqual(k[0], k[4])) ||
			(blend[3] != BLEND_NONE && !IsPixEqual(k[0], k[8])) ||
			(IsPixEqual(k[4], k[3]) && IsPixEqual(k[3], k[2]) && IsPixEqual(k[2], k[1]) && IsPixEqual(k[1], k[8]) && !IsPixEqual(k[0], k[2]))));

	var blendPix: vec3<f32> = select( k[3],  k[1] , (DistYCbCr(k[0], k[1]) <= DistYCbCr(k[0], k[3])));
	(*dst)[1] = mix((*dst)[1], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveSteepLine)));
	(*dst)[2] = mix((*dst)[2], blendPix, select( 0.00,  (select( 1.0 - (M_PI / 4.0),  (select( (select( 0.50,  0.75 , (haveSteepLine))),  (select( 0.75,  5.0 / 6.0 , (haveSteepLine))) , (haveShallowLine))) , (doLineBlend))) , (needBlend)));
	(*dst)[3] = mix((*dst)[3], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveShallowLine)));
}

//---------------------------------------
// Input Pixel Mapping:  --|21|22|23|--
//                       19|06|07|08|09
//                       18|05|00|01|10
//                       17|04|03|02|11
//                       --|15|14|13|--
//
// Output Pixel Mapping:     00|01
//                           03|02
fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let blockStart: vec2<u32> = workgroup_id.xy * vec2<u32>(16u, 16u);
	var gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u, 1u)) + blockStart;

	let outputSize: vec2<u32> = scene.outputSize;
	if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
		return;
	}

	let inputPt: vec2<f32> = scene.inputPt;
	let pos: vec2<f32> = (vec2<f32>(gxy >> vec2<u32>(1u, 1u)) + vec2<f32>(0.5)) * inputPt;

	//    A1 B1 C1
	// A0 A  B  C  C4
	// D0 D  E  F  F4
	// G0 G  H  I  I4
	//    G5 H5 I5

	let t1: vec4<f32> = vec4<f32>(pos.x, pos.x, pos.x, pos.y) + vec4<f32>(-inputPt.x, 0, inputPt.x, -2.0 * inputPt.y); // A1 B1 C1
	let t2: vec4<f32> = vec4<f32>(pos.x, pos.x, pos.x, pos.y) + vec4<f32>(-inputPt.x, 0, inputPt.x, -inputPt.y);		// A  B  C
	let t3: vec4<f32> = vec4<f32>(pos.x, pos.x, pos.x, pos.y) + vec4<f32>(-inputPt.x, 0, inputPt.x, 0);				// D  E  F
	let t4: vec4<f32> = vec4<f32>(pos.x, pos.x, pos.x, pos.y) + vec4<f32>(-inputPt.x, 0, inputPt.x, inputPt.y);		// G  H  I
	let t5: vec4<f32> = vec4<f32>(pos.x, pos.x, pos.x, pos.y) + vec4<f32>(-inputPt.x, 0, inputPt.x, 2.0 * inputPt.y);	// G5 H5 I5
	let t6: vec4<f32> = vec4<f32>(pos.x, pos.y, pos.y, pos.y) + vec4<f32>(-2.0 * inputPt.x, -inputPt.y, 0, inputPt.y);	// A0 D0 G0
	let t7: vec4<f32> = vec4<f32>(pos.x, pos.y, pos.y, pos.y) + vec4<f32>(2.0 * inputPt.x, -inputPt.y, 0, inputPt.y);	// C4 F4 I4

	//---------------------------------------
	// Input Pixel Mapping:  20|21|22|23|24
	//                       19|06|07|08|09
	//                       18|05|00|01|10
	//                       17|04|03|02|11
	//                       16|15|14|13|12

	var src: array<vec3<f32>, 25>;

	src[21] = textureSampleLevel(INPUT, sam, t1.xw, 0.0).rgb;
	src[22] = textureSampleLevel(INPUT, sam, t1.yw, 0.0).rgb;
	src[23] = textureSampleLevel(INPUT, sam, t1.zw, 0.0).rgb;
	src[6] = textureSampleLevel(INPUT, sam, t2.xw, 0.0).rgb;
	src[7] = textureSampleLevel(INPUT, sam, t2.yw, 0.0).rgb;
	src[8] = textureSampleLevel(INPUT, sam, t2.zw, 0.0).rgb;
	src[5] = textureSampleLevel(INPUT, sam, t3.xw, 0.0).rgb;
	src[0] = textureSampleLevel(INPUT, sam, t3.yw, 0.0).rgb;
	src[1] = textureSampleLevel(INPUT, sam, t3.zw, 0.0).rgb;
	src[4] = textureSampleLevel(INPUT, sam, t4.xw, 0.0).rgb;
	src[3] = textureSampleLevel(INPUT, sam, t4.yw, 0.0).rgb;
	src[2] = textureSampleLevel(INPUT, sam, t4.zw, 0.0).rgb;
	src[15] = textureSampleLevel(INPUT, sam, t5.xw, 0.0).rgb;
	src[14] = textureSampleLevel(INPUT, sam, t5.yw, 0.0).rgb;
	src[13] = textureSampleLevel(INPUT, sam, t5.zw, 0.0).rgb;
	src[19] = textureSampleLevel(INPUT, sam, t6.xy, 0.0).rgb;
	src[18] = textureSampleLevel(INPUT, sam, t6.xz, 0.0).rgb;
	src[17] = textureSampleLevel(INPUT, sam, t6.xw, 0.0).rgb;
	src[9] = textureSampleLevel(INPUT, sam, t7.xy, 0.0).rgb;
	src[10] = textureSampleLevel(INPUT, sam, t7.xz, 0.0).rgb;
	src[11] = textureSampleLevel(INPUT, sam, t7.xw, 0.0).rgb;

	var v: array<f32, 9>;
	v[0] = reduce(src[0]);
	v[1] = reduce(src[1]);
	v[2] = reduce(src[2]);
	v[3] = reduce(src[3]);
	v[4] = reduce(src[4]);
	v[5] = reduce(src[5]);
	v[6] = reduce(src[6]);
	v[7] = reduce(src[7]);
	v[8] = reduce(src[8]);

	var blendResult: vec4<i32> = vec4<i32>(0, 0, 0, 0);

	// Preprocess corners
	// Pixel Tap Mapping: --|--|--|--|--
	//                    --|--|07|08|--
	//                    --|05|00|01|10
	//                    --|04|03|02|11
	//                    --|--|14|13|--

	// Corner (1, 1)
	if (!((v[0] == v[1] && v[3] == v[2]) || (v[0] == v[3] && v[1] == v[2]))) {
		var dist_03_01: f32 = DistYCbCr(src[4], src[0]) + DistYCbCr(src[0], src[8]) + DistYCbCr(src[14], src[2]) + DistYCbCr(src[2], src[10]) + (4.0 * DistYCbCr(src[3], src[1]));
		var dist_00_02: f32 = DistYCbCr(src[5], src[3]) + DistYCbCr(src[3], src[13]) + DistYCbCr(src[7], src[1]) + DistYCbCr(src[1], src[11]) + (4.0 * DistYCbCr(src[0], src[2]));
		var dominantGradient: bool = (DOMINANT_DIRECTION_THRESHOLD * dist_03_01) < dist_00_02;
		blendResult.z = select( BLEND_NONE,  (select( BLEND_NORMAL,  BLEND_DOMINANT , (dominantGradient))) , ((dist_03_01 < dist_00_02) && (v[0] != v[1]) && (v[0] != v[3])));
	}

	// Pixel Tap Mapping: --|--|--|--|--
	//                    --|06|07|--|--
	//                    18|05|00|01|--
	//                    17|04|03|02|--
	//                    --|15|14|--|--
	// Corner (0, 1)
	if (!((v[5] == v[0] && v[4] == v[3]) || (v[5] == v[4] && v[0] == v[3]))) {
		var dist_04_00: f32 = DistYCbCr(src[17], src[5]) + DistYCbCr(src[5], src[7]) + DistYCbCr(src[15], src[3]) + DistYCbCr(src[3], src[1]) + (4.0 * DistYCbCr(src[4], src[0]));
		var dist_05_03: f32 = DistYCbCr(src[18], src[4]) + DistYCbCr(src[4], src[14]) + DistYCbCr(src[6], src[0]) + DistYCbCr(src[0], src[2]) + (4.0 * DistYCbCr(src[5], src[3]));
		var dominantGradient: bool = (DOMINANT_DIRECTION_THRESHOLD * dist_05_03) < dist_04_00;
		blendResult.w = select( BLEND_NONE,  (select( BLEND_NORMAL,  BLEND_DOMINANT , (dominantGradient))) , ((dist_04_00 > dist_05_03) && (v[0] != v[5]) && (v[0] != v[3])));
	}

	// Pixel Tap Mapping: --|--|22|23|--
	//                    --|06|07|08|09
	//                    --|05|00|01|10
	//                    --|--|03|02|--
	//                    --|--|--|--|--
	// Corner (1, 0)
	if (!((v[7] == v[8] && v[0] == v[1]) || (v[7] == v[0] && v[8] == v[1]))) {
		var dist_00_08: f32 = DistYCbCr(src[5], src[7]) + DistYCbCr(src[7], src[23]) + DistYCbCr(src[3], src[1]) + DistYCbCr(src[1], src[9]) + (4.0 * DistYCbCr(src[0], src[8]));
		var dist_07_01: f32 = DistYCbCr(src[6], src[0]) + DistYCbCr(src[0], src[2]) + DistYCbCr(src[22], src[8]) + DistYCbCr(src[8], src[10]) + (4.0 * DistYCbCr(src[7], src[1]));
		var dominantGradient: bool = (DOMINANT_DIRECTION_THRESHOLD * dist_07_01) < dist_00_08;
		blendResult.y = select( BLEND_NONE,  (select( BLEND_NORMAL,  BLEND_DOMINANT , (dominantGradient))) , ((dist_00_08 > dist_07_01) && (v[0] != v[7]) && (v[0] != v[1])));
	}

	// Pixel Tap Mapping: --|21|22|--|--
	//                    19|06|07|08|--
	//                    18|05|00|01|--
	//                    --|04|03|--|--
	//                    --|--|--|--|--
	// Corner (0, 0)
	if (!((v[6] == v[7] && v[5] == v[0]) || (v[6] == v[5] && v[7] == v[0]))) {
		var dist_05_07: f32 = DistYCbCr(src[18], src[6]) + DistYCbCr(src[6], src[22]) + DistYCbCr(src[4], src[0]) + DistYCbCr(src[0], src[8]) + (4.0 * DistYCbCr(src[5], src[7]));
		var dist_06_00: f32 = DistYCbCr(src[19], src[5]) + DistYCbCr(src[5], src[3]) + DistYCbCr(src[21], src[7]) + DistYCbCr(src[7], src[1]) + (4.0 * DistYCbCr(src[6], src[0]));
		var dominantGradient: bool = (DOMINANT_DIRECTION_THRESHOLD * dist_05_07) < dist_06_00;
		blendResult.x = select( BLEND_NONE,  (select( BLEND_NORMAL,  BLEND_DOMINANT , (dominantGradient))) , ((dist_05_07 < dist_06_00) && (v[0] != v[5]) && (v[0] != v[7])));
	}

	var dst: array<vec3<f32>, 4>;
	dst[0] = src[0];
	dst[1] = src[0];
	dst[2] = src[0];
	dst[3] = src[0];

	// Scale pixel
	if (IsBlendingNeeded(blendResult)) {
		var k: array<vec3<f32>, 9>;
		var tempDst3: vec3<f32>;

		k[0] = src[0];
		k[1] = src[1];
		k[2] = src[2];
		k[3] = src[3];
		k[4] = src[4];
		k[5] = src[5];
		k[6] = src[6];
		k[7] = src[7];
		k[8] = src[8];
		ScalePixel(blendResult.xyzw, k, &dst);

		k[1] = src[7];
		k[2] = src[8];
		k[3] = src[1];
		k[4] = src[2];
		k[5] = src[3];
		k[6] = src[4];
		k[7] = src[5];
		k[8] = src[6];
		tempDst3 = dst[3];
		dst[3] = dst[2];
		dst[2] = dst[1];
		dst[1] = dst[0];
		dst[0] = tempDst3;
		ScalePixel(blendResult.wxyz, k, &dst);

		k[1] = src[5];
		k[2] = src[6];
		k[3] = src[7];
		k[4] = src[8];
		k[5] = src[1];
		k[6] = src[2];
		k[7] = src[3];
		k[8] = src[4];
		tempDst3 = dst[3];
		dst[3] = dst[2];
		dst[2] = dst[1];
		dst[1] = dst[0];
		dst[0] = tempDst3;
		ScalePixel(blendResult.zwxy, k, &dst);

		k[1] = src[3];
		k[2] = src[4];
		k[3] = src[5];
		k[4] = src[6];
		k[5] = src[7];
		k[6] = src[8];
		k[7] = src[1];
		k[8] = src[2];
		tempDst3 = dst[3];
		dst[3] = dst[2];
		dst[2] = dst[1];
		dst[1] = dst[0];
		dst[0] = tempDst3;
		ScalePixel(blendResult.yzwx, k, &dst);

		// Rotate the destination pixels back to 0 degrees.
		tempDst3 = dst[3];
		dst[3] = dst[2];
		dst[2] = dst[1];
		dst[1] = dst[0];
		dst[0] = tempDst3;
	}

	textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(dst[0], 1.0));

	gxy.x += 1u;
	textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(dst[1], 1.0));

	gxy.y += 1u;
	textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(dst[2], 1.0));

	gxy.x -= 1u;
	textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(dst[3], 1.0));
}
