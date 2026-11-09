// xBRZ 4x — Magpie v4 → WGFX WGSL port
//
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/xBRZ/xBRZ_4x.hlsl
//   which is itself a port of the libretro xBRZ shader:
//     https://github.com/libretro/common-shaders/blob/master/xbrz/shaders/4xbrz.cg
//
// Original license / author:
//   4xBRZ shader — Copyright (C) 2014-2016 DeSmuME team.
//     (GNU General Public License v3, as distributed with libretro/common-shaders)
//   xBR-vertex code and texel mapping — Copyright (C) 2011/2016 Hyllian (MIT).
//   Magpie HLSL port — Copyright (c) Blinue and Magpie contributors.
//   This WGFX WGSL port changes only the pass structure and the sampling idioms;
//   every xBRZ constant and every line of the algorithm is copied verbatim.
//
// Porting notes:
//   1. Magpie's `void Pass1(uint2 blockStart, uint3 threadId)` is a CS-style pass.
//      It maps onto a WGFX compute shader: `blockStart` = workgroup_id.xy * BLOCK_SIZE
//      and `threadId.x` = local_invocation_id.x. Magpie's `//! BLOCK_SIZE 32`
//      + `//! NUM_THREADS 64` means "one 32x32 output-pixel tile per
//      group, 64 threads"; the 8x8 thread grid (Rmp8x8) times the 4x scale factor
//      covers the whole tile.
//   2. The output texture is declared `//! WIDTH INPUT_WIDTH * 4` /
//      `//! HEIGHT INPUT_HEIGHT * 4` so that the host allocates the 4x
//      scaled target; the pass derives the input coordinate with `gxy >> 3`
//      (or `gxy / 4` for the odd variants) exactly like the HLSL.
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
//! SORT_NAME xBRZ_4x

//! TEXTURE
Texture2D INPUT;

//! TEXTURE
//! WIDTH INPUT_WIDTH * 4
//! HEIGHT INPUT_HEIGHT * 4
Texture2D OUTPUT;

//! SAMPLER
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! DESC "xBRZ 4x — high quality pixel art upscaler"
//! IN INPUT
//! OUT OUTPUT
//! BLOCK_SIZE 32
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

//---------------------------------------
// Input Pixel Mapping:  --|21|22|23|--
//                       19|06|07|08|09
//                       18|05|00|01|10
//                       17|04|03|02|11
//                       --|15|14|13|--
//
// Output Pixel Mapping:  06|07|08|09
//                        05|00|01|10
//                        04|03|02|11
//                        15|14|13|12
    var<private> destIdx = array<array<u32, 4>, 4>(
        array<u32, 4>(6, 7, 8, 9),
        array<u32, 4>(5, 0, 1, 10),
        array<u32, 4>(4, 3, 2, 11),
        array<u32, 4>(15, 14, 13, 12),
    );

fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let blockStart: vec2<u32> = workgroup_id.xy * vec2<u32>(32u, 32u);
	var gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(2u, 2u)) + blockStart;

	let outputSize: vec2<u32> = scene.outputSize;
	if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
		return;
	}

	let inputPt: vec2<f32> = scene.inputPt;
	let pos: vec2<f32> = (vec2<f32>(gxy >> vec2<u32>(2u, 2u)) + vec2<f32>(0.5)) * inputPt;

	//    A1 B1 C1
	// A0 A  B  C  C4
	// D0 D  E  F  F4
	// G0 G  H  I  I4
	//    G5 H5 I5

	var t1: vec4<f32> = pos.xxxy + vec4<f32>(-inputPt.x, 0, inputPt.x, -2.0 * inputPt.y); // A1 B1 C1
	var t2: vec4<f32> = pos.xxxy + vec4<f32>(-inputPt.x, 0, inputPt.x, -inputPt.y);		// A  B  C
	var t3: vec4<f32> = pos.xxxy + vec4<f32>(-inputPt.x, 0, inputPt.x, 0);				// D  E  F
	var t4: vec4<f32> = pos.xxxy + vec4<f32>(-inputPt.x, 0, inputPt.x, inputPt.y);		// G  H  I
	var t5: vec4<f32> = pos.xxxy + vec4<f32>(-inputPt.x, 0, inputPt.x, 2.0 * inputPt.y);	// G5 H5 I5
	var t6: vec4<f32> = pos.xyyy + vec4<f32>(-2.0 * inputPt.x, -inputPt.y, 0, inputPt.y);	// A0 D0 G0
	var t7: vec4<f32> = pos.xyyy + vec4<f32>(2.0 * inputPt.x, -inputPt.y, 0, inputPt.y);	// C4 F4 I4

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

	//---------------------------------------
	// Input Pixel Mapping:  20|21|22|23|24
	//                       19|06|07|08|09
	//                       18|05|00|01|10
	//                       17|04|03|02|11
	//                       16|15|14|13|12

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

	var dst: array<vec3<f32>, 16>;
	dst[0] = src[0];
	dst[1] = src[0];
	dst[2] = src[0];
	dst[3] = src[0];
	dst[4] = src[0];
	dst[5] = src[0];
	dst[6] = src[0];
	dst[7] = src[0];
	dst[8] = src[0];
	dst[9] = src[0];
	dst[10] = src[0];
	dst[11] = src[0];
	dst[12] = src[0];
	dst[13] = src[0];
	dst[14] = src[0];
	dst[15] = src[0];

	// Scale pixel
	if (IsBlendingNeeded(blendResult)) {
		var dist_01_04: f32 = DistYCbCr(src[1], src[4]);
		var dist_03_08: f32 = DistYCbCr(src[3], src[8]);
		var haveShallowLine: bool = (STEEP_DIRECTION_THRESHOLD * dist_01_04 <= dist_03_08) && (v[0] != v[4]) && (v[5] != v[4]);
		var haveSteepLine: bool = (STEEP_DIRECTION_THRESHOLD * dist_03_08 <= dist_01_04) && (v[0] != v[8]) && (v[7] != v[8]);
		var needBlend: bool = (blendResult[2] != BLEND_NONE);
		var doLineBlend: bool = (blendResult[2] >= BLEND_DOMINANT ||
			!((blendResult[1] != BLEND_NONE && !IsPixEqual(src[0], src[4])) ||
				(blendResult[3] != BLEND_NONE && !IsPixEqual(src[0], src[8])) ||
				(IsPixEqual(src[4], src[3]) && IsPixEqual(src[3], src[2]) && IsPixEqual(src[2], src[1]) && IsPixEqual(src[1], src[8]) && !IsPixEqual(src[0], src[2]))));

		var blendPix: vec3<f32> = select( src[3],  src[1] , (DistYCbCr(src[0], src[1]) <= DistYCbCr(src[0], src[3])));
		dst[2] = mix(dst[2], blendPix, select( 0.00,  (select( (select( 0.00,  0.25 , (haveSteepLine))),  (select( 0.25,  1.0 / 3.0 , (haveSteepLine))) , (haveShallowLine))) , (needBlend && doLineBlend)));
		dst[9] = mix(dst[9], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveSteepLine)));
		dst[10] = mix(dst[10], blendPix, select( 0.00,  0.75 , (needBlend && doLineBlend && haveSteepLine)));
		dst[11] = mix(dst[11], blendPix, select( 0.00,  (select( 0.08677704501,  (select( (select( 0.50,  0.75 , (haveShallowLine))),  1.00 , (haveSteepLine))) , (doLineBlend))) , (needBlend)));
		dst[12] = mix(dst[12], blendPix, select( 0.00,  (select( 0.6848532563,  1.00 , (doLineBlend))) , (needBlend)));
		dst[13] = mix(dst[13], blendPix, select( 0.00,  (select( 0.08677704501,  (select( (select( 0.50,  0.75 , (haveSteepLine))),  1.00 , (haveShallowLine))) , (doLineBlend))) , (needBlend)));
		dst[14] = mix(dst[14], blendPix, select( 0.00,  0.75 , (needBlend && doLineBlend && haveShallowLine)));
		dst[15] = mix(dst[15], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveShallowLine)));

		dist_01_04 = DistYCbCr(src[7], src[2]);
		dist_03_08 = DistYCbCr(src[1], src[6]);
		haveShallowLine = (STEEP_DIRECTION_THRESHOLD * dist_01_04 <= dist_03_08) && (v[0] != v[2]) && (v[3] != v[2]);
		haveSteepLine = (STEEP_DIRECTION_THRESHOLD * dist_03_08 <= dist_01_04) && (v[0] != v[6]) && (v[5] != v[6]);
		needBlend = (blendResult[1] != BLEND_NONE);
		doLineBlend = (blendResult[1] >= BLEND_DOMINANT ||
			!((blendResult[0] != BLEND_NONE && !IsPixEqual(src[0], src[2])) ||
				(blendResult[2] != BLEND_NONE && !IsPixEqual(src[0], src[6])) ||
				(IsPixEqual(src[2], src[1]) && IsPixEqual(src[1], src[8]) && IsPixEqual(src[8], src[7]) && IsPixEqual(src[7], src[6]) && !IsPixEqual(src[0], src[8]))));

		blendPix = select( src[1],  src[7] , (DistYCbCr(src[0], src[7]) <= DistYCbCr(src[0], src[1])));
		dst[1] = mix(dst[1], blendPix, select( 0.00,  (select( (select( 0.00,  0.25 , (haveSteepLine))),  (select( 0.25,  1.0 / 3.0 , (haveSteepLine))) , (haveShallowLine))) , (needBlend && doLineBlend)));
		dst[6] = mix(dst[6], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveSteepLine)));
		dst[7] = mix(dst[7], blendPix, select( 0.00,  0.75 , (needBlend && doLineBlend && haveSteepLine)));
		dst[8] = mix(dst[8], blendPix, select( 0.00,  (select( 0.08677704501,  (select( (select( 0.50,  0.75 , (haveShallowLine))),  1.00 , (haveSteepLine))) , (doLineBlend))) , (needBlend)));
		dst[9] = mix(dst[9], blendPix, select( 0.00,  (select( 0.6848532563,  1.00 , (doLineBlend))) , (needBlend)));
		dst[10] = mix(dst[10], blendPix, select( 0.00,  (select( 0.08677704501,  (select( (select( 0.50,  0.75 , (haveSteepLine))),  1.00 , (haveShallowLine))) , (doLineBlend))) , (needBlend)));
		dst[11] = mix(dst[11], blendPix, select( 0.00,  0.75 , (needBlend && doLineBlend && haveShallowLine)));
		dst[12] = mix(dst[12], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveShallowLine)));

		dist_01_04 = DistYCbCr(src[5], src[8]);
		dist_03_08 = DistYCbCr(src[7], src[4]);
		haveShallowLine = (STEEP_DIRECTION_THRESHOLD * dist_01_04 <= dist_03_08) && (v[0] != v[8]) && (v[1] != v[8]);
		haveSteepLine = (STEEP_DIRECTION_THRESHOLD * dist_03_08 <= dist_01_04) && (v[0] != v[4]) && (v[3] != v[4]);
		needBlend = (blendResult[0] != BLEND_NONE);
		doLineBlend = (blendResult[0] >= BLEND_DOMINANT ||
			!((blendResult[3] != BLEND_NONE && !IsPixEqual(src[0], src[8])) ||
				(blendResult[1] != BLEND_NONE && !IsPixEqual(src[0], src[4])) ||
				(IsPixEqual(src[8], src[7]) && IsPixEqual(src[7], src[6]) && IsPixEqual(src[6], src[5]) && IsPixEqual(src[5], src[4]) && !IsPixEqual(src[0], src[6]))));

		blendPix = select( src[7],  src[5] , (DistYCbCr(src[0], src[5]) <= DistYCbCr(src[0], src[7])));
		dst[0] = mix(dst[0], blendPix, select( 0.00,  (select( (select( 0.00,  0.25 , (haveSteepLine))),  (select( 0.25,  1.0 / 3.0 , (haveSteepLine))) , (haveShallowLine))) , (needBlend && doLineBlend)));
		dst[15] = mix(dst[15], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveSteepLine)));
		dst[4] = mix(dst[4], blendPix, select( 0.00,  0.75 , (needBlend && doLineBlend && haveSteepLine)));
		dst[5] = mix(dst[5], blendPix, select( 0.00,  (select( 0.08677704501,  (select( (select( 0.50,  0.75 , (haveShallowLine))),  1.00 , (haveSteepLine))) , (doLineBlend))) , (needBlend)));
		dst[6] = mix(dst[6], blendPix, select( 0.00,  (select( 0.6848532563,  1.00 , (doLineBlend))) , (needBlend)));
		dst[7] = mix(dst[7], blendPix, select( 0.00,  (select( 0.08677704501,  (select( (select( 0.50,  0.75 , (haveSteepLine))),  1.00 , (haveShallowLine))) , (doLineBlend))) , (needBlend)));
		dst[8] = mix(dst[8], blendPix, select( 0.00,  0.75 , (needBlend && doLineBlend && haveShallowLine)));
		dst[9] = mix(dst[9], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveShallowLine)));

		dist_01_04 = DistYCbCr(src[3], src[6]);
		dist_03_08 = DistYCbCr(src[5], src[2]);
		haveShallowLine = (STEEP_DIRECTION_THRESHOLD * dist_01_04 <= dist_03_08) && (v[0] != v[6]) && (v[7] != v[6]);
		haveSteepLine = (STEEP_DIRECTION_THRESHOLD * dist_03_08 <= dist_01_04) && (v[0] != v[2]) && (v[1] != v[2]);
		needBlend = (blendResult[3] != BLEND_NONE);
		doLineBlend = (blendResult[3] >= BLEND_DOMINANT ||
			!((blendResult[2] != BLEND_NONE && !IsPixEqual(src[0], src[6])) ||
				(blendResult[0] != BLEND_NONE && !IsPixEqual(src[0], src[2])) ||
				(IsPixEqual(src[6], src[5]) && IsPixEqual(src[5], src[4]) && IsPixEqual(src[4], src[3]) && IsPixEqual(src[3], src[2]) && !IsPixEqual(src[0], src[4]))));

		blendPix = select( src[5],  src[3] , (DistYCbCr(src[0], src[3]) <= DistYCbCr(src[0], src[5])));
		dst[3] = mix(dst[3], blendPix, select( 0.00,  (select( (select( 0.00,  0.25 , (haveSteepLine))),  (select( 0.25,  1.0 / 3.0 , (haveSteepLine))) , (haveShallowLine))) , (needBlend && doLineBlend)));
		dst[12] = mix(dst[12], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveSteepLine)));
		dst[13] = mix(dst[13], blendPix, select( 0.00,  0.75 , (needBlend && doLineBlend && haveSteepLine)));
		dst[14] = mix(dst[14], blendPix, select( 0.00,  (select( 0.08677704501,  (select( (select( 0.50,  0.75 , (haveShallowLine))),  1.00 , (haveSteepLine))) , (doLineBlend))) , (needBlend)));
		dst[15] = mix(dst[15], blendPix, select( 0.00,  (select( 0.6848532563,  1.00 , (doLineBlend))) , (needBlend)));
		dst[4] = mix(dst[4], blendPix, select( 0.00,  (select( 0.08677704501,  (select( (select( 0.50,  0.75 , (haveSteepLine))),  1.00 , (haveShallowLine))) , (doLineBlend))) , (needBlend)));
		dst[5] = mix(dst[5], blendPix, select( 0.00,  0.75 , (needBlend && doLineBlend && haveShallowLine)));
		dst[6] = mix(dst[6], blendPix, select( 0.00,  0.25 , (needBlend && doLineBlend && haveShallowLine)));
	}

	for (var i: u32 = 0u; i < 4u; i += 1u) {

		for (var j: u32 = 0u; j < 4u; j += 1u) {
			textureStore(OUTPUT, vec2<i32>(gxy + vec2<u32>(i, j)), vec4<f32>(dst[destIdx[j][i]], 1.0));
		}
	}
}
