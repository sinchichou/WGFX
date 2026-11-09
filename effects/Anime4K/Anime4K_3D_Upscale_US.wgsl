// Anime4K_3DGraphics_Upscale_x2_US — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/bloc97/Anime4K/blob/master/glsl/Upscale/Anime4K_3DGraphics_Upscale_x2_US.glsl
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
//   - OUTPUT is declared `INPUT_WIDTH * 2` x `INPUT_HEIGHT * 2` (2x upscale), matching the
//   -    HLSL; the intermediate feature maps stay at input resolution.
//   - Pass 3 is the depth-to-space step: every thread writes the 2x2 output quad
//   -    (result.x/.y/.w/.z) and adds the bilinearly upscaled INPUT (LINEAR sampler `sam1`),
//   -    exactly as the HLSL does.


//!VERSION 4
//!SORT_NAME Anime4K_3D_Upscale_US
//!USE MulAdd
//!CAPABILITY FP16

//!TEXTURE
Texture2D INPUT;

//!TEXTURE
//!WIDTH INPUT_WIDTH * 2
//!HEIGHT INPUT_HEIGHT * 2
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
//!DESC Conv-4x3x3x3
//!IN INPUT
//!OUT tex1_storaged
//!BLOCK_SIZE 16
//!NUM_THREADS 64,1,1

fn Pass1(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
	var gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * vec2<u32>(16u);
	var inputSize: vec2<u32> = GetInputSize();
	if (gxy.x >= inputSize.x || gxy.y >= inputSize.y) {
		return;
	}

	var inputPt: vec2<f32> = GetInputPt();
	var i: u32 = 0u;
	var j: u32 = 0u;

	var src: array<array<vec3<f32>, 4>, 4>;
	for (var i: u32 = 0; i <= 2; i += 2u) {
		for (var j: u32 = 0; j <= 2; j += 2u) {
			var tpos: vec2<f32> = (vec2<f32>(gxy + vec2<u32>(i, j))) * inputPt;
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

	for (var i: u32 = 1; i <= 2; i += 1u) {
		for (var j: u32 = 1; j <= 2; j += 1u) {
			var destPos: vec2<u32> = gxy + vec2<u32>(i - 1, j - 1);

			if (i != 1 || j != 1) {
				if (destPos.x >= inputSize.x || destPos.y >= inputSize.y) {
					continue;
				}
			}

			var result: vec4<f32> = vec4<f32>(0.0033260486, 0.0027453774, 0.002706769, -0.0022023292);
			result = mat3x4<f32>(0.020295616, 0.26657984, -0.28531083, 0.041052748, 0.11729335, 0.4042445, -0.5860672, 0.098661005, 0.07756054, 0.32493782, -0.38677853, 0.029859459) * src[i - 1][j - 1] + result;
			result = mat3x4<f32>(-0.28257942, 0.46859467, 0.07703674, -0.05280326, -0.46674755, 0.6592401, 0.28513643, -0.059969574, -0.37685588, 0.54568315, 0.1845521, -0.002367634) * src[i - 1][j] + result;
			result = mat3x4<f32>(-0.14898707, -0.088150576, -0.09723321, 0.48359355, -0.27636692, -0.09316871, -0.15453896, 0.68637407, -0.19012338, -0.13310172, -0.13367014, 0.61284137) * src[i - 1][j + 1] + result;
			result = mat3x4<f32>(0.36225644, 0.14732021, -0.17000826, -0.039856624, 0.50175637, 0.11733988, -0.080232985, -0.059622094, 0.3199943, 0.14199091, -0.08377198, -0.058804136) * src[i][j - 1] + result;
			result = mat3x4<f32>(0.42444956, -0.7508643, 0.77660984, -0.5920436, 0.6205736, -1.1258985, 0.91925675, -0.854362, 0.5576168, -0.84702957, 0.6147012, -0.7004013) * src[i][j] + result;
			result = mat3x4<f32>(0.047896724, -0.018435799, -0.07901341, 0.14234425, 0.12161554, -0.03132901, -0.049536135, 0.16390118, 0.054683182, -0.054823015, -0.019946543, 0.058345634) * src[i][j + 1] + result;
			result = mat3x4<f32>(-0.60845935, -0.0062592463, 0.17769416, 0.00643066, -0.8315386, 0.010405371, 0.227901, 0.054893702, -0.63039875, -0.02128892, 0.20843753, 0.054094747) * src[i + 1][j - 1] + result;
			result = mat3x4<f32>(0.24499738, -0.11924165, -0.16513842, -0.012650152, 0.21676576, -0.14322694, -0.08526706, -0.0122355, 0.26050088, -0.115822464, -0.11147918, -0.018665116) * src[i + 1][j] + result;
			result = mat3x4<f32>(-0.06054074, 0.094478205, -0.23038268, 0.008782032, -0.028404612, 0.22320247, -0.4821615, -0.0055871494, -0.047503285, 0.15098651, -0.26161456, 0.029869653) * src[i + 1][j + 1] + result;

			textureStore(tex1_storaged, vec2<i32>(destPos), result);
		}
	}
}

//!PASS 2
//!DESC Conv-4x3x3x8
//!IN tex1_sampled
//!OUT tex2_storaged
//!BLOCK_SIZE 16
//!NUM_THREADS 64,1,1

fn Pass2(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
	var gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * vec2<u32>(16u);
	var inputSize: vec2<u32> = GetInputSize();
	if (gxy.x >= inputSize.x || gxy.y >= inputSize.y) {
		return;
	}

	var inputPt: vec2<f32> = GetInputPt();
	var i: u32 = 0u;
	var j: u32 = 0u;

	var src: array<array<vec4<f32>, 4>, 4>;
	for (var i: u32 = 0; i < 3; i += 2u) {
		for (var j: u32 = 0; j < 3; j += 2u) {
			var tpos: vec2<f32> = (vec2<f32>(gxy + vec2<u32>(i, j))) * inputPt;
			let sr: vec4<f32> = textureGather(0, tex1_sampled, sam, tpos);
			let sg: vec4<f32> = textureGather(1, tex1_sampled, sam, tpos);
			let sb: vec4<f32> = textureGather(2, tex1_sampled, sam, tpos);
			let sa: vec4<f32> = textureGather(3, tex1_sampled, sam, tpos);

			// w z
			// x y
			src[i][j] = vec4<f32>(sr.w, sg.w, sb.w, sa.w);
			src[i][j + 1] = vec4<f32>(sr.x, sg.x, sb.x, sa.x);
			src[i + 1][j] = vec4<f32>(sr.z, sg.z, sb.z, sa.z);
			src[i + 1][j + 1] = vec4<f32>(sr.y, sg.y, sb.y, sa.y);
		}
	}

	for (var i: u32 = 1; i <= 2; i += 1u) {
		for (var j: u32 = 1; j <= 2; j += 1u) {
			var destPos: vec2<u32> = gxy + vec2<u32>(i - 1, j - 1);

			if (i != 1 || j != 1) {
				if (destPos.x >= inputSize.x || destPos.y >= inputSize.y) {
					continue;
				}
			}

			var result: vec4<f32> = vec4<f32>(-0.0016514715, -0.0038319482, -0.0073834592, -0.0059920377);
			result = mat4x4<f32>(-0.26565167, 0.03297807, 0.08926044, 0.029984077, -0.20809028, 0.00351991, 0.15452495, 0.09963063, -0.13773972, 0.01029457, 0.045066427, 0.1247363, -0.12714206, 0.010659472, -0.06463583, -0.0154152885) * max(src[i - 1][j - 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(0.0030874603, -0.04652791, -0.23016497, -0.46789864, 0.16454394, -0.12389561, -0.026154622, -0.008619661, -0.10648254, -0.10581639, 0.038306713, -0.038339432, 0.09716571, -0.037978917, -0.075691506, -0.0712344) * max(src[i - 1][j], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.22059305, 0.18160829, 0.027175324, -0.09482166, -0.07888123, 0.12431242, 0.02432025, -0.121108614, -0.0720631, -0.0138989305, -0.02100737, -0.027440425, -0.06610268, 0.115165174, 0.02048676, -0.0409675) * max(src[i - 1][j + 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.043652806, 0.0134993605, 0.0030924282, -0.03314738, 0.27716368, 0.05542831, -0.28217152, -0.017624432, -0.11218036, 0.009025405, -0.13453043, -0.05897923, 0.32378325, -0.0070016393, -0.04415195, 0.10701437) * max(src[i][j - 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.23568924, 0.028462708, 0.11686719, 0.025355088, -0.088295944, 0.033592343, 0.22703041, 0.45683298, -0.0860751, -0.17487867, -0.048110463, 0.0030137985, -0.031168368, 0.075628504, 0.24791549, 0.13066974) * max(src[i][j], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(0.04647855, -0.42392284, -0.19600202, -0.07964123, 0.059082046, -0.16060755, -0.12607971, 0.011384242, 0.021775302, -0.10103477, -0.09646511, -0.0333935, 0.017039102, -0.14098461, -0.0981411, -0.0023695084) * max(src[i][j + 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.016526148, -0.0006880693, 0.030872462, -0.018204063, -0.025757499, -0.01037187, 0.21834393, 0.09965417, -0.057800297, -0.029975737, -0.00684766, 0.05800057, 0.1683585, -0.057496388, 0.7087625, 0.2665085) * max(src[i + 1][j - 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(0.059480842, -0.05298332, -0.06537449, -0.06439374, 0.09735456, -0.19591855, -0.42987427, -0.012228564, 0.03898561, -0.22284365, -0.24668825, -0.06651606, -0.087776355, 0.3200755, -0.050433923, -0.100411855) * max(src[i + 1][j], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.053572096, 0.16021295, 0.071161, -0.06335539, 0.015909042, 0.20072727, 0.08304887, 0.05335472, -0.03467755, -0.02854195, 0.044690188, 0.0045897495, -0.02924389, 0.16038977, 0.06578487, 0.027916174) * max(src[i + 1][j + 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(0.11840388, -0.011793132, -0.004042257, -0.116042845, 0.17388786, -0.031351876, -0.1152781, -0.13333467, 0.2354002, -0.045496088, -0.16311322, -0.06960798, 0.11317423, 0.005508318, 0.15634489, 0.061211936) * max(-src[i - 1][j - 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(0.009622364, 0.03682801, 0.20317583, 0.41673073, 0.00662247, 0.11070694, -0.020392824, 0.13875432, 0.036129113, 0.073727116, -0.114370294, 0.042431634, -0.0446233, 0.0734258, 0.09814352, 0.09335129) * max(-src[i - 1][j], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(0.057668976, -0.21368256, -0.029979514, 0.039450984, 0.16685131, -0.15085255, -0.042412136, 0.15757313, 0.110782996, -0.07277766, -0.035268076, 0.06094599, 0.088834405, -0.07145625, 0.0009840949, 0.04932651) * max(-src[i - 1][j + 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.00010679438, 0.0036139274, -0.001123274, -0.10001404, -0.035231482, -0.008196208, 0.24442215, 0.13503289, -0.082656465, -0.061868448, -0.34716147, -0.22398905, -0.00626325, 0.063109376, 0.10340223, 0.030913284) * max(-src[i][j - 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.11526892, -0.01739956, -0.1781953, -0.27035302, 0.13689406, -0.04320452, -0.19705579, -0.16808036, -0.20343314, -0.13549826, -0.08057608, -0.16109829, 0.16905797, 0.00077912246, -0.14143418, 2.5004636e-05) * max(-src[i][j], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.110721245, 0.33579257, 0.14410733, -0.06675498, 0.002022071, 0.18419205, 0.09936456, 0.07311257, 0.0064584822, -0.031634383, 0.095907465, 0.031282485, 0.0051860735, 0.25176755, 0.089756206, 0.052209254) * max(-src[i][j + 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.06346731, 0.009824449, -0.02733463, -0.06341941, 0.17385058, 0.0067438814, -0.15083039, -0.09102664, 0.065680414, -0.06668183, -0.18805085, -0.02785943, 0.06386912, 0.09340653, 0.0032584765, 0.060902752) * max(-src[i + 1][j - 1], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.061221372, -0.0020573896, 0.063335806, 0.046898697, -0.030530764, 0.27301106, 0.23682626, 0.14863367, -0.034555666, 0.024279105, -0.176423, 0.018163674, -0.00015057978, 0.1575829, 0.25034854, 0.054726984) * max(-src[i + 1][j], vec4<f32>(0.0)) + result;
			result = mat4x4<f32>(-0.03807849, -0.0440673, 0.038429137, -0.029738983, 0.035449427, -0.44138825, -0.16462126, -0.015045709, 0.010627751, -0.12178318, -0.07076098, -0.059846148, 0.0009871369, -0.07265447, -0.01998304, -0.018729536) * max(-src[i + 1][j + 1], vec4<f32>(0.0)) + result;

			textureStore(tex2_storaged, vec2<i32>(destPos), result);
		}
	}
}

//!PASS 3
//!DESC Conv-4x3x3x4, Depth-to-Space
//!IN INPUT, tex2_sampled
//!OUT OUTPUT
//!BLOCK_SIZE 16
//!NUM_THREADS 64,1,1

fn Pass3(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
	var gxy: vec2<u32> = (Rmp8x8(local_id.x) << vec2<u32>(1u)) + workgroup_id.xy * vec2<u32>(16u);

	let outputSize: vec2<u32> = GetOutputSize();
	if (gxy.x >= outputSize.x || gxy.y >= outputSize.y) {
		return;
	}

	var inputPt: vec2<f32> = GetInputPt();
	var outputPt: vec2<f32> = GetOutputPt();

	var pos: vec2<f32> = (vec2<f32>(gxy >> vec2<u32>(1u)) + vec2<f32>(0.5, 0.5)) * inputPt;

	// [ a, d, g ]
	// [ b, e, h ]
	// [ c, f, i ]
	var tpos: vec2<f32> = pos - 0.5 * inputPt;
	let sr: vec4<f32> = textureGather(0, tex2_sampled, sam, tpos);
	let sg: vec4<f32> = textureGather(1, tex2_sampled, sam, tpos);
	let sb: vec4<f32> = textureGather(2, tex2_sampled, sam, tpos);
	let sa: vec4<f32> = textureGather(3, tex2_sampled, sam, tpos);

	var a: vec4<f32> = vec4<f32>(sr.w, sg.w, sb.w, sa.w);
	var b: vec4<f32> = vec4<f32>(sr.x, sg.x, sb.x, sa.x);
	var c: vec4<f32> = textureSampleLevel(tex2_sampled, sam, pos + vec2<f32>(-inputPt.x, inputPt.y), 0.0);
	var d: vec4<f32> = vec4<f32>(sr.z, sg.z, sb.z, sa.z);
	var e: vec4<f32> = vec4<f32>(sr.y, sg.y, sb.y, sa.y);
	var f: vec4<f32> = textureSampleLevel(tex2_sampled, sam, pos + vec2<f32>(0.0, inputPt.y), 0.0);
	var g: vec4<f32> = textureSampleLevel(tex2_sampled, sam, pos + vec2<f32>(inputPt.x, -inputPt.y), 0.0);
	var h: vec4<f32> = textureSampleLevel(tex2_sampled, sam, pos + vec2<f32>(inputPt.x, 0.0), 0.0);
	var i: vec4<f32> = textureSampleLevel(tex2_sampled, sam, pos + vec2<f32>(inputPt.x, inputPt.y), 0.0);

	var result: vec4<f32> = vec4<f32>(-0.00016697648, -0.00015957489, 0.00017437353, -0.00019393339);
	result = mat4x4<f32>(0.010651157, 0.0018537974, 0.0051580826, 0.0020901994, -0.41468114, -0.08517094, -0.04803197, 0.008413933, 0.014026587, 0.012820705, 0.01553548, 0.01283508, -0.040812977, -0.015933262, -0.01949977, -0.011502392) * max(a, vec4<f32>(0.0)) + result;
	result = mat4x4<f32>(0.2278103, 0.06374709, 0.062103115, 0.02010158, -0.01932597, 0.01159983, -0.35690293, -0.062822886, 0.29807624, 0.036038283, 0.028565003, -0.025969796, -0.24225195, -0.06635989, -0.06890857, -0.010140013) * max(b, vec4<f32>(0.0)) + result;
	result = mat4x4<f32>(-0.009164529, -0.0036277112, 0.19577485, 0.05511193, 0.025875263, 0.020616682, 0.02919653, 0.030669902, 0.00751339, -0.00551052, 0.2690018, 0.035395138, -0.013392302, -0.03666098, -0.23751089, -0.11222924) * max(c, vec4<f32>(0.0)) + result;
	result = mat4x4<f32>(0.017815048, 0.011143683, -0.002918766, -0.0042153858, -0.01929562, -0.3405707, 0.039267153, 0.0041966257, 0.0026771557, 0.009982735, 0.00087975257, 0.003984122, 0.0062404936, -0.04230462, 0.001981855, -0.0145921325) * max(d, vec4<f32>(0.0)) + result;
	result = mat4x4<f32>(-0.02615864, 0.2872578, 0.08103852, 0.11812216, -0.031988684, -0.08532163, -0.094653316, -0.42660478, 0.00077043264, 0.284553, -0.015517693, 0.036667597, 0.14234932, -0.20201443, -0.017204845, -0.111502305) * max(e, vec4<f32>(0.0)) + result;
	result = mat4x4<f32>(-0.11529456, -0.09024149, -0.2545015, 0.08581955, -0.020051582, -0.031601675, -0.031320736, -0.051691536, -0.019313173, 0.028963564, 0.017521648, 0.3159018, 0.179573, 0.19400181, 0.382411, 0.076367974) * max(f, vec4<f32>(0.0)) + result;
	result = mat4x4<f32>(-0.016741822, -0.001993879, -0.01517403, -0.010949569, 0.01859244, 0.042316843, 0.00025326485, -0.007079785, 0.001070783, 0.000911405, -0.0024269924, -0.0014501393, -0.01206317, 0.007300575, -0.0062582446, -0.0016001783) * max(g, vec4<f32>(0.0)) + result;
	result = mat4x4<f32>(-0.06300321, -0.20786348, 0.019603852, 0.020122321, 0.022107193, 0.036766138, 0.013046632, 0.02677947, -0.0006842017, -0.044024136, 0.0109439045, 0.0040129004, 0.009831765, 0.15807834, -0.05166107, -0.014462446) * max(h, vec4<f32>(0.0)) + result;
	result = mat4x4<f32>(-0.023205867, -0.049788095, -0.08251341, -0.26488927, 0.0029636251, 0.021618038, -0.0057556895, 0.018444102, 0.0064643323, -0.027886944, 0.010029941, -0.041902024, -0.021232832, -0.0072857663, 0.02056806, 0.18491453) * max(i, vec4<f32>(0.0)) + result;

	pos -= 0.5 * outputPt;
	textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(result.x + textureSampleLevel(INPUT, sam1, pos, 0.0).rgb, 1.0));

	gxy.x += 1u;
	pos.x += outputPt.x;
	textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(result.y + textureSampleLevel(INPUT, sam1, pos, 0.0).rgb, 1.0));

	gxy.y += 1u;
	pos.y += outputPt.y;
	textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(result.w + textureSampleLevel(INPUT, sam1, pos, 0.0).rgb, 1.0));

	gxy.x -= 1u;
	pos.x -= outputPt.x;
	textureStore(OUTPUT, vec2<i32>(gxy), vec4<f32>(result.z + textureSampleLevel(INPUT, sam1, pos, 0.0).rgb, 1.0));
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
