// SMAA_Ultra (Enhanced Subpixel Morphological Antialiasing) — Magpie v4 → WGFX WGSL port
//
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/SMAA/SMAA_Ultra.hlsl
//   which includes SMAA.hlsli (a Magpie adaptation of https://github.com/iryoku/smaa):
//     Copyright (C) 2013 Jorge Jimenez, Jose I. Echevarria, Belen Masia,
//     Fernando Navarro, Diego Gutierrez. (MIT)
//
// Preset: SMAA_PRESET_ULTRA  =>  THRESHOLD 0.05, MAX_SEARCH_STEPS 32,
//          MAX_SEARCH_STEPS_DIAG 16, CORNER_ROUNDING 25 (no *_DISABLE_* defines).
//
// Original pass structure (three chained pixel-shader passes):
//     INPUT -> [LumaEdgeDetection] -> edgesTex -> [BlendingWeightCalculation]
//           -> blendTex -> [NeighborhoodBlending] (with INPUT) -> OUTPUT
//
// Porting notes (語意偏離與處理方式):
//   1. `//!STYLE PS` passes become compute shaders, 1 invocation = 1 pixel
//      (`//! NUM_THREADS 8,8,1`). Magpie's PS `pos` is the normalised UV of the
//      pixel centre, so `pos = (vec2<f32>(gid.xy) + 0.5) * pt`; passes 1/2 write
//      INPUT-sized intermediates (pt = scene.inputPt) and pass 3 writes OUTPUT
//      (pt = scene.outputPt). SMAA is a 1:1 effect, so these coincide.
//   2. `SMAA_RT_METRICS = float4(GetInputPt(), GetInputSize())` is not a WGSL
//      module-scope constant because it reads the `scene` uniform (WGSL forbids a
//      module-scope `const` referencing a uniform). It is computed once per pass as
//      `rt = vec4<f32>(scene.inputPt, vec2<f32>(scene.inputSize))` and threaded
//      through the helpers as an extra `rt: vec4<f32>` parameter. Values unchanged.
//      Note `.zz` / `.ww` are used exactly as in the HLSL: the horizontal/vertical
//      distance pairs are both x- (resp. y-) distances, so they are both scaled by
//      the same dimension.
//   3. `SMAASampleLevelZeroOffset(tex, coord, int2(o))` needs a literal offset in
//      WGSL (textureSampleLevel's offset argument must be a const-expression), so
//      every call site inlines its own `vec2<i32>` literal instead of forwarding a
//      helper parameter. Same hardware texel offset as the HLSL.
//   4. `SMAAMovc(boolN cond, inout floatN variable, floatN value)` has no WGSL
//      inout form; it becomes `SMAAMovc2/4(cond, variable, value) -> floatN`,
//      returning `select(variable, value, cond)` — the same conditional move.
//   5. `SMAA_FLATTEN`/`SMAA_BRANCH` ([flatten]/[branch]) have no WGSL equivalent and
//      are dropped (WGSL `if` semantics do not change results).
//   6. `saturate`, `step`, `min`, `max`, `abs`, `floor` keep their names and
//      semantics. `mad(a,b,c)` → `a * b + c`; `lerp` → `mix`; `rcp` is not used.
//      `round` is the one builtin whose tie-breaking rule is not guaranteed to
//      match HLSL exactly: WGSL specifies roundTiesToEven. Every `round` in SMAA
//      either rounds values that can never land on an exact .5 (the diagonal
//      decode and `round(4.0 * e)`), or rounds a distance where an exact .5 tie is
//      a measure-zero event; the worst case is a 1-texel difference on such a tie.
//   7. `SMAADecodeDiagBilinearAccess` returns `round(...)`; the float2 overload
//      writes only `.r` before rounding, which is preserved. `c.yxwz = decode(c)`
//      (a swizzled assignment) is written out as an explicit component shuffle.
//   8. `SMAALumaEdgeDetectionPS` returns `float2(0,0)` instead of `discard`, as in
//      the Magpie source; the edge texture's unused channels are stored as 0.
//      R16G16_FLOAT / R8G8_UNORM edge+blend targets are stored as vec4 here because
//      WebGPU storage textures are always vec4 (see note 9).
//   9. FORMAT: `R16G16_FLOAT` (edgesTex) and `R16G16B16A16_FLOAT` (blendTex) are
//      declared exactly as the source. Dawn rejects `rg16float` storage access, and
//      WGFX substitutes the closest storage-capable format (`rgba16float`) for both
//      the WGSL declaration and the GPUTexture, which keeps the r/g channels' f16
//      precision identical. Low/Medium declare their 8-bit variants (see those files).
//  10. The blend-weight loop's searches sample `edgesTex` with the LINEAR sampler
//      (bilinear taps are part of the algorithm — @PSEUDO_GATHER4) and the edge
//      detection samples the colour input with the POINT sampler, exactly as the
//      source's SMAASampleLevelZero / SMAASamplePoint split.
//  11. The three passes are dispatched back-to-back in one command encoder by the
//      WGFX runtime; WebGPU inserts the storage-write -> sampled-read barriers.
//      The intermediate textures are not explicitly cleared — pass 1 and pass 2
//      overwrite every texel they cover (the dispatch grid covers the full texture).

//! VERSION 4
//! SORT_NAME SMAA_3

//! TEXTURE
Texture2D INPUT;

//! TEXTURE
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! TEXTURE
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
//! FORMAT R16G16_FLOAT
Texture2D edgesTex;

//! TEXTURE
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D blendTex;

//! TEXTURE
//! SOURCE AreaTex.dds
//! FORMAT R8G8B8A8_UNORM
Texture2D areaTex;

//! TEXTURE
//! SOURCE SearchTex.dds
//! FORMAT R8_UNORM
Texture2D searchTex;

//! SAMPLER
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState PointSampler;

//! SAMPLER
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState LinearSampler;

//! COMMON

// ---- SMAA preset ULTRA ----
const SMAA_THRESHOLD: f32 = 0.05;
const SMAA_MAX_SEARCH_STEPS: i32 = 32;
const SMAA_MAX_SEARCH_STEPS_DIAG: i32 = 16;
const SMAA_CORNER_ROUNDING: f32 = 25.0;

// ---- Non-configurable defines ----
const SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR: f32 = 2.0;
const SMAA_AREATEX_MAX_DISTANCE: f32 = 16.0;
const SMAA_AREATEX_MAX_DISTANCE_DIAG: f32 = 20.0;
const SMAA_AREATEX_PIXEL_SIZE: vec2<f32> = vec2<f32>(1.0 / 160.0, 1.0 / 560.0);
const SMAA_AREATEX_SUBTEX_SIZE: f32 = 1.0 / 7.0;
const SMAA_SEARCHTEX_SIZE: vec2<f32> = vec2<f32>(66.0, 33.0);
const SMAA_SEARCHTEX_PACKED_SIZE: vec2<f32> = vec2<f32>(64.0, 16.0);
const SMAA_CORNER_ROUNDING_NORM: f32 = SMAA_CORNER_ROUNDING / 100.0;

// float4(GetInputPt(), GetInputSize()), built component-wise: WGSL has no
// `vec4<f32>(vec2<f32>, vec2<u32>)` overload, and WGFX's u32->f32 constructor
// auto-wrap only rewrites single arguments, so the widening is made explicit here.
fn SMAART() -> vec4<f32> {
    var rt: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    rt.x = scene.inputPt.x;
    rt.y = scene.inputPt.y;
    rt.z = f32(scene.inputSize.x);
    rt.w = f32(scene.inputSize.y);
    return rt;
}

// ---- Porting-function equivalents ----
// #define SMAASampleLevelZero(tex, coord) tex.SampleLevel(SMAA_LINEAR_SAMPLER, coord, 0)
fn SMAASampleLevelZero(tex: texture_2d<f32>, coord: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(tex, LinearSampler, coord, 0.0);
}

// #define SMAASample(tex, coord) tex.SampleLevel(SMAA_LINEAR_SAMPLER, coord, 0)
fn SMAASample(tex: texture_2d<f32>, coord: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(tex, LinearSampler, coord, 0.0);
}

// #define SMAASamplePoint(tex, coord) tex.SampleLevel(SMAA_POINT_SAMPLER, coord, 0)
fn SMAASamplePoint(tex: texture_2d<f32>, coord: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(tex, PointSampler, coord, 0.0);
}

// ---- Misc functions ----

// void SMAAMovc(bool2 cond, inout float2 variable, float2 value)
fn SMAAMovc2(cond: vec2<bool>, variable: vec2<f32>, value: vec2<f32>) -> vec2<f32> {
    return select(variable, value, cond);
}

// void SMAAMovc(bool4 cond, inout float4 variable, float4 value)
fn SMAAMovc4(cond: vec4<bool>, variable: vec4<f32>, value: vec4<f32>) -> vec4<f32> {
    return select(variable, value, cond);
}

// ---- Edge detection ----

// float2 SMAALumaEdgeDetectionPS(float2 texcoord, Texture2D<float4> colorTex)
fn SMAALumaEdgeDetectionPS(texcoord: vec2<f32>, colorTex: texture_2d<f32>, rt: vec4<f32>) -> vec2<f32> {
    var offset: array<vec4<f32>, 3>;
    offset[0] = rt.xyxy * vec4<f32>(-1.0, 0.0, 0.0, -1.0) + texcoord.xyxy;
    offset[1] = rt.xyxy * vec4<f32>(1.0, 0.0, 0.0, 1.0) + texcoord.xyxy;
    offset[2] = rt.xyxy * vec4<f32>(-2.0, 0.0, 0.0, -2.0) + texcoord.xyxy;

    // Calculate the threshold:
    let threshold: vec2<f32> = vec2<f32>(SMAA_THRESHOLD, SMAA_THRESHOLD);

    // Calculate lumas:
    let weights: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
    let L: f32 = dot(SMAASamplePoint(colorTex, texcoord).rgb, weights);

    let Lleft: f32 = dot(SMAASamplePoint(colorTex, offset[0].xy).rgb, weights);
    let Ltop: f32 = dot(SMAASamplePoint(colorTex, offset[0].zw).rgb, weights);

    // We do the usual threshold:
    var delta: vec4<f32> = vec4<f32>(abs(L - Lleft), abs(L - Ltop), 0.0, 0.0);
    var edges: vec2<f32> = step(threshold, delta.xy);

    // Then discard if there is no edge:
    if (dot(edges, vec2<f32>(1.0, 1.0)) == 0.0) {
        return vec2<f32>(0.0, 0.0); // 不使用 discard
    } else {
        // Calculate right and bottom deltas:
        let Lright: f32 = dot(SMAASamplePoint(colorTex, offset[1].xy).rgb, weights);
        let Lbottom: f32 = dot(SMAASamplePoint(colorTex, offset[1].zw).rgb, weights);
        delta = vec4<f32>(delta.x, delta.y, abs(L - Lright), abs(L - Lbottom));

        // Calculate the maximum delta in the direct neighborhood:
        let maxDelta0: vec2<f32> = max(delta.xy, delta.zw);

        // Calculate left-left and top-top deltas:
        let Lleftleft: f32 = dot(SMAASamplePoint(colorTex, offset[2].xy).rgb, weights);
        let Ltoptop: f32 = dot(SMAASamplePoint(colorTex, offset[2].zw).rgb, weights);
        let delta2: vec2<f32> = abs(vec2<f32>(Lleft, Ltop) - vec2<f32>(Lleftleft, Ltoptop));

        // Calculate the final maximum delta:
        let maxDelta: vec2<f32> = max(maxDelta0, delta2);
        let finalDelta: f32 = max(maxDelta.x, maxDelta.y);

        // Local contrast adaptation:
        edges = edges * step(vec2<f32>(finalDelta, finalDelta), SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta.xy);

        return edges;
    }
}

// ---- Diagonal search functions ----

// float2 SMAADecodeDiagBilinearAccess(float2 e)
fn SMAADecodeDiagBilinearAccess2(e: vec2<f32>) -> vec2<f32> {
    // Bilinear access for fetching 'e' have a 0.25 offset, and we are
    // interested in the R and G edges
    return round(vec2<f32>(e.r * abs(5.0 * e.r - 5.0 * 0.75), e.g));
}

// float4 SMAADecodeDiagBilinearAccess(float4 e)
fn SMAADecodeDiagBilinearAccess4(e: vec4<f32>) -> vec4<f32> {
    let rb: vec2<f32> = e.rb * abs(5.0 * e.rb - 5.0 * 0.75);
    return round(vec4<f32>(rb.x, e.g, rb.y, e.a));
}

struct SMAADiagResult {
    zw: vec2<f32>,
    e: vec2<f32>,
};

// float2 SMAASearchDiag1(Texture2D<float2> edgesTex, float2 texcoord, float2 dir, out float2 e)
fn SMAASearchDiag1(edgesTex: texture_2d<f32>, texcoord: vec2<f32>, dir: vec2<f32>, rt: vec4<f32>) -> SMAADiagResult {
    var coord: vec4<f32> = vec4<f32>(texcoord, -1.0, 1.0);
    let t: vec3<f32> = vec3<f32>(rt.x, rt.y, 1.0);
    var e: vec2<f32> = vec2<f32>(0.0, 0.0); // `out` param; the loop always runs at least once
    while (coord.z < f32(SMAA_MAX_SEARCH_STEPS_DIAG - 1) && coord.w > 0.9) {
        coord = vec4<f32>(t * vec3<f32>(dir, 1.0) + coord.xyz, coord.w);
        e = SMAASampleLevelZero(edgesTex, coord.xy).xy;
        coord.w = dot(e, vec2<f32>(0.5, 0.5));
    }
    return SMAADiagResult(coord.zw, e);
}

// float2 SMAASearchDiag2(Texture2D<float2> edgesTex, float2 texcoord, float2 dir, out float2 e)
fn SMAASearchDiag2(edgesTex: texture_2d<f32>, texcoord: vec2<f32>, dir: vec2<f32>, rt: vec4<f32>) -> SMAADiagResult {
    var coord: vec4<f32> = vec4<f32>(texcoord, -1.0, 1.0);
    coord.x += 0.25 * rt.x; // See @SearchDiag2Optimization
    let t: vec3<f32> = vec3<f32>(rt.x, rt.y, 1.0);
    var e: vec2<f32> = vec2<f32>(0.0, 0.0);
    while (coord.z < f32(SMAA_MAX_SEARCH_STEPS_DIAG - 1) && coord.w > 0.9) {
        coord = vec4<f32>(t * vec3<f32>(dir, 1.0) + coord.xyz, coord.w);

        // @SearchDiag2Optimization
        // Fetch both edges at once using bilinear filtering:
        e = SMAASampleLevelZero(edgesTex, coord.xy).xy;
        e = SMAADecodeDiagBilinearAccess2(e);

        coord.w = dot(e, vec2<f32>(0.5, 0.5));
    }
    return SMAADiagResult(coord.zw, e);
}

// float2 SMAAAreaDiag(Texture2D<float4> areaTex, float2 dist, float2 e, float offset)
fn SMAAAreaDiag(areaTex: texture_2d<f32>, dist: vec2<f32>, e: vec2<f32>, offset: f32) -> vec2<f32> {
    var texcoord: vec2<f32> = vec2<f32>(SMAA_AREATEX_MAX_DISTANCE_DIAG, SMAA_AREATEX_MAX_DISTANCE_DIAG) * e + dist;

    // We do a scale and bias for mapping to texel space:
    texcoord = SMAA_AREATEX_PIXEL_SIZE * texcoord + 0.5 * SMAA_AREATEX_PIXEL_SIZE;

    // Diagonal areas are on the second half of the texture:
    texcoord.x += 0.5;

    // Move to proper place, according to the subpixel offset:
    texcoord.y += SMAA_AREATEX_SUBTEX_SIZE * offset;

    // Do it!
    return SMAASampleLevelZero(areaTex, texcoord).rg;
}

// float2 SMAACalculateDiagWeights(Texture2D<float2> edgesTex, Texture2D<float4> areaTex,
//                                 float2 texcoord, float2 e, float4 subsampleIndices)
fn SMAACalculateDiagWeights(
    edgesTex: texture_2d<f32>,
    areaTex: texture_2d<f32>,
    texcoord: vec2<f32>,
    eIn: vec2<f32>,
    subsampleIndices: vec4<f32>,
    rt: vec4<f32>,
) -> vec2<f32> {
    var e: vec2<f32> = eIn;
    var weights: vec2<f32> = vec2<f32>(0.0, 0.0);

    // Search for the line ends:
    var d: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var end: vec2<f32> = vec2<f32>(0.0, 0.0);
    if (e.r > 0.0) {
        let r1 = SMAASearchDiag1(edgesTex, texcoord, vec2<f32>(-1.0, 1.0), rt);
        end = r1.e;
        d = vec4<f32>(r1.zw.x + select(0.0, 1.0, end.y > 0.9), d.y, r1.zw.y, d.w);
    } else {
        d = vec4<f32>(0.0, d.y, 0.0, d.w);
    }
    let r2 = SMAASearchDiag1(edgesTex, texcoord, vec2<f32>(1.0, -1.0), rt);
    d = vec4<f32>(d.x, r2.zw.x, d.z, r2.zw.y);

    if (d.x + d.y > 2.0) { // d.x + d.y + 1 > 3
        // Fetch the crossing edges:
        let coords: vec4<f32> = vec4<f32>(-d.x + 0.25, d.x, d.y, -d.y - 0.25) * rt.xyxy + texcoord.xyxy;
        let s1 = textureSampleLevel(edgesTex, LinearSampler, coords.xy, 0.0, vec2<i32>(-1, 0));
        let s2 = textureSampleLevel(edgesTex, LinearSampler, coords.zw, 0.0, vec2<i32>(1, 0));
        var c: vec4<f32> = vec4<f32>(s1.r, s1.g, s2.r, s2.g);
        let dec = SMAADecodeDiagBilinearAccess4(c);
        c = vec4<f32>(dec.y, dec.x, dec.w, dec.z); // c.yxwz = decode(c)

        // Merge crossing edges at each side into a single value:
        var cc: vec2<f32> = vec2<f32>(2.0, 2.0) * c.xz + c.yw;

        // Remove the crossing edge if we didn't found the end of the line:
        // (HLSL `bool2(step(0.9, d.zw))`: step yields 0.0/1.0, so `!= 0.0` is identical)
        cc = SMAAMovc2(step(vec2<f32>(0.9, 0.9), d.zw) != vec2<f32>(0.0, 0.0), cc, vec2<f32>(0.0, 0.0));

        // Fetch the areas for this line:
        weights += SMAAAreaDiag(areaTex, d.xy, cc, subsampleIndices.z);
    }

    // Search for the line ends:
    let r3 = SMAASearchDiag2(edgesTex, texcoord, vec2<f32>(-1.0, -1.0), rt);
    d = vec4<f32>(r3.zw.x, d.y, r3.zw.y, d.w);
    if (textureSampleLevel(edgesTex, LinearSampler, texcoord, 0.0, vec2<i32>(1, 0)).r > 0.0) {
        let r4 = SMAASearchDiag2(edgesTex, texcoord, vec2<f32>(1.0, 1.0), rt);
        end = r4.e;
        d = vec4<f32>(d.x, r4.zw.x + select(0.0, 1.0, end.y > 0.9), d.z, r4.zw.y);
    } else {
        d = vec4<f32>(d.x, 0.0, d.z, 0.0);
    }

    if (d.x + d.y > 2.0) { // d.x + d.y + 1 > 3
        // Fetch the crossing edges:
        let coords: vec4<f32> = vec4<f32>(-d.x, -d.x, d.y, d.y) * rt.xyxy + texcoord.xyxy;
        let s3 = textureSampleLevel(edgesTex, LinearSampler, coords.xy, 0.0, vec2<i32>(-1, 0));
        let s4 = textureSampleLevel(edgesTex, LinearSampler, coords.xy, 0.0, vec2<i32>(0, -1));
        let s5 = textureSampleLevel(edgesTex, LinearSampler, coords.zw, 0.0, vec2<i32>(1, 0));
        let c: vec4<f32> = vec4<f32>(s3.g, s4.r, s5.g, s5.r);
        var cc: vec2<f32> = vec2<f32>(2.0, 2.0) * c.xz + c.yw;

        // Remove the crossing edge if we didn't found the end of the line:
        cc = SMAAMovc2(step(vec2<f32>(0.9, 0.9), d.zw) != vec2<f32>(0.0, 0.0), cc, vec2<f32>(0.0, 0.0));

        // Fetch the areas for this line:
        let area: vec2<f32> = SMAAAreaDiag(areaTex, d.xy, cc, subsampleIndices.w);
        weights += area.gr;
    }

    return weights;
}

// ---- Horizontal/Vertical search functions ----

// float SMAASearchLength(Texture2D<float> searchTex, float2 e, float offset)
fn SMAASearchLength(searchTex: texture_2d<f32>, e: vec2<f32>, offset: f32) -> f32 {
    // The texture is flipped vertically, with left and right cases taking half
    // of the space horizontally:
    var scale: vec2<f32> = SMAA_SEARCHTEX_SIZE * vec2<f32>(0.5, -1.0);
    var bias: vec2<f32> = SMAA_SEARCHTEX_SIZE * vec2<f32>(offset, 1.0);

    // Scale and bias to access texel centers:
    scale += vec2<f32>(-1.0, 1.0);
    bias += vec2<f32>(0.5, -0.5);

    // Convert from pixel coordinates to texcoords:
    // (We use SMAA_SEARCHTEX_PACKED_SIZE because the texture is cropped)
    scale *= vec2<f32>(1.0, 1.0) / SMAA_SEARCHTEX_PACKED_SIZE;
    bias *= vec2<f32>(1.0, 1.0) / SMAA_SEARCHTEX_PACKED_SIZE;

    // Lookup the search texture:
    return SMAASampleLevelZero(searchTex, scale * e + bias).r;
}

// float SMAASearchXLeft(Texture2D<float2> edgesTex, Texture2D<float> searchTex, float2 texcoord, float end)
fn SMAASearchXLeft(
    edgesTex: texture_2d<f32>,
    searchTex: texture_2d<f32>,
    texcoordIn: vec2<f32>,
    end: f32,
    rt: vec4<f32>,
) -> f32 {
    // @PSEUDO_GATHER4: this texcoord has been offset by (-0.25, -0.125) in the
    // vertex shader equivalent, to sample between edges.
    var texcoord: vec2<f32> = texcoordIn;
    var e: vec2<f32> = vec2<f32>(0.0, 1.0);
    while (texcoord.x > end &&
        e.g > 0.8281 && // Is there some edge not activated?
        e.r == 0.0) { // Or is there a crossing edge that breaks the line?
        e = SMAASampleLevelZero(edgesTex, texcoord).xy;
        texcoord = vec2<f32>(-2.0, 0.0) * rt.xy + texcoord;
    }

    let offset: f32 = -(255.0 / 127.0) * SMAASearchLength(searchTex, e, 0.0) + 3.25;
    return rt.x * offset + texcoord.x;
}

// float SMAASearchXRight(Texture2D<float2> edgesTex, Texture2D<float> searchTex, float2 texcoord, float end)
fn SMAASearchXRight(
    edgesTex: texture_2d<f32>,
    searchTex: texture_2d<f32>,
    texcoordIn: vec2<f32>,
    end: f32,
    rt: vec4<f32>,
) -> f32 {
    var texcoord: vec2<f32> = texcoordIn;
    var e: vec2<f32> = vec2<f32>(0.0, 1.0);
    while (texcoord.x < end &&
        e.g > 0.8281 &&
        e.r == 0.0) {
        e = SMAASampleLevelZero(edgesTex, texcoord).xy;
        texcoord = vec2<f32>(2.0, 0.0) * rt.xy + texcoord;
    }
    let offset: f32 = -(255.0 / 127.0) * SMAASearchLength(searchTex, e, 0.5) + 3.25;
    return -rt.x * offset + texcoord.x;
}

// float SMAASearchYUp(Texture2D<float2> edgesTex, Texture2D<float> searchTex, float2 texcoord, float end)
fn SMAASearchYUp(
    edgesTex: texture_2d<f32>,
    searchTex: texture_2d<f32>,
    texcoordIn: vec2<f32>,
    end: f32,
    rt: vec4<f32>,
) -> f32 {
    var texcoord: vec2<f32> = texcoordIn;
    var e: vec2<f32> = vec2<f32>(1.0, 0.0);
    while (texcoord.y > end &&
        e.r > 0.8281 &&
        e.g == 0.0) {
        e = SMAASampleLevelZero(edgesTex, texcoord).xy;
        texcoord = vec2<f32>(0.0, -2.0) * rt.xy + texcoord;
    }
    let offset: f32 = -(255.0 / 127.0) * SMAASearchLength(searchTex, e.gr, 0.0) + 3.25;
    return rt.y * offset + texcoord.y;
}

// float SMAASearchYDown(Texture2D<float2> edgesTex, Texture2D<float> searchTex, float2 texcoord, float end)
fn SMAASearchYDown(
    edgesTex: texture_2d<f32>,
    searchTex: texture_2d<f32>,
    texcoordIn: vec2<f32>,
    end: f32,
    rt: vec4<f32>,
) -> f32 {
    var texcoord: vec2<f32> = texcoordIn;
    var e: vec2<f32> = vec2<f32>(1.0, 0.0);
    while (texcoord.y < end &&
        e.r > 0.8281 &&
        e.g == 0.0) {
        e = SMAASampleLevelZero(edgesTex, texcoord).xy;
        texcoord = vec2<f32>(0.0, 2.0) * rt.xy + texcoord;
    }
    let offset: f32 = -(255.0 / 127.0) * SMAASearchLength(searchTex, e.gr, 0.5) + 3.25;
    return -rt.y * offset + texcoord.y;
}

// float2 SMAAArea(Texture2D<float4> areaTex, float2 dist, float e1, float e2, float offset)
fn SMAAArea(areaTex: texture_2d<f32>, dist: vec2<f32>, e1: f32, e2: f32, offset: f32) -> vec2<f32> {
    // Rounding prevents precision errors of bilinear filtering:
    var texcoord: vec2<f32> = vec2<f32>(SMAA_AREATEX_MAX_DISTANCE, SMAA_AREATEX_MAX_DISTANCE) * round(4.0 * vec2<f32>(e1, e2)) + dist;

    // We do a scale and bias for mapping to texel space:
    texcoord = SMAA_AREATEX_PIXEL_SIZE * texcoord + 0.5 * SMAA_AREATEX_PIXEL_SIZE;

    // Move to proper place, according to the subpixel offset:
    texcoord.y = SMAA_AREATEX_SUBTEX_SIZE * offset + texcoord.y;

    // Do it!
    return SMAASampleLevelZero(areaTex, texcoord).rg;
}

// ---- Corner detection functions ----

// `rounding.x + rounding.y` is provably >= 1 (step(d.xy, d.yx) cannot be (0,0)),
// so the division below never produces the HLSL 0/0 NaN that `saturate` would heal.
fn SMAADetectHorizontalCornerPattern(
    edgesTex: texture_2d<f32>,
    weightsIn: vec2<f32>,
    texcoord: vec4<f32>,
    d: vec2<f32>,
) -> vec2<f32> {
    let leftRight: vec2<f32> = step(d.xy, d.yx);
    var rounding: vec2<f32> = (1.0 - SMAA_CORNER_ROUNDING_NORM) * leftRight;

    rounding = rounding / (leftRight.x + leftRight.y); // Reduce blending for pixels in the center of a line.

    var factor: vec2<f32> = vec2<f32>(1.0, 1.0);
    factor.x -= rounding.x * textureSampleLevel(edgesTex, LinearSampler, texcoord.xy, 0.0, vec2<i32>(0, 1)).r;
    factor.x -= rounding.y * textureSampleLevel(edgesTex, LinearSampler, texcoord.zw, 0.0, vec2<i32>(1, 1)).r;
    factor.y -= rounding.x * textureSampleLevel(edgesTex, LinearSampler, texcoord.xy, 0.0, vec2<i32>(0, -2)).r;
    factor.y -= rounding.y * textureSampleLevel(edgesTex, LinearSampler, texcoord.zw, 0.0, vec2<i32>(1, -2)).r;

    return weightsIn * saturate(factor);
}

fn SMAADetectVerticalCornerPattern(
    edgesTex: texture_2d<f32>,
    weightsIn: vec2<f32>,
    texcoord: vec4<f32>,
    d: vec2<f32>,
) -> vec2<f32> {
    let leftRight: vec2<f32> = step(d.xy, d.yx);
    var rounding: vec2<f32> = (1.0 - SMAA_CORNER_ROUNDING_NORM) * leftRight;

    rounding = rounding / (leftRight.x + leftRight.y);

    var factor: vec2<f32> = vec2<f32>(1.0, 1.0);
    factor.x -= rounding.x * textureSampleLevel(edgesTex, LinearSampler, texcoord.xy, 0.0, vec2<i32>(1, 0)).g;
    factor.x -= rounding.y * textureSampleLevel(edgesTex, LinearSampler, texcoord.zw, 0.0, vec2<i32>(1, 1)).g;
    factor.y -= rounding.x * textureSampleLevel(edgesTex, LinearSampler, texcoord.xy, 0.0, vec2<i32>(-2, 0)).g;
    factor.y -= rounding.y * textureSampleLevel(edgesTex, LinearSampler, texcoord.zw, 0.0, vec2<i32>(-2, 1)).g;

    return weightsIn * saturate(factor);
}

// ---- Blending weight calculation ----

// float4 SMAABlendingWeightCalculationPS(float2 texcoord, Texture2D<float2> edgesTex,
//     Texture2D<float4> areaTex, Texture2D<float> searchTex, float4 subsampleIndices)
fn SMAABlendingWeightCalculationPS(
    texcoord: vec2<f32>,
    edgesTex: texture_2d<f32>,
    areaTex: texture_2d<f32>,
    searchTex: texture_2d<f32>,
    subsampleIndices: vec4<f32>,
    rt: vec4<f32>,
) -> vec4<f32> {
    let pixcoord: vec2<f32> = texcoord * rt.zw;

    var offset: array<vec4<f32>, 3>;
    // We will use these offsets for the searches later on (see @PSEUDO_GATHER4):
    offset[0] = rt.xyxy * vec4<f32>(-0.25, -0.125, 1.25, -0.125) + texcoord.xyxy;
    offset[1] = rt.xyxy * vec4<f32>(-0.125, -0.25, -0.125, 1.25) + texcoord.xyxy;

    // And these for the searches, they indicate the ends of the loops:
    offset[2] = rt.xxyy * (vec4<f32>(-2.0, 2.0, -2.0, 2.0) * f32(SMAA_MAX_SEARCH_STEPS)) + vec4<f32>(offset[0].xz, offset[1].yw);

    var weights: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    var e: vec2<f32> = SMAASample(edgesTex, texcoord).rg;

    if (e.g > 0.0) { // Edge at north
        // Diagonals have both north and west edges, so searching for them in
        // one of the boundaries is enough.
        let dw: vec2<f32> = SMAACalculateDiagWeights(edgesTex, areaTex, texcoord, e, subsampleIndices, rt);
        weights = vec4<f32>(dw.x, dw.y, weights.z, weights.w);

        // We give priority to diagonals, so if we find a diagonal we skip
        // horizontal/vertical processing.
        if (weights.r == -weights.g) { // weights.r + weights.g == 0.0
            var d: vec2<f32> = vec2<f32>(0.0, 0.0);

            // Find the distance to the left:
            var coords: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
            coords.x = SMAASearchXLeft(edgesTex, searchTex, offset[0].xy, offset[2].x, rt);
            coords.y = offset[1].y; // offset[1].y = texcoord.y - 0.25 * SMAA_RT_METRICS.y (@CROSSING_OFFSET)
            d.x = coords.x;

            // Now fetch the left crossing edges, two at a time using bilinear filtering.
            let e1: f32 = SMAASampleLevelZero(edgesTex, coords.xy).r;

            // Find the distance to the right:
            coords.z = SMAASearchXRight(edgesTex, searchTex, offset[0].zw, offset[2].y, rt);
            d.y = coords.z;

            // We want the distances to be in pixel units (doing this here allow to
            // better interleave arithmetic and memory accesses):
            d = abs(round(rt.zz * d + vec2<f32>(-pixcoord.x, -pixcoord.x)));

            // SMAAArea below needs a sqrt, as the areas texture is compressed
            // quadratically:
            let sqrt_d: vec2<f32> = sqrt(d);

            // Fetch the right crossing edges:
            let e2: f32 = textureSampleLevel(edgesTex, LinearSampler, coords.zy, 0.0, vec2<i32>(1, 0)).r;

            // Ok, we know how this pattern looks like, now it is time for getting
            // the actual area:
            let area: vec2<f32> = SMAAArea(areaTex, sqrt_d, e1, e2, subsampleIndices.y);
            weights = vec4<f32>(area.x, area.y, weights.z, weights.w);

            // Fix corners:
            coords.y = texcoord.y;
            let w2: vec2<f32> = SMAADetectHorizontalCornerPattern(
                edgesTex, weights.xy, vec4<f32>(coords.x, coords.y, coords.z, coords.y), d);
            weights = vec4<f32>(w2.x, w2.y, weights.z, weights.w);
        } else {
            e.r = 0.0; // Skip vertical processing.
        }
    }

    if (e.r > 0.0) { // Edge at west
        var d: vec2<f32> = vec2<f32>(0.0, 0.0);

        // Find the distance to the top:
        var coords: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
        coords.y = SMAASearchYUp(edgesTex, searchTex, offset[1].xy, offset[2].z, rt);
        coords.x = offset[0].x; // offset[1].x = texcoord.x - 0.25 * SMAA_RT_METRICS.x;
        d.x = coords.y;

        // Fetch the top crossing edges:
        let e1: f32 = SMAASampleLevelZero(edgesTex, coords.xy).g;

        // Find the distance to the bottom:
        coords.z = SMAASearchYDown(edgesTex, searchTex, offset[1].zw, offset[2].w, rt);
        d.y = coords.z;

        // We want the distances to be in pixel units:
        d = abs(round(rt.ww * d + vec2<f32>(-pixcoord.y, -pixcoord.y)));

        // SMAAArea below needs a sqrt, as the areas texture is compressed
        // quadratically:
        let sqrt_d: vec2<f32> = sqrt(d);

        // Fetch the bottom crossing edges:
        let e2: f32 = textureSampleLevel(edgesTex, LinearSampler, coords.xz, 0.0, vec2<i32>(0, 1)).g;

        // Get the area for this direction:
        let area: vec2<f32> = SMAAArea(areaTex, sqrt_d, e1, e2, subsampleIndices.x);
        weights = vec4<f32>(weights.x, area.x, area.y, weights.w);

        // Fix corners:
        coords.x = texcoord.x;
        let w2: vec2<f32> = SMAADetectVerticalCornerPattern(
            edgesTex, weights.ba, vec4<f32>(coords.x, coords.y, coords.x, coords.z), d);
        weights = vec4<f32>(weights.x, w2.x, w2.y, weights.w);
    }

    return weights;
}

// ---- Neighborhood blending ----

// float4 SMAANeighborhoodBlendingPS(float2 texcoord, Texture2D<float4> colorTex, Texture2D<float4> blendTex)
fn SMAANeighborhoodBlendingPS(
    colorTex: texture_2d<f32>,
    blendTex: texture_2d<f32>,
    texcoord: vec2<f32>,
    rt: vec4<f32>,
) -> vec4<f32> {
    let offset: vec4<f32> = rt.xyxy * vec4<f32>(1.0, 0.0, 0.0, 1.0) + texcoord.xyxy;

    // Fetch the blending weights for current pixel:
    var a: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    a.x = SMAASample(blendTex, offset.xy).a; // Right
    a.y = SMAASample(blendTex, offset.zw).g; // Top
    let s: vec4<f32> = SMAASample(blendTex, texcoord);
    a.z = s.z; // Bottom
    a.w = s.x; // Left        (a.wz = SMAASample(blendTex, texcoord).xz)

    // Is there any blending weight with a value greater than 0.0?
    if (dot(a, vec4<f32>(1.0, 1.0, 1.0, 1.0)) < 1e-5) {
        return SMAASampleLevelZero(colorTex, texcoord);
    } else {
        let h: bool = max(a.x, a.z) > max(a.y, a.w); // max(horizontal) > max(vertical)

        // Calculate the blending offsets:
        var blendingOffset: vec4<f32> = vec4<f32>(0.0, a.y, 0.0, a.w);
        var blendingWeight: vec2<f32> = a.yw;
        blendingOffset = SMAAMovc4(vec4<bool>(h, h, h, h), blendingOffset, vec4<f32>(a.x, 0.0, a.z, 0.0));
        blendingWeight = SMAAMovc2(vec2<bool>(h, h), blendingWeight, a.xz);
        blendingWeight = blendingWeight / dot(blendingWeight, vec2<f32>(1.0, 1.0));

        // Calculate the texture coordinates:
        let blendingCoord: vec4<f32> = blendingOffset * vec4<f32>(rt.x, rt.y, -rt.x, -rt.y) + texcoord.xyxy;

        // We exploit bilinear filtering to mix current pixel with the chosen neighbor:
        var color: vec4<f32> = blendingWeight.x * SMAASampleLevelZero(colorTex, blendingCoord.xy);
        color += blendingWeight.y * SMAASampleLevelZero(colorTex, blendingCoord.zw);

        return color;
    }
}
//! END COMMON

//! PASS 1
//! DESC "SMAA luma edge detection"
//! IN INPUT
//! OUT edgesTex
//! NUM_THREADS 8,8,1

fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= scene.inputSize.x || gid.y >= scene.inputSize.y) {
        return;
    }
    let rt: vec4<f32> = SMAART();
    let pos: vec2<f32> = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.inputPt;
    let edges: vec2<f32> = SMAALumaEdgeDetectionPS(pos, INPUT, rt);
    textureStore(edgesTex, vec2<i32>(gid.xy), vec4<f32>(edges.x, edges.y, 0.0, 0.0));
}

//! PASS 2
//! DESC "SMAA blending weight calculation"
//! IN edgesTex, areaTex, searchTex
//! OUT blendTex
//! NUM_THREADS 8,8,1

fn Pass2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= scene.inputSize.x || gid.y >= scene.inputSize.y) {
        return;
    }
    let rt: vec4<f32> = SMAART();
    let pos: vec2<f32> = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.inputPt;
    let weights: vec4<f32> = SMAABlendingWeightCalculationPS(pos, edgesTex, areaTex, searchTex, vec4<f32>(0.0), rt);
    textureStore(blendTex, vec2<i32>(gid.xy), weights);
}

//! PASS 3
//! DESC "SMAA neighborhood blending"
//! IN INPUT, blendTex
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

fn Pass3(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= scene.outputSize.x || gid.y >= scene.outputSize.y) {
        return;
    }
    let rt: vec4<f32> = SMAART();
    let pos: vec2<f32> = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.outputPt;
    let color: vec4<f32> = SMAANeighborhoodBlendingPS(INPUT, blendTex, pos, rt);
    textureStore(OUTPUT, vec2<i32>(gid.xy), color);
}
//! END
