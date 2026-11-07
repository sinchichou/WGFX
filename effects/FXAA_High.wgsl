// FXAA High（PRESET 4）— 移植自 effects_src/FXAA/FXAA.hlsli + FXAA_High.hlsl（Magpie v4）
// 上游：libretro slang-shaders fxaa.slang（Timothy Lottes FXAA 3.11 簡化版）
//
// 移植說明：4 次 GatherRed/Green/Blue 展開為 4x4 clamp load 網格
// （採樣點在紋理角點，D3D 四點順序還原為 src[a][b] = load(gxy + a - 1, gxy + b - 1)）。
//! VERSION 4

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! SAMPLER sam
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "FXAA PRESET 4（搜尋 24 步）"

// --- PRESET 4 常數 ---
const FXAA_EDGE_THRESHOLD = 1.0 / 8.0;
const FXAA_EDGE_THRESHOLD_MIN = 1.0 / 24.0;
const FXAA_SEARCH_STEPS = 24u;
const FXAA_SEARCH_THRESHOLD = 1.0 / 4.0;
const FXAA_SUBPIX_CAP = 3.0 / 4.0;
const FXAA_SUBPIX_TRIM = 1.0 / 4.0;
const FXAA_SUBPIX_TRIM_SCALE = 1.0 / (1.0 - FXAA_SUBPIX_TRIM);

// luma 估計（省正規化，範圍 0..2.96）
fn FxaaLuma(rgb: vec3<f32>) -> f32 {
    return rgb.y * (0.587 / 0.299) + rgb.x;
}

fn FxaaLerp3(a: vec3<f32>, b: vec3<f32>, amountOfA: f32) -> vec3<f32> {
    return (a - b) * amountOfA + b;
}

fn Load(x: i32, y: i32) -> vec3<f32> {
    let size = vec2<i32>(scene.inputSize);
    return textureLoad(INPUT, clamp(vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let gxy = vec2<i32>(global_id.xy);
    let inputPt = scene.inputPt;
    let pos = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) * inputPt;

    // 4x4 鄰域（gather 展開）
    var src: array<array<vec3<f32>, 4>, 4>;
    for (var a: u32 = 0u; a < 4u; a++) {
        for (var b: u32 = 0u; b < 4u; b++) {
            src[a][b] = Load(gxy.x + i32(a) - 1, gxy.y + i32(b) - 1);
        }
    }
    let ci = 1u; // 中心索引

    let lumaN = FxaaLuma(src[ci][ci - 1u]);
    let lumaW = FxaaLuma(src[ci - 1u][ci]);
    let lumaM = FxaaLuma(src[ci][ci]);
    let lumaE = FxaaLuma(src[ci + 1u][ci]);
    let lumaS = FxaaLuma(src[ci][ci + 1u]);
    let rangeMin = min(lumaM, min(min(lumaN, lumaW), min(lumaS, lumaE)));
    let rangeMax = max(lumaM, max(max(lumaN, lumaW), max(lumaS, lumaE)));

    let range = rangeMax - rangeMin;
    if (range < max(FXAA_EDGE_THRESHOLD_MIN, rangeMax * FXAA_EDGE_THRESHOLD)) {
        // 局部對比不足 → 原樣輸出
        textureStore(OUTPUT, gxy, vec4<f32>(src[ci][ci], 1.0));
        return;
    }

    var rgbL = src[ci][ci - 1u] + src[ci - 1u][ci] + src[ci][ci] + src[ci + 1u][ci] + src[ci][ci + 1u];

    let lumaL = (lumaN + lumaW + lumaE + lumaS) * 0.25;
    let rangeL = abs(lumaL - lumaM);
    var blendL = max(0.0, (rangeL / range) - FXAA_SUBPIX_TRIM) * FXAA_SUBPIX_TRIM_SCALE;
    blendL = min(FXAA_SUBPIX_CAP, blendL);

    rgbL += (src[ci - 1u][ci - 1u] + src[ci + 1u][ci - 1u] + src[ci - 1u][ci + 1u] + src[ci + 1u][ci + 1u]);
    rgbL *= 1.0 / 9.0;

    let lumaNW = FxaaLuma(src[ci - 1u][ci - 1u]);
    let lumaNE = FxaaLuma(src[ci + 1u][ci - 1u]);
    let lumaSW = FxaaLuma(src[ci - 1u][ci + 1u]);
    let lumaSE = FxaaLuma(src[ci + 1u][ci + 1u]);

    let edgeVert = abs((0.25 * lumaNW) + (-0.5 * lumaN) + (0.25 * lumaNE))
                 + abs((0.50 * lumaW) + (-1.0 * lumaM) + (0.50 * lumaE))
                 + abs((0.25 * lumaSW) + (-0.5 * lumaS) + (0.25 * lumaSE));
    let edgeHorz = abs((0.25 * lumaNW) + (-0.5 * lumaW) + (0.25 * lumaSW))
                 + abs((0.50 * lumaN) + (-1.0 * lumaM) + (0.50 * lumaS))
                 + abs((0.25 * lumaNE) + (-0.5 * lumaE) + (0.25 * lumaSE));

    let horzSpan = edgeHorz >= edgeVert;
    var lengthSign = select(-inputPt.x, -inputPt.y, horzSpan);

    var lumaN2 = lumaN;
    var lumaS2 = lumaS;
    if (!horzSpan) {
        lumaN2 = lumaW;
        lumaS2 = lumaE;
    }

    var gradientN = abs(lumaN2 - lumaM);
    let gradientS = abs(lumaS2 - lumaM);
    lumaN2 = (lumaN2 + lumaM) * 0.5;
    lumaS2 = (lumaS2 + lumaM) * 0.5;

    if (gradientN < gradientS) {
        lumaN2 = lumaS2;
        gradientN = gradientS;
        lengthSign *= -1.0;
    }

    var posN = vec2<f32>(
        pos.x + select(lengthSign * 0.5, 0.0, horzSpan),
        pos.y + select(0.0, lengthSign * 0.5, horzSpan)
    );

    gradientN *= FXAA_SEARCH_THRESHOLD;

    let posP = posN;
    let offNP = select(vec2<f32>(0.0, inputPt.y), vec2<f32>(inputPt.x, 0.0), horzSpan);
    var lumaEndN = lumaN2;
    var lumaEndP = lumaN2;
    var doneN = false;
    var doneP = false;
    var posN2 = posN + offNP * vec2<f32>(-1.0, -1.0);
    var posP2 = posP + offNP * vec2<f32>(1.0, 1.0);

    for (var step: u32 = 0u; step < FXAA_SEARCH_STEPS; step++) {
        if (!doneN) {
            lumaEndN = FxaaLuma(textureSampleLevel(INPUT, sam, posN2, 0.0).xyz);
        }
        if (!doneP) {
            lumaEndP = FxaaLuma(textureSampleLevel(INPUT, sam, posP2, 0.0).xyz);
        }

        doneN = doneN || (abs(lumaEndN - lumaN2) >= gradientN);
        doneP = doneP || (abs(lumaEndP - lumaN2) >= gradientN);

        if (doneN && doneP) {
            break;
        }
        if (!doneN) {
            posN2 -= offNP;
        }
        if (!doneP) {
            posP2 += offNP;
        }
    }

    let dstN = select(pos.y - posN2.y, pos.x - posN2.x, horzSpan);
    let dstP = select(posP2.x - pos.x, posP2.y - pos.y, horzSpan);
    let directionN = dstN < dstP;
    var lumaEnd = select(lumaEndP, lumaEndN, directionN);

    if (((lumaM - lumaN2) < 0.0) == ((lumaEnd - lumaN2) < 0.0)) {
        lengthSign = 0.0;
    }

    var spanLength = dstP + dstN;
    let dstFinal = select(dstP, dstN, directionN);
    let subPixelOffset = (0.5 + (dstFinal * (-1.0 / spanLength))) * lengthSign;
    let rgbF = textureSampleLevel(INPUT, sam, vec2<f32>(
        pos.x + select(subPixelOffset, 0.0, horzSpan),
        pos.y + select(0.0, subPixelOffset, horzSpan)), 0.0).xyz;

    textureStore(OUTPUT, gxy, vec4<f32>(FxaaLerp3(rgbL, rgbF, blendL), 1.0));
}
