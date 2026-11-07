// LCAS — 移植自 effects_src/Sharpen/LCAS.hlsl（Magpie v4）
// 線性插值 + 輕量 CAS 的混合銳化
//
// 移植偏差（記錄）：wRGB 的 mxRGB 除法加 max(mxRGB, 1e-6) 退化域防護——
// 全黑鄰域（mxRGB=0）時上游會產生 0/0 → sqrt(NaN)；防護後全黑輸出正確歸 0，
// 且 mxRGB ≥ 1e-6（內容 ≥ 1/255 = 3.9e-3）時數值與上游完全一致。
//! VERSION 4

//! PARAMETER sharpness
//! LABEL "Sharpness"
//! MIN 0
//! MAX 1
//! DEFAULT 0.5
//! STEP 0.01
float sharpness;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
Texture2D OUTPUT;

//! SAMPLER sam
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "LCAS：線性插值 + 輕量 CAS 銳化"

fn LCAS(ip: vec2<u32>, peak: f32) -> vec3<f32> {
    let pos = (vec2<f32>(ip) + vec2<f32>(0.5)) * scene.outputPt;
    let inputPt = scene.inputPt;

    // 3x3 鄰域
    //  a b c
    //  d e f
    //  g h i
    let a = textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x, -inputPt.y), 0.0).rgb;
    let b = textureSampleLevel(INPUT, sam, pos + vec2<f32>(0.0, -inputPt.y), 0.0).rgb;
    let c = textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x, -inputPt.y), 0.0).rgb;
    let d = textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x, 0.0), 0.0).rgb;
    let f = textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x, 0.0), 0.0).rgb;
    let g = textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x, inputPt.y), 0.0).rgb;
    let h = textureSampleLevel(INPUT, sam, pos + vec2<f32>(0.0, inputPt.y), 0.0).rgb;
    let i = textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x, inputPt.y), 0.0).rgb;

    let x = a + c + g + i;
    let y = b + d + f + h;

    // e 由四個對角 0.25/0.5 偏移的雙線性取樣平均（亞像素細節）
    var e = textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x * 0.25, inputPt.y * 0.5), 0.0).rgb;
    e += textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x * 0.25, -inputPt.y * 0.5), 0.0).rgb;
    e += textureSampleLevel(INPUT, sam, pos + vec2<f32>(inputPt.x * 0.5, -inputPt.y * 0.25), 0.0).rgb;
    e += textureSampleLevel(INPUT, sam, pos + vec2<f32>(-inputPt.x * 0.5, inputPt.y * 0.25), 0.0).rgb;
    e = e / 4.0;

    // 軟 min/max
    let mnRGB = min(min(min(min(d, e), min(f, b)), h), min(min(a, i), min(c, g)));
    let mxRGB = max(max(max(max(d, e), max(f, b)), h), max(max(a, i), max(c, g)));

    // 銳化量整形（退化域防護見檔頭）
    let wRGB = sqrt(min(mnRGB, 1.0 - mxRGB) / max(mxRGB, vec3<f32>(1e-6))) * peak;

    // 濾波形狀
    //  w w w
    //  w 1 w
    //  w w w
    let color = ((x + y) * wRGB + (e * 5.0 - (x + y * 2.0 + e * 4.0) / 4.0)) / (1.0 + 8.0 * wRGB);
    return (color + clamp(color, mnRGB, mxRGB) * 4.0) / 5.0;
}

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let peak = mix(0.0, -0.1111111111111111, uniforms.sharpness);
    textureStore(OUTPUT, vec2<i32>(global_id.xy), vec4<f32>(LCAS(global_id.xy, peak), 1.0));
}
