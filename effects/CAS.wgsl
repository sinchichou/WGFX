// CAS（AMD FidelityFX Contrast Adaptive Sharpening）— 移植自 effects_src/CAS/CAS.hlsl
// 上游：GPUOpen FidelityFX-CAS ffx_cas.h（scalar 路徑；Magpie 的 FP16 打包路徑不適用於 WGSL 單通道語義）
//
// 移植偏差（記錄）：rcp(mxRGB) 加 max(mxRGB, vec3(1e-6)) 退化域防護——全黑鄰域上游為
// 0×Inf=NaN（靠 HLSL saturate 治癒）；防護後全黑輸出正確 = e，mxRGB ≥ 1e-6 時與上游一致。
//! VERSION 4

//! PARAMETER sharpness
//! LABEL "Sharpness"
//! MIN 0
//! MAX 1
//! DEFAULT 0.4
//! STEP 0.01
float sharpness;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "CAS 對比自適應銳化（無取樣器，clamp load）"

fn Load(x: i32, y: i32) -> vec3<f32> {
    let size = vec2<i32>(scene.inputSize);
    return textureLoad(INPUT, clamp(vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let gx = i32(global_id.x);
    let gy = i32(global_id.y);

    // a b c
    // d e f
    // g h i
    let a = Load(gx - 1, gy - 1);
    let b = Load(gx, gy - 1);
    let c = Load(gx + 1, gy - 1);
    let d = Load(gx - 1, gy);
    let e = Load(gx, gy);
    let f = Load(gx + 1, gy);
    let g = Load(gx - 1, gy + 1);
    let h = Load(gx, gy + 1);
    let i = Load(gx + 1, gy + 1);

    // 軟 min/max（中央十字，2 倍縮放已因式分解）
    let mnR = min(min(min(d.r, e.r), f.r), min(b.r, h.r));
    let mnG = min(min(min(d.g, e.g), f.g), min(b.g, h.g));
    let mnB = min(min(min(d.b, e.b), f.b), min(b.b, h.b));
    let mxR = max(max(max(d.r, e.r), f.r), max(b.r, h.r));
    let mxG = max(max(max(d.g, e.g), f.g), max(b.g, h.g));
    let mxB = max(max(max(d.b, e.b), f.b), max(b.b, h.b));

    // 訊號上限的平滑最小距離 ÷ 平滑最大值（退化域防護見檔頭）
    let ampR = saturate(min(mnR, 1.0 - mxR) / max(mxR, 1e-6));
    let ampG = saturate(min(mnG, 1.0 - mxG) / max(mxG, 1e-6));
    let ampB = saturate(min(mnB, 1.0 - mxB) / max(mxB, 1e-6));

    // 銳化整形
    let wG = sqrt(ampG) * (-1.0 / mix(8.0, 5.0, uniforms.sharpness)); // checker:ok — mix ∈ [5,8]（sharpness clamp 0..1），分母有界

    // 濾波（僅用綠色係數，依賴死碼消除）
    //  0 w 0
    //  w 1 w
    //  0 w 0
    let rcpWeight = 1.0 / (1.0 + 4.0 * wG); // checker:ok — wG ∈ [-0.2, 0]，分母 ≥ 0.2
    let outColor = vec3<f32>(
        saturate((b.r * wG + d.r * wG + f.r * wG + h.r * wG + e.r) * rcpWeight),
        saturate((b.g * wG + d.g * wG + f.g * wG + h.g * wG + e.g) * rcpWeight),
        saturate((b.b * wG + d.b * wG + f.b * wG + h.b * wG + e.b) * rcpWeight)
    );

    textureStore(OUTPUT, vec2<i32>(global_id.xy), vec4<f32>(outColor, 1.0));
}
