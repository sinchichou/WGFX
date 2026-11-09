// WGFX regression fixture — u32 scene-size arguments inside vecN<f32>( ) constructors
// ---
// 最小回歸 fixture：覆蓋 WGSLCodeGenerator._wrapU32ArgsInF32VectorCtors 的
// 「過度包裹（over-wrapping）」bug（SMAA 移植時實際踩到，2026-11 修正）。
//
// 症狀（修正前）：
//   `vec4<f32>(scene.inputPt, vec2<f32>(scene.inputSize))`
//   的第二個引數本身已是 f32 向量建構式，卻因「文字提到 scene.inputSize」被再包一層
//   → `f32(vec2<f32>(scene.inputSize))`
//   → Tint: `no matching constructor for 'f32(vec2<f32>)'`（Pass 1/3 皆中）。
//
// 本 fixture 同時要求兩件事（單向斷言不足以防退化）：
//   (a) 已是 f32 向量建構式的引數**不得**再被包裹；
//   (b) 內層真正的 u32 純量引數（scene.inputSize.x/.y 等）**仍必須**被補上 f32()，
//       否則 WGSL 的多引數向量建構子（HLSL 隱式轉換）不合法。
// 由 test/test_runtime.js 的 Test F 以「編譯成功 + 產物字串 + 實跑像素」三重驗證。
//! VERSION 4

//! TEXTURE INPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "u32 scene-size args inside vecN<f32> ctors (over-wrap regression)"
fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    // (1) 混合引數：vec2<f32> + vec2<f32>(vec2<u32>) → 後者不得再被 f32() 包裹（bug 本體）
    let rt = vec4<f32>(scene.inputPt, vec2<f32>(scene.inputSize));
    // (2) 內層純量引數：仍需 f32() 包裹（u32 → f32 轉換，SMAA 檔頭的逐分量繞法）
    let sz = vec2<f32>(scene.inputSize.x, scene.inputSize.y);
    // (3) 巢狀於外層建構式內的純量引數：遞迴由內而外必須處理
    let nested = vec4<f32>(scene.inputPt, vec2<f32>(scene.inputSize.y, scene.outputSize.y));

    let src = textureLoad(INPUT, vec2<i32>(gid.xy), 0);
    // 讓 (1)(2)(3) 的值實際影響輸出（避免被最佳化掉而失去驗證意義）；
    // 各因子皆在 [0.5, 1.5] 內，輸出必為非零且不溢位 f16。
    let a = 0.5 + 0.25 * clamp(rt.x * 64.0, 0.0, 1.0);
    let b = 0.5 + 0.25 * clamp(sz.y / 256.0, 0.0, 1.0);
    let c = 0.5 + 0.25 * clamp(nested.z / 256.0, 0.0, 1.0);
    textureStore(OUTPUT, vec2<i32>(gid.xy), vec4<f32>(src.rgb * a * b * c, 1.0));
}
