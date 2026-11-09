// FSR_RCAS — Magpie v4 → WGFX WGSL port（全新移植）
// Ported from effects_src/FSR/FSR_RCAS.hlsl（Magpie v4）
// 上游：AMD FidelityFX Super Resolution 1.0（ffx-fsr/ffx_fsr1.h，RCAS 通道）
//   https://github.com/GPUOpen-Effects/FidelityFX-FSR
//   （a21ffb8f6c13233ba336352bdff293894c706575）
// 原授權 / 作者：Copyright (c) 2021 Advanced Micro Devices, Inc.
//   （MIT — AMD FidelityFX-FSR）；Magpie HLSL 移植：MIT（Blinue 與 Magpie 貢獻者）。
// 本移植採用純量（scalar，非 FP16 packed）路徑；RCAS 數學與常數逐字保留。
//
// Porting notes:
//   - 粒度比照 Magpie 原效果庫：RCAS 為獨立的 1:1 銳化效果
//     （接在 FSR_EASU 或其他縮放/濾波之後），與 examples/FSR_x2.wgsl 的
//     合併兩 pass 示範不同。
//   - 參數名保留 Magpie 原名 sharpness（uniforms.sharpness）。
//   - 鄰域讀取用 textureLoad（原碼 INPUT.Load；D3D 與 WebGPU 的越界讀取皆回 0，
//     語意一致，不做夾限）。POINT sampler 只出現在原碼宣告中（未使用），不需移植。
//   - hitMin/hitMax 的退化保護：全黑（4·mx4 = 0）或全白（4·mn4 − 4 = 0）時原碼
//     形成 0·Inf = NaN，HLSL 比較語意的 min/max 會癒合；WGSL 的 min/max 依規格
//     傳播 NaN，故以 select 顯式保護（該通道 mx4 == 0 → hitMin = 0；
//     mn4 == 1 → hitMax = 0）。退化情形（鄰域均勻全黑/全白）的輸出與原碼癒合
//     結果一致（輸出 = e，銳化關閉）。此模式沿用 examples/FSR_x2.wgsl 的
//     已驗證寫法（該檔以 test/impulse_response.mjs 對官方數學做過 9/9 對照），
//     見 docs/porting-guide.md §6。
//   - APrxMedRcpF1（ffx_a.h 位元技巧 0x7ef19fff + Newton 步）逐字還原：總函式
//     （輸入有限 → 輸出有限），用於 nz 與 rcpL 的中精度倒數（原碼註解：resolve
//     需要中精度 rcp 以避免可見的色調變化）。
//   - 與 examples/FSR_x2.wgsl 的差異：該示範檔在輸出前加了「四鄰域 min/max
//     除彩鉗位」（其檔頭記錄的刻意偏差，針對 4:2:0 影像的 chroma fireflies）；
//     本移植忠於 Magpie 原碼，**不含**該額外鉗位。
//   - Magpie 為 BLOCK_SIZE 16 / 64 threads（每 thread 經 Rmp8x8 負責 2×2 像素，
//     FP16 路徑一次算 2 像素）；本移植 1 invocation 1 像素（//! NUM_THREADS 8,8,1）。
//! VERSION 4
//! SORT_NAME FSR_RCAS

//! PARAMETER sharpness
//! TYPE float
//! LABEL "Sharpness"
//! MIN 0
//! MAX 1
//! DEFAULT 0.87
//! STEP 0.01
float sharpness;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! PASS 1
//! DESC "FidelityFX RCAS：穩健的對比自適應銳化"
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

// NOTE: 進入點必須是 pass 區塊的第一個函式（產生器把
// @compute @workgroup_size(...) 掛在區塊第一個 fn 上）。
fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outputSize = scene.outputSize;
    if (global_id.x >= outputSize.x || global_id.y >= outputSize.y) {
        return;
    }

    let p = vec2<i32>(global_id.xy);
    // 演算法使用最小 3×3 鄰域：
    //    b
    //  d e f
    //    h
    let b = textureLoad(INPUT, p + vec2<i32>(0, -1), 0).rgb;
    let d = textureLoad(INPUT, p + vec2<i32>(-1, 0), 0).rgb;
    let e = textureLoad(INPUT, p, 0).rgb;
    let f = textureLoad(INPUT, p + vec2<i32>(1, 0), 0).rgb;
    let h = textureLoad(INPUT, p + vec2<i32>(0, 1), 0).rgb;

    textureStore(OUTPUT, global_id.xy, vec4<f32>(FsrRcasF(b, d, e, f, h), 1.0));
}

// 為避免不自然的銳化結果而設的上限（原碼常數）。
const FSR_RCAS_LIMIT: f32 = 0.25 - (1.0 / 16.0);

// 中精度近似倒數（AMD ffx_a.h，逐字：IADD + FMA + MUL）。
// Newton 步使 APrxMedRcpF1(0) ≈ 3.2e38 —— 有限值，不同於精確 1/0 的 +Inf。
fn APrxMedRcpF1(a: f32) -> f32 {
    let b = bitcast<f32>(0x7ef19fffu - bitcast<u32>(a));
    return b * (-b * a + 2.0);
}

// FidelityFX RCAS 核心。(AMD FidelityFX，純量路徑)
fn FsrRcasF(b: vec3<f32>, d: vec3<f32>, e: vec3<f32>, f: vec3<f32>, h: vec3<f32>) -> vec3<f32> {
    // Luma×2
    let bL = b.b * 0.5 + (b.r * 0.5 + b.g);
    let dL = d.b * 0.5 + (d.r * 0.5 + d.g);
    let eL = e.b * 0.5 + (e.r * 0.5 + e.g);
    let fL = f.b * 0.5 + (f.r * 0.5 + f.g);
    let hL = h.b * 0.5 + (h.r * 0.5 + h.g);

    // 噪聲檢測
    var nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    nz = saturate(abs(nz) * APrxMedRcpF1(max(max(max(bL, dL), eL), max(fL, hL)) - min(min(min(bL, dL), eL), min(fL, hL))));
    nz = -0.5 * nz + 1.0;

    // 環帶（b,d,f,h）的 min/max
    let mn4R = min(min(min(b.r, d.r), f.r), h.r);
    let mn4G = min(min(min(b.g, d.g), f.g), h.g);
    let mn4B = min(min(min(b.b, d.b), f.b), h.b);
    let mx4R = max(max(max(b.r, d.r), f.r), h.r);
    let mx4G = max(max(max(b.g, d.g), f.g), h.g);
    let mx4B = max(max(max(b.b, d.b), f.b), h.b);
    // peak 範圍的立即常數：peakC = { 1.0, -1.0 * 4.0 }
    // 限制器（需要高精度 rcp）。
    // 退化鄰域保護：4·mx4 == 0（全黑）或 4·mn4 − 4 == 0（全白）時原碼同樣形成
    // 0·Inf = NaN，但 HLSL 的比較語意 min/max 會癒合；WGSL 依規格傳播 NaN，
    // 故以 select 顯式保護，結果與官方癒合結果一致（見檔頭 porting note）。
    let hitMinR = select(min(mn4R, e.r) * (1.0 / (4.0 * mx4R)), 0.0, mx4R == 0.0);
    let hitMinG = select(min(mn4G, e.g) * (1.0 / (4.0 * mx4G)), 0.0, mx4G == 0.0);
    let hitMinB = select(min(mn4B, e.b) * (1.0 / (4.0 * mx4B)), 0.0, mx4B == 0.0);
    let hitMaxR = select((1.0 - max(mx4R, e.r)) * (1.0 / (4.0 * mn4R + (-4.0))), 0.0, mn4R == 1.0);
    let hitMaxG = select((1.0 - max(mx4G, e.g)) * (1.0 / (4.0 * mn4G + (-4.0))), 0.0, mn4G == 1.0);
    let hitMaxB = select((1.0 - max(mx4B, e.b)) * (1.0 / (4.0 * mn4B + (-4.0))), 0.0, mn4B == 1.0);
    let lobeR = max(-hitMinR, hitMaxR);
    let lobeG = max(-hitMinG, hitMaxG);
    let lobeB = max(-hitMinB, hitMaxB);
    var lobe = max(-FSR_RCAS_LIMIT, min(max(max(lobeR, lobeG), lobeB), 0.0)) * uniforms.sharpness;

    // 套用噪聲去除
    lobe *= nz;

    // Resolve：需要中精度 rcp 近似以避免可見的色調變化
    let rcpL = APrxMedRcpF1(4.0 * lobe + 1.0);
    let res = vec3<f32>(
        (lobe * b.r + lobe * d.r + lobe * h.r + lobe * f.r + e.r) * rcpL,
        (lobe * b.g + lobe * d.g + lobe * h.g + lobe * f.g + e.g) * rcpL,
        (lobe * b.b + lobe * d.b + lobe * h.b + lobe * f.b + e.b) * rcpL
    );

    return res;
}

//! END
