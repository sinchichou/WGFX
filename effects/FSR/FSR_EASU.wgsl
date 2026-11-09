// FSR_EASU — Magpie v4 → WGFX WGSL port（全新移植）
// Ported from effects_src/FSR/FSR_EASU.hlsl（Magpie v4）
// 上游：AMD FidelityFX Super Resolution 1.0（ffx-fsr/ffx_fsr1.h，EASU 通道）
//   https://github.com/GPUOpen-Effects/FidelityFX-FSR
//   （a21ffb8f6c13233ba336352bdff293894c706575）
// 原授權 / 作者：Copyright (c) 2021 Advanced Micro Devices, Inc.
//   （MIT — AMD FidelityFX-FSR）；Magpie HLSL 移植：MIT（Blinue 與 Magpie 貢獻者）。
// 本移植採用純量（scalar，非 FP16 packed）路徑；EASU 數學與常數逐字保留。
//
// Porting notes:
//   - 粒度比照 Magpie 原效果庫：EASU（本檔，任意倍率放大）與 RCAS
//     （effects/FSR/FSR_RCAS.wgsl，1:1 銳化）為兩個獨立效果，由宿主串接。
//   - Magpie 原效果檔不宣告 OUTPUT 尺寸（倍率由 Magpie 設定決定）；WGFX 端
//     預設宣告 2x（INPUT_WIDTH * 2），宿主可以外部 OUTPUT 紋理或
//     WGFX.create 的 outWidth/outHeight 覆寫。EASU 核心以 scene.inputSize /
//     scene.outputSize 的**實際比例**計算 con0..con3，任意輸出尺寸（含 1:1）皆正確。
//   - WGSL 無 GatherRed/Green/Blue：原碼 4 次 gather 的採樣點都在整數紋理角點
//     （D3D 四點順序 w=左上、z=右上、x=左下、y=右下），以 textureLoad 逐 texel
//     還原為 bczz/ijfe/klhg/zzon 四組 vec4；邊界以 clamp load 對齊 Magpie 的
//     CLAMP sampler 位址模式。僅填原碼實際讀取的分量（其餘為 0 且永不讀取）。
//   - AMD ffx_a.h 的近似倒數/平方根（APrxLoRcpF1 / APrxLoRsqF1）以 bitcast
//     位元技巧逐字還原（0x7ef07ebb / 0x5f347d74）：對任意有限輸入皆為有限值
//     （rcp(0) ≈ 1.6e38，不產生 +Inf），移除精確 1/x 在 0·Inf 處的 NaN 路徑
//     （WGSL 的 min/max 依規格傳播 NaN，不像 HLSL 的比較語意會癒合）。
//     zro 保護（dirR < 1/32768 → dir=(1,0)、rsqrt 跳過）逐字保留。
//   - Magpie 為 BLOCK_SIZE 16 / 64 threads（每 thread 經 Rmp8x8 負責 2×2 像素）；
//     本移植 1 invocation 1 像素（//! NUM_THREADS 8,8,1），演算法不變。
//   - 原碼的 POINT sampler 只用於 gather；gather 改 textureLoad 後本檔不需 sampler。
//! VERSION 4
//! SORT_NAME FSR_EASU

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH * 2
//! HEIGHT INPUT_HEIGHT * 2
Texture2D OUTPUT;

//! PASS 1
//! DESC "FidelityFX EASU：邊緣自適應 12-tap 放大（任意倍率）"
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

    let color = FsrEasu(global_id.xy);
    textureStore(OUTPUT, global_id.xy, vec4<f32>(color, 1.0));
}

// 載入一個 INPUT texel，夾限在影像邊界（對齊 D3D CLAMP 位址模式的 gather 邊緣行為）。
fn EasuLoad(pos: vec2<i32>) -> vec3<f32> {
    let size = vec2<i32>(scene.inputSize);
    return textureLoad(INPUT, clamp(pos, vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

// 近似倒數（AMD ffx_a.h，逐字：單一 IADD 位元技巧）。
// 總函式：任何有限輸入都回傳有限值（rcp(0) ≈ 1.6e38），
// 消除精確 1/x 會產生的 Inf→NaN（0·Inf）路徑。
fn APrxLoRcpF1(a: f32) -> f32 {
    return bitcast<f32>(0x7ef07ebbu - bitcast<u32>(a));
}

// 近似倒數平方根（AMD ffx_a.h，逐字：IADD + shift）。
fn APrxLoRsqF1(a: f32) -> f32 {
    return bitcast<f32>(0x5f347d74u - (bitcast<u32>(a) >> 1u));
}

// 單一 tap 的濾波（純量路徑）。(AMD FidelityFX EASU)
fn FsrEasuTapF(
    aC: ptr<function, vec3<f32>>, // 累積顏色（含負 lobe）
    aW: ptr<function, f32>,       // 累積權重
    off: vec2<f32>,               // tap 相對 resolve 位置的偏移
    dir: vec2<f32>,               // 梯度方向
    len: vec2<f32>,               // 長度
    lob: f32,                     // 負 lobe 強度
    clp: f32,                     // 裁切點
    c: vec3<f32>,                 // tap 顏色
) {
    // 以方向旋轉偏移
    var v: vec2<f32>;
    v.x = (off.x * (dir.x)) + (off.y * dir.y);
    v.y = (off.x * (-dir.y)) + (off.y * dir.x);
    // 各向異性
    v *= len;
    // 距離²
    let d2 = min(v.x * v.x + v.y * v.y, clp);
    // 以無 sin()/rcp()/sqrt() 近似 lanczos2：
    //  (25/16 * (2/5 * x^2 - 1)^2 - (25/16 - 1)) * (1/4 * x^2 - 1)^2
    var wB = 2.0 / 5.0 * d2 - 1.0;
    var wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = 25.0 / 16.0 * wB - (25.0 / 16.0 - 1.0);
    let w = wB * wA;
    // 加權平均
    *aC += c * w;
    *aW += w;
}

// 累積方向與長度。(AMD FidelityFX EASU)
fn FsrEasuSetF(
    dir: ptr<function, vec2<f32>>,
    len: ptr<function, f32>,
    pp: vec2<f32>,
    biS: bool,
    biT: bool,
    biU: bool,
    biV: bool,
    lA: f32,
    lB: f32,
    lC: f32,
    lD: f32,
    lE: f32,
) {
    // 雙線性權重（謂詞為編譯期常數，分支會被折疊）
    //  s t
    //  u v
    var w = 0.0;
    if (biS) { w = (1.0 - pp.x) * (1.0 - pp.y); }
    if (biT) { w = pp.x * (1.0 - pp.y); }
    if (biU) { w = (1.0 - pp.x) * pp.y; }
    if (biV) { w = pp.x * pp.y; }
    // 方向是 '+' 差分：
    //    a
    //  b c d
    //    e
    let dc = lD - lC;
    let cb = lC - lB;
    var lenX = max(abs(dc), abs(cb));
    lenX = APrxLoRcpF1(lenX);
    let dirX = lD - lB;
    lenX = saturate(abs(dirX) * lenX);
    lenX *= lenX;
    // y 軸同理
    let ec = lE - lC;
    let ca = lC - lA;
    var lenY = max(abs(ec), abs(ca));
    lenY = APrxLoRcpF1(lenY);
    let dirY = lE - lA;
    lenY = saturate(abs(dirY) * lenY);
    lenY *= lenY;
    *dir += vec2<f32>(dirX * w, dirY * w);
    *len += lenX * w;
    *len += lenY * w;
}

// FidelityFX EASU 核心：12-tap 邊緣自適應放大。(AMD FidelityFX，純量路徑)
fn FsrEasu(ip: vec2<u32>) -> vec3<f32> {
    let inputSize = scene.inputSize;
    let outputSize = scene.outputSize;
    let inputPt = scene.inputPt;

    // 輸出整數位置 → 視窗像素位置（con0）
    let con0x = f32(inputSize.x) / f32(outputSize.x);
    let con0y = f32(inputSize.y) / f32(outputSize.y);
    var pp = vec2<f32>(
        f32(ip.x) * con0x + (0.5 * con0x - 0.5),
        f32(ip.y) * con0y + (0.5 * con0y - 0.5)
    );
    let fp = floor(pp);
    pp = pp - fp;

    // 12-tap 核。
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    // 原碼以 4 次 GatherRed/Green/Blue 讀取（採樣點在紋理角點，分量順序
    // x=左下 y=右下 z=右上 w=左上）：gather(0)@fp+(1,-1) → {b,c}、
    // gather(1)@fp+(0,1) → {i,j,f,e}、gather(2)@fp+(2,1) → {k,l,h,g}、
    // gather(3)@fp+(1,3) → {o,n}。以下逐 texel 還原（僅填原碼讀取的分量）。
    let b = EasuLoad(vec2<i32>(fp) + vec2<i32>(0, -1));
    let c = EasuLoad(vec2<i32>(fp) + vec2<i32>(1, -1));
    let e = EasuLoad(vec2<i32>(fp) + vec2<i32>(-1, 0));
    let f = EasuLoad(vec2<i32>(fp) + vec2<i32>(0, 0));
    let g = EasuLoad(vec2<i32>(fp) + vec2<i32>(1, 0));
    let h = EasuLoad(vec2<i32>(fp) + vec2<i32>(2, 0));
    let i = EasuLoad(vec2<i32>(fp) + vec2<i32>(-1, 1));
    let j = EasuLoad(vec2<i32>(fp) + vec2<i32>(0, 1));
    let k = EasuLoad(vec2<i32>(fp) + vec2<i32>(1, 1));
    let l = EasuLoad(vec2<i32>(fp) + vec2<i32>(2, 1));
    let n = EasuLoad(vec2<i32>(fp) + vec2<i32>(0, 2));
    let o = EasuLoad(vec2<i32>(fp) + vec2<i32>(1, 2));

    let bczzR = vec4<f32>(b.r, c.r, 0.0, 0.0);
    let bczzG = vec4<f32>(b.g, c.g, 0.0, 0.0);
    let bczzB = vec4<f32>(b.b, c.b, 0.0, 0.0);
    let ijfeR = vec4<f32>(i.r, j.r, f.r, e.r);
    let ijfeG = vec4<f32>(i.g, j.g, f.g, e.g);
    let ijfeB = vec4<f32>(i.b, j.b, f.b, e.b);
    let klhgR = vec4<f32>(k.r, l.r, h.r, g.r);
    let klhgG = vec4<f32>(k.g, l.g, h.g, g.g);
    let klhgB = vec4<f32>(k.b, l.b, h.b, g.b);
    let zzonR = vec4<f32>(0.0, 0.0, o.r, n.r);
    let zzonG = vec4<f32>(0.0, 0.0, o.g, n.g);
    let zzonB = vec4<f32>(0.0, 0.0, o.b, n.b);

    // 最簡多通道近似 luma（luma×2，2 個 FMA/MAD）
    let bczzL = bczzB * 0.5 + (bczzR * 0.5 + bczzG);
    let ijfeL = ijfeB * 0.5 + (ijfeR * 0.5 + ijfeG);
    let klhgL = klhgB * 0.5 + (klhgR * 0.5 + klhgG);
    let zzonL = zzonB * 0.5 + (zzonR * 0.5 + zzonG);
    // 重新命名
    let bL = bczzL.x;
    let cL = bczzL.y;
    let iL = ijfeL.x;
    let jL = ijfeL.y;
    let fL = ijfeL.z;
    let eL = ijfeL.w;
    let kL = klhgL.x;
    let lL = klhgL.y;
    let hL = klhgL.z;
    let gL = klhgL.w;
    let oL = zzonL.z;
    let nL = zzonL.w;
    // 雙線性插值累積
    var dir = vec2<f32>(0.0);
    var len = 0.0;
    FsrEasuSetF(&dir, &len, pp, true, false, false, false, bL, eL, fL, gL, jL);
    FsrEasuSetF(&dir, &len, pp, false, true, false, false, cL, fL, gL, hL, kL);
    FsrEasuSetF(&dir, &len, pp, false, false, true, false, fL, iL, jL, kL, nL);
    FsrEasuSetF(&dir, &len, pp, false, false, false, true, gL, jL, kL, lL, oL);
    // 以近似值正規化，並清理接近零的情況
    let dir2 = dir * dir;
    var dirR = dir2.x + dir2.y;
    let zro = dirR < 1.0 / 32768.0;
    dirR = select(APrxLoRsqF1(dirR), 1.0, zro);
    dir.x = select(dir.x, 1.0, zro);
    dir *= dirR;
    // {0..2} → {0..1}，再以平方整形
    len = len * 0.5;
    len *= len;
    // 拉伸核 {1.0 垂直|水平 → 對角 sqrt(2)}
    let stretch = (dir.x * dir.x + dir.y * dir.y) * APrxLoRcpF1(max(abs(dir.x), abs(dir.y)));
    // 旋轉後的各向異性長度
    //  x := 1.0 在邊緣 lerp 到 'stretch'
    //  y := 1.0 在邊緣 lerp 到 2x
    let len2 = vec2<f32>(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    // 依 'edge' 程度，窗口從 ±sqrt(2) 移到略超過 2.0
    let lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    // 距離² 裁切點設在可調窗口末端
    let clp = APrxLoRcpF1(lob);
    // 累積並與 4 近點的 min/max 混合（原碼的 dering）
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    let min4 = min(min(vec3<f32>(ijfeR.z, ijfeG.z, ijfeB.z), vec3<f32>(klhgR.w, klhgG.w, klhgB.w)),
                   min(vec3<f32>(ijfeR.y, ijfeG.y, ijfeB.y), vec3<f32>(klhgR.x, klhgG.x, klhgB.x)));
    let max4 = max(max(vec3<f32>(ijfeR.z, ijfeG.z, ijfeB.z), vec3<f32>(klhgR.w, klhgG.w, klhgB.w)),
                   max(vec3<f32>(ijfeR.y, ijfeG.y, ijfeB.y), vec3<f32>(klhgR.x, klhgG.x, klhgB.x)));
    // 累積
    var aC = vec3<f32>(0.0);
    var aW = 0.0;
    FsrEasuTapF(&aC, &aW, vec2<f32>(0.0, -1.0) - pp, dir, len2, lob, clp, vec3<f32>(bczzR.x, bczzG.x, bczzB.x)); // b
    FsrEasuTapF(&aC, &aW, vec2<f32>(1.0, -1.0) - pp, dir, len2, lob, clp, vec3<f32>(bczzR.y, bczzG.y, bczzB.y)); // c
    FsrEasuTapF(&aC, &aW, vec2<f32>(-1.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfeR.x, ijfeG.x, ijfeB.x)); // i
    FsrEasuTapF(&aC, &aW, vec2<f32>(0.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfeR.y, ijfeG.y, ijfeB.y)); // j
    FsrEasuTapF(&aC, &aW, vec2<f32>(0.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfeR.z, ijfeG.z, ijfeB.z)); // f
    FsrEasuTapF(&aC, &aW, vec2<f32>(-1.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfeR.w, ijfeG.w, ijfeB.w)); // e
    FsrEasuTapF(&aC, &aW, vec2<f32>(1.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(klhgR.x, klhgG.x, klhgB.x)); // k
    FsrEasuTapF(&aC, &aW, vec2<f32>(2.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(klhgR.y, klhgG.y, klhgB.y)); // l
    FsrEasuTapF(&aC, &aW, vec2<f32>(2.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(klhgR.z, klhgG.z, klhgB.z)); // h
    FsrEasuTapF(&aC, &aW, vec2<f32>(1.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(klhgR.w, klhgG.w, klhgB.w)); // g
    FsrEasuTapF(&aC, &aW, vec2<f32>(1.0, 2.0) - pp, dir, len2, lob, clp, vec3<f32>(zzonR.z, zzonG.z, zzonB.z)); // o
    FsrEasuTapF(&aC, &aW, vec2<f32>(0.0, 2.0) - pp, dir, len2, lob, clp, vec3<f32>(zzonR.w, zzonG.w, zzonB.w)); // n
    // 正規化與 dering（aW 由 AMD 核結構保證遠離 0；APrxLo* 均為總函式）
    return min(max4, max(min4, aC * APrxLoRcpF1(aW)));
}

//! END
