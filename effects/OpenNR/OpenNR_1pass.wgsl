// OpenNR → WGFX：完整 1pass f16 U-Net（37 pass：stage + global 分支 + encoder + middle
// + decoder + out + composite）。
//
// 語義正本：
//   effects_src/OpenNR-ReShade/upstream/src/opennr_shaders.hpp   （kConv/kPool/kUpCat/kGlobal/kStage/kComposite）
//   effects_src/OpenNR-ReShade/upstream/src/opennr_addon.cpp:2247-2346 （精確 pass 序列與 ping-pong 換手）
//   effects_src/OpenNR-ReShade/upstream/src/opennr_model.hpp     （層表 / buffer 尺寸）
//   docs/opennr-pass-graph.md                                     （本專案規格正本）
//
// 設計：
//   * 全部中間特徵圖與權重皆為 storage buffer（uint2 = 4 個 packed f16；bias 為 float4 = f32）。
//   * 網路維度 NW×NH 由 external defines 提供（注入為 WGSL const）。通道數為模型常數（字面）。
//   * 每個 pass 只引用它需要的 buffer；產生器會依 pass 實際引用裁剪宣告。
//   * ping-pong 以明確 A/B 命名表達（WGFX 無執行期別名），換手序列與 addon.cpp 的 swap 完全一致。
//   * helper 以 `ptr<storage, array<...>, read_write>` 收 buffer，避免 helper 內直接引用 buffer 名
//     （否則每個呼叫該 helper 的 pass 都會被迫宣告所有被引用的 buffer）。所有 buffer 一律
//     MODE read_write 以統一 helper 簽名。
//! VERSION 4

//! SORT_NAME OpenNR 1pass
//! CAPABILITY FP16

// ============================ 參數（composite / stage） ============================
//! PARAMETER Strength
//! TYPE int
//! LABEL "Strength"
//! MIN 0
//! MAX 100
//! DEFAULT 100
//! STEP 1
int Strength;

//! PARAMETER Style
//! TYPE int
//! LABEL "Colour Style (0 Standard / 1 Natural / 2 Cinematic)"
//! MIN 0
//! MAX 2
//! DEFAULT 0
//! STEP 1
int Style;

//! PARAMETER Tone
//! TYPE int
//! LABEL "Local Tone Strength"
//! MIN 0
//! MAX 100
//! DEFAULT 100
//! STEP 1
int Tone;

//! PARAMETER Structure
//! TYPE int
//! LABEL "Local Structure Strength"
//! MIN 0
//! MAX 100
//! DEFAULT 100
//! STEP 1
int Structure;

//! PARAMETER Detail
//! TYPE int
//! LABEL "Detail Strength"
//! MIN 0
//! MAX 100
//! DEFAULT 100
//! STEP 1
int Detail;

//! PARAMETER Colour
//! TYPE int
//! LABEL "Colour Strength"
//! MIN 0
//! MAX 100
//! DEFAULT 100
//! STEP 1
int Colour;

//! PARAMETER Apply
//! TYPE int
//! LABEL "Apply (0 = passthrough)"
//! MIN 0
//! MAX 1
//! DEFAULT 1
//! STEP 1
int Apply;

// ============================ 紋理 ============================
//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D OUTPUT;

// ============================ Storage buffer 宣告 ============================
// 特徵圖 / 跳接 / 輸出 / 全幀分支 bias。
//! BUFFER FB_CH
//! MODE read_write
//! TYPE u32x2
//! BUFFER FB_A
//! MODE read_write
//! TYPE u32x2
//! BUFFER FB_B
//! MODE read_write
//! TYPE u32x2
//! BUFFER FB_S0
//! MODE read_write
//! TYPE u32x2
//! BUFFER FB_S1
//! MODE read_write
//! TYPE u32x2
//! BUFFER FB_S2
//! MODE read_write
//! TYPE u32x2
//! BUFFER FB_S3
//! MODE read_write
//! TYPE u32x2
//! BUFFER FB_HEAD
//! MODE read_write
//! TYPE u32x2
//! BUFFER GB
//! MODE read_write
//! TYPE f32x4
//! BUFFER WFC
//! MODE read_write
//! TYPE f32

// 逐層權重（w = packed f16 uint2；b = f32 float4）
//! BUFFER WT_enc0a
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_enc0a
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_enc0b
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_enc0b
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_enc1a
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_enc1a
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_enc1b
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_enc1b
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_enc2a
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_enc2a
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_enc2b
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_enc2b
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_enc3a
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_enc3a
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_enc3b
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_enc3b
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_mid_a
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_mid_a
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_mid_b
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_mid_b
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_dec3a
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_dec3a
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_dec3b
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_dec3b
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_dec2a
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_dec2a
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_dec2b
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_dec2b
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_dec1a
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_dec1a
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_dec1b
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_dec1b
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_dec0a
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_dec0a
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_dec0b
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_dec0b
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_out
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_out
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_g0
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_g0
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_g1
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_g1
//! MODE read_write
//! TYPE f32x4
//! BUFFER WT_g2
//! MODE read_write
//! TYPE u32x2
//! BUFFER BS_g2
//! MODE read_write
//! TYPE f32x4

// ============================ COMMON ============================
//! COMMON
const TM: u32 = 4u;

// ---- f16 ↔ f32（bit-level，與上游 f16tof32/f32tof16 一致；不依賴 shader-f16）----
fn f16tof32(h: u32) -> f32 {
    let s = select(1.0, -1.0, ((h >> 15u) & 1u) == 1u);
    let e = (h >> 10u) & 31u;
    let m = h & 1023u;
    if (e == 0u) {
        return s * f32(m) * 5.9604644775390625e-8; // 2^-24（subnormal）
    }
    if (e == 31u) {
        // bitcast 帶入 runtime 位元，避免 Naga 對 NaN 常數的折疊拒絕
        let nanBits = 0x7fc00000u | (m & 1023u);
        return select(s * 3.4028234663852886e38, bitcast<f32>(nanBits), m != 0u);
    }
    return s * exp2(f32(e) - 15.0) * (1.0 + f32(m) / 1024.0);
}

fn f32tof16(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let sign = (bits >> 16u) & 0x8000u;
    let exp = (bits >> 23u) & 0xffu;
    let man = bits & 0x7fffffu;
    if (exp == 255u) {
        return sign | 0x7c00u | select(0u, 0x200u, man != 0u);
    }
    var e = i32(exp) - 112;
    if (e >= 31) {
        return sign | 0x7c00u;
    }
    if (e >= 1) {
        var m10 = (man + 0x0fffu + ((man >> 13u) & 1u)) >> 13u;
        if (m10 > 0x3ffu) {
            m10 = 0u;
            e = e + 1;
            if (e >= 31) {
                return sign | 0x7c00u;
            }
        }
        return sign | (u32(e) << 10u) | m10;
    }
    if (exp == 0u) {
        return sign;
    }
    let full = man | 0x800000u;
    let sh = u32(126 - i32(exp));
    if (sh > 34u) {
        return sign;
    }
    let q = full >> sh;
    let rem = full & ((1u << sh) - 1u);
    let halfB = 1u << (sh - 1u);
    let m10 = q + select(0u, 1u, (rem > halfB) || ((rem == halfB) && ((q & 1u) == 1u)));
    if (m10 >= 0x400u) {
        return sign | 0x400u;
    }
    return sign | m10;
}

fn h4(v: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(
        f16tof32(v.x & 0xffffu),
        f16tof32(v.x >> 16u),
        f16tof32(v.y & 0xffffu),
        f16tof32(v.y >> 16u)
    );
}

fn p4(v: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(f32tof16(v.x) | (f32tof16(v.y) << 16u),
                     f32tof16(v.z) | (f32tof16(v.w) << 16u));
}

// GELU（tanh 形式；clamp 防 tanh 指數溢位成 Inf/Inf = NaN）
fn gelu(v: vec4<f32>) -> vec4<f32> {
    let t = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    return 0.5 * v * (1.0 + tanh(clamp(t, vec4<f32>(-10.0), vec4<f32>(10.0))));
}

fn gelu1(x: f32) -> f32 {
    let t = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(clamp(t, -10.0, 10.0)));
}

fn mix32(v: u32) -> u32 {
    var x = v;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn u01(v: u32) -> f32 {
    return f32(v & 0x00ffffffu) / 16777216.0 + 1.0 / 33554432.0;
}

// ---- 3x3 conv + bias + 可選 GELU（kConvHLSL；TM=4、f32 累加、cin/cout4 由呼叫端提供）----
fn conv3x3(
    x: ptr<storage, array<vec2<u32>>, read_write>,
    w: ptr<storage, array<vec2<u32>>, read_write>,
    b: ptr<storage, array<vec4<f32>>, read_write>,
    o: ptr<storage, array<vec2<u32>>, read_write>,
    W: u32, npix: u32, cin: u32, cout4: u32, act: u32, zeroPad: u32,
    gid: vec3<u32>
) {
    let p0 = (gid.x / cout4) * TM;
    let n4 = gid.x % cout4;
    if (p0 >= npix) {
        return;
    }
    let cin4 = cin / 4u;
    var acc: array<vec4<f32>, 4>;
    for (var z: u32 = 0u; z < TM; z++) {
        acc[z] = vec4<f32>(0.0);
    }
    for (var tap: u32 = 0u; tap < 9u; tap++) {
        let dy = i32(tap / 3u) - 1;
        let dx = i32(tap % 3u) - 1;
        var qm: array<u32, 4>;
        var ok: array<bool, 4>;
        for (var m: u32 = 0u; m < TM; m++) {
            let pm = i32(p0) + i32(m);
            if (zeroPad != 0u) {
                let px = (pm % i32(W)) + dx;
                let py = (pm / i32(W)) + dy;
                ok[m] = (px >= 0) && (py >= 0) && (px < i32(W)) && (py < i32(npix / W));
                qm[m] = select(0u, u32(py * i32(W) + px), ok[m]);
            } else {
                ok[m] = true;
                qm[m] = u32(clamp(pm + dy * i32(W) + dx, 0, i32(npix) - 1));
            }
        }
        for (var k4: u32 = 0u; k4 < cin4; k4++) {
            var xv: array<vec4<f32>, 4>;
            for (var m: u32 = 0u; m < TM; m++) {
                xv[m] = select(vec4<f32>(0.0), h4((*x)[qm[m] * cin4 + k4]), ok[m]);
            }
            for (var kk: u32 = 0u; kk < 4u; kk++) {
                let wv = h4((*w)[(tap * cin + k4 * 4u + kk) * cout4 + n4]);
                for (var a: u32 = 0u; a < TM; a++) {
                    acc[a] += xv[a][kk] * wv;
                }
            }
        }
    }
    let bv = (*b)[n4];
    for (var i: u32 = 0u; i < TM; i++) {
        let p = p0 + i;
        if (p < npix) {
            let v = acc[i] + bv;
            (*o)[p * cout4 + n4] = select(p4(v), p4(gelu(v)), act != 0u);
        }
    }
}

// ---- 2x2 平均池化（kPoolHLSL）----
fn pool2x2(
    x: ptr<storage, array<vec2<u32>>, read_write>,
    o: ptr<storage, array<vec2<u32>>, read_write>,
    W: u32, H: u32, cin: u32, gid: vec3<u32>
) {
    let cin4 = cin / 4u;
    let ow = W / 2u;
    let oh = H / 2u;
    let n = ow * oh * cin4;
    if (gid.x >= n) {
        return;
    }
    let c = gid.x % cin4;
    let p = gid.x / cin4;
    let y = (p / ow) * 2u;
    let xx = (p % ow) * 2u;
    let a = (y * W + xx) * cin4 + c;
    (*o)[gid.x] = p4(0.25 * (h4((*x)[a]) + h4((*x)[a + cin4]) +
                             h4((*x)[a + W * cin4]) + h4((*x)[a + W * cin4 + cin4])));
}

// ---- nearest 上採樣 + skip 串接（kUpCatHLSL）；W/H 為輸出層尺寸 ----
fn upcat(
    lo: ptr<storage, array<vec2<u32>>, read_write>,
    sk: ptr<storage, array<vec2<u32>>, read_write>,
    o: ptr<storage, array<vec2<u32>>, read_write>,
    W: u32, H: u32, clo: u32, cs: u32, scale: u32, gid: vec3<u32>
) {
    let clo4 = clo / 4u;
    let cs4 = cs / 4u;
    let cout4 = clo4 + cs4;
    let n = W * H * cout4;
    if (gid.x >= n) {
        return;
    }
    let c = gid.x % cout4;
    let p = gid.x / cout4;
    let y = p / W;
    let xx = p % W;
    if (c < clo4) {
        (*o)[gid.x] = (*lo)[((y / scale) * (W / scale) + xx / scale) * clo4 + c];
    } else {
        (*o)[gid.x] = (*sk)[p * cs4 + (c - clo4)];
    }
}

// ---- 全幀分支尾段（kGlobalHLSL）：空間均值 → fc1 → gelu → fc2 → + mid_a bias ----
// 單執行緒版本（1 workgroup、128 threads，僅 thread 0 執行）；gC/gH 皆 ≤ 64（見 model.hpp 檢查）。
fn globalbias(
    f: ptr<storage, array<vec2<u32>>, read_write>,
    fc: ptr<storage, array<f32>, read_write>,
    mb: ptr<storage, array<vec4<f32>>, read_write>,
    ob: ptr<storage, array<vec4<f32>>, read_write>,
    gN: u32, gC: u32, gH: u32, gO: u32
) {
    let c4 = gC / 4u;
    var m: array<f32, 64>;
    for (var t: u32 = 0u; t < gC; t++) {
        var s = 0.0;
        for (var i: u32 = 0u; i < gN; i++) {
            s += h4((*f)[i * c4 + t / 4u])[t % 4u];
        }
        m[t] = s / f32(gN);
    }
    var hid: array<f32, 64>;
    for (var t: u32 = 0u; t < gH; t++) {
        var a = (*fc)[gH * gC + t];
        for (var c: u32 = 0u; c < gC; c++) {
            a += (*fc)[t * gC + c] * m[c];
        }
        hid[t] = gelu1(a);
    }
    let w2 = gH * gC + gH;
    let b2 = w2 + gO * gH;
    for (var t: u32 = 0u; t < gO / 4u; t++) {
        var r = (*mb)[t];
        for (var j: u32 = 0u; j < 4u; j++) {
            let oo = t * 4u + j;
            var a = (*fc)[b2 + oo];
            for (var i: u32 = 0u; i < gH; i++) {
                a += (*fc)[w2 + oo * gH + i] * hid[i];
            }
            r[j] += a;
        }
        (*ob)[t] = r;
    }
}

// ---- kStageHLSL：16 輸入通道（種子噪聲 + 降採樣色彩 + 設定）----
fn stage(ch: ptr<storage, array<vec2<u32>>, read_write>, gid: vec3<u32>) {
    let gW = NW;
    let gH = NH;
    if (gid.x >= gW || gid.y >= gH) {
        return;
    }
    let p = gid.y * gW + gid.x;
    let s = mix32((gid.x * 0x9e3779b9u) ^ (gid.y * 0x85ebca6bu) ^ (scene.frame * 0xc2b2ae35u));
    let u0 = u01(mix32(s + 1u));
    let u1 = u01(mix32(s + 2u));
    let u2 = u01(mix32(s + 3u));
    let u3 = u01(mix32(s + 4u));
    let r0 = sqrt(-2.0 * log(u0));
    let r1 = sqrt(-2.0 * log(u1));
    let TAU = 6.283185307179586;

    let os = scene.inputSize;
    var c: vec3<f32>;
    if (gW == os.x && gH == os.y) {
        c = textureLoad(INPUT, vec2<i32>(gid.xy), 0).rgb;
    } else {
        let fx = (f32(gid.x) + 0.5) * (f32(os.x) / f32(gW)) - 0.5;
        let fy = (f32(gid.y) + 0.5) * (f32(os.y) / f32(gH)) - 0.5;
        let x0 = i32(floor(fx));
        let y0 = i32(floor(fy));
        let wx = fx - f32(x0);
        let wy = fy - f32(y0);
        let x1 = clamp(x0 + 1, 0, i32(os.x) - 1);
        let y1 = clamp(y0 + 1, 0, i32(os.y) - 1);
        let xc = clamp(x0, 0, i32(os.x) - 1);
        let yc = clamp(y0, 0, i32(os.y) - 1);
        c = mix(mix(textureLoad(INPUT, vec2<i32>(xc, yc), 0).rgb,
                    textureLoad(INPUT, vec2<i32>(x1, yc), 0).rgb, wx),
                mix(textureLoad(INPUT, vec2<i32>(xc, y1), 0).rgb,
                    textureLoad(INPUT, vec2<i32>(x1, y1), 0).rgb, wx), wy);
    }
    let v = (c - vec3<f32>(0.5)) * 2.0 * (1.0 / 16.0);

    let tone = f32(uniforms.Tone) / 100.0;
    let structure = f32(uniforms.Structure) / 100.0;

    (*ch)[p * 4u + 0u] = p4(vec4<f32>(r1 * cos(TAU * u3), r1 * sin(TAU * u3), r0 * cos(TAU * u2), 1.0));
    (*ch)[p * 4u + 1u] = p4(vec4<f32>(v.r, v.g, v.b, v.r));
    (*ch)[p * 4u + 2u] = p4(vec4<f32>(v.g, v.b, 0.0, tone));
    (*ch)[p * 4u + 3u] = p4(vec4<f32>(structure, -1.0, -1.0, 0.0));
}

// ============================ Composite（kCompositeHLSL，SDR 路徑）============================
fn fetchhead(x: u32, y: u32) -> vec3<f32> {
    // FB_HEAD 是模組層級 storage var（非指標），直接索引。
    return h4(FB_HEAD[(y * NW + x) * 4u]).rgb;
}

fn head_at(pos: vec2<f32>, ow: u32, oh: u32) -> vec3<f32> {
    if (NW == ow && NH == oh) {
        let x = min(u32(pos.x), NW - 1u);
        let y = min(u32(pos.y), NH - 1u);
        return fetchhead(x, y);
    }
    let fx = pos.x * (f32(NW) / f32(ow)) - 0.5;
    let fy = pos.y * (f32(NH) / f32(oh)) - 0.5;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let wx = fx - f32(x0);
    let wy = fy - f32(y0);
    let xc = u32(clamp(x0, 0, i32(NW) - 1));
    let x1 = u32(clamp(x0 + 1, 0, i32(NW) - 1));
    let yc = u32(clamp(y0, 0, i32(NH) - 1));
    let y1 = u32(clamp(y0 + 1, 0, i32(NH) - 1));
    return mix(mix(fetchhead(xc, yc), fetchhead(x1, yc), wx),
               mix(fetchhead(xc, y1), fetchhead(x1, y1), wx), wy);
}

fn grade_smooth(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

fn grade_hue(p: f32, q: f32, t: f32) -> f32 {
    var tt = t;
    if (tt < 0.0) { tt += 1.0; }
    if (tt > 1.0) { tt -= 1.0; }
    if (tt < 1.0 / 6.0) { return p + (q - p) * 6.0 * tt; }
    if (tt < 0.5) { return q; }
    if (tt < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - tt) * 6.0; }
    return p;
}

fn grade_rgb2hsl(c: vec3<f32>) -> vec3<f32> {
    let mx = max(max(c.r, c.g), c.b);
    let mn = min(min(c.r, c.g), c.b);
    let L = (mx + mn) * 0.5;
    var H = 0.0;
    var S = 0.0;
    if (mx > mn) {
        let d = mx - mn;
        S = select(d / (mx + mn), d / (2.0 - mx - mn), L > 0.5);
        if (mx == c.r) {
            H = ((c.g - c.b) / d + select(6.0, 0.0, c.g >= c.b)) / 6.0;
        } else if (mx == c.g) {
            H = ((c.b - c.r) / d + 2.0) / 6.0;
        } else {
            H = ((c.r - c.g) / d + 4.0) / 6.0;
        }
    }
    return vec3<f32>(H, S, L);
}

fn grade_hsl2rgb(hsl: vec3<f32>) -> vec3<f32> {
    let H = hsl.x;
    let S = hsl.y;
    let L = hsl.z;
    if (!(S > 0.0)) {
        return vec3<f32>(L, L, L);
    }
    let q = select(L * (1.0 + S), L + S - L * S, L >= 0.5);
    let p = 2.0 * L - q;
    return vec3<f32>(grade_hue(p, q, H + 1.0 / 3.0), grade_hue(p, q, H), grade_hue(p, q, H - 1.0 / 3.0));
}

fn grade_tint_target(c: vec3<f32>, temperature: bool, positive: bool) -> vec3<f32> {
    let L = (max(max(c.r, c.g), c.b) + min(min(c.r, c.g), c.b)) * 0.5;
    let q = select(2.0 * L, 1.0, L >= 0.5);
    let p = 2.0 * L - q;
    if (temperature) {
        if (positive) {
            return vec3<f32>(q, p + (q - p) * 6.0 * 0.061110001057386398315, p);
        }
        return vec3<f32>(p, p + (q - p) * 6.0 * 0.10555666685104370117, q);
    }
    if (positive) {
        return vec3<f32>(p + (q - p) * 6.0 * 0.022223353385925292969, q, p);
    }
    return vec3<f32>(p + (q - p) * 6.0 * 0.14444339275360107422, p, q);
}

// 風格欄位（kGradeField）：Standard 基準 + Natural/Cinematic 偏移，依 Local Tone 由基準縮放。
fn grade(n: vec3<f32>, style: u32, tone: f32) -> vec3<f32> {
    var f: array<f32, 14>;
    for (var i: u32 = 0u; i < 14u; i++) {
        f[i] = 0.0;
    }
    f[1] = 1.0;
    if (style == 1u) {
        f[2] = -0.1 * tone;
        f[4] = -0.25 * tone;
        f[5] = -0.1 * tone;
    } else if (style >= 2u) {
        f[5] = -0.15 * tone;
    }

    var c = saturate(n);
    c = saturate((c - f[0]) * (1.0 / (f[1] - f[0] + 1e-10)));
    if (abs(f[7]) >= 1e-6) {
        c = saturate(c + (grade_tint_target(c, true, f[7] > 0.0) - c) * abs(f[7]));
    }
    if (abs(f[8]) >= 1e-6) {
        c = saturate(c + (grade_tint_target(c, false, f[8] > 0.0) - c) * abs(f[8]));
    }
    c = saturate(c * exp2(f[2]));
    c = saturate(c + (c * c * (3.0 - 2.0 * c) - c) * f[4]);
    var g3: vec3<f32>;
    for (var k: u32 = 0u; k < 3u; k++) {
        let v = c[k];
        let b0 = grade_smooth(saturate(1.0 - 4.0 * v));
        let b1 = grade_smooth(saturate(4.0 * v)) * grade_smooth(saturate(2.0 - 4.0 * v));
        let b2 = grade_smooth(saturate(4.0 * v - 1.0)) * grade_smooth(saturate(3.0 - 4.0 * v));
        let b3 = grade_smooth(saturate(4.0 * v - 2.0)) * grade_smooth(saturate(4.0 - 4.0 * v));
        let b4 = grade_smooth(saturate(4.0 * v - 3.0));
        g3[k] = b0 * f[9] + b1 * f[10] + b2 * f[11] + b3 * f[12] + b4 * f[13];
    }
    c = exp2(exp2(-g3) * log2(c));
    c = exp2(exp2(-f[3]) * log2(max(c, vec3<f32>(0.0))));
    var hsl = grade_rgb2hsl(c);
    hsl.y = saturate(hsl.y * (1.0 + f[5]));
    c = grade_hsl2rgb(hsl);
    hsl = grade_rgb2hsl(c);
    hsl.y = saturate(exp2(exp2(-f[6]) * log2(hsl.y)));
    return saturate(grade_hsl2rgb(hsl));
}

// Standard: s + correction。Natural/Cinematic: 修正後的畫面經 grade 再依 Strength blend。
// 之後 Detail 縮放整體、Colour 縮放非亮度部分。
fn correction(s: vec3<f32>, hv: vec3<f32>) -> vec3<f32> {
    let strength = f32(uniforms.Strength) / 100.0;
    var corr = 0.25 * hv * strength;
    let style = min(u32(uniforms.Style), 2u);
    if (style != 0u) {
        let tone = f32(min(uniforms.Tone, 100)) / 100.0;
        corr = (grade(saturate(s + 0.25 * hv), style, tone) - s) * strength;
    }
    let Y = dot(corr, vec3<f32>(0.2126, 0.7152, 0.0722));
    let detail = f32(uniforms.Detail) / 100.0;
    let colour = f32(uniforms.Colour) / 100.0;
    return detail * (Y + colour * (corr - Y));
}
//! END

// ============================================================
// PASS 1 — stage：INPUT（frame）→ FB_CH（16ch @ NW×NH）
// ============================================================
//! PASS 1
//! DESC "stage: 16 input channels (noise + colour + settings)"
//! IN INPUT
//! DISPATCH (NW+7)/8, (NH+7)/8
//! NUM_THREADS 8,8,1
fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    stage(&FB_CH, gid);
}

// ============================================================
// 全幀分支（global）：3x pool → g0 → pool → g1 → pool → g2 → globalbias → GB
// ============================================================
//! PASS 2
//! DESC "global pool 1"
//! DISPATCH ((NW>>1)*(NH>>1)*4+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass2(@builtin(global_invocation_id) gid: vec3<u32>) {
    pool2x2(&FB_CH, &FB_B, NW, NH, 16u, gid);
}

//! PASS 3
//! DESC "global pool 2"
//! DISPATCH ((NW>>2)*(NH>>2)*4+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass3(@builtin(global_invocation_id) gid: vec3<u32>) {
    pool2x2(&FB_B, &FB_A, NW >> 1u, NH >> 1u, 16u, gid);
}

//! PASS 4
//! DESC "global pool 3"
//! DISPATCH ((NW>>3)*(NH>>3)*4+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass4(@builtin(global_invocation_id) gid: vec3<u32>) {
    pool2x2(&FB_A, &FB_B, NW >> 2u, NH >> 2u, 16u, gid);
}

//! PASS 5
//! DESC "global conv g0 (cin16 cout32) zeroPad"
//! DISPATCH (((NW>>3)*(NH>>3)/4)*8+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass5(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_g0, &BS_g0, &FB_A, NW >> 3u, (NW >> 3u) * (NH >> 3u), 16u, 8u, 1u, 1u, gid);
}

//! PASS 6
//! DESC "global pool g0->g1"
//! DISPATCH ((NW>>4)*(NH>>4)*8+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass6(@builtin(global_invocation_id) gid: vec3<u32>) {
    pool2x2(&FB_A, &FB_B, NW >> 3u, NH >> 3u, 32u, gid);
}

//! PASS 7
//! DESC "global conv g1 (cin32 cout64) zeroPad"
//! DISPATCH (((NW>>4)*(NH>>4)/4)*16+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass7(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_g1, &BS_g1, &FB_A, NW >> 4u, (NW >> 4u) * (NH >> 4u), 32u, 16u, 1u, 1u, gid);
}

//! PASS 8
//! DESC "global pool g1->g2"
//! DISPATCH ((NW>>5)*(NH>>5)*16+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass8(@builtin(global_invocation_id) gid: vec3<u32>) {
    pool2x2(&FB_A, &FB_B, NW >> 4u, NH >> 4u, 64u, gid);
}

//! PASS 9
//! DESC "global conv g2 (cin64 cout64) zeroPad"
//! DISPATCH (((NW>>5)*(NH>>5)/4)*16+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass9(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_g2, &BS_g2, &FB_A, NW >> 5u, (NW >> 5u) * (NH >> 5u), 64u, 16u, 1u, 1u, gid);
}

//! PASS 10
//! DESC "global head: mean -> fc1 -> gelu -> fc2 -> +mid_a bias -> GB"
//! DISPATCH 1, 1
//! NUM_THREADS 128,1,1
fn Pass10(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u) {
        globalbias(&FB_A, &WFC, &BS_mid_a, &GB, (NW >> 5u) * (NH >> 5u), 64u, 64u, 128u);
    }
}

// ============================================================
// Encoder
// ============================================================
//! PASS 11
//! DESC "enc0a CH->B (cin16 pad16)"
//! DISPATCH (NW*NH+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass11(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_CH, &WT_enc0a, &BS_enc0a, &FB_B, NW, NW * NH, 16u, 4u, 1u, 0u, gid);
}

//! PASS 12
//! DESC "enc0b B->S0"
//! DISPATCH (NW*NH+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass12(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_enc0b, &BS_enc0b, &FB_S0, NW, NW * NH, 16u, 4u, 1u, 0u, gid);
}

//! PASS 13
//! DESC "pool S0->A"
//! DISPATCH ((NW>>1)*(NH>>1)*4+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass13(@builtin(global_invocation_id) gid: vec3<u32>) {
    pool2x2(&FB_S0, &FB_A, NW, NH, 16u, gid);
}

//! PASS 14
//! DESC "enc1a A->B (cin16 cout32)"
//! DISPATCH (((NW>>1)*(NH>>1)/4)*8+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass14(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_A, &WT_enc1a, &BS_enc1a, &FB_B, NW >> 1u, (NW >> 1u) * (NH >> 1u), 16u, 8u, 1u, 0u, gid);
}

//! PASS 15
//! DESC "enc1b B->S1"
//! DISPATCH (((NW>>1)*(NH>>1)/4)*8+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass15(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_enc1b, &BS_enc1b, &FB_S1, NW >> 1u, (NW >> 1u) * (NH >> 1u), 32u, 8u, 1u, 0u, gid);
}

//! PASS 16
//! DESC "pool S1->A"
//! DISPATCH ((NW>>2)*(NH>>2)*8+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass16(@builtin(global_invocation_id) gid: vec3<u32>) {
    pool2x2(&FB_S1, &FB_A, NW >> 1u, NH >> 1u, 32u, gid);
}

//! PASS 17
//! DESC "enc2a A->B (cin32 cout64)"
//! DISPATCH (((NW>>2)*(NH>>2)/4)*16+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass17(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_A, &WT_enc2a, &BS_enc2a, &FB_B, NW >> 2u, (NW >> 2u) * (NH >> 2u), 32u, 16u, 1u, 0u, gid);
}

//! PASS 18
//! DESC "enc2b B->S2"
//! DISPATCH (((NW>>2)*(NH>>2)/4)*16+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass18(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_enc2b, &BS_enc2b, &FB_S2, NW >> 2u, (NW >> 2u) * (NH >> 2u), 64u, 16u, 1u, 0u, gid);
}

//! PASS 19
//! DESC "pool S2->A"
//! DISPATCH ((NW>>3)*(NH>>3)*16+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass19(@builtin(global_invocation_id) gid: vec3<u32>) {
    pool2x2(&FB_S2, &FB_A, NW >> 2u, NH >> 2u, 64u, gid);
}

//! PASS 20
//! DESC "enc3a A->B (cin64 cout128)"
//! DISPATCH (((NW>>3)*(NH>>3)/4)*32+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass20(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_A, &WT_enc3a, &BS_enc3a, &FB_B, NW >> 3u, (NW >> 3u) * (NH >> 3u), 64u, 32u, 1u, 0u, gid);
}

//! PASS 21
//! DESC "enc3b B->S3"
//! DISPATCH (((NW>>3)*(NH>>3)/4)*32+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass21(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_enc3b, &BS_enc3b, &FB_S3, NW >> 3u, (NW >> 3u) * (NH >> 3u), 128u, 32u, 1u, 0u, gid);
}

// ============================================================
// Middle（mid_a 吃 GB）
// ============================================================
//! PASS 22
//! DESC "mid_a S3->A (bias=GB)"
//! DISPATCH (((NW>>3)*(NH>>3)/4)*32+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass22(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_S3, &WT_mid_a, &GB, &FB_A, NW >> 3u, (NW >> 3u) * (NH >> 3u), 128u, 32u, 1u, 0u, gid);
}

//! PASS 23
//! DESC "mid_b A->B"
//! DISPATCH (((NW>>3)*(NH>>3)/4)*32+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass23(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_A, &WT_mid_b, &BS_mid_b, &FB_B, NW >> 3u, (NW >> 3u) * (NH >> 3u), 128u, 32u, 1u, 0u, gid);
}

// ============================================================
// Decoder
// ============================================================
//! PASS 24
//! DESC "upcat L3 B+S3->A (scale1, cin256)"
//! DISPATCH ((NW>>3)*(NH>>3)*256+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass24(@builtin(global_invocation_id) gid: vec3<u32>) {
    upcat(&FB_B, &FB_S3, &FB_A, NW >> 3u, NH >> 3u, 128u, 128u, 1u, gid);
}

//! PASS 25
//! DESC "dec3a A->B (cin256 cout128)"
//! DISPATCH (((NW>>3)*(NH>>3)/4)*32+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass25(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_A, &WT_dec3a, &BS_dec3a, &FB_B, NW >> 3u, (NW >> 3u) * (NH >> 3u), 256u, 32u, 1u, 0u, gid);
}

//! PASS 26
//! DESC "dec3b B->A"
//! DISPATCH (((NW>>3)*(NH>>3)/4)*32+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass26(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_dec3b, &BS_dec3b, &FB_A, NW >> 3u, (NW >> 3u) * (NH >> 3u), 128u, 32u, 1u, 0u, gid);
}

//! PASS 27
//! DESC "upcat L2 A+S2->B (scale2, cin128)"
//! DISPATCH ((NW>>2)*(NH>>2)*128+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass27(@builtin(global_invocation_id) gid: vec3<u32>) {
    upcat(&FB_A, &FB_S2, &FB_B, NW >> 2u, NH >> 2u, 128u, 64u, 2u, gid);
}

//! PASS 28
//! DESC "dec2a B->A (cin192 cout64)"
//! DISPATCH (((NW>>2)*(NH>>2)/4)*16+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass28(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_dec2a, &BS_dec2a, &FB_A, NW >> 2u, (NW >> 2u) * (NH >> 2u), 192u, 16u, 1u, 0u, gid);
}

//! PASS 29
//! DESC "dec2b A->B"
//! DISPATCH (((NW>>2)*(NH>>2)/4)*16+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass29(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_A, &WT_dec2b, &BS_dec2b, &FB_B, NW >> 2u, (NW >> 2u) * (NH >> 2u), 64u, 16u, 1u, 0u, gid);
}

//! PASS 30
//! DESC "upcat L1 B+S1->A (scale2, cin96)"
//! DISPATCH ((NW>>1)*(NH>>1)*96+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass30(@builtin(global_invocation_id) gid: vec3<u32>) {
    upcat(&FB_B, &FB_S1, &FB_A, NW >> 1u, NH >> 1u, 64u, 32u, 2u, gid);
}

//! PASS 31
//! DESC "dec1a A->B (cin96 cout32)"
//! DISPATCH (((NW>>1)*(NH>>1)/4)*8+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass31(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_A, &WT_dec1a, &BS_dec1a, &FB_B, NW >> 1u, (NW >> 1u) * (NH >> 1u), 96u, 8u, 1u, 0u, gid);
}

//! PASS 32
//! DESC "dec1b B->A"
//! DISPATCH (((NW>>1)*(NH>>1)/4)*8+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass32(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_dec1b, &BS_dec1b, &FB_A, NW >> 1u, (NW >> 1u) * (NH >> 1u), 32u, 8u, 1u, 0u, gid);
}

//! PASS 33
//! DESC "upcat L0 A+S0->B (scale2, cin48)"
//! DISPATCH ((NW*NH*48)+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass33(@builtin(global_invocation_id) gid: vec3<u32>) {
    upcat(&FB_A, &FB_S0, &FB_B, NW, NH, 32u, 16u, 2u, gid);
}

//! PASS 34
//! DESC "dec0a B->A (cin48 cout16)"
//! DISPATCH (NW*NH+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass34(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_dec0a, &BS_dec0a, &FB_A, NW, NW * NH, 48u, 4u, 1u, 0u, gid);
}

//! PASS 35
//! DESC "dec0b A->B"
//! DISPATCH (NW*NH+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass35(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_A, &WT_dec0b, &BS_dec0b, &FB_B, NW, NW * NH, 16u, 4u, 1u, 0u, gid);
}

//! PASS 36
//! DESC "out B->FB_HEAD (cin16 cout4 pad16, linear)"
//! DISPATCH (NW*NH+63)/64, 1
//! NUM_THREADS 64,1,1
fn Pass36(@builtin(global_invocation_id) gid: vec3<u32>) {
    conv3x3(&FB_B, &WT_out, &BS_out, &FB_HEAD, NW, NW * NH, 16u, 4u, 0u, 0u, gid);
}

// ============================================================
// Composite
// ============================================================
//! PASS 37
//! DESC "composite: correction lifted to output resolution (SDR)"
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
fn Pass37(@builtin(global_invocation_id) gid: vec3<u32>) {
    let os = scene.outputSize;
    if (gid.x >= os.x || gid.y >= os.y) {
        return;
    }
    let raw = textureLoad(INPUT, vec2<i32>(gid.xy), 0);
    let s = raw.rgb;
    var outCol: vec4<f32>;
    if (uniforms.Apply == 0) {
        outCol = raw;
    } else {
        let corr = correction(s, head_at(vec2<f32>(gid.xy) + vec2<f32>(0.5), os.x, os.y));
        outCol = vec4<f32>(saturate(s + corr), raw.a);
    }
    textureStore(OUTPUT, gid.xy, outCol);
}
