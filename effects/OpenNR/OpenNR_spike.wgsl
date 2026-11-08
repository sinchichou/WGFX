// OpenNR conv spike — 驗證 WGFX storage-buffer 支援（單層 3x3 conv，packed f16）
//
// 對照 opennr_shaders.hpp 的 kConvHLSL：特徵圖與權重皆為 flat storage buffer，
// uint2 = 4 個 packed f16；累加器 f32；TM=4 像素/執行緒。
//! VERSION 4

//! COMMON
// 手寫 f16 ↔ f32 轉換（不依賴 shader-f16 特性；bit-level 與上游 f16tof32/f32tof16 一致）
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
        return select(s * 3.4028234663852886e38, bitcast<f32>(nanBits), m != 0u); // Inf / NaN
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
        return sign; // f32 denormal < f16 最小 subnormal → 四捨五入為 0
    }
    let full = man | 0x800000u;
    let s = u32(126 - i32(exp));
    if (s > 34u) {
        return sign;
    }
    let q = full >> s;
    let rem = full & ((1u << s) - 1u);
    let halfB = 1u << (s - 1u);
    let m10 = q + select(0u, 1u, (rem > halfB) || ((rem == halfB) && ((q & 1u) == 1u)));
    if (m10 >= 0x400u) {
        return sign | 0x400u; // 進位到最小 normal 2^-14
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

fn f16bits(v: f32) -> u32 {
    return f32tof16(v);
}

fn p4(v: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(f16bits(v.x) | (f16bits(v.y) << 16u), f16bits(v.z) | (f16bits(v.w) << 16u));
}

// GELU（tanh 形式；clamp 防 tanh 指數溢位成 Inf/Inf = NaN）
fn gelu(v: vec4<f32>) -> vec4<f32> {
    let t = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    return 0.5 * v * (1.0 + tanh(clamp(t, vec4<f32>(-10.0), vec4<f32>(10.0))));
}
//! END

//! BUFFER x
//! MODE read
//! TYPE u32x2
//! BUFFER w
//! MODE read
//! TYPE u32x2
//! BUFFER b
//! MODE read
//! TYPE f32x4
//! BUFFER o
//! MODE read_write
//! TYPE u32x2

//! PASS 1
//! DESC "3x3 conv（packed f16 buffers）"
//! DISPATCH N_GROUPS, 1
//! NUM_THREADS 64,1,1

const TM = 4u;

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let npix = NW * NH;
    let cin4 = CIN / 4u;
    let cout4 = COUT / 4u;
    let p0 = (global_id.x / cout4) * TM;
    let n4 = global_id.x % cout4;
    if (p0 >= npix) {
        return;
    }

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
            // 主網邊界：clamp 攤平索引（gZeroPad = 0 路徑；global 分支用零填充）
            qm[m] = u32(clamp(pm + dy * i32(NW) + dx, 0, i32(npix) - 1));
            ok[m] = true;
        }
        for (var k4: u32 = 0u; k4 < cin4; k4++) {
            var xv: array<vec4<f32>, 4>;
            for (var m: u32 = 0u; m < TM; m++) {
                xv[m] = h4(x[qm[m] * cin4 + k4]);
            }
            for (var kk: u32 = 0u; kk < 4u; kk++) {
                let wv = h4(w[(tap * CIN + k4 * 4u + kk) * cout4 + n4]);
                for (var a: u32 = 0u; a < TM; a++) {
                    acc[a] += xv[a][kk] * wv;
                }
            }
        }
    }
    let bv = b[n4];
    for (var i: u32 = 0u; i < TM; i++) {
        let p = p0 + i;
        if (p < npix) {
            let v = acc[i] + bv;
            // ACT 由宿主以 external define 提供（1 = GELU，0 = 線性）；
            // 全網效果檔在每個 pass 以字面常數內嵌同一邏輯。
            o[p * cout4 + n4] = select(p4(v), p4(gelu(v)), ACT == 1u);
        }
    }
}
