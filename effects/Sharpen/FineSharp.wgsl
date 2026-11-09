// FineSharp — Magpie v4 → WGFX WGSL port（effects-port 遺產搬移 + 修正）
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/Sharpen/FineSharp.hlsl
// 上游：Didée 的 AviSynth FineSharp 腳本（doom9 論壇；madVR 16-bit 精度鏈）。
// 原授權 / 作者：Magpie 效果庫為 GPL-3.0（作者 Blinue）；FineSharp 原腳本作者 Didée。
// 本移植僅改寫 pass 結構與取樣語意；演算法常數逐字保留。
//
// 管線：P1 RemoveGrain11（RGB→YUV + Y 高斯平滑）→ P2 RemoveGrain4（Y 中值濾波）→
//       P3 非線性銳化 + 均衡（cstr）→ P4 XSharpen + Repair → P5 還原 RGB
//
// Porting notes:
//   1. 【修正（相對 effects-port 版本）】HLSL `mul(M, v)`（M 以三個「列」向量建構）
//      的逐字 WGSL 等價是 `v * m`：WGSL 矩陣建構子吃「欄」，數字逐字照抄時把向量
//      放在乘法左側（本工作區以 Dawn node-webgpu 實測驗證；參見
//      docs/porting-guide.md §2.1 的乘法順序反轉規則——該規則僅適用於 HLSL
//      `v * M` / MulAdd(v, M) 形式，mul(M, v) 不需要反轉但需要向量在左）。
//      effects-port 版寫成 `m * v`，等價於套用**轉置**矩陣：RGB→YUV 與 YUV→RGB
//      的線性往返雖為恆等（兩次轉置互相抵消，untouched 像素不受影響），但銳化、
//      中值與 XSharpen 全部作用在混入色度係數的錯誤「Y」軸上
//      （BT.601：誤植軸 = 0.299R − 0.169G + 0.500B，正確 Y = 0.299R + 0.587G + 0.114B）。
//      本移植已改為 `v * m`，軸向與 Magpie 原碼一致。
//   2. median9 / sort9_partial2 以「等價結果」實作——全排序取中位（P2）與
//      次小/次大（P4 Repair 的夾限界），不逐宏複製排序網路（結果相同）。
//   3. gather 展開為 clamp load 網格（採樣點在紋理角點，
//      src[a][b] = load(gxy + a - 1, gxy + b - 1)；gba/ra 的 (gxy+1) 角點亦同）。
//   4. 1 invocation 1 像素（Magpie 的 4-pixel 佈局為效能細節）。
//   5. tex1/tex2 宣告 INPUT_WIDTH × INPUT_HEIGHT：現行框架的紋理尺寸運算式僅接受
//      defines（INPUT_*，OUTPUT_* 尚未支援，見 effects/Scalers/SSimDownscaler.wgsl
//      檔頭的框架缺口說明）。FineSharp 為 1:1 效果（OUTPUT 尺寸 = INPUT 尺寸），
//      語意等價。
//! VERSION 4
//! SORT_NAME FineSharp

//! PARAMETER sstr
//! TYPE float
//! LABEL "Sharpness"
//! MIN 0
//! MAX 8
//! DEFAULT 2.0
//! STEP 0.01
float sstr;

//! PARAMETER cstr
//! TYPE float
//! LABEL "Equalisation"
//! MIN 0
//! MAX 2
//! DEFAULT 0.9
//! STEP 0.01
float cstr;

//! PARAMETER xstr
//! TYPE float
//! LABEL "XSharpen"
//! MIN 0
//! MAX 1
//! DEFAULT 0.19
//! STEP 0.01
float xstr;

//! PARAMETER xrep
//! TYPE float
//! LABEL "Repair"
//! MIN 0
//! MAX 1
//! DEFAULT 0.25
//! STEP 0.01
float xrep;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! TEXTURE tex1
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D tex1;

//! TEXTURE tex2
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D tex2;

//! SAMPLER
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! DESC "RemoveGrain11：RGB→YUV（SD/HD 矩陣）+ Y 高斯平滑"
//! IN INPUT
//! OUT tex1
//! NUM_THREADS 8,8,1

// BT.601（≤576 行）或 BT.709 矩陣。建構子參數 = HLSL float3x3 的三個「列」向量
// （逐字照抄）；配 `c * m`（向量在左）即為 HLSL `mul(M, v)` 的等價（見檔頭 note 1）。
fn RGBtoYUVmat() -> mat3x3<f32> {
    if (scene.inputSize.y <= 576u) {
        // Kb=0.114, Kr=0.299
        return mat3x3<f32>(
            vec3<f32>(0.299, 1.0 - 0.299 - 0.114, 0.114),
            vec3<f32>(-0.299, 0.299 + 0.114 - 1.0, 1.0 - 0.114) / vec3<f32>(2.0 * (1.0 - 0.114)),
            vec3<f32>(1.0 - 0.299, 0.299 + 0.114 - 1.0, -0.114) / vec3<f32>(2.0 * (1.0 - 0.299))
        );
    } else {
        // Kb=0.0722, Kr=0.2126
        return mat3x3<f32>(
            vec3<f32>(0.2126, 1.0 - 0.2126 - 0.0722, 0.0722),
            vec3<f32>(-0.2126, 0.2126 + 0.0722 - 1.0, 1.0 - 0.0722) / vec3<f32>(2.0 * (1.0 - 0.0722)),
            vec3<f32>(1.0 - 0.2126, 0.2126 + 0.0722 - 1.0, -0.0722) / vec3<f32>(2.0 * (1.0 - 0.2126))
        );
    }
}

fn Load(x: i32, y: i32) -> vec3<f32> {
    let size = vec2<i32>(scene.inputSize);
    return textureLoad(INPUT, clamp(vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.inputSize.x || global_id.y >= scene.inputSize.y) {
        return;
    }
    let gxy = vec2<i32>(global_id.xy);

    var src: array<array<vec3<f32>, 4>, 4>;
    let m = RGBtoYUVmat();
    for (var a: u32 = 0u; a < 4u; a++) {
        for (var b: u32 = 0u; b < 4u; b++) {
            let c = Load(gxy.x + i32(a) - 1, gxy.y + i32(b) - 1);
            src[a][b] = c * m + vec3<f32>(0.0, 0.5, 0.5);
        }
    }

    for (var a: u32 = 1u; a <= 2u; a++) {
        for (var b: u32 = 1u; b <= 2u; b++) {
            var o = vec4<f32>(src[a][b], src[a][b].x); // rgbr
            o.x += o.x;
            o.x += src[a][b - 1u].x + src[a - 1u][b].x + src[a + 1u][b].x + src[a][b + 1u].x;
            o.x += o.x;
            o.x += src[a - 1u][b - 1u].x + src[a + 1u][b - 1u].x + src[a - 1u][b + 1u].x + src[a + 1u][b + 1u].x;
            o.x *= 0.0625;
            textureStore(tex1, gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), o);
        }
    }
}

//! PASS 2
//! DESC "RemoveGrain4：Y 高斯 + 9 點中值"
//! IN tex1
//! OUT tex2
//! NUM_THREADS 8,8,1

fn Pass2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.inputSize.x || global_id.y >= scene.inputSize.y) {
        return;
    }
    let gxy = vec2<i32>(global_id.xy);

    var src: array<array<vec4<f32>, 4>, 4>;
    for (var a: u32 = 0u; a < 4u; a++) {
        for (var b: u32 = 0u; b < 4u; b++) {
            let p = textureLoad(tex1, clamp(gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec2<i32>(0), vec2<i32>(scene.inputSize) - vec2<i32>(1)), 0);
            src[a][b] = vec4<f32>(p.r, 0.0, 0.0, 0.0);
        }
    }
    for (var a: u32 = 1u; a <= 2u; a++) {
        for (var b: u32 = 1u; b <= 2u; b++) {
            let p = textureLoad(tex1, clamp(gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec2<i32>(0), vec2<i32>(scene.inputSize) - vec2<i32>(1)), 0);
            src[a][b] = vec4<f32>(p.r, p.g, p.b, p.a);
        }
    }

    for (var a: u32 = 1u; a <= 2u; a++) {
        for (var b: u32 = 1u; b <= 2u; b++) {
            var o = src[a][b];
            o.x += o.x;
            o.x += src[a][b - 1u].x + src[a - 1u][b].x + src[a + 1u][b].x + src[a][b + 1u].x;
            o.x += o.x;
            o.x += src[a - 1u][b - 1u].x + src[a + 1u][b - 1u].x + src[a - 1u][b + 1u].x + src[a + 1u][b + 1u].x;
            o.x *= 0.0625;

            // 9 點中值（等價結果：全排序取 [4]；上游為排序網路宏）
            var v: array<f32, 9>;
            v[0] = src[a - 1u][b - 1u].x; v[1] = src[a][b - 1u].x; v[2] = src[a + 1u][b - 1u].x;
            v[3] = src[a - 1u][b].x;      v[4] = o.x;              v[5] = src[a + 1u][b].x;
            v[6] = src[a - 1u][b + 1u].x; v[7] = src[a][b + 1u].x; v[8] = src[a + 1u][b + 1u].x;
            for (var p: u32 = 0u; p < 8u; p++) {
                for (var q: u32 = 0u; q < 8u - p; q++) {
                    if (v[q] > v[q + 1u]) {
                        let t = v[q]; v[q] = v[q + 1u]; v[q + 1u] = t;
                    }
                }
            }
            o.x = v[4];
            textureStore(tex2, gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), o);
        }
    }
}

//! PASS 3
//! DESC "Part A：非線性銳化 + 均衡"
//! IN tex2
//! OUT tex1
//! NUM_THREADS 8,8,1

const LSTR = 1.49;  // 非線性銳化修飾
const PSTR = 1.272; // 非線性銳化指數

fn SharpDiff(c: vec4<f32>) -> f32 {
    let ldmp = uniforms.sstr + 0.1; // low damp：抑制平坦區噪聲過增強
    let t = c.a - c.x;
    return sign(t) * (uniforms.sstr / 255.0) * pow(abs(t) / (LSTR / 255.0), 1.0 / PSTR) * ((t * t) / (t * t + ldmp / (255.0 * 255.0)));
}

fn Pass3(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.inputSize.x || global_id.y >= scene.inputSize.y) {
        return;
    }
    let gxy = vec2<i32>(global_id.xy);

    var src: array<array<vec4<f32>, 4>, 4>;
    var sharpDiffs: array<array<f32, 4>, 4>;
    for (var a: u32 = 0u; a < 4u; a++) {
        for (var b: u32 = 0u; b < 4u; b++) {
            let p = textureLoad(tex2, clamp(gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec2<i32>(0), vec2<i32>(scene.inputSize) - vec2<i32>(1)), 0);
            src[a][b] = vec4<f32>(p.r, 0.0, 0.0, p.a);
            sharpDiffs[a][b] = SharpDiff(src[a][b]);
        }
    }
    for (var a: u32 = 1u; a <= 2u; a++) {
        for (var b: u32 = 1u; b <= 2u; b++) {
            let p = textureLoad(tex2, clamp(gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec2<i32>(0), vec2<i32>(scene.inputSize) - vec2<i32>(1)), 0);
            src[a][b] = vec4<f32>(p.r, p.g, p.b, p.a);
        }
    }

    for (var a: u32 = 1u; a <= 2u; a++) {
        for (var b: u32 = 1u; b <= 2u; b++) {
            var o = src[a][b];
            var sd = sharpDiffs[a][b];
            o.x = o.a + sd;
            sd += sd;
            sd += sharpDiffs[a][b - 1u] + sharpDiffs[a - 1u][b] + sharpDiffs[a + 1u][b] + sharpDiffs[a][b + 1u];
            sd += sd;
            sd += sharpDiffs[a - 1u][b - 1u] + sharpDiffs[a + 1u][b - 1u] + sharpDiffs[a - 1u][b + 1u] + sharpDiffs[a + 1u][b + 1u];
            sd *= 0.0625;
            o.x -= uniforms.cstr * sd;
            o.a = o.x;
            textureStore(tex1, gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), o);
        }
    }
}

//! PASS 4
//! DESC "Part B：XSharpen 銳化 + Repair 修復"
//! IN tex1
//! OUT tex2
//! NUM_THREADS 8,8,1

fn Pass4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.inputSize.x || global_id.y >= scene.inputSize.y) {
        return;
    }
    let gxy = vec2<i32>(global_id.xy);

    var src: array<array<vec4<f32>, 4>, 4>;
    for (var a: u32 = 0u; a < 4u; a++) {
        for (var b: u32 = 0u; b < 4u; b++) {
            let p = textureLoad(tex1, clamp(gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec2<i32>(0), vec2<i32>(scene.inputSize) - vec2<i32>(1)), 0);
            src[a][b] = vec4<f32>(0.0, 0.0, 0.0, p.a);
        }
    }
    for (var a: u32 = 1u; a <= 2u; a++) {
        for (var b: u32 = 1u; b <= 2u; b++) {
            let p = textureLoad(tex1, clamp(gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec2<i32>(0), vec2<i32>(scene.inputSize) - vec2<i32>(1)), 0);
            src[a][b] = vec4<f32>(p.r, p.g, p.b, p.a);
        }
    }

    for (var a: u32 = 1u; a <= 2u; a++) {
        for (var b: u32 = 1u; b <= 2u; b++) {
            var o = src[a][b];

            // 9 點（含中心）排序——Repair 需次小/次大（等價結果實作，見檔頭）
            var v: array<f32, 9>;
            v[0] = src[a - 1u][b - 1u].a; v[1] = src[a][b - 1u].a; v[2] = src[a + 1u][b - 1u].a;
            v[3] = src[a - 1u][b].a;      v[4] = o.a;              v[5] = src[a + 1u][b].a;
            v[6] = src[a - 1u][b + 1u].a; v[7] = src[a][b + 1u].a; v[8] = src[a + 1u][b + 1u].a;
            for (var p: u32 = 0u; p < 8u; p++) {
                for (var q: u32 = 0u; q < 8u - p; q++) {
                    if (v[q] > v[q + 1u]) {
                        let t = v[q]; v[q] = v[q + 1u]; v[q + 1u] = t;
                    }
                }
            }

            o.x += v[0] + v[1] + v[2] + v[3] + v[5] + v[6] + v[7] + v[8];
            o.x /= 9.0;
            o.x = o.a + 9.9 * (o.a - o.x);

            // Repair：夾限在鄰域次小/次大（排除中心自身）
            o.x = max(o.x, min(v[1], o.a));
            o.x = min(o.x, max(v[7], o.a));
            textureStore(tex2, gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), o);
        }
    }
}

//! PASS 5
//! DESC "Part C：XSharpen 混合 + YUV→RGB 還原"
//! IN tex2
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

fn YUVtoRGBmat() -> mat3x3<f32> {
    if (scene.inputSize.y <= 576u) {
        // Kb=0.114, Kr=0.299
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 2.0 * (1.0 - 0.299)),
            vec3<f32>(0.114 + 0.299 - 1.0, 2.0 * (1.0 - 0.114) * 0.114, 2.0 * 0.299 * (1.0 - 0.299)) / vec3<f32>(0.114 + 0.299 - 1.0),
            vec3<f32>(1.0, 2.0 * (1.0 - 0.114), 0.0)
        );
    } else {
        // Kb=0.0722, Kr=0.2126
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 2.0 * (1.0 - 0.2126)),
            vec3<f32>(0.0722 + 0.2126 - 1.0, 2.0 * (1.0 - 0.0722) * 0.0722, 2.0 * 0.2126 * (1.0 - 0.2126)) / vec3<f32>(0.0722 + 0.2126 - 1.0),
            vec3<f32>(1.0, 2.0 * (1.0 - 0.0722), 0.0)
        );
    }
}

fn Pass5(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }
    let gxy = vec2<i32>(global_id.xy);

    var src: array<array<vec4<f32>, 4>, 4>;
    for (var a: u32 = 0u; a < 4u; a++) {
        for (var b: u32 = 0u; b < 4u; b++) {
            let p = textureLoad(tex2, clamp(gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec2<i32>(0), vec2<i32>(scene.inputSize) - vec2<i32>(1)), 0);
            src[a][b] = vec4<f32>(p.r, 0.0, 0.0, 0.0);
        }
    }
    for (var a: u32 = 1u; a <= 2u; a++) {
        for (var b: u32 = 1u; b <= 2u; b++) {
            let p = textureLoad(tex2, clamp(gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec2<i32>(0), vec2<i32>(scene.inputSize) - vec2<i32>(1)), 0);
            src[a][b] = vec4<f32>(p.r, p.g, p.b, p.a);
        }
    }

    for (var a: u32 = 1u; a <= 2u; a++) {
        for (var b: u32 = 1u; b <= 2u; b++) {
            var o = src[a][b];
            let edge = abs(src[a][b - 1u].x + src[a - 1u][b].x + src[a + 1u][b].x + src[a][b + 1u].x - 4.0 * o.x);
            o.x = mix(o.a, o.x, uniforms.xstr * (1.0 - saturate(edge * uniforms.xrep)));
            textureStore(OUTPUT, gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec4<f32>((o.xyz - vec3<f32>(0.0, 0.5, 0.5)) * YUVtoRGBmat(), 1.0));
        }
    }
}

//! END
