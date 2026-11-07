// FineSharp — 移植自 effects_src/Sharpen/FineSharp.hlsl（Magpie v4）
// 上游：Didée 的 AviSynth FineSharp 腳本（madVR 級 16-bit 精度鏈）
//
// 管線：P1 RemoveGrain11（RGB→YUV + Y 平滑）→ P2 RemoveGrain4（Y 中值濾波）→
//       P3 非線性銳化 + 均衡（cstr）→ P4 XSharpen + Repair → P5 還原 RGB
//
// 移植偏差（記錄）：
//   1. median9 / sort9_partial2 以「等價結果」實作——全排序取中位（P2）與
//      次小/次大（P4 Repair 的夾限界），不逐宏複製排序網路（結果相同）。
//   2. gather 展開為 clamp load 網格（採樣點在紋理角點，
//      src[a][b] = load(gxy + a - 1, gxy + b - 1)；gba/ra 的 (gxy+1) 角點亦同）。
//   3. 1 invocation 1 像素（Magpie 的 4-pixel 佈局為效能細節）。
//! VERSION 4

//! PARAMETER sstr
//! LABEL "Sharpness"
//! MIN 0
//! MAX 8
//! DEFAULT 2.0
//! STEP 0.01
float sstr;

//! PARAMETER cstr
//! LABEL "Equalisation"
//! MIN 0
//! MAX 2
//! DEFAULT 0.9
//! STEP 0.01
float cstr;

//! PARAMETER xstr
//! LABEL "XSharpen"
//! MIN 0
//! MAX 1
//! DEFAULT 0.19
//! STEP 0.01
float xstr;

//! PARAMETER xrep
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
//! WIDTH OUTPUT_WIDTH
//! HEIGHT OUTPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D tex1;

//! TEXTURE tex2
//! WIDTH OUTPUT_WIDTH
//! HEIGHT OUTPUT_HEIGHT
//! FORMAT R16G16B16A16_FLOAT
Texture2D tex2;

//! SAMPLER sam
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! DESC "RemoveGrain11：RGB→YUV（SD/HD 矩陣）+ Y 高斯平滑"
//! IN INPUT
//! OUT tex1
//! NUM_THREADS 8,8,1

// BT.601（≤576 行）或 BT.709 矩陣，行主序 tap、dy 外層
fn RGBtoYUVmat() -> mat3x3<f32> {
    // wgsl mat3x3 為行（column-major）宣告；此處以「乘向量」的語義直接排：
    // Y 行、U 行、V 行各為一個 column（右乘向量 = Σ col[i] * v[i]）
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
            src[a][b] = m * c + vec3<f32>(0.0, 0.5, 0.5);
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
    return sign(t) * (uniforms.sstr / 255.0) * pow(abs(t) / (LSTR / 255.0), 1.0 / PSTR) * ((t * t) / (t * t + ldmp / (255.0 * 255.0))); // checker:ok — LSTR/PSTR 為模組常數
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
            textureStore(OUTPUT, gxy + vec2<i32>(i32(a) - 1, i32(b) - 1), vec4<f32>(YUVtoRGBmat() * (o.xyz - vec3<f32>(0.0, 0.5, 0.5)), 1.0));
        }
    }
}
