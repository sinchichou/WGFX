// Lanczos6 — 移植自 effects_src/Lanczos.hlsl（Magpie v4）
// 上游：libretro common-shaders lanczos6.cg
// 移植說明：HLSL 的 9 次 GatherRed/Green/Blue 展開為 6x6 clamp load 網格——
// gather 採樣點恰位於紋理角點（pos = floor(pos0+0.5) - 2 + {0,2,4}），
// D3D 四點順序（x=BL,y=BR,z=TR,w=TL）還原後即 src[i][j] = load(base-3+i, base-3+j)。
//! VERSION 4

//! PARAMETER ARStrength
//! LABEL "Anti-ringing Strength"
//! MIN 0
//! MAX 1
//! DEFAULT 0.5
//! STEP 0.01
float ARStrength;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
Texture2D OUTPUT;

//! SAMPLER sam
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "Lanczos6 六階 sinc 插值 + 可調抗振鈴"

fn FIXc(c: f32) -> f32 {
    return max(abs(c), 1e-5);
}

// Lanczos3 權重（外部已正規化，故不乘 radius）
fn weight3(x: f32) -> vec3<f32> {
    let rcpRadius = 1.0 / 3.0;
    let s = vec3<f32>(FIXc(2.0 * 3.14159265359 * (x - 1.5)), FIXc(2.0 * 3.14159265359 * (x - 0.5)), FIXc(2.0 * 3.14159265359 * (x + 0.5)));
    // checker:ok — FIXc 保證 |s| ≥ 1e-5，rcp(s*s) 分母 ≥ 1e-10
    return sin(s) * sin(s * rcpRadius) / (s * s);
}

fn Load(x: i32, y: i32) -> vec3<f32> {
    let size = vec2<i32>(scene.inputSize);
    return textureLoad(INPUT, clamp(vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    var pos = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) * scene.outputPt;
    let inputPt = scene.inputPt;

    pos = pos * vec2<f32>(scene.inputSize);
    let fr = fract(pos + 0.5);
    var linetaps1 = weight3(0.5 - fr.x * 0.5);
    var linetaps2 = weight3(1.0 - fr.x * 0.5);
    var columntaps1 = weight3(0.5 - fr.y * 0.5);
    var columntaps2 = weight3(1.0 - fr.y * 0.5);

    // 讓所有 tap 總和恰為 1.0，避免極小失真
    let suml = dot(linetaps1, vec3<f32>(1.0)) + dot(linetaps2, vec3<f32>(1.0));
    let sumc = dot(columntaps1, vec3<f32>(1.0)) + dot(columntaps2, vec3<f32>(1.0));
    linetaps1 = linetaps1 / suml;
    linetaps2 = linetaps2 / suml;
    columntaps1 = columntaps1 / sumc;
    columntaps2 = columntaps2 / sumc;

    let base = vec2<i32>(floor(pos - fr - 1.5)); // = floor(pos0 + 0.5) - 2

    // 6x6 採樣網格（gather 展開後即 clamp load）
    var src: array<array<vec3<f32>, 6>, 6>;
    for (var i: u32 = 0u; i < 6u; i++) {
        for (var j: u32 = 0u; j < 6u; j++) {
            src[i][j] = Load(base.x - 3 + i32(i), base.y - 3 + i32(j));
        }
    }

    // 可分離 6-tap 摺積：Σ_y columntaps · (Σ_x linetaps · src[x][y])
    var color = vec3<f32>(0.0);
    for (var yi: u32 = 0u; yi < 3u; yi++) {
        let j0 = 2u * yi;
        let rowA = linetaps1.x * src[0][j0] + linetaps1.y * src[2][j0] + linetaps1.z * src[4][j0]
                 + linetaps2.x * src[1][j0] + linetaps2.y * src[3][j0] + linetaps2.z * src[5][j0];
        let rowB = linetaps1.x * src[0][j0 + 1u] + linetaps1.y * src[2][j0 + 1u] + linetaps1.z * src[4][j0 + 1u]
                 + linetaps2.x * src[1][j0 + 1u] + linetaps2.y * src[3][j0 + 1u] + linetaps2.z * src[5][j0 + 1u];
        color += rowA * columntaps1[yi] + rowB * columntaps2[yi];
    }

    // 抗振鈴：夾限在中心 2x2 鄰域範圍
    let min_sample = min(min(src[2][2], src[3][2]), min(src[2][3], src[3][3]));
    let max_sample = max(max(src[2][2], src[3][2]), max(src[2][3], src[3][3]));
    color = mix(color, clamp(color, min_sample, max_sample), uniforms.ARStrength);

    textureStore(OUTPUT, vec2<i32>(global_id.xy), vec4<f32>(color, 1.0));
}
