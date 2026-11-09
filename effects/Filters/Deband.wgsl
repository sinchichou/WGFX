// Deband — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/Deband.hlsl
//   （Magpie 的 HLSL 移植自 mpv 的 deband.glsl；該 GLSL 由
//    https://github.com/haasn/gentoo-conf/blob/xor/home/nand/.mpv/shaders/deband.glsl
//    保存，原始檔未附授權聲明，版權依 mpv / 該來源庫）
// 原授權 / 作者：Magpie 效果庫為 GPL-3.0（作者 Blinue）；
//   deband 演算法基於 mpv 的 flash3kyuu_deband（mpv shader 系列為 GPL v2+）。
// 本移植僅改寫 pass 結構與取樣語意；PRNG 與所有常數逐字保留。
//
// Porting notes:
//   - Magpie 的 `//!STYLE PS` pass 傳入的 `pos` 是**正規化 UV**（輸出 texel 中心
//     (x+0.5)/outputSize）。WGFX 端：`pos = (gid.xy + 0.5) * scene.outputPt`。
//   - **PRNG 種子與原始 GLSL 不同（刻意保留 Magpie 的行為）**：原始 mpv GLSL 以
//     `vec3 m = vec3(pos, random) + vec3(1.0)` 播種，其中 `pos` 是**像素座標**
//     且帶一個每幀隨機的 `random` uniform；Magpie 的 HLSL 改成
//     `float3 m = float3(pos, 4.6)`，此時 `pos` 是**正規化 UV**（0..1）且沒有
//     隨機 uniform。本移植逐字照抄 Magpie 的版本，因此每個像素的雜訊樣式由
//     正規化 UV 決定（不同解析度下花紋不同），這是與原始 GLSL 的已知差異。
//   - HLSL 的 `inout float h` 參數在 WGSL 不存在：`average()` 改為回傳
//     `struct { avg, h }`，呼叫端以區域變數保存更新後的 PRNG 狀態，
//     亂數取用順序與次數完全不變。
//   - HLSL `frac` → WGSL `fract`；HLSL `lerp` → WGSL `mix`；
//     HLSL `step(thres, diff)` 的語意（WGSL 相同：`x < edge ? 0 : 1`）不變。
//   - HLSL 的 `rand()` helper 改名為 `randHash()`（避免與 WGSL 內建名稱衝突；
//     公式與取用順序完全不變）。
//   - `INPUT.SampleLevel(sam, uv, 0)` → `textureSampleLevel(INPUT, sam, uv, 0.0)`
//     （compute stage 不可用隱含 LOD 的 textureSample）。
//   - Magpie 使用兩個取樣器：線性 `sam`（鄰域平均）與點取樣 `sam1`（來源像素），
//     對應 `//! FILTER LINEAR` / `//! FILTER POINT`（皆 CLAMP）。
//   - Magpie 的 OUTPUT 尺寸 = INPUT；WGFX 明確宣告
//     `//! WIDTH INPUT_WIDTH` / `//! HEIGHT INPUT_HEIGHT`。

//! VERSION 4
//! SORT_NAME Deband

//! PARAMETER threshold
//! TYPE float
//! LABEL "Threshold"
//! MIN 0
//! MAX 512
//! DEFAULT 64
//! STEP 4
// The threshold of difference below which a pixel is considered to be part of
// a gradient. Higher = more debanding, but setting it too high diminishes image
// details.
float threshold;

//! PARAMETER range
//! TYPE float
//! LABEL "Range"
//! MIN 0
//! MAX 16
//! DEFAULT 8
//! STEP 0.1
// The range (in source pixels) at which to sample for neighbours. Higher values
// will find more gradients, but lower values will deband more aggressively.
float range;

//! PARAMETER iterations
//! TYPE int
//! LABEL "Iterations"
//! MIN 1
//! MAX 10
//! DEFAULT 4
//! STEP 1
// The number of debanding iterations to perform. Each iteration samples from
// random positions, so increasing the number of iterations is likely to
// increase the debanding quality. Conversely, it slows the shader down.
// (Each iteration will use a multiple of the configured range, and a
// successively lower THRESHOLD - so setting it much higher has little effect)
int iterations;

//! PARAMETER grain
//! TYPE float
//! LABEL "Grain"
//! MIN 0
//! MAX 256
//! DEFAULT 48
//! STEP 4
// (Optional) Add some extra noise to the image. This significantly helps cover
// up remaining banding and blocking artifacts, at comparatively little visual
// quality. Higher = more grain. Setting it to 0 disables the effect.
float grain;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! SAMPLER
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState sam;

//! SAMPLER
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam1;

//! PASS 1
//! DESC "Debanding (stochastic gradient flattening + grain)"
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

// Wide usage friendly PRNG, shamelessly stolen from a GLSL tricks forum post
fn mod289(x: f32) -> f32 { return x - floor(x / 289.0) * 289.0; }
fn permute(x: f32) -> f32 { return mod289((34.0 * x + 1.0) * x); }
fn randHash(x: f32) -> f32 { return fract(x / 41.0); }

struct AverageResult {
    avg: vec3<f32>,
    h: f32,
}

// Helper: Calculate a stochastic approximation of the avg color around a pixel
fn average(pos: vec2<f32>, r: f32, hIn: f32) -> AverageResult {
    let pt = scene.inputPt;
    // Compute a random rangle and distance
    var h = hIn;
    let dist = randHash(h) * r;         h = permute(h);
    let dir  = randHash(h) * 6.2831853; h = permute(h);

    let o = vec2<f32>(cos(dir), sin(dir)) * pt * dist;

    // Sample at quarter-turn intervals around the source pixel
    let ref0 = textureSampleLevel(INPUT, sam, pos + vec2<f32>( o.x,  o.y), 0.0).rgb;
    let ref1 = textureSampleLevel(INPUT, sam, pos + vec2<f32>(-o.y,  o.x), 0.0).rgb;
    let ref2 = textureSampleLevel(INPUT, sam, pos + vec2<f32>(-o.x, -o.y), 0.0).rgb;
    let ref3 = textureSampleLevel(INPUT, sam, pos + vec2<f32>( o.y, -o.x), 0.0).rgb;

    // Return the (normalized) average
    var out: AverageResult;
    out.avg = (ref0 + ref1 + ref2 + ref3) / 4.0;
    out.h = h;
    return out;
}

fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outSize = scene.outputSize;
    if (gid.x >= outSize.x || gid.y >= outSize.y) {
        return;
    }
    // Magpie PS 的 pos = 正規化 UV（輸出 texel 中心）
    let pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.outputPt;

    // Initialize the PRNG by hashing the position + a random uniform
    let m = vec3<f32>(pos, 4.6);
    var h = permute(permute(permute(m.x) + m.y) + m.z);

    // Sample the source pixel
    var col = textureSampleLevel(INPUT, sam1, pos, 0.0).rgb;

    for (var i: i32 = 1; i <= uniforms.iterations; i++) {
        // Use the average instead if the difference is below the threshold
        let av = average(pos, f32(i) * uniforms.range, h);
        let avg = av.avg;
        h = av.h;
        let diff = abs(col - avg);
        let thres = uniforms.threshold / (f32(i) * 16384.0);
        // WGSL 的 step 兩參數必須同型別（HLSL 允許純量廣播），故顯式展開成 vec3。
        col = mix(avg, col, step(vec3<f32>(thres), diff));
    }

    // Add some random noise to the output
    var noise = vec3<f32>(0.0);
    noise.x = randHash(h); h = permute(h);
    noise.y = randHash(h); h = permute(h);
    noise.z = randHash(h); h = permute(h);
    col += (uniforms.grain / 8192.0) * (noise - 0.5);

    textureStore(OUTPUT, vec2<i32>(gid.xy), vec4<f32>(col, 1.0));
}

//! END
