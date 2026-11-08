// Nearest — Magpie v4 → WGFX WGSL port
// Ported from https://github.com/Blinue/Magpie/blob/main/src/Effects/Nearest.hlsl
//
// Porting notes:
//   - Magpie 的 `//!STYLE PS` pass 傳入的 `pos` 是**正規化 UV**（texel 中心位置
//     (x+0.5)/W），不是 texel 座標。證據：同作者的 LumaSharpen 直接以
//     `INPUT.SampleLevel(sam, pos, 0)` 取樣，並以 `GetInputPt()` 作為
//     「一個 texel 的 UV 尺寸」加偏移。
//     WGFX 端以 compute 實作：`pos = (gid.xy + 0.5) * scene.outputPt`。
//   - Magpie 的 `SamplerState sam`（POINT / CLAMP）逐字對應
//     `//! FILTER POINT` + `//! ADDRESS CLAMP`。
//   - `textureSampleLevel` 在 compute shader 中合法（隱含 LOD 的 `textureSample`
//     僅限 fragment stage）。

//! VERSION 4
//! SORT_NAME Nearest

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//! SAMPLER
//! FILTER POINT
//! ADDRESS CLAMP
SamplerState sam;

//! PASS 1
//! DESC "Nearest-neighbour scaler (POINT + CLAMP)"
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1

fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outSize = scene.outputSize;
    if (gid.x >= outSize.x || gid.y >= outSize.y) {
        return;
    }
    // Magpie PS 的 pos = 正規化 UV（輸出 texel 中心）
    let pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * scene.outputPt;
    let color = textureSampleLevel(INPUT, sam, pos, 0.0);
    textureStore(OUTPUT, vec2<i32>(gid.xy), color);
}

//! END
