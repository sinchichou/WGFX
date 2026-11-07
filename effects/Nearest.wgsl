// Nearest — 移植自 effects_src/Nearest.hlsl（Magpie v4）
//! VERSION 4

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
//! DESC "最近鄰取樣複製（縮放語義由宿主輸出尺寸決定）"

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.outputSize.x || global_id.y >= scene.outputSize.y) {
        return;
    }

    let pos = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) * scene.outputPt;
    let color = textureSampleLevel(INPUT, sam, pos, 0.0);
    textureStore(OUTPUT, vec2<i32>(global_id.xy), color);
}
