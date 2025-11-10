// 這是一個 WGFX 範例檔案。
// 它演示了效果的基本結構，包括元資料、
// 參數、資源和計算通道。

// --- 標頭區塊 ---
// 定義效果的全局元資料。
//! MAGPIE EFFECT
//! VERSION 1.0
//! SORT_NAME 演示效果

// --- 參數區塊 ---
// 定義一個可由 UI 控制的統一參數。
//! PARAMETER Strength
//! LABEL 效果強度
//! DEFAULT 0.5
//! MIN 0.0
//! MAX 1.0
//! STEP 0.01
float Strength; // 參數的 HLSL 風格宣告。

// --- 紋理區塊 ---
// 定義紋理資源。
//! TEXTURE InputTexture
//! FORMAT R8G8B8A8_UNORM
//! WIDTH 1920
//! HEIGHT 1080
Texture2D InputTexture;

// --- 取樣器區塊 ---
// 定義紋理的取樣器。
//! SAMPLER LinearSampler
//! FILTER LINEAR
SamplerState LinearSampler;

// --- 通用區塊 ---
// 包含所有通道都可以使用的共享程式碼。
// 內部程式碼假定為 WGSL。
//! COMMON
// 通用函數或常數可以在這裡。
fn get_tex_coord(pos: vec2<f32>) -> vec2<f32> {
    return pos;
}

// --- 通道區塊 ---
// 定義單個計算通道。
//! PASS 1
//! IN InputTexture
//! OUT OutputTexture
//! BLOCK_SIZE 8 8 1
//! NUM_THREADS 8 8 1
//! STYLE Compute
// 下面的程式碼假定為 WGSL，並將放置在
// 生成的入口點函數中。
let tex_coord = get_tex_coord(vec2<f32>(global_id.xy));
let color = textureSample(InputTexture, LinearSampler, tex_coord);
textureStore(OutputTexture, vec2<i32>(global_id.xy), color * uniforms.Strength);