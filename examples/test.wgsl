// MAGPIE EFFECT WGFX
// Canonical WGFX core example: single pass, one float + one int parameter.
// WGFX 標準核心範例：單一通道，含一個 float 與一個 int 參數。
//! VERSION 4

//! PARAMETER InvertAmount
//! LABEL "Invert Amount"
//! MIN 0.0
//! MAX 1.0
//! DEFAULT 1.0
//! STEP 0.01
float InvertAmount;

//! PARAMETER Threshold
//! TYPE int
//! MIN 0
//! MAX 255
//! DEFAULT 128
//! STEP 1
int Threshold;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
Texture2D OUTPUT;

//! SAMPLER LinearSampler
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState LinearSampler;

//! COMMON
// Shared helpers / 共用輔助函式
// Standard uniforms are auto-injected by WGFX:
//   scene    (@group(0) @binding(4)) - inputSize/inputPt/outputSize/outputPt/scale
//   uniforms (@group(0) @binding(1)) - the parameters declared above
fn Luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
//! DESC "A simple color inversion effect with an int threshold."

fn Pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tex_coords = vec2<f32>(global_id.xy) * scene.outputPt;
    let original_color = textureSampleLevel(INPUT, LinearSampler, tex_coords, 0.0);

    var out_color = vec4<f32>(
        mix(original_color.r, 1.0 - original_color.r, uniforms.InvertAmount),
        mix(original_color.g, 1.0 - original_color.g, uniforms.InvertAmount),
        mix(original_color.b, 1.0 - original_color.b, uniforms.InvertAmount),
        original_color.a
    );

    // Zero out pixels darker than the integer threshold parameter.
    if (i32(round(Luma(out_color.rgb) * 255.0)) < uniforms.Threshold) {
        out_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    textureStore(OUTPUT, global_id.xy, out_color);
}
