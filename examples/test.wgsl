//! MAGPIE EFFECT WGFX
//! VERSION 4

//! PARAMETER InvertAmount
//! LABEL "Invert Amount"
//! MIN 0.0
//! MAX 1.0
//! DEFAULT 1.0
//! STEP 0.01
float InvertAmount;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
Texture2D OUTPUT;

//! SAMPLER LinearSampler
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState LinearSampler;

//! COMMON
struct Constants {
    input_size: vec2<f32>,
    output_size: vec2<f32>,
};
@group(0) @binding(0) var<uniform> constants: Constants;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! STYLE CS
//! NUM_THREADS 8,8,1
//! DESC "A simple color inversion effect."

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tex_coords = vec2<f32>(global_id.xy) / constants.output_size;
    let original_color = textureSample(INPUT, LinearSampler, tex_coords);
    let inverted_color = vec4<f32>(
        mix(original_color.r, 1.0 - original_color.r, InvertAmount),
        mix(original_color.g, 1.0 - original_color.g, InvertAmount),
        mix(original_color.b, 1.0 - original_color.b, InvertAmount),
        original_color.a
    );
    textureStore(OUTPUT, global_id.xy, inverted_color);
}
