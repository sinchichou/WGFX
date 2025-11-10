// examples/demo_runtime.js

/**
 * @fileoverview A demonstration of how to use the WGFX runtime API.
 * This script simulates a WebGPU environment by using mock objects and
 * shows the typical workflow: compile, update uniforms, and dispatch passes.
 */

import {compile, dispatchPass, getOutput, updateUniform} from '../src/index.js';
// FileUtils would be used in a real Node.js environment to load the .fx file.
// import { FileUtils } from '../src/utils/FileUtils.js';

// #region MOCK WebGPU OBJECTS
// These classes simulate the WebGPU API for demonstration purposes.
// They log their method calls to the console instead of interacting with a real GPU.

class MockGPUDevice {
    get queue() {
        return {
            writeBuffer: (buffer, offset, data) => {
                console.log(`MockGPUQueue: writeBuffer to ${buffer.label} at offset ${offset}`);
            },
            submit: (commandBuffers) => {
                console.log("MockGPUQueue: submit", commandBuffers);
            }
        };
    }

    createTexture(descriptor) {
        console.log("MockGPUDevice: createTexture", descriptor);
        return {
            width: descriptor.size[0],
            height: descriptor.size[1],
            destroy: () => console.log("MockGPUTexture: destroy"),
            createView: () => ({
                label: `View for ${descriptor.label || 'texture'}`,
            })
        };
    }

    createBuffer(descriptor) {
        console.log("MockGPUDevice: createBuffer", descriptor);
        return {label: `Buffer (${descriptor.size} bytes)`, destroy: () => console.log("MockGPUBuffer: destroy")};
    }

    createSampler(descriptor) {
        console.log("MockGPUDevice: createSampler", descriptor);
        return {label: 'MockSampler'};
    }

    createShaderModule(descriptor) {
        console.log("MockGPUDevice: createShaderModule", {code: descriptor.code.substring(0, 80) + "..."});
        return {label: 'MockShaderModule'};
    }

    createBindGroupLayout(descriptor) {
        console.log("MockGPUDevice: createBindGroupLayout", descriptor);
        return {label: 'MockBindGroupLayout'};
    }

    createPipelineLayout(descriptor) {
        console.log("MockGPUDevice: createPipelineLayout", descriptor);
        return {label: 'MockPipelineLayout'};
    }

    async createComputePipelineAsync(descriptor) {
        console.log("MockGPUDevice: createComputePipelineAsync", descriptor);
        return {label: `Pipeline for ${descriptor.label}`};
    }

    createBindGroup(descriptor) {
        console.log("MockGPUDevice: createBindGroup", descriptor);
        return {label: 'MockBindGroup'};
    }

    createCommandEncoder() {
        console.log("MockGPUDevice: createCommandEncoder");
        return new MockGPUCommandEncoder();
    }
}

class MockGPUCommandEncoder {
    beginComputePass() {
        console.log("MockGPUCommandEncoder: beginComputePass");
        return new MockGPUComputePassEncoder();
    }

    finish() {
        console.log("MockGPUCommandEncoder: finish");
        return {label: 'MockCommandBuffer'};
    }
}

class MockGPUComputePassEncoder {
    setPipeline(pipeline) {
        console.log("MockGPUComputePassEncoder: setPipeline", pipeline.label);
    }

    setBindGroup(index, bindGroup) {
        console.log(`MockGPUComputePassEncoder: setBindGroup at index ${index}`, bindGroup.label);
    }

    dispatchWorkgroups(x, y, z) {
        console.log(`MockGPUComputePassEncoder: dispatchWorkgroups (${x}, ${y}, ${z})`);
    }

    end() {
        console.log("MockGPUComputePassEncoder: end");
    }
}

// #endregion

/**
 * The main demonstration function.
 */
async function runDemo() {
    console.log("--- WGFX Runtime Demo ---");

    // In a real application, this would be your actual GPUDevice.
    const mockDevice = new MockGPUDevice();

    // 1. Load the FX file content.
    // In a real app, you would load this from a file or network request.
    // e.g., const fxCode = await FileUtils.readFile('./demo.fx');
    const fxCode = `
        //! MAGPIE EFFECT
        //! VERSION 1.0
        //! SORT_NAME Demo Effect

        //! PARAMETER Strength
        //! LABEL Effect Strength
        //! DEFAULT 0.5
        //! MIN 0.0
        //! MAX 1.0
        //! STEP 0.01
        float Strength;

        //! TEXTURE InputTexture
        //! FORMAT R8G8B8A8_UNORM
        Texture2D InputTexture;

        //! SAMPLER LinearSampler
        //! FILTER LINEAR
        SamplerState LinearSampler;

        //! COMMON
        fn get_tex_coord(pos: vec2<f32>) -> vec2<f32> {
            return pos;
        }

        //! PASS 1
        //! IN InputTexture
        //! OUT OutputTexture
        //! NUM_THREADS 8 8 1
        let tex_coord = get_tex_coord(vec2<f32>(global_id.xy));
        let color = textureSample(InputTexture, LinearSampler, tex_coord);
        textureStore(OutputTexture, vec2<i32>(global_id.xy), color * uniforms.Strength);
    `;
    console.log("1. Loaded FX code.");

    // 2. Compile the effect.
    // This parses the code, generates WGSL, creates resources, and builds pipelines.
    try {
        await compile(fxCode, mockDevice);
        console.log("\n2. Effect compiled successfully.");
    } catch (error) {
        console.error("Compilation failed:", error);
        return;
    }

    // 3. Update a uniform value.
    const newStrength = 0.75;
    updateUniform('Strength', newStrength);
    console.log(`\n3. Updated uniform 'Strength' to ${newStrength}.`);

    // 4. Dispatch a pass.
    // Create a command encoder and dispatch the desired pass by name.
    console.log("\n4. Dispatching pass 'PASS_1'.");
    const commandEncoder = mockDevice.createCommandEncoder();
    dispatchPass('PASS_1', commandEncoder);
    mockDevice.queue.submit([commandEncoder.finish()]);

    // 5. Get the output.
    // Retrieve the final texture view for rendering or further processing.
    const finalOutput = getOutput();
    console.log("\n5. Retrieved final output view:", finalOutput);

    console.log("\n--- Demo Finished ---");
}

runDemo();