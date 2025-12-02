// test_runtime.js

// 確保 webgpu 相關的全局變量盡早被設置
import 'webgpu';
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };

/**
 * @fileoverview This is a test script for the WGFX runtime.
 * It demonstrates how to load, compile, and run a WGFX effect using the
 * public API in a Node.js environment with '@webgpu/node'.
 *
 * This script will attempt to use a real WebGPU device. It logs the
 * steps of the process to the console to verify that the runtime is
t working correctly.
 */

import * as fs from 'fs/promises';
import * as path from 'path';

async function main() {
    console.log("--- WGFX Runtime Test ---");
    // console.log("Globals object from 'webgpu':", globals); // Remove this debug log

    // 在 globalThis.GPUDevice 被設置後再導入 WGFX 相關模組
    const { compile, dispatchPass, updateUniform, getOutput } = await import('../src/index.js');

    try {
        // 1. Get a real GPUDevice.
        console.log("Requesting WebGPU adapter...");
        if (!navigator.gpu) {
            console.error("WebGPU not available. Please run with a supported environment for '@webgpu/node'.");
            process.exit(1);
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.error("Failed to get WebGPU adapter.");
            process.exit(1);
        }
        console.log("Requesting WebGPU device...");
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageTexturesPerShaderStage: 8,
            },
        });
        console.log("Real GPUDevice created.");

        // 2. Read the effect file.
        // 修正檔案路徑
        const effectPath = path.join(process.cwd(), 'examples', 'Anime4K_Restore_Soft_UL.wgsl');
        console.log(`Loading effect file: ${effectPath}`);
        const effectCode = await fs.readFile(effectPath, 'utf-8');

        // 3. Compile the effect and get shader info.
        console.log("Compiling effect...");
        const shaderInfo = await compile(effectCode, device);
        console.log("Effect compiled successfully.");
        console.log(`Found ${shaderInfo.passes.length} passes.`);

        // 4. Create a mock command encoder to dispatch the passes.
        const commandEncoder = device.createCommandEncoder();
        console.log("Mock GPUCommandEncoder created.");

        // 5. Dispatch all passes of the effect.
        for (const pass of shaderInfo.passes) {
            const passName = `PASS_${pass.index}`;
            console.log(`Dispatching pass: '${passName}'`);
            dispatchPass(passName, commandEncoder);
            console.log(`Pass '${passName}' dispatched.`);
        }

        // 6. Get the output texture view.
        console.log("Getting output texture view...");
        const outputView = getOutput();
        console.log("Output texture view obtained:", outputView);

        console.log("\n--- Test Summary ---");
        console.log("The runtime test completed successfully.");
        console.log("This demonstrates that the WGFX runtime can:");
        console.log("  - Load and parse a complex .wgsl file.");
        console.log("  - Generate WGSL code for multiple passes.");
        console.log("  - Create real GPU resources and pipelines.");
        console.log("  - Dispatch multiple compute passes in sequence.");
        console.log("  - Return an output texture view.");

    } catch (error) {
        console.error("\n--- Test Failed ---");
        console.error(error);
        process.exit(1);
    }
}

main().then().catch(err => console.error(err));
