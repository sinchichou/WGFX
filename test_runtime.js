// test_runtime.js

/**
 * @fileoverview This is a test script for the WGFX runtime.
 * It demonstrates how to load, compile, and run a WGFX effect using the
 * public API.
 *
 * This script uses the mock WebGPU environment, so it will not produce
 * any visual output. Instead, it logs the steps of the process to the
 * console to verify that the runtime is working correctly.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { compile, dispatchPass, updateUniform, getOutput } from './src/index.js';
import { GPUDevice } from './src/runtime/WebGPU-mock.js';

async function main() {
    console.log("--- WGFX Runtime Test ---");

    try {
        // 1. Create a mock GPUDevice. In a real application, this would be
        //    obtained from the browser's WebGPU adapter.
        const device = new GPUDevice();
        console.log("Mock GPUDevice created.");

        // 2. Read the effect file.
        const effectPath = path.join('examples', 'Anime4K_Restore_Soft_UL.wgsl');
        console.log(`Loading effect file: ${effectPath}`);
        const effectCode = await fs.readFile(effectPath, 'utf-8');

        // 3. Compile the effect.
        console.log("Compiling effect...");
        await compile(effectCode, device);
        console.log("Effect compiled successfully.");

        // 4. Update a uniform parameter.
        const uniformName = 'InvertAmount';
        const newValue = 0.5;
        console.log(`Updating uniform '${uniformName}' to ${newValue}`);
        updateUniform(uniformName, newValue);

        // 5. Create a mock command encoder to dispatch the pass.
        const commandEncoder = device.createCommandEncoder();
        console.log("Mock GPUCommandEncoder created.");

        // 6. Dispatch the first pass of the effect.
        const passName = 'pass_1';
        console.log(`Dispatching pass: '${passName}'`);
        dispatchPass(passName, commandEncoder);
        console.log("Pass dispatched.");

        // 7. Get the output texture view.
        console.log("Getting output texture view...");
        const outputView = getOutput();
        console.log("Output texture view obtained:", outputView);

        console.log("\n--- Test Summary ---");
        console.log("The runtime test completed successfully.");
        console.log("This demonstrates that the WGFX runtime can:");
        console.log("  - Load and parse a .wgsl file.");
        console.log("  - Generate WGSL code.");
        console.log("  - Create mock GPU resources and pipelines.");
        console.log("  - Update uniform parameters.");
        console.log("  - Dispatch a compute pass.");
        console.log("  - Return an output texture view.");

    } catch (error) {
        console.error("\n--- Test Failed ---");
        console.error(error);
        process.exit(1);
    }
}

main().then().catch(err => console.error(err));
