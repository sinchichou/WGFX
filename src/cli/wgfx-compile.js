// src/cli/wgfx-compile.js

/**
 * @fileoverview The main entry point for the WGFX command-line interface (CLI).
 * This script orchestrates the static compilation process of a .fx file into
 * a distributable package.
 *
 * @example
 * To run from the project root:
 * node src/cli/wgfx-compile.js --input examples/demo.fx --output build/demo.zip
 */
import {createRequire} from 'module';

const require = createRequire(import.meta.url);

import {WGSLCodeGenerator} from '../runtime/WGSLCodeGenerator.js';
import {PipelineMetadataGenerator} from './PipelineMetadataGenerator.js';
import {OutputPackager} from './OutputPackager.js';
import {FileUtils} from '../utils/FileUtils.js';

// Directly use the runtime parser.
const shaderParser = require('../runtime/ShaderParser.cjs');

/**
 * A simple command-line argument parser.
 * Parses arguments in the format `--key value`.
 * @param {string[]} args - The array of command-line arguments (e.g., from `process.argv.slice(2)`).
 * @returns {Object.<string, string|boolean>} An object mapping keys to their values.
 */
function parseArgs(args) {
    const options = {};
    for (let i = 0; i < args.length; i++) {
        if (args[i].startsWith('--')) {
            const key = args[i].substring(2);
            // If the next argument is not a key, it's the value for the current key.
            const value = (i + 1 < args.length && !args[i + 1].startsWith('--')) ? args[i + 1] : true;
            options[key] = value;
        }
    }
    return options;
}

/**
 * The main function for the CLI script.
 */
async function main() {
    console.log("--- WGFX CLI Compiler ---");

    const options = parseArgs(process.argv.slice(2));

    const inputFxFile = options.input;
    const outputZipFile = options.output;

    if (!inputFxFile || !outputZipFile) {
        console.error("Error: Missing required arguments.");
        console.error("Usage: node wgfx-compile.js --input <path/to/effect.fx> --output <path/to/output.zip>");
        process.exit(1);
    }

    try {
        // 1. Read the source .fx file.
        console.log(`Reading FX file: ${inputFxFile}`);
        const fxCode = await FileUtils.readFile(inputFxFile);

        // 2. Parse the file into an Intermediate Representation (IR) using the runtime parser directly.
        console.log("Parsing FX file...");
        const shaderInfo = shaderParser.parse(fxCode);

        // 3. Generate WGSL code from the IR using the runtime generator directly.
        console.log("Generating WGSL code and pass-specific resource bindings...");
        const codeGenerator = new WGSLCodeGenerator();
        // The runtime generator produces an array of modules, each containing WGSL code and its bound resources.
        const generatedModules = codeGenerator.generate(shaderInfo);

        // 4. Generate metadata JSON from the IR.
        console.log("Generating pipeline metadata...");
        const metadataGenerator = new PipelineMetadataGenerator();
        const {pipeline, metadata} = metadataGenerator.generate(shaderInfo);

        // Augment pipeline metadata with pass-specific resource binding information
        pipeline.passes.forEach(p => {
            const correspondingModule = generatedModules.find(m => m.passIndex === p.index);
            if (correspondingModule) {
                p.resources = correspondingModule.resources;
            }
        });

        // 5. Package all artifacts for distribution.
        console.log(`Packaging output to ${outputZipFile}...`);
        const packager = new OutputPackager();
        await packager.package(generatedModules, pipeline, metadata, outputZipFile);

        console.log(`
Successfully compiled ${inputFxFile}.`);

    } catch (error) {
        console.error("\nCLI Compilation failed:", error);
        process.exit(1);
    }
}

export {main as mainCLI};
