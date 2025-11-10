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

import {StaticParser} from './StaticParser.js';
import {WGSLCodeGeneratorCLI} from './WGSLCodeGeneratorCLI.js';
import {PipelineMetadataGenerator} from './PipelineMetadataGenerator.js';
import {OutputPackager} from './OutputPackager.js';
import {FileUtils} from '../utils/FileUtils.js';

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

        // 2. Parse the file into an Intermediate Representation (IR).
        console.log("Parsing FX file...");
        const staticParser = new StaticParser();
        const shaderInfo = staticParser.parse(fxCode);

        // 3. Generate WGSL code from the IR.
        console.log("Generating WGSL code...");
        const codeGenerator = new WGSLCodeGeneratorCLI();
        const wgslCode = codeGenerator.generate(shaderInfo);

        // 4. Generate metadata JSON from the IR.
        console.log("Generating pipeline metadata...");
        const metadataGenerator = new PipelineMetadataGenerator();
        const {pipeline, metadata} = metadataGenerator.generate(shaderInfo);

        // 5. Package all artifacts for distribution.
        console.log(`Packaging output to ${outputZipFile}...`);
        const packager = new OutputPackager();
        await packager.package(wgslCode, pipeline, metadata, outputZipFile);

        console.log(`
Successfully compiled ${inputFxFile}.`);

    } catch (error) {
        console.error("\nCLI Compilation failed:", error);
        process.exit(1);
    }
}

main();
