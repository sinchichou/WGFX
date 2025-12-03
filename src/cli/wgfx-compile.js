// src/cli/wgfx-compile.js

/**
 * @fileoverview
 * - EN: The main entry point for the WGFX command-line interface (CLI).
 *   This script orchestrates the static compilation process of a .fx file into
 *   a distributable package.
 * - TW: WGFX 命令列介面 (CLI) 的主要入口點。
 *   此腳本協調 .fx 檔案到可分發套件的靜態編譯過程。
 *
 * @example
 * - EN: To run from the project root:
 *   node src/cli/wgfx-compile.js --input examples/demo.fx --output build/demo.zip
 * - TW: 從專案根目錄運行：
 *   node src/cli/wgfx-compile.js --input examples/demo.fx --output build/demo.zip
 */
import {createRequire} from 'module';

const require = createRequire(import.meta.url);

import {WGSLCodeGenerator} from '../runtime/WGSLCodeGenerator.js';
import {PipelineMetadataGenerator} from './PipelineMetadataGenerator.js';
import {OutputPackager} from './OutputPackager.js';
import {FileUtils} from '../utils/FileUtils.js';

/**
 * - EN: Directly use the runtime parser.
 * - TW: 直接使用運行時解析器。
 */
import {parse} from '../runtime/ShaderParser.js';

/**
 * - EN: A simple command-line argument parser.
 *   Parses arguments in the format `--key value`.
 * - TW: 一個簡單的命令列參數解析器。
 *   解析 `--key value` 格式的參數。
 * @param {string[]} args
 * - EN: The array of command-line arguments (e.g., from `process.argv.slice(2)`).
 * - TW: 命令列參數陣列 (例如，來自 `process.argv.slice(2)`）。
 * @returns {Object.<string, string|boolean>}
 * - EN: An object mapping keys to their values.
 * - TW: 將鍵映射到其值的物件。
 */
function parseArgs(args) {
    const options = {};
    for (let i = 0; i < args.length; i++) {
        if (args[i].startsWith('--')) {
            const key = args[i].substring(2);
            /**
             * - EN: If the next argument is not a key, it's the value for the current key.
             * - TW: 如果下一個參數不是鍵，則它是當前鍵的值。
             */
            const value = (i + 1 < args.length && !args[i + 1].startsWith('--')) ? args[i + 1] : true;
            options[key] = value;
        }
    }
    return options;
}

/**
 * - EN: The main function for the WGFX CLI script.
 * - TW: WGFX CLI 腳本的主函數。
 */
async function main() {
    /**
     * - EN: Console output for CLI start.
     * - TW: CLI 啟動的控制台輸出。
     */
    console.log("--- WGFX CLI Compiler --- \n");
    /**
     * - EN: Parse command-line arguments.
     * - TW: 解析命令列參數。
     */
    const options = parseArgs(process.argv.slice(2));

    const inputFxFile = options.input;
    const outputZipFile = options.output;

    /**
     * - EN: Validate required arguments.
     * - TW: 驗證所需參數。
     */
    if (!inputFxFile || !outputZipFile) {
        console.error("Error: Missing required arguments.");
        console.error("Usage: node wgfx-compile.js --input <path/to/effect.fx> --output <path/to/output.zip>");
        /**
         * - EN: Exit with an error code.
         * - TW: 以錯誤代碼退出。
         */
        process.exit(1);
    }

    try {
        /**
         * - EN: 1. Read the source .wgsl file.
         * - TW: 1. 讀取源 .wgsl 檔案。
         */
        console.log(`Reading WGSL file: ${inputFxFile}`);
        const fxCode = await FileUtils.readFile(inputFxFile);

        /**
         * - EN: 2. Parse the file into an Intermediate Representation (IR) using the runtime parser directly.
         * - TW: 2. 使用運行時解析器直接將檔案解析為中介表示 (IR)。
         */
        console.log("Parsing WGSL file...");
        const shaderInfo = parse(fxCode);

        /**
         * - EN: 3. Generate WGSL code from the IR using the runtime generator directly.
         * - TW: 3. 使用運行時生成器直接從 IR 生成 WGSL 程式碼。
         */
        console.log("Generating WGSL code and pass-specific resource bindings...");
        const codeGenerator = new WGSLCodeGenerator();
        /**
         * - EN: The runtime generator produces an array of modules, each containing WGSL code and its bound resources.
         * - TW: 運行時生成器生成一個模組陣列，每個模組包含 WGSL 程式碼及其綁定資源。
         */
        const generatedModules = codeGenerator.generate(shaderInfo);

        /**
         * - EN: 4. Generate metadata JSON from the IR.
         * - TW: 4. 從 IR 生成元資料 JSON。
         */
        console.log("Generating pipeline metadata...");
        const metadataGenerator = new PipelineMetadataGenerator();
        const {pipeline, metadata} = metadataGenerator.generate(shaderInfo);

        /**
         * - EN: Augment pipeline metadata with pass-specific resource binding information.
         * - TW: 使用通道特定的資源綁定資訊擴充管線元資料。
         */
        pipeline.passes.forEach(p => {
            const correspondingModule = generatedModules.find(m => m.passIndex === p.index);
            if (correspondingModule) {
                p.resources = correspondingModule.resources;
            }
        });

        /**
         * - EN: 5. Package all artifacts for distribution.
         * - TW: 5. 將所有構件打包以便分發。
         */
        console.log(`Packaging output to ${outputZipFile}...`);
        const packager = new OutputPackager();
        await packager.package(generatedModules, pipeline, metadata, outputZipFile);

        console.log(`
Successfully compiled ${inputFxFile}.`);

    } catch (error) {
        /**
         * - EN: Log compilation failure.
         * - TW: 記錄編譯失敗。
         */
        console.error("\nCLI Compilation failed:", error);
        /**
         * - EN: Exit with an error code.
         * - TW: 以錯誤代碼退出。
         */
        process.exit(1);
    }
}

export {main as mainCLI};
/**
 * - EN: Export the main function for CLI usage.
 * - TW: 導出主函數以供 CLI 使用。
 */