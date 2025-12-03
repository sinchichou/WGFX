// src/cli/OutputPackager.js
import fs from 'fs';
import archiver from 'archiver';

/**
 * @fileoverview
 * - EN: Packages final build artifacts into a distributable zip archive.
 * - TW: 將最終的建置產物打包成可分發的 zip 壓縮檔。
 */
export class OutputPackager {
    constructor() {
        /**
         * - EN: This class is stateless.
         * - TW: 此類別是無狀態的。
         */
    }

    /**
     * - EN: Packages compiled WGSL code, pipeline metadata, and general metadata into a zip file.
     * - TW: 將已編譯的 WGSL 程式碼、管線元資料和一般元資料打包成一個 zip 檔案。
     *
     * @param {Array<{ wgslCode: string, passIndex: number }>} generatedModules
     * - EN: An array of generated WGSL modules for each pass.
     * - TW: 每個通道生成的 WGSL 模組陣列。
     * @param {object} pipelineMetadata
     * - EN: Pipeline metadata JSON object.
     * - TW: 管線元資料 JSON 物件。
     * @param {object} generalMetadata
     * - EN: General effect metadata JSON object.
     * - TW: 一般效果元資料 JSON 物件。
     * @param {string} outputPath
     * - EN: The path to the output .zip file.
     * - TW: 輸出 .zip 檔案的路徑。
     * @returns {Promise<void>}
     * - EN: A promise that resolves when the archive is successfully created.
     * - TW: 當壓縮檔成功建立時解析的 Promise。
     */
    async package(generatedModules, pipelineMetadata, generalMetadata, outputPath) {
        console.log("OutputPackager: Packaging build artifacts to zip archive.");

        return new Promise((resolve, reject) => {
            const output = fs.createWriteStream(outputPath);
            const archive = archiver('zip', {
                zlib: {level: 9} // Sets the compression level.
            });

            output.on('close', () => {
                console.log(`OutputPackager: Successfully created archive with ${archive.pointer()} total bytes.`);
                resolve();
            });

            archive.on('warning', (err) => {
                if (err.code === 'ENOENT') {
                    console.warn('Archiver warning: ', err);
                } else {
                    reject(err);
                }
            });

            archive.on('error', (err) => {
                reject(err);
            });

            archive.pipe(output);

            // Add individual WGSL files for each pass
            generatedModules.forEach(module => {
                archive.append(module.wgslCode, {name: `pass_${module.passIndex}.wgsl`});
            });

            // Add pipeline metadata (now includes resource bindings)
            archive.append(JSON.stringify(pipelineMetadata, null, 2), {name: 'pipeline.json'});

            // Add general metadata
            archive.append(JSON.stringify(generalMetadata, null, 2), {name: 'metadata.json'});

            archive.finalize();
        });
    }
}
