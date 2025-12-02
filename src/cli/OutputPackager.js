// src/cli/OutputPackager.js

/**
 * @fileoverview
 * - EN: Packages final build artifacts into a distributable format.
 *   This class receives generated WGSL code and metadata JSON and uses FileUtils
 *   to save them.
 * - TW: 將最終的建置產物打包成可分發的格式。
 *   此類別接收生成的 WGSL 程式碼和元資料 JSON，並使用 FileUtils
 *   將它們儲存。
 */

import {FileUtils} from '../utils/FileUtils.js';

export class OutputPackager {
    constructor() {
        /**
         * - EN: This class is stateless.
         * - TW: 此類別是無狀態的。
         */
    }

    /**
     * - EN: Packages compiled WGSL code, pipeline metadata, and general metadata.
     *
     *   In a full implementation, this would create a .zip archive. However, since
     *   compression requires external Node.js dependencies that cannot be added here,
     *   it uses a placeholder `FileUtils.zipFiles` function, which writes content to
     *   a directory instead of a zip file.
     * - TW: 打包已編譯的 WGSL 程式碼、管線元資料和一般元資料。
     *
     *   在完整的實作中，這將建立一個 .zip 壓縮檔。然而，由於
     *   壓縮需要外部 Node.js 依賴項，而此處無法添加，
     *   因此它使用佔位符 `FileUtils.zipFiles`，該函數將內容寫入
     *   目錄而不是壓縮檔。
     *
     * @param {Array<{ wgslCode: string, passIndex: number }>} generatedModules - EN: An array of generated WGSL modules for each pass. - TW: 每個通道生成的 WGSL 模組陣列。
     * @param {object} pipelineMetadata - EN: Pipeline metadata JSON object. - TW: 管線元資料 JSON 物件。
     * @param {object} generalMetadata - EN: General effect metadata JSON object. - TW: 一般效果元資料 JSON 物件。
     * @param {string} outputPath - EN: The path to the output ".zip" file (will be created as a directory). - TW: 輸出 ".zip" 檔案的路徑 (將建立為目錄)。
     * @returns {Promise<void>}
     */
    async package(generatedModules, pipelineMetadata, generalMetadata, outputPath) {
        /**
         * - EN: Packaging build artifacts.
         * - TW: 正在打包建置產物。
         */
        console.log("OutputPackager: Packaging build artifacts.");
        const filesToPackage = [];

        // Add individual WGSL files for each pass
        generatedModules.forEach(module => {
            filesToPackage.push({
                name: `pass_${module.passIndex}.wgsl`,
                content: module.wgslCode
            });
        });

        // Add pipeline metadata (now includes resource bindings)
        filesToPackage.push({
            name: 'pipeline.json',
            content: JSON.stringify(pipelineMetadata, null, 2)
        });

        // Add general metadata
        filesToPackage.push({
            name: 'metadata.json',
            content: JSON.stringify(generalMetadata, null, 2)
        });

        // Use FileUtils to write the package.
        await FileUtils.zipFiles(filesToPackage, outputPath);

        /**
         * - EN: Packaging complete.
         * - TW: 打包完成。
         */
        console.log("OutputPackager: Packaging complete.");
    }
}