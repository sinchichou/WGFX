// src/cli/OutputPackager.js

/**
 * @fileoverview 將最終的建置產物打包成可分發的格式。
 * 此類別接收生成的 WGSL 程式碼和元資料 JSON，並使用 FileUtils
 * 將它們儲存。
 */

import {FileUtils} from '../utils/FileUtils.js';

export class OutputPackager {
    constructor() {
        // 此類別是無狀態的。
    }

    /**
     * 打包已編譯的 WGSL 程式碼、管線元資料和一般元資料。
     *
     * 在完整的實作中，這將建立一個 .zip 壓縮檔。然而，由於
     * 壓縮需要外部 Node.js 依賴項，而此處無法添加，
     * 因此它使用佔位符 `FileUtils.zipFiles`，該函數將內容寫入
     * 目錄而不是壓縮檔。
     *
     * @param {string} wgslCode - 已編譯的 WGSL 著色器程式碼。
     * @param {object} pipelineMetadata - 管線元資料 JSON 物件。
     * @param {object} generalMetadata - 一般效果元資料 JSON 物件。
     * @param {string} outputPath - 輸出 ".zip" 檔案的路徑 (將建立為目錄)。
     * @returns {Promise<void>}
     */
    async package(generatedModules, pipelineMetadata, generalMetadata, outputPath) {
        console.log("OutputPackager: 正在打包建置產物...");

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

        console.log("OutputPackager: 打包完成。");
    }
}