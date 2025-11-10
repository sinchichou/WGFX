// src/utils/FileUtils.js

/**
 * @fileoverview 適用於 Node.js 環境的檔案系統工具集合。
 * 此類別主要由 CLI 工具用於讀取 FX 檔案和寫入輸出套件。
 * 它使用 Node 內建的 `fs/promises` 和 `path` 模組。
 */

// Node.js 實作
import fs from 'fs/promises';
import path from 'path';

// 注意：壓縮功能需要外部依賴項，例如 'archiver'。
// 由於我們無法添加新的依賴項，因此這將保留為佔位符。

export class FileUtils {
    /**
     * 非同步讀取檔案的全部內容。
     * @param {string} filePath - 檔案的絕對或相對路徑。
     * @returns {Promise<string>} 解析為檔案內容 (UTF-8 字串) 的 Promise。
     */
    static async readFile(filePath) {
        try {
            return await fs.readFile(filePath, 'utf-8');
        } catch (error) {
            console.error(`讀取檔案 ${filePath} 時發生錯誤:`, error);
            throw error; // 重新拋出錯誤以供呼叫者處理。
        }
    }

    /**
     * 非同步將內容寫入檔案。
     * 如果目錄結構不存在，它將建立目錄結構。
     * @param {string} filePath - 要寫入的檔案路徑。
     * @param {string} content - 要寫入檔案的內容。
     * @returns {Promise<void>} 檔案寫入完成時解析的 Promise。
     */
    static async writeFile(filePath, content) {
        try {
            const dir = path.dirname(filePath);
            await fs.mkdir(dir, {recursive: true}); // 確保目錄存在。
            await fs.writeFile(filePath, content, 'utf-8');
        } catch (error) {
            console.error(`寫入檔案 ${filePath} 時發生錯誤:`, error);
            throw error;
        }
    }

    /**
     * 壓縮檔案的佔位符。
     * 由於無法添加新的依賴項 (例如 'archiver')，此函數
     * 透過將檔案寫入目錄來模擬打包過程。
     * @param {Array<{name: string, content: string}>} files - 檔案物件陣列，每個物件包含名稱和內容。
     * @param {string} outputPath - 輸出 .zip 檔案的路徑。 .zip 副檔名將替換為目錄。
     * @returns {Promise<void>}
     */
    static async zipFiles(files, outputPath) {
        console.warn("FileUtils.zipFiles 是一個佔位符。壓縮需要外部庫，例如 'archiver'。");

        const dir = path.dirname(outputPath);
        await fs.mkdir(dir, {recursive: true});

        // 建立一個以預期 zip 檔案命名的目錄。
        const outDir = outputPath.replace('.zip', '');
        await fs.mkdir(outDir, {recursive: true});

        // 將每個檔案寫入新目錄。
        for (const file of files) {
            await fs.writeFile(path.join(outDir, file.name), file.content, 'utf-8');
        }
        console.log(`[佔位符] 套件內容已寫入目錄: ${outDir}`);
    }

    /**
     * 檢查給定路徑是否存在檔案或目錄。
     * @param {string} path - 要檢查的路徑。
     * @returns {Promise<boolean>} 解析為 true (如果路徑存在) 或 false (否則) 的 Promise。
     */
    static async pathExists(path) {
        try {
            await fs.access(path);
            return true;
        } catch {
            return false;
        }
    }
}