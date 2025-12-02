// src/utils/FileUtils.js

/**
 * @fileoverview
 * - EN: A collection of file system utilities for Node.js environments.
 *   This class is primarily used by CLI tools for reading FX files and writing output packages.
 *   It uses Node's built-in `fs/promises` and `path` modules.
 * - TW: 適用於 Node.js 環境的檔案系統工具集合。
 *   此類別主要由 CLI 工具用於讀取 FX 檔案和寫入輸出套件。
 *   它使用 Node 內建的 `fs/promises` 和 `path` 模組。
 */

// Node.js 實作
import fs from 'fs/promises';
import path from 'path';

/**
 * - EN: Note: Compression functionality would require external dependencies, such as 'archiver'.
 *   Since we cannot add new dependencies, this will remain a placeholder.
 * - TW: 注意：壓縮功能需要外部依賴項，例如 'archiver'。
 *   由於我們無法添加新的依賴項，因此這將保留為佔位符。
 */

export class FileUtils {
    /**
     * - EN: Asynchronously reads the entire content of a file.
     * - TW: 非同步讀取檔案的全部內容。
     * @param {string} filePath - EN: The absolute or relative path to the file. - TW: 檔案的絕對或相對路徑。
     * @returns {Promise<string>} - EN: A Promise that resolves with the file's content (UTF-8 string). - TW: 解析為檔案內容 (UTF-8 字串) 的 Promise。
     */
    static async readFile(filePath) {
        try {
            return await fs.readFile(filePath, 'utf-8');
        } catch (error) {
            console.error(`讀取檔案 ${filePath} 時發生錯誤:`, error);
            /**
             * - EN: Re-throw the error for the caller to handle.
             * - TW: 重新拋出錯誤以供呼叫者處理。
             */
            throw error;
        }
    }

    /**
     * - EN: Asynchronously writes content to a file.
     *   If the directory structure does not exist, it will create it.
     * - TW: 非同步將內容寫入檔案。
     *   如果目錄結構不存在，它將建立目錄結構。
     * @param {string} filePath - EN: The path to the file to write. - TW: 要寫入的檔案路徑。
     * @param {string} content - EN: The content to write to the file. - TW: 要寫入檔案的內容。
     * @returns {Promise<void>} - EN: A Promise that resolves when the file write is complete. - TW: 檔案寫入完成時解析的 Promise。
     */
    static async writeFile(filePath, content) {
        try {
            const dir = path.dirname(filePath);
            await fs.mkdir(dir, {recursive: true});
            /**
             * - EN: Ensure the directory exists.
             * - TW: 確保目錄存在。
             */
            await fs.writeFile(filePath, content, 'utf-8');
        } catch (error) {
            console.error(`寫入檔案 ${filePath} 時發生錯誤:`, error);
            throw error;
        }
    }

    /**
     * - EN: Placeholder for zipping files.
     *   Since new dependencies (e.g., 'archiver') cannot be added, this function
     *   simulates the packaging process by writing files to a directory.
     * - TW: 壓縮檔案的佔位符。
     *   由於我們無法添加新的依賴項 (例如 'archiver')，此函數
     *   透過將檔案寫入目錄來模擬打包過程。
     * @param {Array<{name: string, content: string}>} files - EN: An array of file objects, each containing a name and content. - TW: 檔案物件陣列，每個物件包含名稱和內容。
     * @param {string} outputPath - EN: The path to the output .zip file. The .zip extension will be replaced with a directory. - TW: 輸出 .zip 檔案的路徑。 .zip 副檔名將替換為目錄。
     * @returns {Promise<void>} - EN: A Promise that resolves when the packaging is complete. - TW: 檔案寫入完成時解析的 Promise。
     */
    static async zipFiles(files, outputPath) {
        console.warn("FileUtils.zipFiles is a placeholder. Compression requires external libraries like 'archiver'.",
            /**
             * - EN: FileUtils.zipFiles is a placeholder. Compression requires external libraries like 'archiver'.
             * - TW: FileUtils.zipFiles 是一個佔位符。壓縮需要外部庫，例如 'archiver'。
             */
            "");
        const dir = path.dirname(outputPath);
        await fs.mkdir(dir, {recursive: true});

        /**
         * - EN: Create a directory named after the expected zip file.
         * - TW: 建立一個以預期 zip 檔案命名的目錄。
         */
        const outDir = outputPath.replace('.zip', '');
        await fs.mkdir(outDir, {recursive: true});

        /**
         * - EN: Write each file into the new directory.
         * - TW: 將每個檔案寫入新目錄。
         */
        for (const file of files) {
            await fs.writeFile(path.join(outDir, file.name), file.content, 'utf-8');
        }
        console.log(`[佔位符] 套件內容已寫入目錄: ${outDir}`);
    }

    /**
     * - EN: Checks if a file or directory exists at the given path.
     * - TW: 檢查給定路徑是否存在檔案或目錄。
     * @param {string} path - EN: The path to check. - TW: 要檢查的路徑。
     * @returns {Promise<boolean>} - EN: A Promise that resolves to true if the path exists, or false otherwise. - TW: 解析為 true (如果路徑存在) 或 false (否則) 的 Promise。
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