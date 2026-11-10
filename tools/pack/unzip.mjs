// unzip.mjs — 最小 zip 讀取器（純 JS，可用於瀏覽器與 Node）
//
// 用途：`.wgx` 套件包在執行期的解壓。刻意不使用 Node 專用 API 或外部相依，
// 以便同一份程式碼在 Chrome 擴充套件（content script）與 Node 中都能跑。
//
// 支援：zip 的 local file header / central directory 解析、stored(0) 與
// deflate-raw(8) 兩種壓縮方法、ZIP64 之外的一般 zip（32 位元尺寸）。
//
// 不支援：加密 zip、zip64（>4GB）、多磁碟 archive。對 `.wgx` 而言綽綽有餘。

/** 解壓所需的 inflate 實作 —— 優先用瀏覽器內建的 DecompressionStream。 */
async function inflateRaw(data) {
    // Node 18+ / 現代瀏覽器皆有 DecompressionStream
    if (typeof DecompressionStream === 'function') {
        const ds = new DecompressionStream('deflate-raw');
        const stream = new Blob([data]).stream().pipeThrough(ds);
        const buf = await new Response(stream).arrayBuffer();
        return new Uint8Array(buf);
    }
    // Node 的 fallback（Node 18 的 DecompressionStream 已可用，這是保險）
    const zlib = await import('node:zlib');
    return new Uint8Array(zlib.inflateRawSync(Buffer.from(data)));
}

/**
 * 讀取 zip 內容。
 * @param {ArrayBuffer|Uint8Array} buffer zip bytes
 * @returns {Promise<Map<string, Uint8Array>>} 檔名 → 解壓後內容
 */
export async function unzip(buffer) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    // 從尾端找 End of Central Directory (EOCD, 0x06054b50)
    let eocd = -1;
    for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 22 - 0xffff); i--) {
        if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error('不是有效的 zip：找不到 EOCD');

    const entryCount = view.getUint16(eocd + 10, true);
    let cdOffset = view.getUint32(eocd + 16, true);

    const files = new Map();
    let ptr = cdOffset;

    for (let i = 0; i < entryCount; i++) {
        if (view.getUint32(ptr, true) !== 0x02014b50) {
            throw new Error(`zip central directory 損毀（entry ${i}）`);
        }
        const method = view.getUint16(ptr + 10, true);
        const compressedSize = view.getUint32(ptr + 20, true);
        const uncompressedSize = view.getUint32(ptr + 24, true);
        const nameLen = view.getUint16(ptr + 28, true);
        const extraLen = view.getUint16(ptr + 30, true);
        const commentLen = view.getUint16(ptr + 32, true);
        const localOffset = view.getUint32(ptr + 42, true);
        const name = new TextDecoder('utf-8').decode(bytes.subarray(ptr + 46, ptr + 46 + nameLen));

        // 跳到 local header 取得實際資料起點
        if (view.getUint32(localOffset, true) !== 0x04034b50) {
            throw new Error(`zip local header 損毀：${name}`);
        }
        const lNameLen = view.getUint16(localOffset + 26, true);
        const lExtraLen = view.getUint16(localOffset + 28, true);
        const dataStart = localOffset + 30 + lNameLen + lExtraLen;
        const raw = bytes.subarray(dataStart, dataStart + compressedSize);

        if (!name.endsWith('/')) {
            let content;
            if (method === 0) {
                content = raw;
            } else if (method === 8) {
                content = await inflateRaw(raw);
            } else {
                throw new Error(`不支援的 zip 壓縮方法 ${method}：${name}`);
            }
            // 以 central directory 記錄的原始大小做健全性檢查
            if (uncompressedSize > 0 && content.length !== uncompressedSize) {
                throw new Error(
                    `zip entry 大小不符：${name}（預期 ${uncompressedSize}，實得 ${content.length}）`);
            }
            files.set(name, content);
        }

        ptr += 46 + nameLen + extraLen + commentLen;
    }

    return files;
}

/**
 * 讀取 zip 內的單一檔案為文字。
 * @param {Map<string, Uint8Array>} files
 * @param {string} name
 */
export function readText(files, name) {
    const data = files.get(name);
    if (!data) throw new Error(`套件包內找不到 ${name}`);
    return new TextDecoder('utf-8').decode(data);
}
