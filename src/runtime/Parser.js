// src/runtime/Parser.js

/**
 * @fileoverview WGFX 著色器檔案的核心解析器。
 * 此解析器讀取 .fx 檔案字串，並將其轉換為結構化的
 * 中介表示 (IR) 物件，即 `WGFXShaderInfo`。
 * 它遵循 README 中定義的基於區塊的指令系統 (`//! DIRECTIVE`)。
 */

/**
 * 描述著色器的統一參數。
 * @typedef {Object} WGFXParameter
 * @property {string} name - 著色器程式碼中的變數名稱 (例如："Strength")。
 * @property {string} id - 參數區塊的識別符 (例如："Strength")。
 * @property {string} type - 資料類型 ('float' 或 'int')。
 * @property {string} [label] - 此參數的 UI 標籤 (例如："Effect Strength")。
 * @property {number} default - 預設值。
 * @property {number} min - 允許的最小值。
 * @property {number} max - 允許的最大值。
 * @property {number} step - 滑桿步進值。
 */

/**
 * 描述紋理資源。
 * @typedef {Object} WGFXTexture
 * @property {string} name - 紋理資源的名稱 (例如："InputTexture")。
 * @property {string} id - 紋理區塊的識別符 (例如："InputTexture")。
 * @property {string} [source] - 紋理的來源 (如果由檔案支援)。
 * @property {string} [format] - 紋理格式 (例如："R8G8B8A8_UNORM")。
 * @property {string} [width] - 寬度表達式 (例如："INPUT_WIDTH")。
 * @property {string} [height] - 高度表達式 (例如："INPUT_HEIGHT")。
 */

/**
 * 描述取樣器資源。
 * @typedef {Object} WGFXSampler
 * @property {string} name - 取樣器資源的名稱 (例如："LinearSampler")。
 * @property {string} id - 取樣器區塊的識別符 (例如："LinearSampler")。
 * @property {'LINEAR' | 'POINT'} filter - 濾波模式。
 * @property {'CLAMP' | 'WRAP'} [address] - 位址模式。
 */

/**
 * 描述單個計算通道。
 * @typedef {Object} WGFXPass
 * @property {number} index - 通道索引，從 1 開始。
 * @property {string[]} in - 用作輸入的資源名稱列表。
 * @property {string[]} out - 用作輸出的資源名稱列表。
 * @property {[number, number]} [blockSize] - 區塊大小 [w, h]。
 * @property {[number, number, number]} [numThreads] - 執行緒數量 [x, y, z]。
 * @property {'PS' | 'CS'} [style] - 著色器樣式。'CS' 代表計算著色器，'PS' 代表像素著色器樣式。
 * @property {string} [desc] - 通道的描述。
 * @property {string} code - 此通道的 WGSL 程式碼。
 */

/**
 * 已解析 WGFX 著色器檔案的中介表示 (IR)。
 * @typedef {Object} WGFXShaderInfo
 * @property {Object.<string, any>} metadata - 一般元資料 (版本、排序名稱、使用、功能)。
 * @property {WGFXParameter[]} parameters - 統一參數列表。
 * @property {WGFXTexture[]} textures - 紋理資源列表。
 * @property {WGFXSampler[]} samplers - 取樣器資源列表。
 * @property {WGFXPass[]} passes - 渲染通道列表。
 * @property {string} commonCode - 要預先添加的通用 WGSL 程式碼區塊。
 */

/**
 * 基於狀態機的 WGFX 著色器檔案解析器。
 */
export class ShaderParser {
    /**
     * @param {boolean} [debug=false] - 如果為 true，則將解析步驟記錄到控制台。
     */
    constructor(debug = false) {
        this.debug = debug;
    }

    /**
     * 將 WGFX/MagpieFX 著色器字串解析為結構化的 IR。
     * @param {string} shaderCode - 完整的著色器程式碼字串。
     * @returns {WGFXShaderInfo} 表示著色器 IR 的結構化物件。
     */
    parse(shaderCode) {
        if (this.debug) console.log("--- 開始解析 WGFX 著色器 ---");

        const lines = shaderCode.split('\n');

        /** @type {WGFXShaderInfo} */
        const info = {
            metadata: {use: {}, capability: {}},
            parameters: [],
            textures: [],
            samplers: [],
            passes: [],
            commonCode: '',
        };

        let currentBlock = null; // 當前區塊的類型，例如 'PARAMETER'、'TEXTURE'、'PASS'。
        let currentData = null;  // 為當前區塊建立的物件。
        let passCodeBuffer = []; // 用於在 PASS 區塊內累積程式碼行的緩衝區。

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const trimmedLine = line.trim();

            // 這行是指令嗎？
            if (trimmedLine.startsWith('//!')) {
                // 如果我們在 PASS 區塊中，則其程式碼在此結束。
                if (currentBlock === 'PASS' && passCodeBuffer.length > 0) {
                    currentData.code = passCodeBuffer.join('\n');
                    passCodeBuffer = [];
                }
                // 新指令總是提交前一個區塊的資料。
                if (currentData) {
                    this._commitBlock(info, currentBlock, currentData);
                    currentData = null;
                }

                // 解析指令本身。
                const [directive, ...args] = trimmedLine.substring(3).trim().split(/\s+/);
                const value = args.join(' ');

                // 這是頂層區塊指令嗎？
                if (!currentBlock || ['PARAMETER', 'TEXTURE', 'SAMPLER', 'COMMON', 'PASS'].includes(directive)) {
                    currentBlock = directive;
                    if (currentBlock === 'COMMON') {
                        // COMMON 區塊很特殊；它只收集原始行。
                        currentData = {lines: []};
                    } else {
                        // 其他區塊以 ID 開頭。
                        currentData = {id: value};
                        if (currentBlock === 'PASS') {
                            currentData.index = parseInt(value, 10);
                        }
                    }
                } else {
                    // 這是當前區塊的子指令 (例如，PARAMETER 的 //! MIN)。
                    this._parseSubDirective(currentData, directive, value);
                }
            } else if (currentBlock === 'COMMON') {
                // 如果我們在 COMMON 區塊中，只需附加該行。
                currentData.lines.push(line);
            } else if (currentBlock === 'PASS') {
                // 如果我們在 PASS 區塊中，將該行附加到其程式碼緩衝區。
                passCodeBuffer.push(line);
            } else if (trimmedLine && !trimmedLine.startsWith('//')) {
                // 這是一行宣告 (例如，"float Strength;")。
                // 它為當前區塊提供 `type` 和 `name`。
                if (currentData) {
                    const declMatch = trimmedLine.match(/(\w+)\s+(\w+);/);
                    if (declMatch) {
                        const [, type, name] = declMatch;
                        currentData.type = type;
                        currentData.name = name;
                    }
                }
            }
        }

        // 提交檔案末尾的任何剩餘資料。
        if (currentBlock === 'PASS' && passCodeBuffer.length > 0) {
            currentData.code = passCodeBuffer.join('\n');
        }
        if (currentData) {
            this._commitBlock(info, currentBlock, currentData);
        }

        this._validate(info);

        if (this.debug) {
            console.log('--- 解析完成 ---');
            console.log('最終 ShaderInfo 物件:', JSON.stringify(info, null, 2));
        }

        return info;
    }

    /**
     * 解析區塊內的子指令，並將其添加到當前資料物件中。
     * @param {Object} data - 正在建立的當前區塊的物件。
     * @param {string} directive - 子指令鍵 (例如："MIN"、"FILTER")。
     * @param {string} value - 子指令的值。
     * @private
     */
    _parseSubDirective(data, directive, value) {
        const lowerDirective = directive.toLowerCase();
        switch (lowerDirective) {
            case 'in':
            case 'out':
                data[lowerDirective] = value.split(',').map(s => s.trim());
                break;
            case 'block_size':
                data.blockSize = value.split(',').map(s => parseInt(s.trim(), 10));
                if (data.blockSize.length === 1) data.blockSize.push(data.blockSize[0]); // 允許單個值表示正方形大小
                break;
            case 'num_threads':
                data.numThreads = value.split(',').map(s => parseInt(s.trim(), 10));
                while (data.numThreads.length < 3) data.numThreads.push(1); // 填充到 3 個維度
                break;
            case 'default':
            case 'min':
            case 'max':
            case 'step':
                data[lowerDirective] = parseFloat(value);
                break;
            default:
                data[lowerDirective] = value;
        }
    }

    /**
     * 將已完成的區塊資料物件提交到主要的著色器資訊 IR 中。
     * @param {WGFXShaderInfo} info - 主要 IR 物件。
     * @param {string} blockType - 正在提交的區塊類型 (例如："PARAMETER")。
     * @param {Object} data - 區塊的資料物件。
     * @private
     */
    _commitBlock(info, blockType, data) {
        switch (blockType) {
            case 'MAGPIE': // 來自 "MAGPIE EFFECT"
                // 這是魔術檢查，不作為資料儲存。
                break;
            case 'VERSION':
                info.metadata.version = parseInt(data.id, 10);
                break;
            case 'SORT_NAME':
                info.metadata.sortName = data.id;
                break;
            case 'USE':
                data.id.split(',').map(s => s.trim().toUpperCase()).forEach(flag => info.metadata.use[flag] = true);
                break;
            case 'CAPABILITY':
                data.id.split(',').map(s => s.trim().toUpperCase()).forEach(flag => info.metadata.capability[flag] = true);
                break;
            case 'PARAMETER':
                info.parameters.push(data);
                break;
            case 'TEXTURE':
                info.textures.push(data);
                break;
            case 'SAMPLER':
                info.samplers.push(data);
                break;
            case 'COMMON':
                info.commonCode = data.lines.join('\n');
                break;
            case 'PASS':
                info.passes.push(data);
                break;
        }
    }

    /**
     * 對最終 IR 執行驗證檢查。
     * @param {WGFXShaderInfo} info - 要驗證的 IR。
     * @private
     */
    _validate(info) {
        if (info.metadata.version === undefined) {
            throw new Error("`//! VERSION` 是必需的。 উত্ত");
        }

        // 驗證通道索引是否連續。
        info.passes.sort((a, b) => a.index - b.index);
        const expectedStartIndex = info.passes.length > 0 && info.passes[0].index === 0 ? 0 : 1;
        for (let i = 0; i < info.passes.length; i++) {
            if (info.passes[i].index !== i + expectedStartIndex) {
                throw new Error(`通道索引必須連續。預期為 ${i + expectedStartIndex}，但找到 ${info.passes[i].index}。`);
            }
        }

        // TODO: 添加 README 中的更多驗證檢查，例如：
        // - PARAMETER 的必需欄位 (DEFAULT, MIN, MAX, STEP)。
        // - 通道 IN/OUT 規則。
        // - 非 PS 樣式的 BLOCK_SIZE/NUM_THREADS 規則。
    }
}
