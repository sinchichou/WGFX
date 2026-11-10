// manifest.mjs — `.wgx` 套件包 manifest 的 schema 定義、產生與驗證
//
// `.wgx` 是 WGFX 效果的「預編」套件包（zip 容器）。manifest 描述套件包內的
// 一切：各 pass 的預生成 WGSL、權重二進位、外部紋理與資源需求。
//
// 設計原則
// ---
// 1. **純資料**：manifest 是 JSON，不含任何程式碼 → MV3 CSP 完全適用。
// 2. **自足**：解包後不需要再回頭找原始 `.wgsl` 或 `effects_src/`。
// 3. **可驗證**：欄位結構與跨檔案一致性由 {@link validateManifest} 檢查。

/** 目前的 manifest 格式版本。 */
export const MANIFEST_VERSION = 1;

/**
 * 資源需求描述。
 * @typedef {object} WgxResource
 * @property {'texture'|'storageBuffer'} kind
 * @property {string} name            邏輯名稱（框架內的資源名）
 * @property {number} [binding]       binding 編號
 * @property {string} [format]         紋理格式（WebGPU 格式名）
 * @property {number[]} [size]        [寬, 高]（紋理）或位元組數（buffer）
 * @property {number} [stride]         buffer 結構 stride（位元組）
 * @property {string} [file]           套件包內的相對路徑（二進位資料）
 * @property {'read'|'read_write'} [mode]
 * @property {number} [count]          元素數（packed f16 時為 u32 對數）
 * @property {boolean} [runtimeSized]  大小於執行期依輸入尺寸決定（CNN 特徵圖等）
 * @property {string} [sizeHint]       大小推導規則的識別碼（供 runtime 計算）
 */

/**
 * 一個 pass 的描述。
 * @typedef {object} WgxPass
 * @property {number} index            pass 序號（對應 WGFX 的 pass.index）
 * @property {string} file              套件包內的 WGSL 檔案路徑
 * @property {string[]} [dispatch]      //! DISPATCH 運算式（若有）
 * @property {number[]} [workgroupSize]
 * @property {string[]} [inputs]        邏輯紋理名
 * @property {string[]} [outputs]       邏輯紋理名
 * @property {WgxResource[]} [resources] 此 pass 實際綁定的資源
 */

/**
 * `.wgx` manifest。
 * @typedef {object} WgxManifest
 * @property {number} version           格式版本
 * @property {string} id                效果識別碼
 * @property {string} name              顯示名稱
 * @property {string} [family]          家族（Anime4K / CuNNy / …）
 * @property {string} [description]
 * @property {string} [source]          原始效果來源（URL 或路徑）
 * @property {string[]} [tags]
 * @property {object} [metadata]        原始效果 metadata（sortName / version / use / capability）
 * @property {object} [parameters]      可調參數（沿用 WGFX 的 ParameterInfo 形狀）
 * @property {WgxPass[]} passes
 * @property {WgxResource[]} [resources] 全域資源（權重、外部紋理）
 * @property {object} [limits]          此效果建議的裝置上限
 * @property {string} [buildTool]       產生此套件包的工具版本
 * @property {string} [builtAt]         ISO 時間戳
 */

/**
 * 產生一份 manifest。
 * @param {Partial<WgxManifest>} fields
 * @returns {WgxManifest}
 */
export function createManifest(fields) {
    return {
        version: MANIFEST_VERSION,
        id: fields.id,
        name: fields.name || fields.id,
        family: fields.family,
        description: fields.description,
        source: fields.source,
        tags: fields.tags || [],
        metadata: fields.metadata || {},
        parameters: fields.parameters || [],
        passes: fields.passes || [],
        resources: fields.resources || [],
        limits: fields.limits,
        buildTool: fields.buildTool,
        builtAt: fields.builtAt || new Date().toISOString(),
    };
}

/**
 * 驗證 manifest 的結構與跨欄位一致性。
 * @param {WgxManifest} manifest
 * @param {object} [opts]
 * @param {Set<string>} [opts.files] 套件包內實際存在的檔案集合（做交叉檢查）
 * @returns {string[]} 問題清單（空陣列代表通過）
 */
export function validateManifest(manifest, opts = {}) {
    const issues = [];

    if (!manifest || typeof manifest !== 'object') {
        issues.push('manifest 不是物件');
        return issues;
    }
    if (manifest.version !== MANIFEST_VERSION) {
        issues.push(`manifest 版本不支援: ${manifest.version}（期望 ${MANIFEST_VERSION}）`);
    }
    if (!manifest.id || typeof manifest.id !== 'string') {
        issues.push('manifest 缺少 id');
    }
    if (!Array.isArray(manifest.passes)) {
        issues.push('manifest.passes 必須是陣列');
        return issues;
    }
    if (manifest.passes.length === 0) {
        issues.push('manifest.passes 為空 — 套件包至少要有一個 pass');
    }

    // pass index 必須連續且不重複（WGFX 以 index 為鍵）
    const seen = new Set();
    manifest.passes.forEach((p, i) => {
        if (typeof p.index !== 'number') {
            issues.push(`pass[${i}] 缺少 index`);
            return;
        }
        if (seen.has(p.index)) {
            issues.push(`pass index 重複: ${p.index}`);
        }
        seen.add(p.index);
        if (!p.file) {
            issues.push(`pass ${p.index} 缺少 WGSL 檔案路徑`);
        } else if (opts.files && !opts.files.has(p.file)) {
            issues.push(`pass ${p.index} 指向不存在的檔案: ${p.file}`);
        }
        if (p.dispatch && !Array.isArray(p.dispatch)) {
            issues.push(`pass ${p.index} 的 dispatch 必須是陣列`);
        }
        if (p.dispatch && p.dispatch.some((d) => typeof d !== 'string')) {
            issues.push(`pass ${p.index} 的 dispatch 含有非字串項`);
        }
    });

    // 資源：二進位資源必須有檔案,且檔案需存在
    const allResources = [...(manifest.resources || [])];
    for (const p of manifest.passes) {
        for (const r of p.resources || []) allResources.push(r);
    }
    for (const r of allResources) {
        if (!r.name) {
            issues.push('資源缺少 name');
            continue;
        }
        if (r.file) {
            if (opts.files && !opts.files.has(r.file)) {
                issues.push(`資源 ${r.name} 指向不存在的檔案: ${r.file}`);
            }
        } else if (r.kind === 'storageBuffer') {
            // 沒有二進位檔的 storage buffer 必須宣告大小。
            // 例外：`runtimeSized` 者（如 CNN 的特徵圖緩衝區）大小取決於輸入
            // 尺寸，只能在執行期決定 —— 這類資源由 manifest 記錄「來源」，
            // 實際大小由宿主依輸入尺寸配置。
            if (!r.runtimeSized && (typeof r.size !== 'number' || r.size <= 0)) {
                issues.push(`storage buffer ${r.name} 缺少 size`);
            }
        }
    }

    // 參數：型別與範圍
    for (const p of manifest.parameters || []) {
        if (!p.name) issues.push('參數缺少 name');
        if (p.type && !['float', 'int', 'bool'].includes(p.type)) {
            issues.push(`參數 ${p.name} 的型別未知: ${p.type}`);
        }
    }

    return issues;
}

/**
 * 資源查表：依邏輯名稱找出資源描述。
 * @param {WgxManifest} manifest
 * @returns {Map<string, WgxResource>}
 */
export function indexResources(manifest) {
    const map = new Map();
    for (const r of manifest.resources || []) map.set(r.name, r);
    for (const p of manifest.passes || []) {
        for (const r of p.resources || []) {
            if (!map.has(r.name)) map.set(r.name, r);
        }
    }
    return map;
}
