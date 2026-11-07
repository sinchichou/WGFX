// =====================================================================================
// IMPORTS & DEPENDENCIES
// =====================================================================================
import * as monaco from "monaco-editor";
// @ts-ignore
import { WGFX, LogLevel } from "wgfx";

// WESL WebAssembly imports
// 1. 修改 Import路徑：現在從src目錄下載入了相對路徑(移除錯誤的named export)
import initWesl, { run } from "./wesl-web/wesl_web.js";

// 2. 關鍵：使用?url告訴Vite將這個當作外部資源網址
// @ts-ignore (如果沒有設定d.ts文件能會報錯，這裡忽略即可)
import wasmUrl from "./wesl-web/wesl_web_bg.wasm?url";

// =====================================================================================
// CONSTANTS & DEFAULT VALUES
// =====================================================================================

// 編輯器初始佔位內容（實際著色器由 loadFileList() + checkExternalShader() 流程自動載入，
// 預設會自動選取 examples/test.wgsl）
const INITIAL_SHADER_PLACEHOLDER =
    "// 請從左上角選單載入 examples 下的 .wgsl 特效檔，或直接貼上 WGFX DSL 著色器";

// =====================================================================================
// DOM ELEMENTS
// =====================================================================================

// Core canvas and input elements
const canvas = document.getElementById("output-canvas") as HTMLCanvasElement;
const fileInput = document.getElementById("input-media") as HTMLInputElement;
const compileBtn = document.getElementById("btn-compile") as HTMLButtonElement;
const statusText = document.getElementById("status-text") as HTMLSpanElement;
const fpsMeter = document.getElementById("fps-meter") as HTMLDivElement;

// Video control elements
const videoControls = document.getElementById(
    "video-controls"
) as HTMLDivElement;
const btnPlayPause = document.getElementById(
    "btn-play-pause"
) as HTMLButtonElement;
const btnStop = document.getElementById("btn-stop") as HTMLButtonElement;
const seekBar = document.getElementById("seek-bar") as HTMLInputElement;
const timeDisplay = document.getElementById("time-display") as HTMLSpanElement;

// Shader and UI control elements
const chkAutoSync = document.getElementById(
    "chk-auto-sync"
) as HTMLInputElement;
const btnReload = document.getElementById("btn-reload") as HTMLButtonElement;
const shaderSelector = document.getElementById(
    "shader-selector"
) as HTMLSelectElement;
const chkColorFilters = document.getElementById(
    "chk-color-filters"
) as HTMLInputElement;

// Volume control elements
const volumeDisplay = document.getElementById(
    "volume-display"
) as HTMLSpanElement;
const volumeBar = document.getElementById("volume-bar") as HTMLInputElement;

// Split view control elements
const splitHandle = document.getElementById("split-handle") as HTMLDivElement;
const splitControls = document.getElementById(
    "split-controls"
) as HTMLDivElement;

// Panel resizing elements
const panelResizer = document.getElementById("panel-resizer") as HTMLDivElement;
const debugInfoResizer = document.getElementById(
    "debug-info-resizer"
) as HTMLDivElement;

// Debug output elements
const logContent = document.getElementById("log-content") as HTMLDivElement;
const irContent = document.getElementById("ir-content") as HTMLDivElement;
const wgslTextContent = document.getElementById(
    "wgsl-text"
) as HTMLDivElement;

// Uniform 參數面板（由 compileShader() 依 initialize() 回傳的 uniforms 動態重建）
const uniformPanel = document.getElementById(
    "uniform-panel"
) as HTMLDivElement;

// =====================================================================================
// APPLICATION STATE
// =====================================================================================

// WebGPU and rendering state
let wgfx: any = null;
let device: GPUDevice | null = null;
let inputSource: ImageBitmap | VideoFrame | HTMLVideoElement | null = null;
let currentVideoElement: HTMLVideoElement | null = null; // Track active video element

// Editor state
let editor: monaco.editor.IStandaloneCodeEditor;
let hasUserEdits = false; // Track if user has made edits since last external load
let lastExternalContent = ""; // Track last external content for smart undo detection

// WESL validation state（用於驗證「產生的 WGSL」，不驗證原始 DSL 編輯器內容）
let weslReady = false; // WESL WASM 引擎是否已初始化
let wgslEditor: monaco.editor.IStandaloneCodeEditor | null = null; // 「Generated WGSL」分頁內的唯讀編輯器

// Rendering and animation state
let isAnimating = false;
let frameCount = 0;
let lastTime = 0;

// UI state
let useColorFilters = true; // Track color filter state
let splitRatio = 0.5; // Track split ratio (0.0 = all original, 1.0 = all processed)
let displayMode: "split" | "single" = "split"; // Display mode
let panelRatio = 0.5; // Track panel ratio (0.0 = all editor, 1.0 = all preview)
let isPanelDragging = false; // Track panel resizer drag state
let debugInfoHeight = 320; // Track debug info panel height
let isDebugInfoDragging = false; // Track debug info resizer drag state

// UI elements for dynamic display
const currentRatioDisplay = document.getElementById(
    "current-ratio"
) as HTMLSpanElement;

// =====================================================================================
// RENDERING SYSTEM STATE
// =====================================================================================

// Blit Pipeline for displaying results (used for split-screen and single view rendering)
let blitPipeline: GPURenderPipeline | null = null;
let blitSampler: GPUSampler | null = null;

// Cached blit resources（重用資源：bind group 與 uniform 僅在變動時重建/上傳，避免每幀配置）
let blitUniformBuffer: GPUBuffer | null = null;
let blitBindGroupLayout: GPUBindGroupLayout | null = null;
let blitBindGroup: GPUBindGroup | null = null;
let blitBindGroupKeyIn: GPUTexture | null = null; // bind group 建立時的 INPUT 紋理
let blitBindGroupKeyOut: GPUTexture | null = null; // bind group 建立時的 OUTPUT 紋理
const blitUniformData = new Float32Array(4); // 16-byte uniform: useFilters, splitRatio, displayMode
let blitUniformsDirty = true; // UI 狀態變動旗標
let canvasCtx: GPUCanvasContext | null = null; // 重用的 canvas context（不必每幀 getContext）

// =====================================================================================
// UTILITY FUNCTIONS
// =====================================================================================

/**
 * Logging helper function that outputs to both the log panel and status bar
 */
function log(msg: string, type: "info" | "error" | "success" = "info") {
    const div = document.createElement("div");
    div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    div.className = `log-${type}`;
    logContent.appendChild(div);
    logContent.scrollTop = logContent.scrollHeight;

    statusText.textContent = msg;
    if (type === "error") statusText.style.color = "#f48771";
    else statusText.style.color = "#888";
}

/**
 * Format time in HH:MM:SS format
 */
function formatTime(seconds: number): string {
    const h = Math.floor(seconds / 3600)
        .toString()
        .padStart(2, "0");
    const m = Math.floor((seconds % 3600) / 60)
        .toString()
        .padStart(2, "0");
    const s = Math.floor(seconds % 60)
        .toString()
        .padStart(2, "0");
    return `${h}:${m}:${s}`;
}

/**
 * Update the volume display based on current video element volume
 */
function updateVolumeDisplay() {
    if (currentVideoElement) {
        const volumePercent = Math.round(currentVideoElement.volume * 100);
        volumeDisplay.textContent = `VOL: ${volumePercent}%`;
    }
}

// =====================================================================================
// UI CONTROL FUNCTIONS
// =====================================================================================

/**
 * Update split handle position and display based on current split ratio and display mode
 */
function updateSplitHandle() {
    if (displayMode === "single") {
        splitHandle.style.display = "none";
        if (splitControls) splitControls.style.display = "none";
    } else {
        splitHandle.style.display = "block";
        if (splitControls) splitControls.style.display = "flex";

        // Constrain split handle height to canvas dimensions
        const canvasRect = canvas.getBoundingClientRect();
        const canvasWrapperRect = canvas.parentElement!.getBoundingClientRect();

        // Calculate the offset of canvas within its wrapper
        const canvasOffsetTop = canvasRect.top - canvasWrapperRect.top;

        // Position split handle to align exactly with canvas
        splitHandle.style.top = `${canvasOffsetTop}px`;
        splitHandle.style.height = `${canvasRect.height}px`;

        // Ensure split handle doesn't extend beyond canvas bounds
        const maxHeight = canvasRect.height;
        if (parseFloat(splitHandle.style.height) > maxHeight) {
            splitHandle.style.height = `${maxHeight}px`;
        }

        // Calculate position relative to canvas
        const canvasRectForPosition = canvas.getBoundingClientRect();
        const canvasWrapperRectForPosition = canvas.parentElement!.getBoundingClientRect();
        const canvasOffsetLeft = canvasRectForPosition.left - canvasWrapperRectForPosition.left;

        // 計算控制桿的中心位置（相對於 canvas wrapper）
        // 控制桿使用 transform: translateX(-50%)，所以 left 值是中心位置
        // 控制桿最大寬度為 10px（dragging 狀態），所以中心至少距離邊緣 5px
        const handleHalfWidth = 5; // 控制桿最大寬度的一半
        const canvasWidth = canvasRectForPosition.width;

        // 計算控制桿中心在 canvas 內的位置（相對於 canvas 左邊）
        const centerInCanvas = splitRatio * canvasWidth;

        // 限制控制桿中心在 canvas 範圍內（考慮控制桿寬度）
        const clampedCenter = Math.max(handleHalfWidth, Math.min(canvasWidth - handleHalfWidth, centerInCanvas));

        // 轉換為相對於 wrapper 的位置
        const leftPosition = canvasOffsetLeft + clampedCenter;

        splitHandle.style.left = `${leftPosition}px`;

        if (currentRatioDisplay) {
            const percentage = splitRatio * 100;
            currentRatioDisplay.textContent = `${Math.round(percentage)}%`;
        }
    }
}

// =====================================================================================
// SPLIT VIEW DRAGGING SYSTEM
// =====================================================================================

let isDragging = false; // Track if split handle is being dragged

/**
 * Start split handle dragging
 */
function startSplitDrag(e: MouseEvent) {
    isDragging = true;
    document.body.style.cursor = "ew-resize";
    document.body.style.userSelect = "none";
    splitHandle.classList.add("dragging");
    updateSplitFromMouse(e);
}

/**
 * Update split ratio based on mouse position during drag
 */
function updateSplitFromMouse(e: MouseEvent) {
    const canvas = document.getElementById("output-canvas") as HTMLCanvasElement;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    // 計算滑鼠相對於 canvas 左邊的位置
    const x = e.clientX - rect.left;
    // 限制在 canvas 範圍內（0 到 canvas 寬度）
    const clampedX = Math.max(0, Math.min(rect.width, x));
    // 計算比例
    const ratio = clampedX / rect.width;
    if (ratio !== splitRatio) {
        splitRatio = ratio;
        blitUniformsDirty = true; // split 比例變動 → 下一幀重新上傳 blit uniform
    }
    updateSplitHandle();
}

/**
 * End split handle dragging
 */
function endSplitDrag() {
    isDragging = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    splitHandle.classList.remove("dragging");
}

/**
 * Update display mode button states
 */
function updateModeButtons() {
    document
        .querySelectorAll(".mode-btn")
        .forEach((btn) => btn.classList.remove("active"));
    document.getElementById(`mode-${displayMode}`)?.classList.add("active");
}

/**
 * Handle preset button clicks for split ratio
 */
function handlePresetClick(ratio: number) {
    if (ratio !== splitRatio) {
        splitRatio = ratio;
        blitUniformsDirty = true; // split 比例變動 → 下一幀重新上傳 blit uniform
    }
    updateSplitHandle();

    // Update active button
    document
        .querySelectorAll(".preset-btn")
        .forEach((btn) => btn.classList.remove("active"));
    document.querySelector(`[data-split="${ratio}"]`)?.classList.add("active");
}

// =====================================================================================
// PANEL RESIZING SYSTEM
// =====================================================================================

/**
 * Update panel sizes based on current panel ratio and screen size
 * Supports both desktop (horizontal) and mobile (vertical) layouts
 */
function updatePanelSizes() {
    const editorPanel = document.querySelector(
        ".panel:first-child"
    ) as HTMLElement;
    const isMobile = window.innerWidth <= 768;

    if (isMobile) {
        // Mobile: vertical layout
        const totalHeight = window.innerHeight;
        const editorHeight = totalHeight * panelRatio;

        editorPanel.style.width = "100%";
        editorPanel.style.height = `${editorHeight}px`;
        editorPanel.style.flex = "none";

        if (panelResizer) {
            panelResizer.style.cursor = "ns-resize";
            panelResizer.style.width = "100%";
            panelResizer.style.height = "8px";
        }
    } else {
        // Desktop: horizontal layout
        const totalWidth = window.innerWidth;
        const editorWidth = totalWidth * panelRatio;

        editorPanel.style.width = `${editorWidth}px`;
        editorPanel.style.height = "100vh";
        editorPanel.style.flex = "none";

        if (panelResizer) {
            panelResizer.style.cursor = "ew-resize";
            panelResizer.style.width = "8px";
            panelResizer.style.height = "100vh";
        }
    }
}

// =====================================================================================
// PANEL DRAGGING SYSTEM
// =====================================================================================

/**
 * Start panel resizing drag
 */
function startPanelDrag(e: MouseEvent) {
    isPanelDragging = true;
    const isMobile = window.innerWidth <= 768;
    document.body.style.cursor = isMobile ? "ns-resize" : "ew-resize";
    document.body.style.userSelect = "none";
    if (panelResizer) panelResizer.classList.add("dragging");
    updatePanelFromMouse(e);
}

/**
 * Update panel ratio based on mouse position during drag
 */
function updatePanelFromMouse(e: MouseEvent) {
    const isMobile = window.innerWidth <= 768;
    if (isMobile) {
        // Mobile: vertical drag
        const totalHeight = window.innerHeight;
        const newRatio = Math.max(0.2, Math.min(0.8, e.clientY / totalHeight));
        panelRatio = newRatio;
    } else {
        // Desktop: horizontal drag
        const totalWidth = window.innerWidth;
        const newRatio = Math.max(0.2, Math.min(0.8, e.clientX / totalWidth));
        panelRatio = newRatio;
    }
    updatePanelSizes();
}

/**
 * End panel resizing drag
 */
function endPanelDrag() {
    isPanelDragging = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    if (panelResizer) panelResizer.classList.remove("dragging");
}

// =====================================================================================
// DEBUG INFO PANEL RESIZING SYSTEM
// =====================================================================================

/**
 * Start debug info panel resizing drag
 */
function startDebugInfoDrag(e: MouseEvent) {
    isDebugInfoDragging = true;
    document.body.style.cursor = "ns-resize";
    document.body.style.userSelect = "none";
    if (debugInfoResizer) debugInfoResizer.classList.add("dragging");
    updateDebugInfoFromMouse(e);
}

/**
 * Update debug info panel height based on mouse position during drag
 */
function updateDebugInfoFromMouse(e: MouseEvent) {
    const container = document.querySelector(".panel:last-child") as HTMLElement;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const newHeight = Math.max(
        150,
        Math.min(600, containerRect.bottom - e.clientY)
    );

    debugInfoHeight = newHeight;
    updateDebugInfoHeight();
}

/**
 * Update debug info panel height in DOM
 */
function updateDebugInfoHeight() {
    const debugInfo = document.querySelector(".debug-info") as HTMLElement;
    if (debugInfo) {
        debugInfo.style.height = `${debugInfoHeight}px`;
    }
}

/**
 * End debug info panel resizing drag
 */
function endDebugInfoDrag() {
    isDebugInfoDragging = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    if (debugInfoResizer) debugInfoResizer.classList.remove("dragging");
}

// =====================================================================================
// WESL VALIDATION SYSTEM（驗證「產生的 WGSL」）
// =====================================================================================

/**
 * 將字元偏移量（offset）轉換為 Monaco 的行列位置
 */
function offsetToPosition(
    lines: string[],
    offset: number
): { lineNumber: number; column: number } {
    let charCount = 0;
    for (let i = 0; i < lines.length; i++) {
        const lineLength = lines[i].length + 1; // +1 for newline
        if (charCount + lineLength > offset) {
            return {
                lineNumber: i + 1,
                column: Math.max(1, offset - charCount + 1),
            };
        }
        charCount += lineLength;
    }
    return {
        lineNumber: lines.length,
        column: Math.max(1, lines[lines.length - 1]?.length || 1),
    };
}

/**
 * 初始化 WESL 引擎，並在「Generated WGSL」分頁內建立唯讀的 WGSL 檢視編輯器
 * 注意：不再對 DSL 原始編輯器內容做即時驗證（//! 指示詞會造成誤報），
 * 驗證對象改為編譯後產生的 WGSL，見 validateGeneratedWGSL()
 */
async function setupWeslValidator() {
    try {
        // 1. Initialize WASM module (using module_or_path parameter to avoid warnings)
        await initWesl({ module_or_path: wasmUrl });
        weslReady = true;
        console.log("WESL 語法檢查引擎已啟動");

        // 2. 在「Generated WGSL」分頁內建立唯讀迷你編輯器（約分頁高度 40%）
        const container = document.getElementById("wgsl-editor");
        if (!container) return;

        wgslEditor = monaco.editor.create(container, {
            value: "// 尚未編譯：成功編譯後此處會顯示 WGFX 產生的 WGSL 並標記 WESL 驗證錯誤",
            language: "wgsl", // 使用註冊的WGSL語法
            theme: "wgsl-dark", // 使用自定義的WGSL主題
            readOnly: true,
            minimap: { enabled: false },
            lineNumbers: "on",
            lineNumbersMinChars: 3,
            glyphMargin: true, // 啟用以顯示錯誤標記
            folding: true,
            renderLineHighlight: "line",
            scrollBeyondLastLine: false,
            automaticLayout: true, // 分頁切換顯示/隱藏時自動重新計算版面
        });
    } catch (e) {
        console.error("無法啟動 WESL 檢查器:", e);
        log("WESL Validator failed to load. Check console.", "error");
    }
}

/**
 * 使用 WESL 引擎編譯驗證 WGFX 產生的 WGSL，並將錯誤標記在唯讀編輯器上
 * @param wgslText 產生的 WGSL 程式碼（可含多個 PASS）
 * @returns 錯誤數量（引擎未就緒或無法驗證時回傳 0）
 */
export function validateGeneratedWGSL(wgslText: string): number {
    // 更新唯讀編輯器內容
    const model = wgslEditor?.getModel() ?? null;
    if (model) {
        model.setValue(wgslText);
    }

    if (!weslReady) {
        console.warn("WESL 引擎尚未就緒，略過產生 WGSL 的驗證");
        return 0;
    }

    // 使用 run 參數 Compile 命令對產生的 WGSL 進行驗證
    let diagnostics: any[] = [];
    try {
        // 嘗試編譯，如果成功則沒有診斷信息
        void run({
            command: "Compile",
            files: { "main.wgsl": wgslText },
            root: "main.wgsl",
            validate: true,
            sourcemap: false,
            imports: false,
            condcomp: false,
            generics: false,
            strip: false,
            lower: false,
            naga: false,
            lazy: false,
            keep_root: false,
            mangle_root: false,
            features: {},
        });

        // 如果編譯成功，沒有診斷信息
        diagnostics = [];
    } catch (err: any) {
        // 如果編譯失敗，錯誤對象包含 diagnostics
        if (err && err.diagnostics && Array.isArray(err.diagnostics)) {
            diagnostics = err.diagnostics;
        } else if (err && err.message) {
            // 如果沒有 diagnostics，以錯誤訊息建立通用診斷（標記在檔案開頭）
            diagnostics = [
                {
                    title: err.message || "Compilation error",
                    span: { start: 0, end: 0 },
                },
            ];
        } else {
            console.error("WESL Analysis failed:", err);
            return 0;
        }
    }

    // 將 WESL 診斷資訊轉換為 Monaco 標記
    const lines = wgslText.split("\n");
    const markers = diagnostics.map((d: any) => {
        const startOffset =
            typeof d.span?.start === "number" ? d.span.start : 0;
        const start = offsetToPosition(lines, startOffset);

        // 預設結束位置：同一行行尾（涵蓋整行）
        let end = {
            lineNumber: start.lineNumber,
            column: Math.max(
                start.column + 1,
                (lines[start.lineNumber - 1]?.length || start.column) + 1
            ),
        };
        if (
            d.span &&
            typeof d.span.end === "number" &&
            d.span.end > startOffset
        ) {
            end = offsetToPosition(lines, d.span.end);
        }

        return {
            message: d.title || d.message || "語法錯誤",
            severity: monaco.MarkerSeverity.Error,
            startLineNumber: start.lineNumber,
            startColumn: start.column,
            endLineNumber: end.lineNumber,
            endColumn: end.column,
        };
    });

    console.log(`生成 WGSL 驗證：${markers.length} 個錯誤`);
    // 在唯讀編輯器上設置紅色錯誤標記（Owner 為 'wesl-generated' 以免與其他 owner 衝突）
    if (model) {
        monaco.editor.setModelMarkers(model, "wesl-generated", markers);
    }
    return markers.length;
}

// =====================================================================================
// MONACO EDITOR CONFIGURATION
// =====================================================================================

// Configure Monaco Editor workers to avoid warnings
// Use simple setup for worker main thread (any environment can accept)
// @ts-ignore
(window as any).MonacoEnvironment = {
    getWorkerUrl: function () {
        // Return an empty string for Monaco main thread
        // To avoid warning, any environment can accept
        return "";
    },
};

// Register WGSL language support
// 1. 註冊語言
monaco.languages.register({ id: "wgsl" });

monaco.languages.setMonarchTokensProvider("wgsl", {
    // 宣告關鍵字 (維持深灰色)
    keywords: [
        "fn",
        "let",
        "var",
        "const",
        "alias",
        "struct",
        "enable",
        "diagnostic",
    ],
    // 控制流程 (維持紫色)
    controlKeywords: [
        "if",
        "else",
        "loop",
        "for",
        "while",
        "break",
        "continue",
        "return",
        "discard",
        "switch",
        "case",
        "default",
    ],
    // 內建類型 (維持青色)
    types: [
        "bool",
        "i32",
        "u32",
        "f32",
        "f16",
        "vec2",
        "vec3",
        "vec4",
        "mat2x2",
        "mat3x3",
        "mat4x4",
        "texture_2d",
        "texture_3d",
        "texture_cube",
        "texture_depth_2d",
        "sampler",
        "sampler_comparison",
        "atomic",
        "array",
        "ptr",
    ],

    tokenizer: {
        root: [
            // A. 函式名稱 (維持橘色)
            [/[a-zA-Z_][\w]*(?=\s*\()/, "function"],

            // B. 屬性 (@黃色, 名稱跟隨變數顏色)
            [/(@)([a-zA-Z_]\w*)/, ["keyword.operator", "attribute"]],

            // C. 識別字分析
            [
                /[a-zA-Z_]\w*/,
                {
                    cases: {
                        "@keywords": "keyword", // 1. 宣告 -> 深灰
                        "@controlKeywords": "keyword.control", // 2. 控制 -> 紫色
                        "@types": "type", // 3. 內建類型 -> 青色
                        "/^[A-Z][w]*$/": "type.custom", // 4. 自定義類型 -> 紅色
                        "@default": "identifier", // 5. 變數 -> 暗藍色
                    },
                },
            ],

            // D. 數字 (維持淺綠)
            [/\b0x[\da-fA-F]+/, "number.hex"],
            [/\b\d+\.\d+([eE][\-+]?\d+)?[fh]?\b/, "number.float"],
            [/\b\d+([eE][\-+]?\d+)?[fh]?\b/, "number"],
            [/\b\d+[iu]\b/, "number"],

            // E. 註釋 (維持淺灰)
            [/\/\/.*$/, "comment"],
            [/\/\*[\s\S]*?\*\//, "comment"],

            // F. 運算符
            [/[{}()\[\]]/, "@brackets"],
            [/[+\-*/%=<>!&|^~]+/, "operator"],
            [/[,;:]/, "delimiter"],
        ],
    },
});

// 定義主題
monaco.editor.defineTheme("wgsl-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [
        // 1. 變數 (Identifier) -> 改為暗一點的藍色
        // 原本是 #9CDCFE (淡藍)，改為 #569CD6 (標準藍) 或 #4D96C9
        { token: "identifier", foreground: "#569CD6" },

        // 2. 宣告關鍵字 -> 深灰色
        { token: "keyword", foreground: "#666666" },

        // 3. 註解 -> 淺灰色
        { token: "comment", foreground: "#999999" },

        // --- 其他維持不變 ---

        // 控制流程 -> 紫色
        { token: "keyword.control", foreground: "#C586C0" },

        // 自定義類型 -> 紅色
        { token: "type.custom", foreground: "#FF6B6B" },

        // 內建類型 -> 青色
        { token: "type", foreground: "#4EC9B0" },

        // 函式 -> 橘色
        { token: "function", foreground: "#FF9E64" },

        // 屬性名稱 -> 跟隨變數顏色 (暗藍)
        { token: "attribute", foreground: "#569CD6" },
        // @符號 -> 黃色
        { token: "keyword.operator", foreground: "#DCDCAA" },

        // 數字 -> 淺綠色
        { token: "number", foreground: "#B5CEA8" },

        // 符號 -> 淺灰/白
        { token: "operator", foreground: "#D4D4D4" },
        { token: "delimiter", foreground: "#D4D4D4" },
    ],
    colors: {
        "editor.background": "#1e1e1e",
        "editor.foreground": "#d4d4d4",
        "editorLineNumber.foreground": "#858585",
        "editor.selectionBackground": "#264f78",
        "editor.lineHighlightBackground": "#2f3136",
    },
});
// =====================================================================================
// EDITOR INITIALIZATION SYSTEM
// =====================================================================================

/**
 * Initialize Monaco Editor with WGSL language support and custom configuration
 */
function initEditor() {
    editor = monaco.editor.create(document.getElementById("editor")!, {
        value: INITIAL_SHADER_PLACEHOLDER,
        language: "wgsl", // 使用註冊的WGSL語法
        theme: "wgsl-dark", // 使用自定義的WGSL主題
        minimap: { enabled: false },
        lineNumbers: "on",
        lineNumbersMinChars: 3,
        glyphMargin: true, // 啟用以顯示錯誤標記
        folding: true,
        renderLineHighlight: "line",
        scrollBeyondLastLine: false,
        wordWrap: "off",
        tabSize: 4,
        insertSpaces: true,
        detectIndentation: false,
    });

    // Set up keyboard shortcuts
    // Compile on Ctrl+S
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
        compileShader();
    });

    // 初始化 WESL 引擎與「Generated WGSL」唯讀檢視編輯器（不驗證 DSL 原始內容）
    setupWeslValidator();

    // Track user edits to preserve undo stack with smart detection
    editor.onDidChangeModelContent(() => {
        // Check if current content matches last external content
        // If it matches (e.g., user undid back to original state), allow auto-sync to continue
        if (editor.getValue() === lastExternalContent) {
            hasUserEdits = false;
        } else {
            hasUserEdits = true;
        }
    });
}

// =====================================================================================
// WEBGPU INITIALIZATION SYSTEM
// =====================================================================================

/**
 * Initialize WebGPU context and create device with maximum supported limits
 * Falls back to default limits if maximum limits are not supported
 * @returns Promise<boolean> - true if initialization successful, false otherwise
 */
async function initWebGPU(): Promise<boolean> {
    // Check WebGPU support
    if (!navigator.gpu) {
        log("WebGPU not supported in this browser.", "error");
        return false;
    }

    // Request WebGPU adapter
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        log("Failed to get WebGPU adapter.", "error");
        return false;
    }

    // Request maximum limits supported by the hardware
    // We explicitly copy the limits to a plain object to avoid potential issues
    // with passing the GPUSupportedLimits object directly.
    const requiredLimits: Record<string, number> = {};
    // @ts-ignore - Iterate over limits
    for (const key in adapter.limits) {
        // @ts-ignore
        const value = adapter.limits[key];
        if (typeof value === "number") {
            requiredLimits[key] = value;
        }
    }

    // Try to create device with maximum limits, fallback to default if needed
    try {
        device = await adapter.requestDevice({
            requiredLimits,
        });
    } catch (err) {
        log(`Failed to request max limits: ${err}. Fallback to default.`, "error");
        device = await adapter.requestDevice();
    }

    // 註冊未捕捉的 GPU 錯誤事件，將錯誤顯示在 log 面板（addEventListener 與 on- 屬性兩種途徑）
    device.addEventListener?.("uncapturederror", (event: Event) => {
        const err = (event as GPUUncapturedErrorEvent).error;
        log(`GPU 錯誤: ${err?.message || err}`, "error");
    });
    device.onuncapturederror = (e) => {
        log(`GPU 錯誤: ${e.error?.message || e.error}`, "error");
    };

    // 裝置遺失（如驅動重置、裝置關閉）時於 log 面板提示（一次性 Promise）
    device.lost.then((info) => {
        log(`裝置遺失: ${info.message}`, "error");
    });

    log("WebGPU initialized successfully.", "success");
    return true;
}

// =====================================================================================
// MEDIA HANDLING SYSTEM
// =====================================================================================

/**
 * Handle file selection and load media (video or image) for processing
 * Sets up video controls for video files and prepares canvas for rendering
 */
async function handleFileSelect(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);

    // Cleanup previous video element
    if (currentVideoElement) {
        currentVideoElement.pause();
        currentVideoElement.src = "";
        currentVideoElement = null;
    }

    if (file.type.startsWith("video")) {
        await loadVideoFile(file, url);
    } else {
        await loadImageFile(file, url);
    }

    // Re-compile shader to update dimensions and start rendering
    // Always attempt compile when media changes, as it might be the first input
    compileShader();
}

/**
 * Load and configure a video file for processing
 */
async function loadVideoFile(file: File, url: string) {
    const video = document.createElement("video");
    video.src = url;
    video.loop = true;
    video.muted = false; // Allow audio
    video.autoplay = true;

    // Set up video event handlers
    video.addEventListener("ended", () => {
        btnPlayPause.textContent = "⏯";
        seekBar.value = "0";
        timeDisplay.textContent = `00:00 / ${formatTime(video.duration)}`;
    });

    // Wait for metadata to load and get duration
    await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => {
            seekBar.max = video.duration.toString();
            seekBar.value = "0";
            timeDisplay.textContent = `00:00 / ${formatTime(video.duration)}`;
            resolve();
        };
    });

    await video.play();
    inputSource = video;
    currentVideoElement = video;

    // Set initial volume from UI
    video.volume = volumeBar.valueAsNumber / 100;

    // Show video controls
    videoControls.style.display = "flex";
    btnPlayPause.textContent = "⏸";
    updateVolumeDisplay();

    log(
        `Video loaded: ${file.name} (${video.videoWidth}x${video.videoHeight})`,
        "success"
    );
    resizeCanvas(video.videoWidth, video.videoHeight);
}

/**
 * Load and process an image file for processing
 */
async function loadImageFile(file: File, url: string) {
    const img = new Image();
    img.src = url;
    // 修正：以包裝的 Promise 等待 onload（直接指定 resolve 會有型別不相容問題）
    await new Promise<void>((res) => {
        img.onload = () => res();
    });
    const bitmap = await createImageBitmap(img);

    // 釋放先前的輸入來源（ImageBitmap / VideoFrame 需手動 close，避免資源洩漏）
    (inputSource as any)?.close?.();
    inputSource = bitmap;

    // Hide video controls for images
    videoControls.style.display = "none";

    log(
        `Image loaded: ${file.name} (${bitmap.width}x${bitmap.height})`,
        "success"
    );
    resizeCanvas(bitmap.width, bitmap.height);
}

// =====================================================================================
// CANVAS AND RENDERING SYSTEM
// =====================================================================================

let isCanvasConfigured = false; // Track if canvas WebGPU context is configured

let lastMediaW = 0; // 最近一次媒體（輸入）寬高——供 window resize 時維持正確長寬比
let lastMediaH = 0;

/**
 * Resize canvas to fit container while maintaining aspect ratio of input media
 * Sets both CSS display size and internal WebGPU resolution.
 * 內部解析度可與媒體尺寸不同（升頻特效的 OUTPUT 為 2x），CSS 顯示尺寸依媒體長寬比。
 * @param w media width / 媒體寬（決定長寬比）
 * @param h media height / 媒體高（決定長寬比）
 * @param internalW internal WebGPU resolution width / 內部解析度寬（預設同 w）
 * @param internalH internal WebGPU resolution height / 內部解析度高（預設同 h）
 */
function resizeCanvas(w: number, h: number, internalW: number = w, internalH: number = h) {
    const aspect = w / h;
    lastMediaW = w;
    lastMediaH = h;

    const container = canvas.parentElement!;

    // Fit to container while maintaining aspect ratio
    let dispW = container.clientWidth;
    let dispH = dispW / aspect;

    if (dispH > container.clientHeight) {
        dispH = container.clientHeight;
        dispW = dispH * aspect;
    }

    // Set display size (CSS) - controls how big the canvas appears on screen
    canvas.style.width = `${dispW}px`;
    canvas.style.height = `${dispH}px`;

    // Set internal resolution (may be the effect's OUTPUT size, e.g. 2x for SR)
    // 內部解析度 = 效果實際輸出解析度（升頻特效由此如實呈現超解析度結果）
    canvas.width = internalW;
    canvas.height = internalH;

    // Reconfigure canvas context with new dimensions
    initCanvasContext();

    // Update split control UI
    updateSplitHandle();
}

/**
 * Initialize or reconfigure WebGPU canvas context
 * Called when canvas is resized or WebGPU device becomes available
 */
function initCanvasContext() {
    if (!device) {
        return;
    }

    const ctx = canvas.getContext("webgpu") as unknown as GPUCanvasContext;
    if (ctx) {
        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        ctx.configure({
            device,
            format: canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        canvasCtx = ctx;
        isCanvasConfigured = true;
    } else {
    }
}

// Video Control Listeners
btnPlayPause.addEventListener("click", () => {
    if (!currentVideoElement) return;
    if (currentVideoElement.paused) {
        currentVideoElement.play();
        btnPlayPause.textContent = "▶️";
    } else {
        currentVideoElement.pause();
        btnPlayPause.textContent = "⏸";
    }
});

seekBar.addEventListener("input", () => {
    if (!currentVideoElement) return;
    const time = parseFloat(seekBar.value);
    currentVideoElement.currentTime = time;
    timeDisplay.textContent = `${formatTime(time)} / ${formatTime(
        currentVideoElement.duration
    )}`;
});

btnStop.addEventListener("click", () => {
    if (!currentVideoElement) return;
    currentVideoElement.pause();
    currentVideoElement.currentTime = 0;
    seekBar.value = "0";
    timeDisplay.textContent = `00:00 / ${formatTime(
        currentVideoElement.duration
    )}`;
    btnPlayPause.textContent = "⏯";
});

// Volume Control
volumeBar.addEventListener("input", () => {
    if (!currentVideoElement) return;
    currentVideoElement.volume = volumeBar.valueAsNumber / 100;
    updateVolumeDisplay();
});

// =====================================================================================
// UNIFORM PARAMETER PANEL（動態參數面板）
// =====================================================================================

/** initialize() 回傳的單一參數中繼資料 */
interface UniformMeta {
    name: string;
    type: string;
    default: number;
    min: number;
    max: number;
    step: number;
    label?: string;
}

/**
 * 依 initialize() 回傳的參數清單重建動態參數面板
 * float 參數使用中繼資料的 min/max/step；int 參數固定使用 step 1
 * 若沒有任何參數則隱藏面板
 */
function buildUniformPanel(uniforms: UniformMeta[]) {
    // 每次成功編譯都重建面板，避免殘留舊參數的滑桿
    uniformPanel.innerHTML = "";

    if (!uniforms || uniforms.length === 0) {
        uniformPanel.style.display = "none";
        return;
    }

    for (const u of uniforms) {
        const isInt = u.type === "int";

        // 範圍與步進：float 使用中繼資料，int 固定 step 1；min/max 異常時給予安全後備值
        const min = Number.isFinite(u.min) ? u.min : 0;
        const max = Number.isFinite(u.max) && u.max > min ? u.max : min + 1;
        const step = isInt
            ? 1
            : Number.isFinite(u.step) && u.step > 0
            ? u.step
            : 0.01;
        const defaultValue = Number.isFinite(u.default) ? u.default : min;

        const row = document.createElement("div");
        row.className = "uniform-row";

        const label = document.createElement("label");
        label.textContent = u.label || u.name;
        label.title = `${u.name} (${u.type})`;

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = String(min);
        slider.max = String(max);
        slider.step = String(step);
        slider.value = String(defaultValue);

        const valueDisplay = document.createElement("span");
        valueDisplay.className = "uniform-value";
        valueDisplay.textContent = String(defaultValue);

        slider.addEventListener("input", () => {
            let value = parseFloat(slider.value);
            if (!Number.isFinite(value)) return;
            if (isInt) value = Math.round(value); // int 參數僅送出整數
            valueDisplay.textContent = String(value);
            if (wgfx) {
                // 僅送出數值型 uniform
                wgfx.updateUniforms({ [u.name]: value });
            }
        });

        row.appendChild(label);
        row.appendChild(slider);
        row.appendChild(valueDisplay);
        uniformPanel.appendChild(row);
    }

    uniformPanel.style.display = "flex";
}

// Main Compile & Run Logic
async function compileShader() {
    if (!device || !inputSource) {
        if (!inputSource) log("Please load an image or video first.", "error");
        return;
    }

    const code = editor.getValue();
    const inputWidth =
        inputSource instanceof HTMLVideoElement
            ? inputSource.videoWidth
            : (inputSource as ImageBitmap).width;
    const inputHeight =
        inputSource instanceof HTMLVideoElement
            ? inputSource.videoHeight
            : (inputSource as ImageBitmap).height;

    // Use input dimensions for WGFX processing
    const width = inputWidth;
    const height = inputHeight;

    log("Compiling shader...");
    statusText.textContent = "Compiling...";

    try {
        if (wgfx) {
            wgfx.dispose();
            wgfx = null;
        }

        // Enable debug mode in WGFX
        WGFX.setDebug(true);

        wgfx = await WGFX.create({
            device,
            effectCode: code,
            width,
            height,
        });

        // 取得特效資訊（uniforms 參數清單；參數預設值已於編譯時自動上傳，無需手動設定）
        const info = wgfx.initialize();

        // 依 initialize() 回傳的 uniforms 重建動態參數面板
        // （場景資訊 inputSize/outputSize 等已由核心自動注入為 scene，不再手動傳遞）
        buildUniformPanel(info.uniforms);

        // 畫布內部解析度改用效果「實際」的 OUTPUT 尺寸（升頻特效如 FSR/Anime4K
        // 宣告 2x，核心現會採納宣告尺寸），讓預覽呈現真正的超解析度輸出。
        // 之後回到 INFO 層級，避免每幀 dispatch 除錯日誌灌爆 console。
        const outTex = wgfx.runtime.resourceManager.getTexture("OUTPUT");
        if (outTex) {
            resizeCanvas(width, height, outTex.width, outTex.height);
        }
        WGFX.setLevel(LogLevel.INFO);

        log("Compilation successful!", "success");

        // Extract Debug Info
        const runtime = wgfx.runtime;
        irContent.textContent = JSON.stringify(runtime.shaderInfo, null, 2);

        // Show generated WGSL code
        if (runtime.generatedModules && runtime.generatedModules.length > 0) {
            let joinedWgsl = "";
            runtime.generatedModules.forEach(
                (module: { passIndex: any; wgslCode: string }) => {
                    joinedWgsl += `// === PASS ${module.passIndex} ===\n`;
                    joinedWgsl += module.wgslCode;
                    joinedWgsl += "\n\n";
                }
            );
            wgslTextContent.textContent = joinedWgsl;

            // 以 WESL 編譯驗證產生的 WGSL，並在唯讀編輯器上標記錯誤
            const wgslErrorCount = validateGeneratedWGSL(joinedWgsl);
            log(
                `生成 WGSL 驗證：${wgslErrorCount} 個錯誤`,
                wgslErrorCount > 0 ? "error" : "info"
            );
        } else {
            wgslTextContent.textContent =
                "// No generated WGSL modules found.\n// Check that shader compilation was successful.";
            // 清除唯讀編輯器內容與先前殘留的驗證標記
            const wgslModel = wgslEditor?.getModel();
            if (wgslModel) {
                wgslModel.setValue("// 找不到產生的 WGSL 模組");
                monaco.editor.setModelMarkers(wgslModel, "wesl-generated", []);
            }
        }

        startRenderLoop();
    } catch (e: any) {
        log(e.message, "error");
        console.error(e);
    }
}

function initBlitPipeline() {
    if (!device) {
        return;
    }

    blitSampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
    });

    const shaderModule = device.createShaderModule({
        label: "Blit Shader (Split View)",
        code: `
            struct VertexOutput {
                @builtin(position) position : vec4<f32>,
                @location(0) uv : vec2<f32>,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
                var pos = array<vec2<f32>, 3>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>( 3.0, -1.0),
                    vec2<f32>(-1.0,  3.0)
                );

                var output : VertexOutput;
                output.position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
                output.uv = vec2<f32>(output.position.x * 0.5 + 0.5, 0.5 - output.position.y * 0.5);
                return output;
            }

            struct FilterUniforms {
                useFilters: f32,
                splitRatio: f32,
                displayMode: f32, // 0.0 = split, 1.0 = single
            };
            @group(0) @binding(0) var texInput : texture_2d<f32>;
            @group(0) @binding(1) var texOutput : texture_2d<f32>;
            @group(0) @binding(2) var s : sampler;
            @group(0) @binding(3) var<uniform> uniforms: FilterUniforms;

            @fragment
            fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
                if (uniforms.displayMode == 1.0) {
                    // Single view mode: show processed output only
                    // （單一模式不使用棋盤格 fallback，直接呈現真實輸出狀態，避免掩蓋渲染失敗）
                    let color = textureSampleLevel(texOutput, s, uv, 0.0);
                    return select(color, mix(color, vec4<f32>(0.0, 0.2, 0.0, 1.0), 0.05), uniforms.useFilters == 1.0);
                } else {
                    // Split screen mode
                    if (uv.x < uniforms.splitRatio) {
                        // Left portion: processed output texture
                        let color = textureSampleLevel(texOutput, s, uv, 0.0);

                        // Check if output texture appears to be empty/uninitialized
                        let isBlackOrTransparent = (color.r + color.g + color.b) < 0.01;

                        if (isBlackOrTransparent) {
                            // 橘色棋盤格 fallback：表示 OUTPUT 尚未被渲染（輸出為空），
                            // 僅在分割模式的已處理側顯示，用於輔助除錯
                            let patternUV = uv * 10.0;
                            let checker = (floor(patternUV.x) + floor(patternUV.y)) % 2.0;
                            return vec4<f32>(1.0, checker * 0.5, 0.0, 1.0);
                        } else {
                            // Output has content, add subtle green tint for distinction (if filters enabled)
                            return select(color, mix(color, vec4<f32>(0.0, 0.2, 0.0, 1.0), 0.05), uniforms.useFilters == 1.0);
                        }
                    } else {
                        // Right portion: original input texture
                        let color = textureSampleLevel(texInput, s, uv, 0.0);
                        // Add subtle blue tint for distinction (if filters enabled)
                        return select(color, mix(color, vec4<f32>(0.0, 0.0, 0.2, 1.0), 0.05), uniforms.useFilters == 1.0);
                    }
                }
            }
        `,
    });

    // Create uniform buffer for filter, split and display settings (useFilters: f32, splitRatio: f32, displayMode: f32)
    const uniformBuffer = device.createBuffer({
        // 16 bytes：uniform buffer 建議 16-byte 對齊（3 個 f32 共 12 bytes + 4 bytes padding），跨驅動較安全
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create bind group layout manually since we need specific binding for uniforms
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "float" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "float" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: { type: "filtering" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            },
        ],
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
    });

    blitPipeline = device.createRenderPipeline({
        label: "Blit Pipeline",
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: "vs_main",
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fs_main",
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
        },
        primitive: {
            topology: "triangle-list",
        },
    });

    // Store uniform buffer and layout for later updates
    blitUniformBuffer = uniformBuffer;
    blitBindGroupLayout = bindGroupLayout;
    // 新管線 → 舊 bind group 一律作廢，uniform 需重新上傳
    blitBindGroup = null;
    blitBindGroupKeyIn = null;
    blitBindGroupKeyOut = null;
    blitUniformsDirty = true;
}

function startRenderLoop() {
    if (isAnimating) return;
    isAnimating = true;

    async function frame(now: number) {
        if (!wgfx || !inputSource || !device) {
            isAnimating = false;
            return;
        }

        // FPS Calculation
        if (now - lastTime >= 1000) {
            fpsMeter.textContent = `${frameCount} FPS`;
            frameCount = 0;
            lastTime = now;
        }
        frameCount++;

        try {
            // Update UI if video is playing
            if (currentVideoElement) {
                seekBar.value = currentVideoElement.currentTime.toString();
                timeDisplay.textContent = `${formatTime(
                    currentVideoElement.currentTime
                )} / ${formatTime(currentVideoElement.duration)}`;

                // Update play/pause button state
                if (currentVideoElement.paused) {
                    btnPlayPause.textContent = "▶";
                } else {
                    btnPlayPause.textContent = "⏸";
                }
            }

            // copyExternalImageToTexture 直接支援 HTMLVideoElement（會取用目前播放影格），
            // 不必每幀配置/關閉 VideoFrame（減少 GC 壓力）。
            let outputTexture: GPUTexture | null = null;
            try {
                outputTexture = await wgfx.process(inputSource);
            } catch (e: any) {
                log(`WGFX process error: ${e?.message || String(e)}`, "error");
                outputTexture = null;
            }

            const inputTexture = wgfx.runtime.resourceManager.getTexture("INPUT");

            // Draw result to canvas
            if (canvasCtx && isCanvasConfigured && inputTexture && outputTexture) {
                if (!blitPipeline) {
                    initBlitPipeline();
                }

                const currentTexture = canvasCtx.getCurrentTexture();

                const commandEncoder = device.createCommandEncoder();

                const passEncoder = commandEncoder.beginRenderPass({
                    colorAttachments: [
                        {
                            view: currentTexture.createView(),
                            clearValue: { r: 0, g: 0, b: 0, a: 1 },
                            loadOp: "clear",
                            storeOp: "store",
                        },
                    ],
                });

                if (blitPipeline && blitSampler && blitUniformBuffer) {
                    // Uniform 僅在 UI 狀態（split/filters/displayMode）變動時上傳
                    if (blitUniformsDirty) {
                        blitUniformData[0] = useColorFilters ? 1 : 0;
                        blitUniformData[1] = splitRatio;
                        blitUniformData[2] = displayMode === "single" ? 1 : 0;
                        device.queue.writeBuffer(blitUniformBuffer, 0, blitUniformData);
                        blitUniformsDirty = false;
                    }

                    // Bind group 僅在輸入/輸出紋理物件改變（重新編譯）時重建；
                    // 紋理物件在每幀之間是穩定的，因此每幀不再 createView/配置。
                    if (
                        !blitBindGroup ||
                        blitBindGroupKeyIn !== inputTexture ||
                        blitBindGroupKeyOut !== outputTexture
                    ) {
                        blitBindGroup = device.createBindGroup({
                            layout:
                                blitBindGroupLayout ||
                                blitPipeline.getBindGroupLayout(0),
                            entries: [
                                { binding: 0, resource: inputTexture.createView() },
                                { binding: 1, resource: outputTexture.createView() },
                                { binding: 2, resource: blitSampler },
                                {
                                    binding: 3,
                                    resource: { buffer: blitUniformBuffer },
                                },
                            ],
                        });
                        blitBindGroupKeyIn = inputTexture;
                        blitBindGroupKeyOut = outputTexture;
                    }

                    passEncoder.setPipeline(blitPipeline);
                    passEncoder.setBindGroup(0, blitBindGroup);
                    passEncoder.draw(3);
                } else {
                }

                passEncoder.end();
                device.queue.submit([commandEncoder.finish()]);
            } else {
            }
        } catch (e: any) {
            console.error(e);
            isAnimating = false; // Stop on error
            return;
        }

        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

// Event Listeners
fileInput.addEventListener("change", handleFileSelect);
compileBtn.addEventListener("click", compileShader);

// Reload Button
btnReload.addEventListener("click", () => {
    lastModifiedTime = 0;
    checkExternalShader(true);
});

// Shader Selector
shaderSelector.addEventListener("change", () => {
    lastModifiedTime = 0; // Reset
    checkExternalShader(true); // Force load
});

// Color Filters Toggle
chkColorFilters.addEventListener("change", () => {
    useColorFilters = chkColorFilters.checked;
    blitUniformsDirty = true; // 濾色狀態變動 → 下一幀重新上傳 blit uniform
});

// Display Mode Buttons
document.getElementById("mode-split")?.addEventListener("click", () => {
    displayMode = "split";
    blitUniformsDirty = true; // 顯示模式變動 → 下一幀重新上傳 blit uniform
    updateModeButtons();
    updateSplitHandle();
    console.log("Switched to split view mode");
});

document.getElementById("mode-single")?.addEventListener("click", () => {
    displayMode = "single";
    blitUniformsDirty = true; // 顯示模式變動 → 下一幀重新上傳 blit uniform
    updateModeButtons();
    updateSplitHandle();
    console.log("Switched to single view mode");
});

// Split Control Events
splitHandle.addEventListener("mousedown", startSplitDrag);

// Panel Resizer Events
if (panelResizer) {
    panelResizer.addEventListener("mousedown", startPanelDrag);
}

// Debug Info Resizer Events
if (debugInfoResizer) {
    debugInfoResizer.addEventListener("mousedown", startDebugInfoDrag);
}

document.addEventListener("mousemove", (e) => {
    if (isDragging) {
        updateSplitFromMouse(e);
    }
    if (isPanelDragging) {
        updatePanelFromMouse(e);
    }
    if (isDebugInfoDragging) {
        updateDebugInfoFromMouse(e);
    }
});
document.addEventListener("mouseup", () => {
    endSplitDrag();
    endPanelDrag();
    endDebugInfoDrag();
});

// Preset Buttons (delegate event to handle dynamically added buttons)
document.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;
    if (target.classList.contains("preset-btn")) {
        const ratio = parseFloat(target.dataset.split!);
        handlePresetClick(ratio);
    }
});

// Tabs
document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", (e) => {
        const targetId = (e.target as HTMLElement).dataset.target;

        document
            .querySelectorAll(".tab")
            .forEach((x) => x.classList.remove("active"));
        (e.target as HTMLElement).classList.add("active");

        document
            .querySelectorAll(".tab-content")
            .forEach((c) => ((c as HTMLElement).style.display = "none"));
        document.getElementById(`${targetId}-content`)!.style.display = "block";

        // 切換到 WGSL 分頁時，重新計算唯讀編輯器的大小
        if (targetId === "wgsl") {
            wgslEditor?.layout();
        }
    });
});

// Init
initEditor();
loadFileList(); // Load available shaders
updateSplitHandle(); // Initialize split control
updateModeButtons(); // Initialize mode buttons
updatePanelSizes(); // Initialize panel sizes
updateDebugInfoHeight(); // Initialize debug info height

// Handle window resize
window.addEventListener("resize", () => {
    updatePanelSizes();
    updateDebugInfoHeight();
    // Also update canvas if needed
    // 以媒體長寬比重新排版；內部解析度維持目前的 OUTPUT 尺寸
    if (lastMediaW > 0 && lastMediaH > 0) {
        resizeCanvas(lastMediaW, lastMediaH, canvas.width, canvas.height);
    } else if (inputSource && canvas.width > 0 && canvas.height > 0) {
        resizeCanvas(canvas.width, canvas.height);
    }
});

// Initialize current ratio display if element exists
if (currentRatioDisplay) {
    currentRatioDisplay.textContent = "50%";
}
initWebGPU().then((ready) => {
    if (ready) {
        createDefaultImage().then((bitmap) => {
            inputSource = bitmap;
            log("Loaded default test pattern.", "success");
            resizeCanvas(bitmap.width, bitmap.height);
            // Initial compile will happen when shader is loaded via checkExternalShader
        });
    }
});

// --- File List Logic ---
async function loadFileList() {
    try {
        log("Fetching shader list...", "info");
        const res = await fetch("/api/files");

        if (!res.ok) {
            throw new Error(`Server returned ${res.status} ${res.statusText}`);
        }

        const files = await res.json();

        if (!Array.isArray(files)) {
            throw new Error("Invalid response format from server (expected array)");
        }

        if (files.length === 0) {
            log("No .wgsl files found in examples folder.", "error");
            return;
        }

        shaderSelector.innerHTML =
            '<option value="" disabled selected>Select Shader...</option>';
        files.forEach((f: any) => {
            const option = document.createElement("option");
            option.value = f.path; // e.g., "examples/test.wgsl"
            option.textContent = f.name;
            shaderSelector.appendChild(option);
        });

        log(`Loaded ${files.length} shader examples.`, "success");

        // Auto-select first file if none selected
        if (!shaderSelector.value && files.length > 0) {
            shaderSelector.value = files[0].path;
            checkExternalShader(true); // Trigger load
        }
    } catch (e: any) {
        log(`Failed to load shader list: ${e.message}`, "error");
        console.error("Shader list fetch error:", e);
    }
}

// --- Auto-Sync Logic ---
let lastModifiedTime = 0;
async function checkExternalShader(force = false) {
    const selectedFile = shaderSelector.value;

    // Skip if no file selected OR (auto-sync unchecked AND not forced)
    if (!selectedFile || (!chkAutoSync.checked && !force)) return;

    try {
        // Pass the selected file as query param
        const res = await fetch(
            `/api/shader?file=${encodeURIComponent(selectedFile)}`
        );
        if (!res.ok) return;

        const data = await res.json();
        if (data.error) return;

        if (force || data.lastModified > lastModifiedTime) {
            const isFirstLoad = lastModifiedTime === 0;
            lastModifiedTime = data.lastModified;

            const currentVal = editor.getValue();

            // Logic update: if content really changed AND (forced OR no user edits OR current content equals last external content)
            if (
                (force || currentVal !== data.content) &&
                (!hasUserEdits || currentVal === lastExternalContent)
            ) {
                // Update last external content record
                lastExternalContent = data.content;
                // --- Use executeEdits instead of setValue to preserve Ctrl+Z ---
                const model = editor.getModel();
                if (model) {
                    // Calculate the full file range
                    const fullRange = model.getFullModelRange();

                    // Use executeEdits to update content, preserving undo history
                    editor.executeEdits("auto-sync", [
                        {
                            range: fullRange,
                            text: data.content,
                            forceMoveMarkers: true,
                        },
                    ]);

                    // Push undo stop to ensure this is a separate undo step
                    editor.pushUndoStop();
                }
                // --- End of modification ---

                hasUserEdits = false; // Reset user edits flag

                log(
                    `Shader loaded: ${data.path}`,
                    isFirstLoad || force ? "success" : "info"
                );
                compileShader();
            } else if (hasUserEdits && currentVal !== data.content) {
            }
        }
    } catch (e) {
        // Silent fail
    }
}

// Poll every 500ms
setInterval(checkExternalShader, 500);

// Create a simple checkerboard pattern for testing
async function createDefaultImage() {
    const size = 512;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;

    // Draw checkerboard
    const checkSize = 32;
    for (let y = 0; y < size; y += checkSize) {
        for (let x = 0; x < size; x += checkSize) {
            ctx.fillStyle =
                (x / checkSize + y / checkSize) % 2 === 0 ? "#444" : "#888";
            ctx.fillRect(x, y, checkSize, checkSize);
        }
    }

    // Draw text
    ctx.fillStyle = "white";
    ctx.font = "48px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("WGFX Test", size / 2, size / 2);

    return createImageBitmap(canvas);
}

log("System ready. Edit shader in IDE or load media.");
