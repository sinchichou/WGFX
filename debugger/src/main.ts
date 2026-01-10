import * as monaco from 'monaco-editor';
// @ts-ignore
import { WGFX } from 'wgfx';

// 1. 修改 Import路�?：現?��???src ?��??��?了�??�相對路�?(移除?�誤??named export)
import initWesl, { run } from './wesl-web/wesl_web.js';

// 2. ?�鍵：使???url ?�訴 Vite ?�這個當作�??��?源網?�?��?
// @ts-ignore (如�?沒�?設�? d.ts ?�能?�報?��??�這�?忽略?�可)
import wasmUrl from './wesl-web/wesl_web_bg.wasm?url'; 

// Default shader code - using test.wgsl content for debugging
const DEFAULT_SHADER = `//! MAGPIE EFFECT WGFX
//! VERSION 4

//! PARAMETER InvertAmount
//! LABEL "Invert Amount"
//! MIN 0.0
//! MAX 1.0
//! DEFAULT 1.0
//! STEP 0.01
float InvertAmount;

//! TEXTURE INPUT
Texture2D INPUT;

//! TEXTURE OUTPUT
Texture2D OUTPUT;

//! SAMPLER LinearSampler
//! FILTER LINEAR
//! ADDRESS CLAMP
SamplerState LinearSampler;

//! COMMON
struct Constants {
    input_size: vec2<f32>,
    output_size: vec2<f32>,
};
@group(0) @binding(0) var<uniform> constants: Constants;

//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! STYLE CS
//! NUM_THREADS 8,8,1
//! DESC "A simple color inversion effect."

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tex_coords = vec2<f32>(global_id.xy) / constants.output_size;
    let original_color = textureSample(INPUT, LinearSampler, tex_coords);
    let inverted_color = vec4<f32>(
        mix(original_color.r, 1.0 - original_color.r, InvertAmount),
        mix(original_color.g, 1.0 - original_color.g, InvertAmount),
        mix(original_color.b, 1.0 - original_color.b, InvertAmount),
        original_color.a
    );
    textureStore(OUTPUT, global_id.xy, inverted_color);
}
`;

// DOM Elements
const canvas = document.getElementById('output-canvas') as HTMLCanvasElement;
const fileInput = document.getElementById('input-media') as HTMLInputElement;
const compileBtn = document.getElementById('btn-compile') as HTMLButtonElement;
const statusText = document.getElementById('status-text') as HTMLSpanElement;
const fpsMeter = document.getElementById('fps-meter') as HTMLDivElement;

// Video Controls
const videoControls = document.getElementById('video-controls') as HTMLDivElement;
const btnPlayPause = document.getElementById('btn-play-pause') as HTMLButtonElement;
const btnStop = document.getElementById('btn-stop') as HTMLButtonElement;
const seekBar = document.getElementById('seek-bar') as HTMLInputElement;
const timeDisplay = document.getElementById('time-display') as HTMLSpanElement;

const chkAutoSync = document.getElementById('chk-auto-sync') as HTMLInputElement;
const btnReload = document.getElementById('btn-reload') as HTMLButtonElement;
const shaderSelector = document.getElementById('shader-selector') as HTMLSelectElement;
const chkColorFilters = document.getElementById('chk-color-filters') as HTMLInputElement;
const volumeDisplay = document.getElementById('volume-display') as HTMLSpanElement;
const volumeBar = document.getElementById('volume-bar') as HTMLInputElement;
const splitHandle = document.getElementById('split-handle') as HTMLDivElement;
const splitControls = document.getElementById('split-controls') as HTMLDivElement;
const panelResizer = document.getElementById('panel-resizer') as HTMLDivElement;
const debugInfoResizer = document.getElementById('debug-info-resizer') as HTMLDivElement;

const logContent = document.getElementById('log-content') as HTMLDivElement;
const irContent = document.getElementById('ir-content') as HTMLDivElement;
const wgslContent = document.getElementById('wgsl-content') as HTMLDivElement;

// App State
let wgfx: any = null;
let device: GPUDevice | null = null;
let inputSource: ImageBitmap | VideoFrame | HTMLVideoElement | null = null;
let currentVideoElement: HTMLVideoElement | null = null; // Track active video
let editor: monaco.editor.IStandaloneCodeEditor;
let isAnimating = false;
let frameCount = 0;
let lastTime = 0;
let useColorFilters = true; // Track color filter state
let splitRatio = 0.5; // Track split ratio (0.0 = all original, 1.0 = all processed)
let displayMode = 'split'; // 'split' or 'single'
let panelRatio = 0.5; // Track panel ratio (0.0 = all editor, 1.0 = all preview)
let isPanelDragging = false; // Track panel resizer drag state
let debugInfoHeight = 320; // Track debug info panel height
let isDebugInfoDragging = false; // Track debug info resizer drag state

const currentRatioDisplay = document.getElementById('current-ratio') as HTMLSpanElement;

// Blit Pipeline for displaying results
let blitPipeline: GPURenderPipeline | null = null;
let blitSampler: GPUSampler | null = null;

// Logging Helper
function log(msg: string, type: 'info' | 'error' | 'success' = 'info') {
    const div = document.createElement('div');
    div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    div.className = `log-${type}`;
    logContent.appendChild(div);
    logContent.scrollTop = logContent.scrollHeight;

    statusText.textContent = msg;
    if (type === 'error') statusText.style.color = '#f48771';
    else statusText.style.color = '#888';
}

// Format time helper (HH:MM:SS)
function formatTime(seconds: number): string {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
}

// Update volume display
function updateVolumeDisplay() {
    if (currentVideoElement) {
        const volumePercent = Math.round(currentVideoElement.volume * 100);
        volumeDisplay.textContent = `VOL: ${volumePercent}%`;
    }
}

// Update split handle position and display
function updateSplitHandle() {
    if (displayMode === 'single') {
        splitHandle.style.display = 'none';
        if (splitControls) splitControls.style.display = 'none';
    } else {
        splitHandle.style.display = 'block';
        if (splitControls) splitControls.style.display = 'flex';

        const percentage = splitRatio * 100;
        splitHandle.style.left = `${percentage}%`;
        if (currentRatioDisplay) {
            currentRatioDisplay.textContent = `${Math.round(percentage)}%`;
        }
    }
}

// Handle split slider drag
let isDragging = false;
function startSplitDrag(e: MouseEvent) {
    isDragging = true;
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    splitHandle.classList.add('dragging');
    updateSplitFromMouse(e);
}

function updateSplitFromMouse(e: MouseEvent) {
    const canvasWrapper = document.getElementById('output-canvas')?.parentElement;
    if (!canvasWrapper) return;

    const rect = canvasWrapper.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    splitRatio = ratio;
    updateSplitHandle();
}

function endSplitDrag() {
    isDragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    splitHandle.classList.remove('dragging');
}

// Update mode button states
function updateModeButtons() {
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`mode-${displayMode}`)?.classList.add('active');
}

// Handle preset buttons
function handlePresetClick(ratio: number) {
    splitRatio = ratio;
    updateSplitHandle();

    // Update active button
    document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-split="${ratio}"]`)?.classList.add('active');
}

// Update panel sizes
function updatePanelSizes() {
    const editorPanel = document.querySelector('.panel:first-child') as HTMLElement;
    const isMobile = window.innerWidth <= 768;

    if (isMobile) {
        // Mobile: vertical layout
        const totalHeight = window.innerHeight;
        const editorHeight = totalHeight * panelRatio;

        editorPanel.style.width = '100%';
        editorPanel.style.height = `${editorHeight}px`;
        editorPanel.style.flex = 'none';

        if (panelResizer) {
            panelResizer.style.cursor = 'ns-resize';
            panelResizer.style.width = '100%';
            panelResizer.style.height = '8px';
        }
    } else {
        // Desktop: horizontal layout
        const totalWidth = window.innerWidth;
        const editorWidth = totalWidth * panelRatio;

        editorPanel.style.width = `${editorWidth}px`;
        editorPanel.style.height = '100vh';
        editorPanel.style.flex = 'none';

        if (panelResizer) {
            panelResizer.style.cursor = 'ew-resize';
            panelResizer.style.width = '8px';
            panelResizer.style.height = '100vh';
        }
    }
}

// Handle panel resizer drag
function startPanelDrag(e: MouseEvent) {
    isPanelDragging = true;
    const isMobile = window.innerWidth <= 768;
    document.body.style.cursor = isMobile ? 'ns-resize' : 'ew-resize';
    document.body.style.userSelect = 'none';
    if (panelResizer) panelResizer.classList.add('dragging');
    updatePanelFromMouse(e);
}

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

function endPanelDrag() {
    isPanelDragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    if (panelResizer) panelResizer.classList.remove('dragging');
}

// Handle debug info resizer drag
function startDebugInfoDrag(e: MouseEvent) {
    isDebugInfoDragging = true;
    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
    if (debugInfoResizer) debugInfoResizer.classList.add('dragging');
    updateDebugInfoFromMouse(e);
}

function updateDebugInfoFromMouse(e: MouseEvent) {
    const container = document.querySelector('.panel:last-child') as HTMLElement;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const newHeight = Math.max(150, Math.min(600, containerRect.bottom - e.clientY));

    debugInfoHeight = newHeight;
    updateDebugInfoHeight();
}

function updateDebugInfoHeight() {
    const debugInfo = document.querySelector('.debug-info') as HTMLElement;
    if (debugInfo) {
        debugInfo.style.height = `${debugInfoHeight}px`;
    }
}

function endDebugInfoDrag() {
    isDebugInfoDragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    if (debugInfoResizer) debugInfoResizer.classList.remove('dragging');
}

async function setupWeslValidator(editorInstance: monaco.editor.IStandaloneCodeEditor) {    
    try {
        // 1. Initialize WASM (using module_or_path parameter to avoid warnings)
        await initWesl({ module_or_path: wasmUrl });
        console.log("WESL 語法檢查引擎已啟動");

        const model = editorInstance.getModel();
        if (!model) return;

        // 2. Define validation logic
        const performValidation = () => {
            const code = model.getValue();
            
            // 使用 run 參數 Compile 命令語法檢查
            let diagnostics: any[] = [];
            try {
                // 嘗試編譯，如果成功則沒有診斷信息
                void run({
                    command: "Compile",
                    files: { "main.wgsl": code },
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
                console.log("WESL 編譯錯誤:", err);
                if (err && err.diagnostics && Array.isArray(err.diagnostics)) {
                    diagnostics = err.diagnostics;
                    console.log("找到診斷信息:", diagnostics.length, "個錯誤");
                } else if (err && err.message) {
                    // 如果沒有 diagnostics，嘗試從錯誤消息中提取
                    console.warn("WESL 錯誤沒有 diagnostics，錯誤消息:", err.message);
                    // Create a generic diagnostic message
                    diagnostics = [{
                        title: err.message || 'Compilation error',
                        span: { start: 0, end: 0 }
                    }];
                } else {
                    console.error("WESL Analysis failed:", err);
                    return;
                }
            }

            // 3. Convert to Monaco Markers
            // Need to convert to Monaco Markers
            const lines = code.split('\n');
            const markers = diagnostics.map((d: any) => {
                // Calculate line and column positions
                let lineNumber = 1;
                let column = 1;
                let endLineNumber = 1;
                let endColumn = 1;

                if (d.span && typeof d.span.start === 'number') {
                    // Calculate start position from span.start
                    let charCount = 0;
                    for (let i = 0; i < lines.length; i++) {
                        const lineLength = lines[i].length + 1; // +1 for newline
                        if (charCount + lineLength > d.span.start) {
                            lineNumber = i + 1;
                            column = Math.max(1, d.span.start - charCount + 1);
                            break;
                        }
                        charCount += lineLength;
                    }

                    // Calculate end position
                    endLineNumber = lineNumber;
                    endColumn = column;
                    if (d.span.end && typeof d.span.end === 'number' && d.span.end > d.span.start) {
                        let endCharCount = 0;
                        for (let i = 0; i < lines.length; i++) {
                            const lineLength = lines[i].length + 1;
                            if (endCharCount + lineLength > d.span.end) {
                                endLineNumber = i + 1;
                                endColumn = Math.max(1, d.span.end - endCharCount + 1);
                                break;
                            }
                            endCharCount += lineLength;
                        }
                    } else {
                        // If no end position, use entire line
                        endColumn = lines[lineNumber - 1]?.length || column;
                    }
                }
                
                return {
                    message: d.title || d.message || 'Syntax error',
                    severity: monaco.MarkerSeverity.Error,
                    startLineNumber: lineNumber,
                    startColumn: column,
                    endLineNumber: endLineNumber,
                    endColumn: endColumn
                };
            });

            console.log(`設定 ${markers.length} 個語法錯誤標記`);
            // 4. Set red error markers (Owner set to 'wesl' to avoid conflicts with other owners)
            monaco.editor.setModelMarkers(model, "wesl", markers);
        };

        // 5. On change (debounce input)
        let debounceHandle: any;
        editorInstance.onDidChangeModelContent(() => {
            clearTimeout(debounceHandle);
            debounceHandle = setTimeout(performValidation, 500); // 500ms 延遲
        });

        // Every time
        performValidation();

    } catch (e) {
        console.error("無法啟動 WESL 檢查器:", e);
        log("WESL Validator failed to load. Check console.", 'error');
    }
}

// Configure Monaco Editor workers (worker warning)
// Use simple setup for worker main thread (any environment can accept)
// @ts-ignore
(window as any).MonacoEnvironment = {
    getWorkerUrl: function () {
        // Return an empty string for Monaco main thread
        // To avoid warning, any environment can accept
        return '';
    }
};

// Register WGSL language (Monarch syntax highlighting)
monaco.languages.register({ id: 'wgsl' });
monaco.languages.setMonarchTokensProvider('wgsl', {
    tokenizer: {
        root: [
            // Keywords            [/\b(fn|let|var|const|struct|if|else|loop|for|while|break|continue|return|discard|switch|case|default)\b/, 'keyword'],
            // Types
            [/\b(bool|i32|u32|f32|f16|vec2|vec3|vec4|mat2x2|mat3x3|mat4x4|texture_2d|sampler|sampler_comparison|Texture2D|SamplerState)\b/, 'type'],
            // Attributes            [/@\w+/, 'attribute'],
            // Numbers
            [/\b\d+\.?\d*[f]?\b/, 'number'],
            // Strings            [/"[^"]*"/, 'string'],
            // Comments
            [/\/\/.*$/, 'comment'],
            [/\/\*[\s\S]*?\*\//, 'comment'],
            // Operators            [/[+\-*/%=<>!&|]+/, 'operator'],
            // Identifiers            [/\b[a-zA-Z_][a-zA-Z0-9_]*\b/, 'identifier'],
        ]
    }
});

// Define custom WGSL theme (based on vs-dark, but set better colors for WGSL tokens)
monaco.editor.defineTheme('wgsl-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
        // 關鍵字 - 使用亮藍色（類似 TypeScript 的關鍵字）
        { token: 'keyword', foreground: '569cd6', fontStyle: 'bold' },
        // 類型 - 使用青藍色
        { token: 'type', foreground: '4ec9b0' },
        // 屬性 - 使用紫色
        { token: 'attribute', foreground: 'c586c0' },
        // 數字 - 使用淺綠色
        { token: 'number', foreground: 'b5cea8' },
        // 字符串 - 使用橙色
        { token: 'string', foreground: 'ce9178' },
        // 註釋 - 使用灰色
        { token: 'comment', foreground: '6a9955', fontStyle: 'italic' },
        // 運算符 - 使用白色（突出顯示）
        { token: 'operator', foreground: 'd4d4d4' },
        // 標識符 - 使用默認顏色（繼承主題）
        { token: 'identifier', foreground: 'd4d4d4' },
    ],
    colors: {
        'editor.background': '#1e1e1e',
        'editor.foreground': '#d4d4d4',
    }
});

// Initialize Monaco Editor
function initEditor() {
    editor = monaco.editor.create(document.getElementById('editor')!, {
        value: DEFAULT_SHADER,
        language: 'wgsl', // 使用註�???WGSL 語�?
        theme: 'wgsl-dark', // 使用自定義的 WGSL 主題
        minimap: { enabled: false },
        lineNumbers: 'on',
        lineNumbersMinChars: 3,
        glyphMargin: true, // ?�用以顯示錯誤�?�?        folding: true,
        renderLineHighlight: 'line',
        scrollBeyondLastLine: false,
        wordWrap: 'off',
        tabSize: 4,
        insertSpaces: true,
        detectIndentation: false
    });

    // Compile on Ctrl+S
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
        compileShader();
    });
    setupWeslValidator(editor);

}

// Initialize WebGPU
async function initWebGPU() {
    //#region agent log
    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initWebGPU',message:'Starting WebGPU initialization',data:{hypothesisId:'A',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
    //#endregion

    if (!navigator.gpu) {
        log('WebGPU not supported in this browser.', 'error');
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initWebGPU',message:'WebGPU not supported',data:{hypothesisId:'A',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
        return false;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        log('Failed to get WebGPU adapter.', 'error');
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initWebGPU',message:'No WebGPU adapter available',data:{hypothesisId:'A',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
        return false;
    }

    //#region agent log
    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initWebGPU',message:'WebGPU adapter obtained',data:{hypothesisId:'A',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
    //#endregion

    // Request maximum limits supported by the hardware
    // We explicitly copy the limits to a plain object to avoid potential issues
    // with passing the GPUSupportedLimits object directly.
    const requiredLimits: Record<string, number> = {};
    // @ts-ignore - Iterate over limits
    for (const key in adapter.limits) {
        // @ts-ignore
        const value = adapter.limits[key];
        if (typeof value === 'number') {
            requiredLimits[key] = value;
        }
    }

    try {
        device = await adapter.requestDevice({
            requiredLimits
        });
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initWebGPU',message:'Device created with max limits',data:{hypothesisId:'A',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
    } catch (err) {
        log(`Failed to request max limits: ${err}. Fallback to default.`, 'error');
        device = await adapter.requestDevice();
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initWebGPU',message:'Device created with default limits after max limits failed',data:{hypothesisId:'A',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
    }

    log('WebGPU initialized successfully.', 'success');
    return true;
}

// Load Media
async function handleFileSelect(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);

    // Cleanup previous video
    if (currentVideoElement) {
        currentVideoElement.pause();
        currentVideoElement.src = "";
        currentVideoElement = null;
    }

    if (file.type.startsWith('video')) {
        const video = document.createElement('video');
        video.src = url;
        video.loop = true;
        video.muted = false; // Allow audio
        video.autoplay = true;

        // Handle video end
        video.addEventListener('ended', () => {
            btnPlayPause.textContent = '?��?';
            seekBar.value = "0";
            timeDisplay.textContent = `00:00 / ${formatTime(video.duration)}`;
        });
        // Wait for metadata to get duration
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                seekBar.max = video.duration.toString();
                seekBar.value = "0";
                timeDisplay.textContent = `00:00 / ${formatTime(video.duration)}`;
                resolve(null);
            };
        });

        await video.play();
        inputSource = video;
        currentVideoElement = video;

        // Set initial volume
        video.volume = volumeBar.valueAsNumber / 100;

        // Show Controls
        videoControls.style.display = 'flex';
        btnPlayPause.textContent = '?��?';
        updateVolumeDisplay();

        log(`Video loaded: ${file.name} (${video.videoWidth}x${video.videoHeight})`, 'success');
        resizeCanvas(video.videoWidth, video.videoHeight);
    } else {
        const img = new Image();
        img.src = url;
        await new Promise(r => img.onload = r);
        const bitmap = await createImageBitmap(img);
        inputSource = bitmap;

        // Hide Controls
        videoControls.style.display = 'none';

        log(`Image loaded: ${file.name} (${bitmap.width}x${bitmap.height})`, 'success');
        resizeCanvas(bitmap.width, bitmap.height);
    }

    // Re-compile if WGFX exists to update dimensions
    // Always attempt compile when image changes, as it might be the first input
    compileShader();
}

let isCanvasConfigured = false;

function resizeCanvas(w: number, h: number) {
    // Set canvas to display the full video frame
    const aspect = w / h;
    const container = canvas.parentElement!;

    // Fit to container while maintaining aspect ratio
    let dispW = container.clientWidth;
    let dispH = dispW / aspect;

    if (dispH > container.clientHeight) {
        dispH = container.clientHeight;
        dispW = dispH * aspect;
    }

    // Set display size (CSS) - this controls how big the canvas appears on screen
    canvas.style.width = `${dispW}px`;
    canvas.style.height = `${dispH}px`;

    // Set internal resolution to match the video dimensions
    // The shader will handle splitting the display
    canvas.width = w;
    canvas.height = h;

    initCanvasContext();

    // Initialize split control
    updateSplitHandle();
}

function initCanvasContext() {
    if (!device) {
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initCanvasContext',message:'No device available for canvas context',data:{hypothesisId:'B',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
        return;
    }
    const ctx = (canvas.getContext('webgpu') as unknown as GPUCanvasContext);
    if (ctx) {
        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initCanvasContext',message:'Configuring canvas context',data:{canvasFormat, hypothesisId:'B',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
        ctx.configure({
            device,
            format: canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        isCanvasConfigured = true;
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initCanvasContext',message:'Canvas context configured successfully',data:{hypothesisId:'B',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
    } else {
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initCanvasContext',message:'Failed to get WebGPU canvas context',data:{hypothesisId:'B',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
    }
}

// Video Control Listeners
btnPlayPause.addEventListener('click', () => {
    if (!currentVideoElement) return;
    if (currentVideoElement.paused) {
        currentVideoElement.play();
        btnPlayPause.textContent = '?��?';
    } else {
        currentVideoElement.pause();
        btnPlayPause.textContent = '?��?';
    }
});

seekBar.addEventListener('input', () => {
    if (!currentVideoElement) return;
    const time = parseFloat(seekBar.value);
    currentVideoElement.currentTime = time;
    timeDisplay.textContent = `${formatTime(time)} / ${formatTime(currentVideoElement.duration)}`;
});

btnStop.addEventListener('click', () => {
    if (!currentVideoElement) return;
    currentVideoElement.pause();
    currentVideoElement.currentTime = 0;
    seekBar.value = "0";
    timeDisplay.textContent = `00:00 / ${formatTime(currentVideoElement.duration)}`;
    btnPlayPause.textContent = '?��?';
});

// Volume Control
volumeBar.addEventListener('input', () => {
    if (!currentVideoElement) return;
    currentVideoElement.volume = volumeBar.valueAsNumber / 100;
    updateVolumeDisplay();
});

// Main Compile & Run Logic
async function compileShader() {
    if (!device || !inputSource) {
        if (!inputSource) log('Please load an image or video first.', 'error');
        return;
    }

    const code = editor.getValue();
    const inputWidth = (inputSource instanceof HTMLVideoElement) ? inputSource.videoWidth : (inputSource as ImageBitmap).width;
    const inputHeight = (inputSource instanceof HTMLVideoElement) ? inputSource.videoHeight : (inputSource as ImageBitmap).height;

    // Use input dimensions for WGFX processing
    const width = inputWidth;
    const height = inputHeight;

    //#region agent log
    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:compileShader',message:'Starting shader compilation',data:{width, height, codeLength: code.length, hypothesisId:'C',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
    //#endregion

    log('Compiling shader...');
    statusText.textContent = 'Compiling...';

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
            height
        });

        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:compileShader',message:'WGFX instance created',data:{hasWgfx: !!wgfx, hypothesisId:'C',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion

        const info = wgfx.initialize();

        // Set uniforms matching the test shader structure
        const uniforms = {
            input_size: [width, height],
            output_size: [width, height]
        };
        wgfx.updateUniforms(uniforms);

        // Set shader parameters
        wgfx.updateUniforms({ InvertAmount: 1.0 });

        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:compileShader',message:'Uniforms set',data:{uniforms, shaderInfo: wgfx.runtime?.shaderInfo, hypothesisId:'I',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion

        log('Compilation successful!', 'success');

        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:compileShader',message:'WGFX initialized successfully',data:{hasInfo: !!info, uniformsSet: true, hypothesisId:'C',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion

        // Extract Debug Info
        const runtime = wgfx.runtime;
        irContent.textContent = JSON.stringify(runtime.shaderInfo, null, 2);

        // Show generated WGSL code
        if (runtime.generatedModules && runtime.generatedModules.length > 0) {
            let wgslText = "";
            runtime.generatedModules.forEach((module: { passIndex: any; wgslCode: string; }) => {
                wgslText += `// === PASS ${module.passIndex} ===\n`;
                wgslText += module.wgslCode;
                wgslText += "\n\n";
            });
            wgslContent.textContent = wgslText;
        } else {
            wgslContent.textContent = "// No generated WGSL modules found.\n// Check that shader compilation was successful.";
        }

        startRenderLoop();

    } catch (e: any) {
        log(e.message, 'error');
        console.error(e);
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:compileShader',message:'Shader compilation failed',data:{error: e.message, hypothesisId:'C',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
    }
}

function initBlitPipeline() {
    if (!device) {
        //#region agent log
        fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initBlitPipeline',message:'No device available for blit pipeline',data:{hypothesisId:'E',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
        //#endregion
        return;
    }

    //#region agent log
    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initBlitPipeline',message:'Creating blit sampler',data:{hypothesisId:'E',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
    //#endregion

    blitSampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
    });

    const shaderModule = device.createShaderModule({
        label: 'Blit Shader (Split View)',
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
                    let color = textureSampleLevel(texOutput, s, uv, 0.0);

                    // Check if output texture appears to be empty/uninitialized
                    let isBlackOrTransparent = (color.r + color.g + color.b) < 0.01;

                    if (isBlackOrTransparent) {
                        // If output is black/empty, show a pattern to confirm rendering works
                        let patternUV = uv * 10.0;
                        let checker = (floor(patternUV.x) + floor(patternUV.y)) % 2.0;
                        return vec4<f32>(1.0, checker * 0.5, 0.0, 1.0); // Orange checkerboard if output is empty
                    } else {
                        // Output has content, apply color filters if enabled
                        return select(color, mix(color, vec4<f32>(0.0, 0.2, 0.0, 1.0), 0.05), uniforms.useFilters == 1.0);
                    }
                } else {
                    // Split screen mode
                    if (uv.x < uniforms.splitRatio) {
                        // Left portion: processed output texture
                        let color = textureSampleLevel(texOutput, s, uv, 0.0);

                        // Check if output texture appears to be empty/uninitialized
                        let isBlackOrTransparent = (color.r + color.g + color.b) < 0.01;

                        if (isBlackOrTransparent) {
                            // If output is black/empty, show a pattern to confirm rendering works
                            let patternUV = uv * 10.0;
                            let checker = (floor(patternUV.x) + floor(patternUV.y)) % 2.0;
                            return vec4<f32>(1.0, checker * 0.5, 0.0, 1.0); // Orange checkerboard if output is empty
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
        `
    });

    //#region agent log
    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initBlitPipeline',message:'Shader module created, creating render pipeline',data:{hypothesisId:'E',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
    //#endregion

    // Create uniform buffer for filter, split and display settings (useFilters: u32, splitRatio: f32, displayMode: u32)
    const uniformBuffer = device.createBuffer({
        size: 12, // 4 bytes for u32 + 4 bytes for f32 + 4 bytes for u32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create bind group layout manually since we need specific binding for uniforms
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ],
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
    });

    blitPipeline = device.createRenderPipeline({
        label: 'Blit Pipeline',
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vs_main',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
        },
        primitive: {
            topology: 'triangle-list',
        },
    });

    // Store uniform buffer and layout for later updates
    (blitPipeline as any).uniformBuffer = uniformBuffer;
    (blitPipeline as any).bindGroupLayout = bindGroupLayout;

    //#region agent log
    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:initBlitPipeline',message:'Blit pipeline created successfully',data:{hasBlitPipeline: !!blitPipeline, hasBlitSampler: !!blitSampler, hypothesisId:'E',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
    //#endregion
}

function startRenderLoop() {
    if (isAnimating) return;
    isAnimating = true;

    //#region agent log
    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop','message':'Starting render loop','data':{canvasVisible: canvas.offsetWidth > 0 && canvas.offsetHeight > 0, canvasInDOM: !!canvas.parentElement, canvasDisplay: canvas.style.display, hypothesisId:'F',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
    //#endregion

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
                timeDisplay.textContent = `${formatTime(currentVideoElement.currentTime)} / ${formatTime(currentVideoElement.duration)}`;

                // Update play/pause button state
                if (currentVideoElement.paused) {
                    btnPlayPause.textContent = '?��?';
                } else {
                    btnPlayPause.textContent = '?��?';
                }
            }

            //#region agent log
            fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'About to call wgfx.process',data:{hasWgfx: !!wgfx, hasInputSource: !!inputSource, hypothesisId:'D',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
            //#endregion

            //#region agent log
            fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'About to call wgfx.process',data:{inputSourceType: inputSource?.constructor?.name, hasWgfx: !!wgfx, hypothesisId:'D',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
            //#endregion

            // Try different input formats like content.js does
            let processedInput = inputSource;
            if (inputSource instanceof HTMLVideoElement) {
                // Convert video element to VideoFrame like content.js
                if (typeof VideoFrame !== 'undefined') {
                    try {
                        processedInput = new VideoFrame(inputSource);
                    } catch (e) {
                        // If VideoFrame fails, use original video element
                        processedInput = inputSource;
                    }
                }
            }

            let outputTexture;
            try {
                outputTexture = await wgfx.process(processedInput);
            } catch (e: any) {
                log(`WGFX process error: ${e?.message || String(e)}`, 'error');
                outputTexture = null;
            }

            // Clean up VideoFrame or ImageBitmap if we created one
            if (processedInput !== inputSource) {
                // VideoFrame (WebCodecs)
                if (
                    typeof VideoFrame !== "undefined" &&
                    processedInput instanceof VideoFrame &&
                    typeof processedInput.close === "function"
                ) {
                    processedInput.close();
                }
                // ImageBitmap
                else if (
                    typeof ImageBitmap !== "undefined" &&
                    processedInput instanceof ImageBitmap &&
                    typeof processedInput.close === "function"
                ) {
                    processedInput.close();
                }
            }

            //#region agent log
            fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'WGFX processing completed',data:{hasOutputTexture: !!outputTexture, outputWidth: outputTexture?.width, outputHeight: outputTexture?.height, outputFormat: outputTexture?.format, hypothesisId:'D',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
            //#endregion

            const inputTexture = wgfx.runtime.resourceManager.getTexture('INPUT');

            //#region agent log
            fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'WGFX internal state check',data:{hasRuntime: !!wgfx.runtime, runtimeType: typeof wgfx.runtime, pipelineCount: wgfx.runtime?.pipelineManager?.pipelines?.length || 0, hypothesisId:'H',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
            //#endregion

            //#region agent log
            fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'Texture comparison',data:{inputWidth: inputTexture?.width, inputHeight: inputTexture?.height, outputWidth: outputTexture?.width, outputHeight: outputTexture?.height, texturesIdentical: inputTexture?.width === outputTexture?.width && inputTexture?.height === outputTexture?.height, hypothesisId:'G',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
            //#endregion

            //#region agent log
            fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'Got input texture from resource manager',data:{hasInputTexture: !!inputTexture, hypothesisId:'D',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
            //#endregion

            // Draw result to canvas
            const ctx = (canvas.getContext('webgpu') as unknown as GPUCanvasContext);
            if (ctx && isCanvasConfigured && inputTexture) {
                if (!blitPipeline) {
                    initBlitPipeline();
                }

                const currentTexture = ctx.getCurrentTexture();

                //#region agent log
                fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'Starting canvas rendering',data:{hasCtx: !!ctx, isCanvasConfigured, hasBlitPipeline: !!blitPipeline, hasBlitSampler: !!blitSampler, hasCurrentTexture: !!currentTexture, hypothesisId:'E',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
                //#endregion

                const commandEncoder = device.createCommandEncoder();

                const passEncoder = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: currentTexture.createView(),
                        clearValue: { r: 0, g: 0, b: 0, a: 1 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    }]
                });

                if (blitPipeline && blitSampler) {
                    // Update uniform buffer with current filter, split and display settings
                    const displayModeValue = displayMode === 'single' ? 1 : 0;
                    const uniformData = new Float32Array([useColorFilters ? 1 : 0, splitRatio, displayModeValue]);
                    device.queue.writeBuffer((blitPipeline as any).uniformBuffer, 0, uniformData);

                    //#region agent log
                    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:renderLoop',message:'Uniform data updated',data:{useColorFilters, splitRatio, displayMode, displayModeValue, hypothesisId:'J',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
                    //#endregion

                    const bindGroup = device.createBindGroup({
                        layout: (blitPipeline as any).bindGroupLayout || blitPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: inputTexture.createView() },
                            { binding: 1, resource: outputTexture.createView() },
                            { binding: 2, resource: blitSampler },
                            { binding: 3, resource: { buffer: (blitPipeline as any).uniformBuffer } },
                        ],
                    });

                    passEncoder.setPipeline(blitPipeline);
                    passEncoder.setBindGroup(0, bindGroup);
                    passEncoder.draw(3);

                    //#region agent log
                    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'Blit rendering completed',data:{hypothesisId:'E',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
                    //#endregion
                } else {
                    //#region agent log
                    fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'Blit pipeline or sampler not available',data:{hasBlitPipeline: !!blitPipeline, hasBlitSampler: !!blitSampler, hypothesisId:'E',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
                    //#endregion
                }

                passEncoder.end();
                device.queue.submit([commandEncoder.finish()]);

                //#region agent log
                fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'Command buffer submitted',data:{hypothesisId:'E',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
                //#endregion
            } else {
                //#region agent log
                fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'Canvas rendering skipped',data:{hasCtx: !!ctx, isCanvasConfigured, hasInputTexture: !!inputTexture, hypothesisId:'E',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
                //#endregion
            }
        } catch (e: any) {
            console.error(e);
            isAnimating = false; // Stop on error
            //#region agent log
            fetch('http://127.0.0.1:7242/ingest/8299c694-88e8-4ba2-a532-4802af9bb93d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.ts:startRenderLoop',message:'Render loop error',data:{error: e?.message || String(e), hypothesisId:'D',runId:'initial'},timestamp:Date.now(),sessionId:'debug-session'})}).catch(()=>{});
            //#endregion
            return;
        }

        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

// Event Listeners
fileInput.addEventListener('change', handleFileSelect);
compileBtn.addEventListener('click', compileShader);

// Reload Button
btnReload.addEventListener('click', () => {
    lastModifiedTime = 0;
    checkExternalShader(true);
});

// Shader Selector
shaderSelector.addEventListener('change', () => {
    lastModifiedTime = 0; // Reset
    checkExternalShader(true); // Force load
});

// Color Filters Toggle
chkColorFilters.addEventListener('change', () => {
    useColorFilters = chkColorFilters.checked;
});

// Display Mode Buttons
document.getElementById('mode-split')?.addEventListener('click', () => {
    displayMode = 'split';
    updateModeButtons();
    updateSplitHandle();
    console.log('Switched to split view mode');
});

document.getElementById('mode-single')?.addEventListener('click', () => {
    displayMode = 'single';
    updateModeButtons();
    updateSplitHandle();
    console.log('Switched to single view mode');
});

// Split Control Events
splitHandle.addEventListener('mousedown', startSplitDrag);

// Panel Resizer Events
if (panelResizer) {
    panelResizer.addEventListener('mousedown', startPanelDrag);
}

// Debug Info Resizer Events
if (debugInfoResizer) {
    debugInfoResizer.addEventListener('mousedown', startDebugInfoDrag);
}

document.addEventListener('mousemove', (e) => {
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
document.addEventListener('mouseup', () => {
    endSplitDrag();
    endPanelDrag();
    endDebugInfoDrag();
});

// Preset Buttons (delegate event to handle dynamically added buttons)
document.addEventListener('click', (e) => {
    const target = e.target as HTMLElement;
    if (target.classList.contains('preset-btn')) {
        const ratio = parseFloat(target.dataset.split!);
        handlePresetClick(ratio);
    }
});

// Tabs
document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', (e) => {
        const targetId = (e.target as HTMLElement).dataset.target;

        document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
        (e.target as HTMLElement).classList.add('active');

        document.querySelectorAll('.tab-content').forEach(c => (c as HTMLElement).style.display = 'none');
        document.getElementById(`${targetId}-content`)!.style.display = 'block';
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
window.addEventListener('resize', () => {
    updatePanelSizes();
    updateDebugInfoHeight();
    // Also update canvas if needed
    if (inputSource && canvas.width > 0 && canvas.height > 0) {
        resizeCanvas(canvas.width, canvas.height);
    }
});

// Initialize current ratio display if element exists
if (currentRatioDisplay) {
    currentRatioDisplay.textContent = '50%';
}
initWebGPU().then(ready => {
    if (ready) {
        createDefaultImage().then(bitmap => {
            inputSource = bitmap;
            log('Loaded default test pattern.', 'success');
            resizeCanvas(bitmap.width, bitmap.height);
            // Initial compile will happen when shader is loaded via checkExternalShader
        });
    }
});

// --- File List Logic ---
async function loadFileList() {
    try {
        log('Fetching shader list...', 'info');
        const res = await fetch('/api/files');

        if (!res.ok) {
            throw new Error(`Server returned ${res.status} ${res.statusText}`);
        }

        const files = await res.json();

        if (!Array.isArray(files)) {
            throw new Error('Invalid response format from server (expected array)');
        }

        if (files.length === 0) {
            log('No .wgsl files found in examples folder.', 'error');
            return;
        }

        shaderSelector.innerHTML = '<option value="" disabled selected>Select Shader...</option>';
        files.forEach((f: any) => {
            const option = document.createElement('option');
            option.value = f.path; // e.g., "examples/test.wgsl"
            option.textContent = f.name;
            shaderSelector.appendChild(option);
        });

        log(`Loaded ${files.length} shader examples.`, 'success');

        // Auto-select first file if none selected
        if (!shaderSelector.value && files.length > 0) {
            shaderSelector.value = files[0].path;
            checkExternalShader(true); // Trigger load
        }

    } catch (e: any) {
        log(`Failed to load shader list: ${e.message}`, 'error');
        console.error('Shader list fetch error:', e);
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
        const res = await fetch(`/api/shader?file=${encodeURIComponent(selectedFile)}`);
        if (!res.ok) return;

        const data = await res.json();
        if (data.error) return;

        if (force || data.lastModified > lastModifiedTime) {
            const isFirstLoad = lastModifiedTime === 0;
            lastModifiedTime = data.lastModified;

            const currentVal = editor.getValue();
            // Update if content changed OR forced reload
            if (force || currentVal !== data.content) {
                editor.setValue(data.content);
                log(`Shader loaded: ${data.path}`, isFirstLoad || force ? 'success' : 'info');
                compileShader();
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
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;

    // Draw checkerboard
    const checkSize = 32;
    for (let y = 0; y < size; y += checkSize) {
        for (let x = 0; x < size; x += checkSize) {
            ctx.fillStyle = ((x / checkSize + y / checkSize) % 2 === 0) ? '#444' : '#888';
            ctx.fillRect(x, y, checkSize, checkSize);
        }
    }

    // Draw text
    ctx.fillStyle = 'white';
    ctx.font = '48px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('WGFX Test', size / 2, size / 2);

    return createImageBitmap(canvas);
}

log('System ready. Edit shader in IDE or load media.');
