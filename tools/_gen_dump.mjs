// _gen_dump.mjs — 產生指定效果的 WGSL 模組並列出 Tint 全部編譯訊息（除錯用）
// 用法: node tools/_gen_dump.mjs <effect.wgsl>
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const file = process.argv[2];
if (!file) { console.error('usage: node tools/_gen_dump.mjs <effect.wgsl>'); process.exit(2); }

const { WGFXRuntime } = await import('../dist/wgfx.esm.js');
const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();

const code = fs.readFileSync(path.resolve(file), 'utf8');
const runtime = new WGFXRuntime(dev);
try {
    await runtime.compile(code, {
        defines: { INPUT_WIDTH: 256, INPUT_HEIGHT: 256 },
        textures: {
            INPUT: { size: [256, 256], format: 'rgba8unorm', usage: 0x10 | 0x2 },
            OUTPUT: { size: [512, 256], format: 'rgba16float', usage: 0x8 | 0x4 },
        },
    });
} catch (e) {
    console.error('compile threw:', e && e.message);
}
for (const m of runtime.generatedModules || []) {
    const out = `.tmp/gen_pass${m.passIndex}.wgsl`;
    fs.mkdirSync('.tmp', { recursive: true });
    fs.writeFileSync(out, m.wgslCode);
    console.log(`pass ${m.passIndex} → ${out} (${m.wgslCode.split('\n').length} lines)`);
    // 逐一建立 shader module 收集全部訊息
    const sm = dev.createShaderModule({ code: m.wgslCode, label: `probe pass ${m.passIndex}` });
    const info = await sm.getCompilationInfo();
    for (const msg of info.messages) {
        console.log(`  [${msg.type}] L${msg.lineNum}:${msg.linePos} ${msg.message.split('\n')[0]}`);
    }
}
dev.destroy?.();
