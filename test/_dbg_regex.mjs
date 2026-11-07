// 從 dump 的生成 WGSL 直接建 shaderModule，取得完整編譯訊息
import { create, globals } from 'webgpu';
import fs from 'fs';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
process.on('unhandledRejection', e => console.error('[rejection]', (e?.message || String(e)).split('\n').slice(0, 4).join('\n')));
const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();
const file = process.argv[2];
const code = fs.readFileSync(file, 'utf8');
const sm = dev.createShaderModule({ code, label: file });
const info = await sm.getCompilationInfo();
if (info.messages.length === 0) console.log('（無診斷訊息 — 模組有效）');
for (const m of info.messages) {
    const line = code.split('\n')[m.lineNum - 1];
    console.log(`${m.type.toUpperCase()} L${m.lineNum}:${m.linePos} ${m.message}\n    | ${line?.trim()}`);
}
dev.destroy?.();
process.exit(0);
