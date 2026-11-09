// 印出特效 WGSL 編譯診斷（compilationInfo messages）；任何失敗一律 dump 生成碼
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
process.on('uncaughtException', e => console.error('[uncaught]', (e?.message || String(e)).split('\n').slice(0, 6).join('\n')));
process.on('unhandledRejection', e => console.error('[unhandledRejection]', (e?.message || String(e)).split('\n').slice(0, 6).join('\n')));
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const { WGFXRuntime } = await import('../../dist/wgfx.esm.js');
const file = process.argv[2];
if (!file) { console.error('usage: node tools/port/wgsl_diag.mjs <file.wgsl> [--dump]'); process.exit(2); }

function dump(r, base) {
    fs.mkdirSync('test/out_wgsl', { recursive: true });
    for (const m of r.generatedModules) fs.writeFileSync(`test/out_wgsl/${base}_P${m.passIndex}.wgsl`, m.wgslCode);
    console.log(`已 dump 生成碼 -> test/out_wgsl/${base}_P*.wgsl`);
}

const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();
const r = new WGFXRuntime(dev);
const U = GPUTextureUsage;
const opts = {
    defines: { INPUT_WIDTH: 128, INPUT_HEIGHT: 128 },
    textures: { INPUT: { size: [128, 128], format: 'rgba8unorm', usage: U.TEXTURE_BINDING | U.COPY_DST } }
};
try {
    try {
        await r.compile(fs.readFileSync(file, 'utf8'), opts);
    } catch (e) {
        // 效果未宣告 OUTPUT 尺寸 → 以輸入尺寸提供外部 OUTPUT 重試
        if (!/OUTPUT.*WIDTH\/HEIGHT/.test(e?.message || '')) throw e;
        await r.compile(fs.readFileSync(file, 'utf8'), {
            ...opts,
            textures: { ...opts.textures, OUTPUT: { size: [128, 128], format: 'rgba16float', usage: U.STORAGE_BINDING | U.COPY_SRC | U.TEXTURE_BINDING } }
        });
    }
} catch (e) {
    dump(r, path.basename(file, '.wgsl'));
    console.error('[compile 失敗]', (e?.message || String(e)).split('\n').slice(0, 8).join('\n'));
    process.exit(0);
}

const msgs = [];
dev.addEventListener?.('uncapturederror', ev => msgs.push('UNCAPTURED: ' + (ev.error?.message || ev.error)));
if (process.argv.includes('--dump')) dump(r, path.basename(file, '.wgsl'));
for (const m of r.generatedModules) {
    const sm = dev.createShaderModule({ code: m.wgslCode });
    const info = await sm.getCompilationInfo();
    for (const msg of info.messages) {
        const line = m.wgslCode.split('\n')[msg.lineNum - 1];
        msgs.push(`PASS ${m.passIndex} ${msg.type.toUpperCase()} L${msg.lineNum}:${msg.linePos} ${msg.message}\n    | ${line?.trim()}`);
    }
}
console.log(msgs.length ? msgs.join('\n') : '（無診斷訊息）');
dev.destroy?.();
process.exit(0);
