// _storage_probe.mjs — 探測此 Dawn/node-webgpu 支援的 storage texture 格式與存取模式
// 注意:Dawn 的 createTexture/createBindGroupLayout 驗證錯誤**不丟 JS 例外**,
// 只走 uncaptured error / error scope,因此本探測以 pushErrorScope/popErrorScope 判定。
// 用法: node tools/_storage_probe.mjs
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };

const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();

const FORMATS = [
    'r8unorm', 'r8snorm', 'r8uint', 'r8sint',
    'rg8unorm', 'rg8snorm', 'rg8uint', 'rg8sint',
    'rgba8unorm', 'rgba8snorm', 'rgba8uint', 'rgba8sint', 'rgba8unorm-srgb',
    'bgra8unorm', 'bgra8unorm-srgb',
    'r16float', 'rg16float', 'rgba16float',
    'r16unorm', 'rg16unorm', 'rgba16unorm',
    'r16uint', 'rg16uint', 'rgba16uint', 'r16sint', 'rg16sint', 'rgba16sint',
    'r32float', 'rg32float', 'rgba32float', 'r32uint', 'rg32uint', 'rgba32uint',
    'r32sint', 'rg32sint', 'rgba32sint',
    'rg11b10ufloat', 'rgb9e5ufloat', 'rgb10a2unorm',
];
const ACCESSES = ['write-only', 'read-only', 'read-write'];

/** 在 validation error scope 內執行,回傳 [是否通過, 首行訊息] */
async function probe(fn) {
    dev.pushErrorScope('validation');
    let firstMsg = '';
    try { fn(); } catch (e) { firstMsg = String((e && e.message) || e).split('\n')[0]; }
    const err = await dev.popErrorScope();
    if (err && !firstMsg) firstMsg = String(err.message || err).split('\n')[0];
    return [!err && !firstMsg, firstMsg];
}

const results = {};
for (const fmt of FORMATS) {
    results[fmt] = {};
    for (const access of ACCESSES) {
        // 1) BGL storage 存取宣告  2) createTexture with STORAGE_BINDING usage
        const [okBgl, msgBgl] = await probe(() => {
            dev.createBindGroupLayout({
                entries: [{
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { format: fmt, access },
                }],
            });
        });
        const [okTex, msgTex] = await probe(() => {
            const t = dev.createTexture({
                size: [4, 4],
                format: fmt,
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
            });
            t?.destroy?.();
        });
        results[fmt][access] = { ok: okBgl && okTex, msg: (msgBgl || msgTex).slice(0, 110) };
    }
}

console.log('=== Dawn storage texture 支援矩陣(node-webgpu / D3D12) ===');
for (const fmt of FORMATS) {
    const cells = ACCESSES.map((a) => `${a[0]}w=${results[fmt][a].ok ? '✓' : '✗'}`).join(' ');
    const anyOk = ACCESSES.some((a) => results[fmt][a].ok);
    console.log(`  ${anyOk ? 'Y' : 'N'}  ${fmt.padEnd(16)} ${cells}${anyOk ? '' : '   ' + results[fmt]['write-only'].msg}`);
}

// core(不要求任何 feature)判斷:write-only 是否通過
console.log('\n可安全用於 storage(write-only)的格式:');
console.log('  ' + FORMATS.filter((f) => results[f]['write-only'].ok).join(', '));
dev.destroy?.();
process.exit(0);
