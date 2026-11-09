import { create, globals } from 'webgpu';
import fs from 'fs';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');
const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();
const file = process.argv[2];
const code = fs.readFileSync(file, 'utf8');
const runtime = new WGFXRuntime(dev);
await runtime.compile(code, {
    defines: { INPUT_WIDTH: 64, INPUT_HEIGHT: 64 },
    textures: {
        INPUT: { size: [64, 64], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
        OUTPUT: { size: [64, 64], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC },
    },
});
const mod = runtime.generatedModules[0].wgslCode;
const lines = mod.split('\n');
const from = Number(process.argv[3] || 340), to = Number(process.argv[4] || 360);
for (let i = from; i <= to && i <= lines.length; i++) {
    console.log(String(i).padStart(4) + ': ' + lines[i - 1]);
}
