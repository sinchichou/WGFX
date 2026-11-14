/**
 * bench_output_format — measure the OUTPUT-format bandwidth effect.
 *
 * Bilinear 512x512 → 1024x1024 (2x upscale; the last pass writes the OUTPUT
 * texture). Compares rgba16float (8 B/px) vs rgba8unorm (4 B/px) per-frame
 * GPU wall time (submit + onSubmittedWorkDone), N rounds × M frames.
 *
 * Run: node test/bench_output_format.mjs
 */
import { create, globals } from 'webgpu';
import { readFile } from 'node:fs/promises';
Object.assign(globalThis, globals);

const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
const device = await adapter.requestDevice();
const EFFECT = await readFile(new URL('../effects/Scalers/Bilinear.wgsl', import.meta.url), 'utf8');

const W = 512, H = 512, ROUNDS = 5, FRAMES = 30;
const pixels = new Uint8Array(W * H * 4);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4;
    pixels[i] = (x * 255 / W) | 0; pixels[i + 1] = (y * 255 / H) | 0; pixels[i + 3] = 255;
}

async function bench(format) {
    const rt = new WGFXRuntime(device);
    await rt.compile(EFFECT, {
        defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
        textures: {
            INPUT: { size: [W, H], format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
            OUTPUT: { size: [W * 2, H * 2], format,
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.TEXTURE_BINDING },
        },
    }, { honorDeclaredOutput: false, outputFormat: format });
    device.queue.writeTexture({ texture: rt.resourceManager.getTexture('INPUT') }, pixels,
        { bytesPerRow: W * 4, rowsPerImage: H }, [W, H]);
    rt.resourceManager.updateSceneBuffer(W, H, W * 2, H * 2);

    const frame = () => {
        const e = device.createCommandEncoder();
        rt.dispatchAllPasses(e);
        device.queue.submit([e.finish()]);
        return device.queue.onSubmittedWorkDone();
    };
    for (let i = 0; i < 10; i++) await frame(); // warmup
    const times = [];
    for (let r = 0; r < ROUNDS; r++) {
        const t0 = performance.now();
        for (let i = 0; i < FRAMES; i++) await frame();
        times.push((performance.now() - t0) / FRAMES);
    }
    times.sort((a, b) => a - b);
    const med = times[Math.floor(times.length / 2)];
    console.log(`${format.padEnd(12)}  med ${med.toFixed(3)} ms/frame  min ${times[0].toFixed(3)}  FPS(med) ${(1000 / med).toFixed(0)}`);
    return med;
}

console.log(`=== OUTPUT format bench — Bilinear ${W}x${H} → ${W * 2}x${H * 2}, ${ROUNDS}x${FRAMES} frames ===`);
const f = await bench('rgba16float');
const u = await bench('rgba8unorm');
console.log(`\noutput write: ${W * 2}x${H * 2} px/frame`);
console.log(`  rgba16float ${(W * 2) * (H * 2) * 8 / 1e6 | 0} MB/frame  vs  rgba8unorm ${(W * 2) * (H * 2) * 4 / 1e6 | 0} MB/frame`);
console.log(`measured delta: ${(f - u).toFixed(3)} ms/frame  (${(u < f ? '8unorm faster' : '16f faster')})`);
