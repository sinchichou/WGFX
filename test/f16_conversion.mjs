// f16_conversion.mjs — 記錄並驗證本執行時的 f32 → f16 轉換語意
//
// 用途：移植 Magpie HLSL 效果時，CPU 參考需要知道 GPU 把 f32 寫入 rgba16float
// 紋理時的捨入行為。本測試證實：**Dawn（node-webgpu）向零截斷 mantissa**，
// 而非 IEEE 的 round-to-nearest-even。
//
// 這是 WGFX 移植的**基礎前提**（見 docs/porting-guide.md §6.1）：
//   * CPU 參考若不模擬截斷，會出現最大約 1 個 f16 ulp 的系統性偏差；
//   * 通用容差需 ≥ 1e-3（1.0 附近的 1 ulp ≈ 9.8e-4）。
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();

let fail = 0;
const check = (name, ok, detail = '') => {
    console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${name}${detail ? ' — ' + detail : ''}`);
    if (!ok) fail++;
};

// 刻意挑選 mantissa 小數部分 > 0.5 的值：round 與 trunc 的結果必然不同
const consts = [
    0.1, 26 / 255, 0.15, 223 / 255, 0.77777777777, 0.33333333333,
    0.65000000001, 0.40000000001, 0.15000000001,
];
const N = consts.length;

const src = new Float32Array(N * 4);
consts.forEach((v, i) => { src[i * 4] = v; src[i * 4 + 3] = 1; });

const inTex = dev.createTexture({
    size: [N, 1], format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
dev.queue.writeTexture({ texture: inTex }, src, { bytesPerRow: N * 16, rowsPerImage: 1 }, [N, 1, 1]);
const outTex = dev.createTexture({
    size: [N, 1], format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
});

const mod = dev.createShaderModule({
    code: `
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var o: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(${N}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  textureStore(o, vec2<i32>(g.xy), textureLoad(t, vec2<i32>(g.xy), 0));
}`,
});
const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
const bg = dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: inTex.createView() },
        { binding: 1, resource: outTex.createView() },
    ],
});
const enc = dev.createCommandEncoder();
const p = enc.beginComputePass();
p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(1); p.end();
dev.queue.submit([enc.finish()]);

// bytesPerRow 必須是 256 的倍數（WebGPU 規範）
const bytesPerRow = Math.ceil((N * 8) / 256) * 256;
const staging = dev.createBuffer({ size: bytesPerRow, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const e2 = dev.createCommandEncoder();
e2.copyTextureToBuffer({ texture: outTex }, { buffer: staging, bytesPerRow, rowsPerImage: 1 }, [N, 1, 1]);
dev.queue.submit([e2.finish()]);
await staging.mapAsync(GPUMapMode.READ);
const raw = new Uint16Array(staging.getMappedRange().slice(0));
staging.unmap(); staging.destroy();

/** f32 → f16 位元：trunc（截斷）與 round（四捨五入）兩種 */
function f16bits(v) {
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    f32[0] = v;
    const bits = u32[0];
    const s = (bits >>> 16) & 0x8000;
    const e = (bits >>> 23) & 0xff;
    const m = bits & 0x7fffff;
    if (e === 0xff) return { round: s | 0x7c00 | (m ? 0x200 : 0), trunc: s | 0x7c00 | (m ? 0x200 : 0) };
    const ex = e - 127 + 15;
    if (ex >= 31) return { round: s | 0x7c00, trunc: s | 0x7c00 };
    if (ex <= 0) {
        const full = m | 0x800000;
        const sh = 14 - ex;
        if (sh > 24) return { round: s, trunc: s };
        return { round: s | (full >> sh), trunc: s | (full >> sh) };
    }
    let mr = m >> 13;
    const rem = m & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (mr & 1))) mr++;
    return { round: s | (ex << 10) | mr, trunc: s | (ex << 10) | (m >> 13) };
}

let truncOk = 0, roundOk = 0, diffCases = 0;
for (let i = 0; i < N; i++) {
    const got = raw[i * 4];
    const { round, trunc } = f16bits(consts[i]);
    if (got === trunc) truncOk++;
    if (got === round) roundOk++;
    if (round !== trunc) diffCases++;
    console.log(`    v=${consts[i].toFixed(10).padStart(14)}  got=0x${got.toString(16).padStart(4, '0')}` +
        `  round=0x${round.toString(16).padStart(4, '0')}  trunc=0x${trunc.toString(16).padStart(4, '0')}`);
}
check('f32→f16 一律符合「截斷」模型', truncOk === N, `${truncOk}/${N}`);
check('截斷 ≠ 四捨五入（案例確有區別，模型非巧合）', diffCases >= 4, `可區分案例=${diffCases}/${N}`);

console.log(`\n=== 總結: ${fail === 0 ? 'ALL PASS' : fail + ' FAILED'} ===`);
dev.destroy?.();
process.exit(fail === 0 ? 0 : 1);
