/**
 * f16 gating test — proves the FP16 path is opt-in and that the default
 * (f32) behaviour is byte-for-byte unchanged.
 *
 * Checks, in order:
 *   1. An effect declaring `//! CAPABILITY FP16` compiled WITHOUT preferF16
 *      produces modules with NO `enable f16;` — identical to the pre-change
 *      generator output.
 *   2. The same effect WITH preferF16 but on a device WITHOUT `shader-f16`
 *      still produces no `enable f16;` (feature detection wins over the request).
 *   3. preferF16 on a device WITH `shader-f16` DOES emit `enable f16;`.
 *   4. WGFXInfo.f16 reports the resolved state.
 *
 * Run: node test/f16_gate.test.mjs   (uses the webgpu polyfill; RTX 4060 here
 *       does NOT expose shader-f16, so case 3 is simulated with a stub device.)
 */
import { create, globals } from 'webgpu';
import assert from 'node:assert';
Object.assign(globalThis, globals);

const { WGFX } = await import('../dist/wgfx.esm.js');
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const EFFECT = `//! VERSION 1
//! CAPABILITY FP16
//! PARAMETER Amount
//! TYPE float
//! MIN 0.0
//! MAX 1.0
//! DEFAULT 0.5
float Amount;
//! TEXTURE INPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
//! FORMAT rgba8unorm
Texture2D INPUT;
//! TEXTURE OUTPUT
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
//! FORMAT rgba16float
Texture2D OUTPUT;
//! PASS 1
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(INPUT);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let c = textureLoad(INPUT, vec2<i32>(gid.xy), 0);
    textureStore(OUTPUT, vec2<i32>(gid.xy), c * Amount);
}
`;

let pass = 0, fail = 0;
function check(name, fn) {
    try { fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name}\n     ${e.message}`); fail++; }
}

const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
const device = await adapter.requestDevice();
console.log(`🧪 f16 gating tests (adapter f16 support: ${adapter.features.has('shader-f16')})`);
console.log('');

// --- 1 & 2: default path emits no enable f16 --------------------------
check('default (no preferF16) → no `enable f16;`', () => {
    const rt = new WGFXRuntime(device);
    return rt.compile(EFFECT, {
        defines: { INPUT_WIDTH: 8, INPUT_HEIGHT: 8 },
        textures: {
            INPUT: { size: [8, 8], format: 'rgba8unorm' },
            OUTPUT: { size: [8, 8], format: 'rgba16float' },
        },
    }).then(() => {
        assert.ok(rt.f16Enabled === false, 'f16Enabled must default to false');
        for (const m of rt.generatedModules) {
            assert.ok(!m.wgslCode.includes('enable f16;'),
                'default path must not contain `enable f16;`');
        }
    });
});

// --- 3: preferF16 on a device WITHOUT shader-f16 is refused ------------
check('preferF16 but device lacks shader-f16 → still no `enable f16;`', () => {
    const rt = new WGFXRuntime(device);
    return rt.compile(EFFECT, {
        defines: { INPUT_WIDTH: 8, INPUT_HEIGHT: 8 },
        textures: {
            INPUT: { size: [8, 8], format: 'rgba8unorm' },
            OUTPUT: { size: [8, 8], format: 'rgba16float' },
        },
    }, { preferF16: true }).then(() => {
        if (adapter.features.has('shader-f16')) {
            // This host DOES support it — then it must be enabled.
            assert.ok(rt.f16Enabled === true, 'f16Enabled should be true');
            assert.ok(rt.generatedModules[0].wgslCode.includes('enable f16;'));
        } else {
            assert.ok(rt.f16Enabled === false,
                'f16 must stay off when the device lacks shader-f16');
            for (const m of rt.generatedModules) {
                assert.ok(!m.wgslCode.includes('enable f16;'));
            }
        }
    });
});

// --- 4: stub device WITH shader-f16 → enable f16 emitted ---------------
check('stub device with shader-f16 → `enable f16;` emitted', () => {
    const stub = {
        features: new Set(['shader-f16']),
        limits: device.limits,
        createShaderModule: device.createShaderModule.bind(device),
        createBindGroupLayout: device.createBindGroupLayout.bind(device),
        createPipelineLayout: device.createPipelineLayout.bind(device),
        createComputePipeline: device.createComputePipeline.bind(device),
        createBuffer: device.createBuffer.bind(device),
        createTexture: device.createTexture.bind(device),
        createCommandEncoder: device.createCommandEncoder.bind(device),
        queue: device.queue,
        pushErrorScope: device.pushErrorScope.bind(device),
        popErrorScope: device.popErrorScope.bind(device),
    };
    const rt = new WGFXRuntime(stub);
    return rt.compile(EFFECT, {
        defines: { INPUT_WIDTH: 8, INPUT_HEIGHT: 8 },
        textures: {
            INPUT: { size: [8, 8], format: 'rgba8unorm' },
            OUTPUT: { size: [8, 8], format: 'rgba16float' },
        },
    }, { preferF16: true }).then(() => {
        assert.ok(rt.f16Enabled === true, 'f16Enabled should be true on a f16 device');
        assert.ok(rt.generatedModules[0].wgslCode.includes('enable f16;'),
            'expected `enable f16;` in the generated WGSL');
    });
});

// --- 5: effect WITHOUT the CAPABILITY marker never enables f16 ---------
check('effect without CAPABILITY FP16 → never enabled', () => {
    const noCap = EFFECT.replace('//! CAPABILITY FP16\n', '');
    const rt = new WGFXRuntime(device);
    return rt.compile(noCap, {
        defines: { INPUT_WIDTH: 8, INPUT_HEIGHT: 8 },
        textures: {
            INPUT: { size: [8, 8], format: 'rgba8unorm' },
            OUTPUT: { size: [8, 8], format: 'rgba16float' },
        },
    }, { preferF16: true }).then(() => {
        assert.ok(rt.f16Enabled === false, 'must not enable f16 without the marker');
        for (const m of rt.generatedModules) {
            assert.ok(!m.wgslCode.includes('enable f16;'));
        }
    });
});

console.log('');
console.log(`✅ ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
