/**
 * make_smem.mjs — O2 prototype: transform a CuNNy-*-NVL WGSL effect's
 * 3×3-tap passes into workgroup shared-memory tiled loads.
 *
 * The generated Magpie/CuNNy pass bodies are extremely regular:
 *     var s0_K = vec4<f32>(textureSampleLevel(T, SP, pos + (vec2<f32>(dx, dy)) * pt, 0.0));
 * with (dx,dy) ∈ {-1,0,1}² in a FIXED order (dy=-1,0,1 × dx=-1,0,1) and
 * `pos = (vec2<f32>(gxy) + 0.5) * pt`, `gxy = Rmp8x8(tid.x) + blockStart`,
 * BLOCK_SIZE 8, NUM_THREADS 64.
 *
 * Because every tap is an exact texel (integer × pt), sampling is equivalent
 * to textureLoad at clamped integer coordinates — which lets us cooperatively
 * load the 10×10 tile once per workgroup and read taps from shared memory:
 * 64 threads × 9 taps = 576 texture reads → 100 (5.76× fewer).
 *
 * The transform:
 *   1. Inject `var<workgroup> tile: array<vec4<f32>, (BS+2)²>` per pass.
 *   2. Replace the early-return prologue with: cooperative load → barrier → return.
 *   3. Replace each `textureSampleLevel(...)` tap with `tile[idx − Δ]` lookups.
 * Passes that don't match the pattern (the out-shuffle last pass) are untouched.
 *
 * Usage: node .tmp/smem_proto/make_smem.mjs <in.wgsl> <out.wgsl>
 */
import fs from 'node:fs';

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
    console.error('usage: node make_smem.mjs <in.wgsl> <out.wgsl>');
    process.exit(2);
}

let src = fs.readFileSync(inPath, 'utf8');
const passes = src.split(/(?=\/\/!PASS \d+)/); // keep prologue as passes[0]

const TAP = [ // (dx, dy) in the generator's fixed emission order
    [-1, -1], [0, -1], [1, -1],
    [-1, 0], [0, 0], [1, 0],
    [-1, 1], [0, 1], [1, 1],
];

let transformed = 0;
for (let pi = 1; pi < passes.length; pi++) {
    const p = passes[pi];
    const bs = /\/\/!BLOCK_SIZE (\d+)/.exec(p);
    const nt = /\/\/!NUM_THREADS (\d+)/.exec(p);
    if (!bs || bs[1] !== '8' || !nt || nt[1] !== '64') continue; // only 8×8/64 passes
    if (!/fn Pass\d+\(/.test(p)) continue;

    // Gate: exactly 9 generic texel taps, fixed order, one texture.
    const genericTapRe = /textureSampleLevel\((\w+), SP, pos \+ \(vec2<f32>\(\((-?\d+)\.0\), \((-?\d+)\.0\)\)\) \* pt, 0\.0\)/g;
    const taps = [...p.matchAll(genericTapRe)];
    if (taps.length !== 9) continue; // only the canonical 3×3 pattern
    const texName = taps[0][1];
    let ok = true;
    for (let k = 0; k < 9; k++) {
        if (taps[k][1] !== texName ||
            Number(taps[k][2]) !== TAP[k][0] || Number(taps[k][3]) !== TAP[k][1]) { ok = false; break; }
    }
    if (!ok) continue;

    // 1. inject the workgroup array right before `fn Pass`
    const decl = `// O2 shared-memory tile: (8+2)×(8+2) texels loaded cooperatively once.\n` +
        `var<workgroup> smem_tile: array<vec4<f32>, 100>;\n\n`;
    const p1 = p.replace(/(fn Pass\d+\()/, `${decl}$1`);

    // 2. replace the prologue: load loop → barrier → then bounds check
    const prologueRe = /fn (Pass\d+)\(@builtin\(workgroup_id\) workgroup_id: vec3<u32>, @builtin\(local_invocation_id\) tid: vec3<u32>\) \{\s*\n\s*let blockStart = workgroup_id\.xy \* vec2<u32>\(8u, 8u\);\s*\n\s*let pt = GetInputPt\(\);\s*\n\s*var gxy = Rmp8x8\(tid\.x\) \+ blockStart;\s*\n\s*let size = GetInputSize\(\);\s*\n\s*if \(gxy\.x >= size\.x \|\| gxy\.y >= size\.y\) \{\s*\n\s*return;\s*\n\s*\}\s*\n\s*var pos = \(vec2<f32>\(gxy\) \+ vec2<f32>\(0\.5\)\) \* pt;\s*\n/;
    const m = prologueRe.exec(p1);
    if (!m) continue;
    const newPrologue =
        `fn ${m[1]}(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) tid: vec3<u32>) {\n` +
        `    let blockStart = workgroup_id.xy * vec2<u32>(8u, 8u);\n` +
        `    let size = GetInputSize();\n` +
        `    let tileOrg = vec2<i32>(blockStart) - vec2<i32>(1, 1);\n` +
        `    let maxXY = vec2<i32>(size) - vec2<i32>(1, 1);\n` +
        `    for (var k = tid.x; k < 100u; k += 64u) {\n` +
        `        let tx = i32(k % 10u); let ty = i32(k / 10u);\n` +
        `        let g = clamp(tileOrg + vec2<i32>(tx, ty), vec2<i32>(0, 0), maxXY);\n` +
        `        smem_tile[k] = vec4<f32>(textureLoad(${texName}, g, 0));\n` +
        `    }\n` +
        `    workgroupBarrier();\n` +
        `    let gxy = Rmp8x8(tid.x) + blockStart;\n` +
        `    if (gxy.x >= size.x || gxy.y >= size.y) {\n` +
        `        return;\n` +
        `    }\n`;

    // 3. replace ALL texel taps with shared-memory reads (idx = ly*10+lx).
    //    Handles both the standalone `var s0_K = (vec4<f32>)?(textureSampleLevel…)` form
    //    and the inline `dot(…, textureSampleLevel(INPUT, …).rgb …)` form — the call
    //    itself has the same shape everywhere (full-texel offset × pt).
    let p2 = p1.replace(prologueRe, newPrologue);
    p2 = p2.replace(/textureSampleLevel\(\w+, SP, pos \+ \(vec2<f32>\(\((-?\d+)\.0\), \((-?\d+)\.0\)\)\) \* pt, 0\.0\)/g,
        (full, dx, dy) => {
            const delta = (Number(dy) + 1) * 10 + (Number(dx) + 1) - 11;
            return `smem_tile[idx + ${delta}]`;
        });
    // insert the idx computation right after the bounds check
    p2 = p2.replace(/(if \(gxy\.x >= size\.x \|\| gxy\.y >= size\.y\) \{\s*\n\s*return;\s*\n\s*\})/,
        `$1\n    let idx = (i32(gxy.y) - tileOrg.y) * 10 + (i32(gxy.x) - tileOrg.x);`);

    passes[pi] = p2;
    transformed++;
}

fs.writeFileSync(outPath, passes.join(''));
console.log(`transformed ${transformed} pass(es) → ${outPath}`);
