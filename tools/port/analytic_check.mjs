// analytic_check.mjs — uniform-input full-pipeline comparison (smoke-level).
//
// For a constant gray INPUT the CNN reduces to scalar arithmetic, so a tiny
// WGSL-statement interpreter can predict the GPU output (up to 8-bit
// intermediate quantization + f16 output truncation). Validates plumbing that
// port_check cannot see: luma path, scalar mad broadcast, matNxM orientation,
// ReLU placement, YUV corner mapping (r0.x/y/z/w per output quad), tanh head,
// store targets and the pass-split stores.
//
// usage: node tools/port/analytic_check.mjs <effect.wgsl> <family>
//   family: cunny2-yuv | cunny2-rgb | cunny-c
import { create, globals } from 'webgpu';
import fs from 'fs';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const file = process.argv[2];
const family = process.argv[3] || 'cunny2-yuv';
const SIZE = 64;
const C = 0.6;

// ---------- vec/mat helpers (WGSL semantics, verbatim flat numbers) ----------
const vadd = (a, b) => a.map((x, i) => x + b[i]);
const relu = (a) => a.map((x) => Math.max(0, x));
const tanhv = (a) => a.map(Math.tanh);
const clamp01 = (x) => Math.max(0, Math.min(1, x));
// matCxR<f32>(nums) is column-major: (M*v)[i] = Σ_j nums[j*R + i] * v[j]
function matmul(numsA, R, v) {
  if (!Array.isArray(v) || v.length < R || v.some((x) => typeof x !== 'number')) {
    throw new Error('matmul bad v: ' + JSON.stringify(v) + ' R=' + R + ' nums=' + JSON.stringify(numsA.slice(0, 4)));
  }
  const out = new Array(R).fill(0);
  for (let j = 0; j < numsA.length / R; j++)
    for (let i = 0; i < R; i++) out[i] += numsA[j * R + i] * v[j];
  return out;
}
const numsOf = (s) => s.split(',').map((x) => Number(x.trim().replace(/^\+/, '')));
const qUnorm = (x) => Math.max(0, Math.min(1, Math.round(x * 255) / 255));
const qSnorm = (x) => Math.max(-1, Math.min(1, Math.round(x * 127) / 127));

// ---------- parse ----------
const code = fs.readFileSync(file, 'utf8');
const fmtOf = {};
for (const m of code.matchAll(/var (\w+)_storaged: texture_storage_2d<(\w+), write>/g)) fmtOf[m[1]] = m[2];
const passBlocks = [];
for (const m of code.matchAll(/\/\/!PASS (\d+)([\s\S]*?)(?=\/\/!PASS \d+|$)/g)) {
  const body = m[2];
  passBlocks.push({
    index: Number(m[1]),
    out: (body.match(/\/\/!OUT (.+)/) || ['', ''])[1].split(',').map((s) => s.trim()),
    code: body,
  });
}

// ---------- evaluator ----------
// scope: { vars: Map, texVal: {logicalName: [r,g,b,a]} }
function makeEval(scope) {
  function E(exprRaw) {
    const expr = exprRaw.trim();
    let m;
    if ((m = expr.match(/textureSampleLevel\((\w+)_sampled,/))) return scope.texVal[m[1]];
    if ((m = expr.match(/textureSampleLevel\(INPUT,/))) return scope.texVal.INPUT;
    if ((m = expr.match(/^f32\(dot\(vec3<f32>\(([-\d.eE]+), ([-\d.eE]+), ([-\d.eE]+)\), textureSampleLevel\(INPUT, SP, [^)]+\)\.rgb\)\)$/))) {
      const rgb = scope.texVal.INPUT;
      return Number(m[1]) * rgb[0] + Number(m[2]) * rgb[1] + Number(m[3]) * rgb[2];
    }
    if ((m = expr.match(/^f32\(dot\(vec3<f32>\(([-\d.eE]+), ([-\d.eE]+), ([-\d.eE]+)\), textureSampleLevel\(INPUT, SP, [^)]+\)\.rgb\) \+ f32\(([-\d.eE]+)\)\)$/))) {
      const rgb = scope.texVal.INPUT;
      return Number(m[1]) * rgb[0] + Number(m[2]) * rgb[1] + Number(m[3]) * rgb[2] + Number(m[4]);
    }
    // ((matNxM<w>) * (v) + (c)) — MulAdd form
    if ((m = expr.match(/^\(\(mat(\d)x(\d)<f32>\((.*)\)\) \* \((.+)\) \+ \((.+)\)\)$/))) {
      return vadd(E(m[5]), matmul(numsOf(m[3]), Number(m[2]), E(m[4])));
    }
    if ((m = expr.match(/^\(\((.+)\) \* \((.+)\) \+ \((.+)\)\)$/))) {
      const a = E(m[1]);
      const b = E(m[2]);
      const c = E(m[3]);
      if (a && a.mat) return vadd(c, matmul(a.mat, Math.sqrt(a.mat.length), b));
      const bb = Array.isArray(b) ? b : [b, b, b, b].slice(0, Array.isArray(a) ? a.length : 4);
      return vadd(c, (Array.isArray(a) ? a : bb).map((x, i) => (Array.isArray(a) ? x : bb[i]) * (Array.isArray(a) ? bb[i] : x)) ?? null);
    }
    if ((m = expr.match(/^\(\((.+)\) \* \((.+)\)\)$/))) {
      const a = E(m[1]);
      const b = E(m[2]);
      if (a && a.mat) return matmul(a.mat, Math.sqrt(a.mat.length), b);
      if (Array.isArray(a)) return a.map((x, i) => x * (Array.isArray(b) ? b[i] : b));
      return (Array.isArray(b) ? b : [b]).map((x) => a * x);
    }
    if ((m = expr.match(/^\(\(mat(\d)x(\d)<f32>\((.*)\)\) \* \((.+)\)\)$/))) {
      return matmul(numsOf(m[3]), Number(m[2]), E(m[4]));
    }
    if ((m = expr.match(/^max\(\((.+)\), vec4<f32>\(0\.0\)\)$/))) return relu(E(m[1]));
    if ((m = expr.match(/^-max\(\((.+)\), vec4<f32>\(0\.0\)\)$/))) return relu(E(m[1])).map((x) => -x);
    // (dot(...) + f32(b)) — C-family pass-1 luma with bias (single outer paren)
    if ((m = expr.match(/^\(dot\(vec3<f32>\(([-\d.eE]+), ([-\d.eE]+), ([-\d.eE]+)\), textureSampleLevel\(INPUT, SP, .*\)\.rgb\) \+ f32\(([-\d.eE]+)\)$/))) {
      const rgb = scope.texVal.INPUT;
      return Number(m[1]) * rgb[0] + Number(m[2]) * rgb[1] + Number(m[3]) * rgb[2] + Number(m[4]);
    }
    if ((m = expr.match(/^-\((.+)\)$|^-(\w+)$/))) {
      const inner = E(m[1] || m[2]);
      return Array.isArray(inner) ? inner.map((x) => -x) : -inner;
    }
    if ((m = expr.match(/^\((.+)\)$/)) && !/mat\d/.test(expr)) return E(m[1]);
    if ((m = expr.match(/^tanh\((.+)\)$/))) return tanhv(E(m[1]));
    if ((m = expr.match(/^([A-Za-z_]\w*)\((.*)\)$/)) && scope.helpers[m[1]]) {
      return scope.helpers[m[1]](m[2].split(',').map((a) => E(a.trim())));
    }
    if ((m = expr.match(/^vec4<f32>\(([^)]+)\)$/))) {
      const xs = numsOf(m[1]);
      return xs.length === 1 ? [xs[0], xs[0], xs[0], xs[0]] : xs;
    }
    if ((m = expr.match(/^vec3<f32>\((.+)\)$/))) {
      const parts = m[1].split(',').map((p) => E(p.trim()));
      const out = [];
      for (const p of parts) Array.isArray(p) ? out.push(...p) : out.push(p);
      return out;
    }
    if ((m = expr.match(/^saturate\(yuv\.r \+ ([\w.]+)\)$/))) return clamp01(scope.vars.get('yuv')[0] + scalar(m[1]));
    if ((m = expr.match(/^yuv\.r \+ ([\w.]+)$/))) return scope.vars.get('yuv')[0] + scalar(m[1]);
    if (/^[A-Za-z_]\w*$/.test(expr)) {
      if (scope.vars.has(expr)) return scope.vars.get(expr);
      throw new Error('unknown identifier: ' + expr);
    }
    throw new Error('unhandled expr: ' + expr.slice(0, 140));
    function scalar(name) {
      const v = scope.vars.get(name);
      return Array.isArray(v) ? v[0] : v;
    }
  }
  return E;
}

// split a code block into single statements (converter packs several `var`
// declarations on one physical line)
function splitStatements(codeText) {
  const out = [];
  for (const rawLine of codeText.split('\n')) {
    const t = rawLine.trim();
    if (t.startsWith('fn ') || t.startsWith('textureStore')) { out.push(t); continue; }
    for (const piece of t.split(';')) {
      const s = piece.trim();
      if (s) out.push(s);
    }
  }
  return out;
}

function evalPass(pb, texVal, helpers) {
  const scope = { vars: new Map(), texVal, helpers };
  const E = makeEval(scope);
  const stores = [];
  // helper fn bodies must not be executed at pass level
  const codeNoHelpers = pb.code.replace(/fn [A-Za-z_]\w+\([^)]*\) -> vec4<f32> \{[\s\S]*?\n\}/g, '');
  for (const stmt of splitStatements(codeNoHelpers)) {
    const l = stmt;
    if (!l || /^(\/\/|if\b|let\b|fn\b|return\b|\})/.test(l)) continue;
    if (/^(let (pt|sz|size|opt|fpos|blockStart)|var (gxy|pos) =)/.test(l)) continue; // structural header lines
    let m;
    if ((m = l.match(/^var ([\w]+): (f32|vec[34]<f32>);?$/))) { scope.vars.set(m[1], undefined); continue; }
    if ((m = l.match(/^var ([\w]+) = (.+);?$/))) { scope.vars.set(m[1], E(m[2])); continue; }
    if ((m = l.match(/^([\w]+) = (.+);?$/)) && !l.startsWith('textureStore')) {
      const val = E(m[2]);
      if (val === undefined) throw new Error(`PASS stmt left ${m[1]} undefined: ${l.slice(0, 120)}`);
      scope.vars.set(m[1], val);
      continue;
    }
    if ((m = l.match(/^textureStore\((\w+)_storaged, [^,]+, (.+)\);?$/))) {
      const fmt = fmtOf[m[1]];
      const val = E(m[2]);
      if (process.env.DBG) console.log(`  [dbg] PASS${pb.index} store ${m[1]} <-`, JSON.stringify(val).slice(0, 80), '| texVal keys:', Object.keys(texVal).join(','));
      texVal[m[1]] = val.map((x) => (fmt === 'rgba8unorm' ? qUnorm(x) : qSnorm(x)));
      stores.push(m[1]);
      continue;
    }
  }
  return stores;
}

// parse fn helpers per pass block (they never sample; pure math). f0/f1 are
// redefined per pass with different signatures → scope them per block.
function parseHelpers(pb, texVal) {
  const helpers = {};
  for (const m of pb.code.matchAll(/fn ([A-Za-z_]\w+)\(([^)]*)\) -> vec4<f32> \{([\s\S]*?)\n\}/g)) {
    const name = m[1];
    const params = m[2].split(',').map((s) => s.trim().split(':')[0].trim());
    const body = splitStatements(m[3]);
    helpers[name] = (args) => {
      const scope = { vars: new Map(params.map((p, i) => [p, args[i]])), texVal, helpers };
      const E = makeEval(scope);
      for (const l of body) {
        let mm;
        if ((mm = l.match(/^var ([\w]+) = (.+);?$/))) scope.vars.set(mm[1], E(mm[2]));
        else if ((mm = l.match(/^([\w]+) = (.+);?$/))) scope.vars.set(mm[1], E(mm[2]));
        else if ((mm = l.match(/^return (.+);?$/))) return E(mm[1]);
      }
      throw new Error('helper ' + name + ' returned nothing');
    };
  }
  return helpers;
}

// ---------- run conv passes on the CPU ----------
const texVal = { INPUT: [C, C, C, 1] };
const finalBlock = passBlocks.find((p) => p.out[0] === 'OUTPUT');
for (const pb of passBlocks) {
  if (pb === finalBlock) continue;
  evalPass(pb, texVal, parseHelpers(pb, texVal));
}

// ---------- final pass expectation ----------
const finalHelpers = parseHelpers(finalBlock, texVal);
const finalVars = { texVal, helpers: finalHelpers, vars: new Map() };
const FE = makeEval(finalVars);
{
  const scope = finalVars;
  const E = FE;
  const codeNoHelpers = finalBlock.code.replace(/fn [A-Za-z_]\w+\([^)]*\) -> vec4<f32> \{[\s\S]*?\n\}/g, '');
  for (const stmt of splitStatements(codeNoHelpers)) {
    const l = stmt;
    if (!l || /^(\/\/|if\b|fn\b|return\b|\}|let (pt|sz|size|opt|fpos|blockStart)|var (gxy|pos)(\.\w)? =|(pos|gxy)(\.\w)? = )/.test(l)) continue;
    let m;
    if ((m = l.match(/^let ([\w]+) = mat3x3<f32>\((.*)\);?$/))) scope.vars.set(m[1], { mat: numsOf(m[2]) });
    else if ((m = l.match(/^var ([\w]+): (f32|vec[34]<f32>);?$/))) scope.vars.set(m[1], undefined);
    else if ((m = l.match(/^var ([\w]+) = (.+);?$/))) scope.vars.set(m[1], E(m[2]));
    else if ((m = l.match(/^([\w]+) = (.+);?$/)) && !l.startsWith('textureStore')) {
      const val = E(m[2]);
      if (val === undefined) throw new Error(`FINAL stmt left ${m[1]} undefined: ${l.slice(0, 120)}`);
      scope.vars.set(m[1], val);
    }
    else if (l.startsWith('textureStore')) {
      // OUTPUT stores are reconstructed analytically below
    }
  }
}

// The 4 output corners. YUV families: out = yuv2rgb * (sat(Y + rc), yuv.yz);
// rgb family: out = sat(INPUT.rgb + (r0[i], r1[i], r2[i])).
const rFinal = finalVars.vars.get('r') || finalVars.vars.get('r0');
if (!rFinal) throw new Error('final r vector not found');
let expected;
if (family === 'cunny2-rgb') {
  const r0 = finalVars.vars.get('r0'), r1 = finalVars.vars.get('r1'), r2 = finalVars.vars.get('r2');
  if (!r0 || !r1 || !r2) throw new Error('rgb family needs r0..r2');
  const sat = (v) => v.map(clamp01);
  expected = {
    '0,0': sat([C + r0[0], C + r1[0], C + r2[0]]),
    '1,0': sat([C + r0[1], C + r1[1], C + r2[1]]),
    '0,1': sat([C + r0[2], C + r1[2], C + r2[2]]),
    '1,1': sat([C + r0[3], C + r1[3], C + r2[3]]),
  };
} else {
  const matOf = {};
  for (const [k, v] of finalVars.vars) if (v && v.mat) matOf[k] = v.mat;
  const rgb2yuvM = matOf.RY || matOf.rgb2yuv;
  const yuv2rgbM = matOf.YR || matOf.yuv2rgb;
  const yuv = matmul(rgb2yuvM, 3, [C, C, C]);
  const apply = (Y) => matmul(yuv2rgbM, 3, [clamp01(Y), yuv[1], yuv[2]]);
  if (family === 'cunny-c') {
    // C-family walk: (0,0)←r.x, (1,0)←r.y, (1,1)←r.w, (0,1)←r.z
    expected = {
      '0,0': apply(clamp01(yuv[0] + rFinal[0])),
      '1,0': apply(clamp01(yuv[0] + rFinal[1])),
      '1,1': apply(clamp01(yuv[0] + rFinal[3])),
      '0,1': apply(clamp01(yuv[0] + rFinal[2])),
    };
  } else {
    // CuNNy2-family: (0,0)←r0.x, (1,0)←r0.y, (0,1)←r0.z, (1,1)←r0.w
    expected = {
      '0,0': apply(clamp01(yuv[0] + rFinal[0])),
      '1,0': apply(clamp01(yuv[0] + rFinal[1])),
      '0,1': apply(clamp01(yuv[0] + rFinal[2])),
      '1,1': apply(clamp01(yuv[0] + rFinal[3])),
    };
  }
}

// ---------- GPU ----------
const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();
const runtime = new WGFXRuntime(dev);
await runtime.compile(code, {
  defines: { INPUT_WIDTH: SIZE, INPUT_HEIGHT: SIZE },
  textures: {
    INPUT: { size: [SIZE, SIZE], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
    OUTPUT: { size: [SIZE, SIZE], format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC },
  },
});
const rm = runtime.resourceManager;
{
  const data = new Uint8Array(SIZE * SIZE * 4).fill(Math.round(C * 255));
  dev.queue.writeTexture({ texture: rm.getTexture('INPUT') }, data, { bytesPerRow: SIZE * 4, rowsPerImage: SIZE }, [SIZE, SIZE, 1]);
}
rm.updateSceneBuffer(SIZE, SIZE, SIZE, SIZE);
const cmd = dev.createCommandEncoder();
for (const p of runtime.shaderInfo.passes) runtime.dispatchPass(`PASS_${p.index}`, cmd);
dev.queue.submit([cmd.finish()]);

function f16(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
const bytesPerRow = Math.ceil((SIZE * 8) / 256) * 256;
const staging = dev.createBuffer({ size: bytesPerRow * SIZE, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
{
  const enc = dev.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: rm.getTexture('OUTPUT'), origin: [0, 0, 0] },
    { buffer: staging, bytesPerRow, rowsPerImage: SIZE }, [SIZE, SIZE, 1]);
  dev.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
}
const raw = new Uint16Array(staging.getMappedRange().slice(0));
staging.unmap();

// ---------- compare the (0,0) 2x2 quad ----------
const rowStride = bytesPerRow / 2;
let maxDiff = 0;
for (const [key, exp] of Object.entries(expected)) {
  const [bx, by] = key.split(',').map(Number);
  const i = by * rowStride + bx * 4;
  const got = [f16(raw[i]), f16(raw[i + 1]), f16(raw[i + 2])];
  for (let c = 0; c < 3; c++) maxDiff = Math.max(maxDiff, Math.abs(got[c] - exp[c]));
}
const TOL = 0.02;
console.log(`${file} [${family}]: max|GPU−CPU| = ${maxDiff.toFixed(5)} → ${maxDiff <= TOL ? 'MATCH' : 'MISMATCH'}`);
if (maxDiff > TOL) {
  for (const [k, v] of Object.entries(expected)) console.log(`  cpu corner ${k}: [${v.map((x) => x.toFixed(4)).join(', ')}]`);
  console.log(`  gpu corner 0,0: [${f16(raw[0]).toFixed(4)}, ${f16(raw[1]).toFixed(4)}, ${f16(raw[2]).toFixed(4)}]`);
  console.log(`  gpu corner 1,0: [${f16(raw[4]).toFixed(4)}, ${f16(raw[5]).toFixed(4)}, ${f16(raw[6]).toFixed(4)}]`);
  dev.destroy?.();
  process.exit(1);
}
dev.destroy?.();
