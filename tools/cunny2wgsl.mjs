// cunny2wgsl.mjs — Magpie CuNNy / CuNNy2 HLSL → WGFX WGSL converter (task-4)
//
// Usage:
//   node tools/cunny2wgsl.mjs <in.hlsl> <out.wgsl>
//   node tools/cunny2wgsl.mjs --all          (convert every effects_src file)
//
// Conversion strategy (follows effects/FSRCNNX/FSRCNNX.wgsl and
// effects/Anime4K/Anime4K_Upscale_GAN_*.wgsl precedents):
//   - CS passes (blockStart/tid) → compute entry points; blockStart injected from
//     BLOCK_SIZE * workgroup_id.xy; tid = local_invocation_id.
//   - Function-like macros (O / L0..L7 / l0..l7) expanded inline at call sites
//     (they reference pass-locals pos/pt, so inlining keeps semantics verbatim).
//   - mad(a, V4(w), acc)         → acc = a * vec4<f32>(w) + acc   (scalar or vec4 a)
//     MulAdd(v, M4(w), acc)      → acc = mat4x4<f32>(w) * v + acc (weights VERBATIM;
//                                  HLSL row-major / WGSL column-major cancel, see
//                                  porting-guide §2.1 and the Anime4K GAN notes)
//     MulAdd(v, M3x4(w), acc)    → acc = mat3x4<f32>(w) * v + acc
//     mul(RY, x) on MF3x3 consts → mat3x3 built with columns = HLSL rows, M * x
//   - Weights stay INLINE as literals (FSRCNNX pattern); no buffers.
//   - Intermediates keep their Magpie format (R8G8B8A8_UNORM/SNORM → rgba8unorm /
//     rgba8snorm) as _sampled/_storaged alias pairs; explicit //!WIDTH/HEIGHT added
//     (source relies on Magpie's implicit input-size default).
//   - //!USE MulAdd / //!CAPABILITY FP16 kept as markers; arithmetic in f32 (the
//     route used by the passed FSRCNNX / Anime4K ports).
import fs from 'fs';
import path from 'path';

const EFFECTS_SRC = path.resolve(import.meta.dirname, '..', 'effects_src');
const EFFECTS = path.resolve(import.meta.dirname, '..', 'effects');

// ----------------------------------------------------------------------------
// Small parsing utilities
// ----------------------------------------------------------------------------

/** Find the index of the ')' matching the '(' at openIdx. */
function matchParen(s, openIdx) {
  let depth = 0;
  for (let i = openIdx; i < s.length; i++) {
    const c = s[i];
    if (c === '(') depth++;
    else if (c === ')') { depth--; if (depth === 0) return i; }
  }
  return -1;
}

/** Split a comma-separated argument list (top level only). */
function splitArgs(s) {
  const args = [];
  let depth = 0, cur = '';
  for (const c of s) {
    if (c === '(' || c === '[') depth++;
    else if (c === ')' || c === ']') depth--;
    if (c === ',' && depth === 0) { args.push(cur.trim()); cur = ''; }
    else cur += c;
  }
  if (cur.trim().length) args.push(cur.trim());
  return args;
}

const isPureNumber = (s) => /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(s);

// ----------------------------------------------------------------------------
// Pass 1: expand function-like macros (O, L0..L7, l0..l7) inline
// ----------------------------------------------------------------------------

function collectMacros(code) {
  const macros = new Map(); // name -> { params: [..], body }
  const re = /^#define\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s+(.+)$/gm;
  let m;
  while ((m = re.exec(code)) !== null) {
    macros.set(m[1], {
      params: m[2].split(',').map((s) => s.trim()).filter(Boolean),
      body: m[3].trim(),
    });
  }
  return macros;
}

/** Wrap a macro argument unless it is a bare identifier/number (keeps
 *  `INPUT.SampleLevel` receiver parsing intact after expansion). */
function wrapArg(a) {
  let t = (a || '').trim();
  if (!t) return t;
  // avoid double-wrapping when a param was substituted into another macro call
  const redundant = t.match(/^\(([^()]*)\)$/);
  if (redundant) t = redundant[1];
  if (/^[A-Za-z_][\w.]*$/.test(t)) return t;      // identifier / plain number
  return '(' + t + ')';
}

function substituteParams(body, params, args) {
  let out = body;
  for (let i = 0; i < params.length; i++) {
    out = out.replace(new RegExp('\\b' + params[i] + '\\b', 'g'), wrapArg(args[i]));
  }
  return out;
}

function expandFunctionMacros(code, macros) {
  const names = [...macros.keys()];
  if (!names.length) return code;
  const nameAlt = names.map(escapeRe).join('|');
  const re = new RegExp('\\b(' + nameAlt + ')\\s*\\(', 'g');
  let guard = 0;
  let src = code;
  for (;;) {
    if (++guard > 200) throw new Error('macro expansion did not converge');
    re.lastIndex = 0;
    let changed = false;
    let out = '';
    let pos = 0;
    let m;
    while ((m = re.exec(src)) !== null) {
      const name = m[1];
      const openIdx = re.lastIndex - 1;
      const closeIdx = matchParen(src, openIdx);
      if (closeIdx < 0) throw new Error('unbalanced parens in macro call: ' + name);
      const args = splitArgs(src.slice(openIdx + 1, closeIdx));
      const { params, body } = macros.get(name);
      const replacement = substituteParams(body, params, args);
      out += src.slice(pos, m.index) + replacement;
      pos = closeIdx + 1;
      re.lastIndex = pos;
      changed = true;
    }
    out += src.slice(pos);
    src = out;
    if (!changed) break;
  }
  return src;
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// ----------------------------------------------------------------------------
// Pass 2: line-based statement conversion
// ----------------------------------------------------------------------------

const TEXTURE_ZERO = {
  V4: ['var', 'vec4<f32>'],
  V3: ['var', 'vec3<f32>'],
  MF: ['var', 'f32'],
  MF2: ['var', 'vec2<f32>'],
  MF3: ['var', 'vec3<f32>'],
  MF4: ['var', 'vec4<f32>'],
};

const ALIAS_CALLS = [
  [/MF4x4\s*\(/g, 'mat4x4<f32>('],
  [/MF3x4\s*\(/g, 'mat3x4<f32>('],
  [/MF4x3\s*\(/g, 'mat4x3<f32>('],
  [/MF4\s*\(/g, 'vec4<f32>('],
  [/MF3\s*\(/g, 'vec3<f32>('],
  [/MF2\s*\(/g, 'vec2<f32>('],
  [/MF\s*\(/g, 'f32('],
  [/V4\s*\(/g, 'vec4<f32>('],
  [/V3\s*\(/g, 'vec3<f32>('],
  [/M4\s*\(/g, 'mat4x4<f32>('],
  [/M3x4\s*\(/g, 'mat3x4<f32>('],
  [/float4\s*\(/g, 'vec4<f32>('],
  [/float3\s*\(/g, 'vec3<f32>('],
  [/float2\s*\(/g, 'vec2<f32>('],
  [/uint2\s*\(/g, 'vec2<u32>('],
  [/uint3\s*\(/g, 'vec3<u32>('],
  [/int2\s*\(/g, 'vec2<i32>('],
  [/\bMF\b/g, 'f32'],
];

const MAT_TYPES = {
  'mat4x4<f32>': 16,
  'mat3x4<f32>': 12,
  'mat4x3<f32>': 12,
  'mat3x3<f32>': 9,
};

function convertDeclaration(line) {
  // "V4 a, b, c;" / "V4 r0 = 0.0, r1 = 0.0;" / "MF s0_0 = expr;" / "MF3 yuv = expr;"
  const m = line.match(/^(\s*)(V4|V3|MF4|MF3|MF2|MF)\s+(.+);\s*$/);
  if (!m) return null;
  const [, indent, type, rest] = m;
  const [kind, wgslType] = TEXTURE_ZERO[type];
  const items = splitArgs(rest); // top-level comma split
  const stmts = [];
  for (const item of items) {
    const eq = item.indexOf('=');
    if (eq < 0) {
      stmts.push(`${kind} ${item.trim()}: ${wgslType};`);
    } else {
      const name = item.slice(0, eq).trim();
      let expr = item.slice(eq + 1).trim();
      if (isPureNumber(expr)) expr = `${wgslType}(${expr})`;
      stmts.push(`${kind} ${name} = ${expr};`);
    }
  }
  return indent + stmts.join(' ');
}

function convertFunctionDef(line, passBlock, wgName) {
  // "void PassN(uint2 blockStart, uint3 tid) {"
  let m = line.match(/^void\s+Pass(\d+)\s*\(\s*uint2\s+blockStart\s*,\s*uint3\s+tid\s*\)\s*\{\s*$/);
  if (m) {
    const bsRaw = passBlock.blockSize || [8];
    const bs = [bsRaw[0], bsRaw[1] !== undefined ? bsRaw[1] : bsRaw[0]];
    return (
      `fn Pass${m[1]}(@builtin(workgroup_id) workgroup_id: vec3<u32>, ` +
      `@builtin(local_invocation_id) tid: vec3<u32>) {\n` +
      `    let blockStart = workgroup_id.xy * vec2<u32>(${bs[0]}u, ${bs[1]}u);`
    );
  }
  // "V4 f0(MF a, V4 b, ...) {"
  m = line.match(/^(V4|V3)\s+([A-Za-z_]\w*)\s*\((.*)\)\s*\{\s*$/);
  if (m) {
    const ret = m[1] === 'V4' ? 'vec4<f32>' : 'vec3<f32>';
    const params = splitArgs(m[3]).map((p) => {
      const pm = p.match(/^(MF|V4|V3|MF2|MF3|MF4)\s+([A-Za-z_]\w*)$/);
      if (!pm) throw new Error('unhandled param: ' + p);
      const t = { MF: 'f32', V4: 'vec4<f32>', V3: 'vec3<f32>', MF2: 'vec2<f32>', MF3: 'vec3<f32>', MF4: 'vec4<f32>' }[pm[1]];
      return `${pm[2]}: ${t}`;
    });
    return `fn ${m[2]}(${params.join(', ')}) -> ${ret} {`;
  }
  return null;
}

function convertSampleLevel(line, textureNames) {
  // recv.SampleLevel(sampler, uv, 0)  →  textureSampleLevel(recv_sampled, sampler, uv, 0.0)
  // (recv == INPUT stays INPUT; OUTPUT is never sampled)
  let out = '';
  let pos = 0;
  const re = /\.SampleLevel\s*\(/g;
  let m;
  while ((m = re.exec(line)) !== null) {
    // receiver: word right before the dot
    const dotIdx = m.index;
    const recvM = /([A-Za-z_]\w*)\s*$/.exec(line.slice(0, dotIdx));
    if (!recvM) throw new Error('SampleLevel without receiver: ' + line);
    const recv = recvM[1];
    const openIdx = re.lastIndex - 1;
    const closeIdx = matchParen(line, openIdx);
    if (closeIdx < 0) throw new Error('unbalanced SampleLevel');
    const args = splitArgs(line.slice(openIdx + 1, closeIdx)); // [sampler, uv, lod]
    const tex = recv === 'INPUT' ? 'INPUT' : recv + '_sampled';
    if (recv !== 'INPUT' && !textureNames.has(recv)) throw new Error('unknown texture ' + recv);
    const repl = `textureSampleLevel(${tex}, ${args[0]}, ${args[1]}, 0.0)`;
    out += line.slice(pos, dotIdx - recv.length) + repl;
    pos = closeIdx + 1;
    re.lastIndex = pos;
  }
  return out + line.slice(pos);
}

function convertCalls(line) {
  // mad(a, b, c)   → ((a) * (b) + (c))
  // MulAdd(v, M, c) → ((M) * (v) + (c))
  // mul(M, v)      → ((M) * (v))   (only the YUV const matrices)
  // max(x, 0.0)    → max((x), vec4<f32>(0.0))   (all CuNNy max() calls are vec4 ReLU;
  //                  any other max() is kept verbatim)
  let out = '';
  let pos = 0;
  const re = /\b(mad|MulAdd|mul|max)\s*\(/g;
  for (;;) {
    re.lastIndex = 0;
    const m = re.exec(line.slice(pos));
    if (!m) { out += line.slice(pos); break; }
    const name = m[1];
    const start = pos + m.index;
    const open = start + m[0].length - 1;
    const close = matchParen(line, open);
    if (close < 0) throw new Error('unbalanced call: ' + name);
    const args = splitArgs(line.slice(open + 1, close));
    let repl = null;
    if (name === 'mad') {
      repl = `((${args[0]}) * (${args[1]}) + (${args[2]}))`;
    } else if (name === 'MulAdd') {
      repl = `((${args[1]}) * (${args[0]}) + (${args[2]}))`;
    } else if (name === 'mul') {
      repl = `((${args[0]}) * (${args[1]}))`;
    } else if (name === 'max' && args.length === 2 && isPureNumber(args[1]) && Number(args[1]) === 0) {
      repl = `max((${args[0]}), vec4<f32>(0.0))`;
    }
    if (repl !== null) {
      out += line.slice(pos, start) + repl;
      pos = close + 1;
    } else {
      out += line.slice(pos, close + 1); // keep verbatim, continue after it
      pos = close + 1;
    }
  }
  return out;
}

function convertStore(line, textureNames) {
  // t0[gxy] = expr;          → textureStore(t0_storaged, vec2<i32>(gxy), expr);
  // OUTPUT[gxy + int2(1,0)] = ... → textureStore(OUTPUT, vec2<i32>(gxy) + vec2<i32>(1, 0), ...);
  const m = line.match(/^(\s*)([A-Za-z_]\w*)\s*\[([^\]]+)\]\s*=\s*(.+);\s*$/);
  if (!m) return null;
  const [, indent, tex, index, value] = m;
  if (tex !== 'OUTPUT' && !textureNames.has(tex)) return null;
  let idx = index.trim();
  // gxy + int2(a, b) — mixed u32/i32 arithmetic is illegal in WGSL
  // (alias conversion may already have rewritten int2 → vec2<i32>)
  const mix = idx.match(/^gxy\s*\+\s*(?:int2|vec2<i32>)\(([^)]*)\)$/);
  if (mix) idx = `vec2<i32>(gxy) + vec2<i32>(${mix[1]})`;
  else if (/^gxy/.test(idx) && idx !== 'gxy') throw new Error('unhandled store index: ' + idx);
  else idx = `vec2<i32>(${idx})`;
  let val = value.trim();
  val = val.replace(/,\s*1\)$/, ', 1.0)'); // vec4<f32>(..., 1) → ..., 1.0)
  return `${indent}textureStore(${tex === 'OUTPUT' ? 'OUTPUT' : tex + '_storaged'}, ${idx}, ${val});`;
}

function convertLine(line, ctx) {
  const trimmed = line.trim();
  if (!trimmed) return line;
  if (trimmed.startsWith('//')) return line; // keep comments

  // 1. function definitions
  const fnDef = convertFunctionDef(line, ctx.pass, ctx.name);
  if (fnDef !== null) return fnDef;

  let out = line;

  // 2. canonical header lines
  out = out.replace(/^(\s*)float2 pt = float2\(GetInputPt\(\)\);\s*$/, '$1let pt = GetInputPt();');
  out = out.replace(/^(\s*)uint2 gxy = Rmp8x8\(tid\.x\) \+ blockStart;\s*$/, '$1var gxy = Rmp8x8(tid.x) + blockStart;');
  out = out.replace(
    /^(\s*)uint2 gxy = \(Rmp8x8\(tid\.x\) << 1\) \+ blockStart;\s*$/,
    '$1var gxy = (Rmp8x8(tid.x) << vec2<u32>(1u)) + blockStart;'
  );
  out = out.replace(/^(\s*)uint2 (size|sz) = GetInputSize\(\);\s*$/, '$1let $2 = GetInputSize();');
  out = out.replace(/^(\s*)uint2 (size|sz) = GetOutputSize\(\);\s*$/, '$1let $2 = GetOutputSize();');
  out = out.replace(
    /^(\s*)float2 pos = \(gxy \+ 0\.5\) \* pt;\s*$/,
    '$1var pos = (vec2<f32>(gxy) + vec2<f32>(0.5)) * pt;'
  );
  out = out.replace(
    /^(\s*)float2 pos = \(\(gxy >> 1\) \+ 0\.5\) \* pt;\s*$/,
    '$1var pos = (vec2<f32>(gxy >> vec2<u32>(1u)) + vec2<f32>(0.5)) * pt;'
  );
  out = out.replace(/^(\s*)float2 opt = float2\(GetOutputPt\(\)\);\s*$/, '$1let opt = GetOutputPt();');
  out = out.replace(
    /^(\s*)float2 opt = float2\(GetOutputPt\(\)\), fpos = \(float2\(gxy\) \+ 0\.5\) \* opt;\s*$/,
    '$1let opt = GetOutputPt();\n$1let fpos = (vec2<f32>(gxy) + vec2<f32>(0.5)) * opt;'
  );

  // 3. static const MF3x3 name = { ... };  (possibly chained: "A = { .. }, B = { .. };")
  out = out.replace(/^(\s*)static const\s+MF3x3\s+(.+);\s*$/, (mm, ind, list) => {
    const parts = list.split(/\}\s*,\s*/);
    return parts.map((p, i) => {
      const piece = i < parts.length - 1 ? p + '}' : p;
      const m2 = piece.match(/([A-Za-z_]\w*)\s*=\s*\{([^{}]*)\}/);
      if (!m2) throw new Error('unhandled static const piece: ' + piece);
      return `${ind}let ${m2[1]} = mat3x3<f32>(${m2[2].trim()});`;
    }).join('\n');
  });
  if (/static\b/.test(out)) throw new Error('unhandled static: ' + out);

  // 4. brace initialisers for declarations (V4 r = { ... };)
  out = out.replace(/^(\s*)(V4|V3)\s+([A-Za-z_]\w*)\s*=\s*\{([^{}]*)\};\s*$/,
    (mm, ind, ty, name, nums) =>
      `${ind}var ${name} = ${ty === 'V4' ? 'vec4<f32>' : 'vec3<f32>'}(${nums.trim()});`);

  // 5. plain declarations
  const decl = convertDeclaration(out);
  if (decl !== null) out = decl;

  // 6. SampleLevel → textureSampleLevel
  out = convertSampleLevel(out, ctx.textureNames);

  // 7. alias-call conversions
  for (const [re, to] of ALIAS_CALLS) out = out.replace(re, to);

  // 8. float literal suffix
  out = out.replace(/(\d)f(?![\w.])/g, '$1');

  // 9. mad / MulAdd / mul / max ReLU
  out = convertCalls(out);

  // 10. texture stores
  const store = convertStore(out, ctx.textureNames);
  if (store !== null) out = store;

  // 11. mutations
  out = out.replace(/^(\s*)\+\+(\w+)\.(\w+);\s*$/, (mm, ind, v, c) => `${ind}${v}.${c} = ${v}.${c} + 1u;`);
  out = out.replace(/^(\s*)--(\w+)\.(\w+);\s*$/, (mm, ind, v, c) => `${ind}${v}.${c} = ${v}.${c} - 1u;`);
  out = out.replace(/^(\s*)(\w+)\.(\w+) \+= (.+);\s*$/, (mm, ind, v, c, e) => `${ind}${v}.${c} = ${v}.${c} + ${e};`);
  out = out.replace(/^(\s*)(\w+)\.(\w+) -= (.+);\s*$/, (mm, ind, v, c, e) => `${ind}${v}.${c} = ${v}.${c} - ${e};`);
  out = out.replace(/^(\s*)(\w+) -= (.+);\s*$/, (mm, ind, v, e) => `${ind}${v} = ${v} - (${e});`);

  return out;
}

// ----------------------------------------------------------------------------
// File-level conversion
// ----------------------------------------------------------------------------

function parseSource(text) {
  const lines = text.split(/\r?\n/);
  const meta = { sortName: null, desc: [] };
  const textures = []; // {name, width, height, format, hlslName}
  const samplers = []; // {name, filter, address}
  const commonLines = []; // COMMON code lines (#define O / V4 / M4)
  const passes = []; // {index, desc, in, out, blockSize, numThreads, codeLines[]}

  let block = null; // 'TEXTURE' | 'SAMPLER' | 'COMMON' | 'PASS'
  let cur = null;

  for (const raw of lines) {
    const dm = raw.match(/^\/\/!\s*([A-Za-z_]\w*)\s*(.*)$/);
    if (dm) {
      const directive = dm[1].toUpperCase();
      const val = dm[2].trim();
      switch (directive) {
        case 'MAGPIE': block = null; break;
        case 'VERSION': meta.version = val; block = null; break;
        case 'SORT_NAME': meta.sortName = val.replace(/^"|"$/g, ''); block = null; break;
        case 'USE': meta.use = val; block = null; break;
        case 'CAPABILITY': meta.capability = val; block = null; break;
        case 'TEXTURE':
          flushPass();
          block = 'TEXTURE'; cur = {}; break;
        case 'SAMPLER':
          flushPass();
          block = 'SAMPLER'; cur = {}; break;
        case 'COMMON':
          flushPass();
          block = 'COMMON'; break;
        case 'PASS':
          flushPass();
          block = 'PASS';
          cur = { index: parseInt(val, 10), codeLines: [] };
          break;
        case 'DESC':
          if (block === 'PASS') cur.desc = val.replace(/^"|"$/g, '');
          break;
        case 'IN': if (block === 'PASS') cur.in = val; break;
        case 'OUT': if (block === 'PASS') cur.out = val; break;
        case 'BLOCK_SIZE': if (block === 'PASS') cur.blockSize = val.split(',').map((s) => parseInt(s.trim(), 10)); break;
        case 'NUM_THREADS': if (block === 'PASS') cur.numThreads = val; break;
        case 'WIDTH': if (block === 'TEXTURE') cur.width = val; break;
        case 'HEIGHT': if (block === 'TEXTURE') cur.height = val; break;
        case 'FORMAT': if (block === 'TEXTURE') cur.format = val; break;
        case 'FILTER': if (block === 'SAMPLER') cur.filter = val; break;
        case 'ADDRESS': if (block === 'SAMPLER') cur.address = val; break;
        default:
          // unknown directive inside a pass block: keep as code (harmless)
          if (block === 'PASS') cur.codeLines.push(raw);
          break;
      }
      continue;
    }
    // non-directive line
    if (block === 'TEXTURE') {
      const tm = raw.trim().match(/^Texture2D\s+([A-Za-z_]\w*)\s*;/);
      if (tm) { cur.name = tm[1]; textures.push(cur); cur = {}; }
    } else if (block === 'SAMPLER') {
      const sm = raw.trim().match(/^SamplerState\s+([A-Za-z_]\w*)\s*;/);
      if (sm) { cur.name = sm[1]; samplers.push(cur); cur = {}; }
    } else if (block === 'COMMON') {
      commonLines.push(raw);
    } else if (block === 'PASS') {
      cur.codeLines.push(raw);
    }
  }
  flushPass();
  function flushPass() {
    if (block === 'PASS' && cur) passes.push(cur);
  }
  return { meta, textures, samplers, commonLines, passes };
}

const FORMAT_MAP = {
  'R8G8B8A8_UNORM': 'rgba8unorm',
  'R8G8B8A8_SNORM': 'rgba8snorm',
  'R16G16B16A16_FLOAT': 'rgba16float',
};

function convertFile(inPath, outPath) {
  const src = fs.readFileSync(inPath, 'utf8');
  const { meta, textures, samplers, commonLines, passes } = parseSource(src);
  if (!meta.sortName) throw new Error('missing SORT_NAME');
  if (!passes.length) throw new Error('no passes');

  const inter = textures.filter((t) => t.name !== 'INPUT' && t.name !== 'OUTPUT');
  const textureNames = new Set(inter.map((t) => t.name));
  const wgslFormat = new Map(inter.map((t) => [t.name, FORMAT_MAP[t.format || 'R16G16B16A16_FLOAT'] || t.format.toLowerCase()]));

  // ---- header -------------------------------------------------------------
  const isC = /C-NVL/.test(path.basename(inPath));
  const isDN = /-DN\.hlsl$/.test(inPath);
  const family = isC ? 'CuNNy (C)' : 'CuNNy2';
  const out = [];
  out.push(`// ${path.basename(inPath, '.hlsl')} — Magpie v4 → WGFX WGSL port`);
  out.push(`//`);
  out.push(`// Ported from effects_src/${path.relative(EFFECTS_SRC, inPath).split(path.sep).join('/')}`);
  out.push(`//   (upstream: funnyplanter/CuNNy — GNU GPL v3; Magpie HLSL port by the Magpie project).`);
  out.push(`// Every trained weight constant is copied verbatim; only the pass structure and`);
  out.push(`// sampling idioms are translated.`);
  out.push(`//`);
  out.push(`// Weights are kept **inline** as literals, following the passed FSRCNNX /`);
  out.push(`// Anime4K_GAN ports (no storage buffers, no host-side upload).`);
  out.push(`//`);
  out.push(`// Porting notes:`);
  out.push(`//   1. Magpie CS passes (\`void PassN(uint2 blockStart, uint3 tid)\`) become WGFX compute`);
  out.push(`//      entry points: blockStart = workgroup_id.xy * BLOCK_SIZE, tid = local_invocation_id.`);
  out.push(`//      \`//!NUM_THREADS 64\` → @workgroup_size(64,1,1); the Rmp8x8 8x8 thread mapping is kept.`);
  out.push(`//   2. The \`O(...)\` / \`L0..\` / \`l0..\` sampling macros are expanded inline at each call site`);
  out.push(`//      (they reference the pass-local pos/pt, so inlining is semantics-preserving).`);
  out.push(`//   3. \`mad(s, V4(w), acc)\` → \`acc + s * vec4<f32>(w)\` (scalar broadcasts; component-wise,`);
  out.push(`//      identical to HLSL mad).  \`MulAdd(v, M4(w), acc)\` → \`acc + mat4x4<f32>(w) * v\` and`);
  out.push(`//      \`MulAdd(v, M3x4(w), acc)\` → \`acc + mat3x4<f32>(w) * v\`: the HLSL row-major constructor`);
  out.push(`//      and the WGSL column-major constructor cancel in the flipped product order, so every`);
  out.push(`//      weight number is copied verbatim without transposition (porting-guide §2.1).`);
  out.push(`//   4. The YUV colour matrices (\`static const MF3x3\`) are applied as mul(M, v) → M * v with`);
  out.push(`//      WGSL columns = HLSL rows; the flat number sequence is unchanged.`);
  out.push(`//   5. Intermediate textures keep the Magpie format (${family === 'CuNNy (C)' ? 'R8G8B8A8_SNORM → rgba8snorm' : 'R8G8B8A8_UNORM → rgba8unorm'}),`);
  out.push(`//      declared as \`_sampled\`/\`_storaged\` alias pairs (same logical texture, two views).`);
  out.push(`//      Explicit //!WIDTH/HEIGHT were added — the HLSL relies on Magpie's implicit`);
  out.push(`//      input-size default, WGFX requires the declaration.`);
  out.push(`//   6. ReLU \`max(x, 0.0)\` → \`max(x, vec4<f32>(0.0))\`.  NaN note: WGSL max propagates NaN`);
  out.push(`//      while HLSL heals it; every input here comes from a UNORM/SNORM sampler (finite by`);
  out.push(`//      construction) and the weights are finite, so NaN healing is unreachable — no guard needed.`);
  out.push(`//   7. \`tanh\` is a WGSL builtin; used verbatim.${isC ? ' The final conv ends with tanh (NVL head).' : ''}`);
  out.push(`//   8. The last pass is the out-shuffle: BLOCK_SIZE 16, one 2x2 output quad per thread,`);
  out.push(`//      result added${isC ? ' to the luma channel in YUV space (rgb2yuv/yuv2rgb)' : ' directly to INPUT rgb (saturate)'} and stored to the 2x OUTPUT.`);
  out.push(`//      gxy/pos are mutated in place there, so they are declared \`var\`.`);
  out.push(`//   9. //!USE MulAdd / //!CAPABILITY FP16 are capability markers, kept as-is; arithmetic`);
  out.push(`//      runs in f32 with rgba8unorm/rgba8snorm intermediates (same route as the passed`);
  out.push(`//      FSRCNNX / Anime4K ports; node-webgpu f32→f16 texture truncation does not apply).`);
  out.push(`//  10. Float literal suffixes (\`0.5f\`) are dropped; values unchanged.${isDN ? '  (DN = denoise-optimised weights of the same architecture.)' : ''}`);
  out.push(`//  11. Passes writing more than 4 storage textures (Dawn's default per-stage limit) are`);
  out.push(`//      split into consecutive sub-passes with identical math: each output vec4 is an`);
  out.push(`//      independent function of the same 9 samples, so only the textureStore lines are`);
  out.push(`//      partitioned; sub-passes re-read the same taps (marked "(part k/n)" in //!DESC).`);
  out.push(``);
  out.push(`//!VERSION 4`);
  out.push(`//!SORT_NAME ${meta.sortName}`);
  out.push(`//!USE ${meta.use || 'MulAdd'}`);
  out.push(`//!CAPABILITY ${meta.capability || 'FP16'}`);
  out.push(``);

  // ---- INPUT / OUTPUT textures --------------------------------------------
  out.push(`//!TEXTURE`);
  out.push(`Texture2D INPUT;`);
  out.push(``);
  out.push(`//!TEXTURE`);
  out.push(`//!WIDTH INPUT_WIDTH * 2`);
  out.push(`//!HEIGHT INPUT_HEIGHT * 2`);
  out.push(`Texture2D OUTPUT;`);
  out.push(``);

  // ---- samplers ------------------------------------------------------------
  for (const s of samplers) {
    out.push(`//!SAMPLER`);
    out.push(`//!FILTER ${s.filter}`);
    if (s.address) out.push(`//!ADDRESS ${s.address}`);
    out.push(`SamplerState ${s.name};`);
    out.push(``);
  }

  // ---- intermediate textures ----------------------------------------------
  for (const t of inter) {
    out.push(`//!TEXTURE`);
    out.push(`//!WIDTH ${t.width || 'INPUT_WIDTH'}`);
    out.push(`//!HEIGHT ${t.height || 'INPUT_HEIGHT'}`);
    out.push(`//!FORMAT ${t.format}`);
    out.push(`var ${t.name}_sampled: texture_2d<f32>;`);
    out.push(`var ${t.name}_storaged: texture_storage_2d<${wgslFormat.get(t.name)}, write>;`);
    out.push(``);
  }

  // ---- COMMON --------------------------------------------------------------
  out.push(`//!COMMON`);
  out.push(`// --- Magpie StubDefs.hlsli helpers (verbatim WGSL equivalents) ---`);
  out.push(`alias uint2 = vec2<u32>;`);
  out.push(`alias uint3 = vec3<u32>;`);
  out.push(`alias int2 = vec2<i32>;`);
  out.push(`alias MF = f32;`);
  out.push(`alias MF2 = vec2<f32>;`);
  out.push(`alias MF3 = vec3<f32>;`);
  out.push(`alias MF4 = vec4<f32>;`);
  out.push(``);
  out.push(`fn GetInputSize() -> uint2 { return scene.inputSize; }`);
  out.push(`fn GetInputPt() -> MF2 { return scene.inputPt; }`);
  out.push(`fn GetOutputSize() -> uint2 { return scene.outputSize; }`);
  out.push(`fn GetOutputPt() -> MF2 { return scene.outputPt; }`);
  out.push(``);
  out.push(`// Magpie Rmp8x8: linear thread index → 8x8 position inside the group.`);
  out.push(`fn Rmp8x8(a: u32) -> uint2 { return uint2(a / 8u, a % 8u); }`);
  out.push(`//!END COMMON`);
  out.push(``);

  // ---- passes --------------------------------------------------------------
  // IN/OUT lists use the alias names (like the FSRCNNX / Anime4K ports): the
  // code generator grants storage access only when pass.out lists `X_storaged`.
  const aliasInOut = (list, suffix) =>
    (list || '').split(',').map((n) => {
      const name = n.trim();
      if (name === 'INPUT' || name === 'OUTPUT' || !textureNames.has(name)) return name;
      return `${name}_${suffix}`;
    }).filter(Boolean).join(', ');
  const commonMacros = collectMacros(commonLines.join('\n'));

  // Dawn's default device limit is maxStorageTexturesPerShaderStage = 4 (the
  // adapter supports 8, but the harness device uses defaults). Conv passes of
  // the bigger CuNNy2 models write up to 8 rgba8 textures in one pass, so each
  // over-limit pass is split into equal sub-passes: every output vec4 is an
  // independent function of the same samples, so duplicating the sample loads
  // and keeping only this chunk's stores is numerically identical.
  const MAX_STORAGE_PER_PASS = 4;

  // First convert every pass body to WGSL lines (headers emitted later, with
  // renumbered indices once splits are known).
  const converted = [];
  for (const p of passes) {
    let code = p.codeLines.join('\n');
    const macros = new Map([...commonMacros, ...collectMacros(code)]);
    code = code.replace(/^#define[^\n]*\n/gm, '');
    code = expandFunctionMacros(code, macros);

    const ctx = { pass: p, name: p.desc, textureNames };
    const outLines = [];
    for (const line of code.split('\n')) {
      outLines.push(convertLine(line, ctx));
    }
    // WGSL requires braces on if statements; the only bare if in CuNNy is the
    // two-line bounds check `if (cond)\n\treturn;` — merge it.
    const fixed = [];
    for (let i = 0; i < outLines.length; i++) {
      const l = outLines[i];
      const im = l.match(/^(\s*)if \((.+)\)\s*$/);
      if (im && i + 1 < outLines.length && /^(\s*)return;\s*$/.test(outLines[i + 1])) {
        fixed.push(`${im[1]}if (${im[2]}) {`);
        fixed.push(outLines[i + 1]);
        fixed.push(im[1] + '}');
        i++;
      } else {
        fixed.push(l);
      }
    }
    converted.push({ p, lines: fixed });
  }

  // Split over-limit passes; collect the final flat pass list.
  const finalPasses = [];
  for (const { p, lines } of converted) {
    const outList = aliasInOut(p.out, 'storaged').split(',').map((s) => s.trim()).filter(Boolean);
    const storTexs = outList.filter((n) => n.endsWith('_storaged'));
    if (storTexs.length <= MAX_STORAGE_PER_PASS) {
      finalPasses.push({ p, lines, out: outList });
      continue;
    }
    const chunks = [];
    for (let i = 0; i < storTexs.length; i += MAX_STORAGE_PER_PASS) chunks.push(storTexs.slice(i, i + MAX_STORAGE_PER_PASS));
    chunks.forEach((chunk, ci) => {
      const keep = new Set(chunk);
      const sub = lines.filter((l) => {
        const m = l.match(/^\s*textureStore\((\w+),/);
        return !m || keep.has(m[1]);
      });
      finalPasses.push({
        p,
        lines: sub,
        out: outList.filter((n) => keep.has(n)),
        part: `${ci + 1}/${chunks.length}`,
      });
    });
  }

  let passIndex = 0;
  for (const fp of finalPasses) {
    passIndex++;
    const p = fp.p;
    out.push(`//!PASS ${passIndex}`);
    out.push(`//!DESC ${fp.part ? `${p.desc || ''} (part ${fp.part})` : (p.desc || '')}`);
    out.push(`//!BLOCK_SIZE ${(p.blockSize || [8])[0]}`);
    out.push(`//!NUM_THREADS ${p.numThreads || '64'}`);
    out.push(`//!IN ${aliasInOut(p.in, 'sampled')}`);
    out.push(`//!OUT ${fp.out.join(', ')}`);
    out.push(``);

    // renumber the entry point (split sub-passes inherit the source pass's
    // `fn PassN` name; WGFX anchors @compute to `fn Pass<index>`)
    const entryRenumbered = fp.lines.map((l) => l.replace(/^fn Pass\d+\(/, `fn Pass${passIndex}(`));
    const blockText = entryRenumbered.join('\n').replace(/^\n+/, '').replace(/\n+$/, '\n');
    out.push(blockText);
    out.push(``);
  }

  let text = out.join('\n').replace(/\n{3,}/g, '\n\n');
  fs.writeFileSync(outPath, text, 'utf8');
  const stat = fs.statSync(outPath);
  console.log(`OK  ${path.basename(outPath)}  (${passes.length} passes, ${(stat.size / 1024).toFixed(1)} KiB)`);
}

// ----------------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------------

if (process.argv[2] === '--all') {
  let n = 0;
  for (const fam of ['CuNNy2', 'CuNNy']) {
    const dir = path.join(EFFECTS_SRC, fam);
    for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.hlsl')).sort()) {
      const outDir = path.join(EFFECTS, fam);
      fs.mkdirSync(outDir, { recursive: true });
      convertFile(path.join(dir, f), path.join(outDir, f.replace(/\.hlsl$/, '.wgsl')));
      n++;
    }
  }
  console.log(`converted ${n} files`);
} else {
  const [inPath, outPath] = process.argv.slice(2);
  if (!inPath || !outPath) {
    console.error('usage: node tools/cunny2wgsl.mjs <in.hlsl> <out.wgsl>');
    process.exit(2);
  }
  convertFile(path.resolve(inPath), path.resolve(outPath));
}
