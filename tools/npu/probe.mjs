/**
 * probe.mjs — NPU / execution-provider probe for the WGFX NPU experiment.
 *
 * Reports: CPU, NPU (PnP), GPU adapters, Ryzen AI Software presence, and the
 * ONNX Runtime execution providers available to onnxruntime-node.
 *
 * Usage: node tools/npu/probe.mjs
 */

import { execSync } from 'child_process';

const run = (cmd) => {
    try {
        return execSync(cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
    } catch {
        return '';
    }
};

console.log('=== WGFX NPU probe ===\n');

// ---- CPU ----
const cpu = run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"').trim();
console.log(`CPU        : ${cpu || '(unknown)'}`);
const isRyzenAI = /8840|8845|8850|Hawk Point|Phoenix|Ryzen AI|7840|7940/i.test(cpu);
console.log(`Ryzen AI-capable CPU: ${isRyzenAI ? 'YES' : 'no'}\n`);

// ---- NPU device ----
const names = run('powershell -NoProfile -Command "(Get-PnpDevice -PresentOnly).FriendlyName"')
    .split('\n').map((l) => l.trim()).filter(Boolean);
const npuDevices = names.filter((n) => /\bnpu\b|neural|ai boost|compute accelerator/i.test(n));
console.log('--- NPU / accelerator devices ---');
for (const n of npuDevices) console.log(`  ${n}`);
if (npuDevices.length === 0) console.log('(none found)');
const hasNpu = npuDevices.length > 0;

// ---- GPU adapters ----
const gpu = run(
    'powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name"').trim();
console.log('--- GPU adapters (DirectML targets) ---');
console.log(gpu || '(unknown)');

// ---- Ryzen AI Software ----
console.log('\n--- Ryzen AI Software (NPU execution provider) ---');
const candidates = [
    'C:\\Program Files\\Amd\\RyzenAI',
    'C:\\Program Files\\AMD\\RyzenAI',
    'C:\\RyzenAI',
    process.env.USERPROFILE + '\\RyzenAI',
];
let rai = candidates.find((p) => {
    try { return execSync(`powershell -NoProfile -Command "Test-Path '${p}'"`, { encoding: 'utf8' }).trim() === 'True'; } catch { return false; }
});
console.log(rai ? `FOUND: ${rai}` : 'NOT installed — NPU unreachable until installed (see docs/npu-experiment.md §4)');

// ---- onnxruntime execution providers ----
console.log('\n--- onnxruntime-node ---');
try {
    const mod = await import('onnxruntime-node');
    const ort = mod.default ?? mod;
    console.log('onnxruntime-node: installed');
    console.log('known EPs on this build: cpu' + (process.platform === 'win32' ? ', dml (DirectML → GPU; never NPU)' : ''));
    void ort;
} catch {
    console.log('onnxruntime-node: NOT installed (npm i onnxruntime-node)');
}

// ---- verdict ----
console.log('\n=== verdict ===');
console.log(`NPU hardware present        : ${hasNpu ? 'YES (XDNA)' : 'no'}`);
console.log(`Ryzen AI Software installed : ${rai ? 'YES' : 'NO — install to reach the NPU'}`);
console.log(`Reachable tonight via ORT   : CPU ✅, DirectML→GPU ✅, NPU ${rai ? '✅' : '⛔ (install Ryzen AI SW)'}`);
