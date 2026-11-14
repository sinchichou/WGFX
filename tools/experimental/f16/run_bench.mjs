/**
 * run_bench.mjs — drive headless Chrome through the f16 bench page.
 * Probes which adapter each powerPreference resolves to, then runs the bench
 * for each requested target and prints the <pre> report.
 *
 * Usage: node run_bench.mjs [--chrome <path>] [--target low,high] [--headed]
 */
import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';

const CHROME_CANDIDATES = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
];
let CHROME = null;
let TARGETS = ['low', 'high'];
let HEADED = false;
let SIZE = null;
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--chrome') CHROME = argv[++i];
    else if (argv[i] === '--target') TARGETS = argv[++i].split(',');
    else if (argv[i] === '--headed') HEADED = true;
    else if (argv[i] === '--size') SIZE = Number(argv[++i]);
}
if (!CHROME) CHROME = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!CHROME) { console.error('no chrome/edge found'); process.exit(2); }

const URL_BASE = 'http://127.0.0.1:8123/tools/experimental/f16/bench.html';

const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: HEADED ? false : 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=WebGPUDeveloperFeatures',
        '--no-first-run', '--disable-extensions',
        `--user-data-dir=${process.env.TEMP}/wgfx-bench-prof-${Date.now()}`],
});

// ---- probe: what does each powerPreference resolve to? ----
// (must run on a secure context — about:blank has no navigator.gpu)
const probe = await browser.newPage();
await probe.goto(URL_BASE, { waitUntil: 'domcontentloaded' });
const adapters = await probe.evaluate(async () => {
    const out = {};
    for (const [k, pref] of [['high', 'high-performance'], ['low', 'low-power'], ['default', undefined]]) {
        const a = await navigator.gpu.requestAdapter(pref ? { powerPreference: pref } : {});
        const i = a.info || {};
        out[k] = [i.vendor, i.architecture, i.device || i.description].filter(Boolean).join('/');
    }
    return out;
});
console.log('PROBE ' + JSON.stringify(adapters, null, 0));
await probe.close();

// ---- bench per target ----
for (const t of TARGETS) {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.error(`[${t}] PAGE ERROR: ${e.message}`));
    page.on('console', (m) => { if (m.type() === 'error') console.error(`[${t}] console.error: ${m.text()}`); });
    await page.goto(`${URL_BASE}?gpu=${t}${SIZE ? `&size=${SIZE}` : ''}`, { waitUntil: 'domcontentloaded' });
    try {
        await page.waitForFunction('window.__BENCH_DONE === true', { timeout: 240000, polling: 500 });
    } catch {
        console.error(`[${t}] TIMEOUT waiting for bench completion`);
        await page.close();
        continue;
    }
    const report = await page.$eval('#out', (el) => el.textContent);
    console.log(`\n===== target=${t} =====\n${report}`);
    await page.close();
}

await browser.close();
