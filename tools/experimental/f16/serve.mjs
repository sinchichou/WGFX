/**
 * serve.mjs — zero-dependency static file server for the browser f16 bench.
 * Serves the repo root so bench.html can fetch dist/, effects/, and itself.
 *
 * Usage: node tools/experimental/f16/serve.mjs [port=8123]
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../..', import.meta.url));
const PORT = Number(process.argv[2] || 8123);
const MIME = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.ts': 'text/plain', '.css': 'text/css', '.json': 'application/json',
    '.wgsl': 'text/plain; charset=utf-8', '.png': 'image/png', '.svg': 'image/svg+xml',
};

createServer(async (req, res) => {
    try {
        const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
        let p = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, '');
        if (p === '' || p.endsWith('/') || p.endsWith('\\')) p += 'index.html';
        const abs = join(ROOT, p);
        if (!abs.startsWith(ROOT)) { res.writeHead(403); res.end('forbidden'); return; }
        const data = await readFile(abs);
        res.writeHead(200, { 'content-type': MIME[extname(abs).toLowerCase()] || 'application/octet-stream' });
        res.end(data);
    } catch {
        res.writeHead(404); res.end('not found');
    }
}).listen(PORT, '127.0.0.1', () => console.log(`serving ${ROOT} at http://127.0.0.1:${PORT}/`));
