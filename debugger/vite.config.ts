import { defineConfig } from 'vite';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Custom plugin to serve shader files
const shaderWatcherPlugin = () => {
  return {
    name: 'shader-watcher',
    configureServer(server) {
      // API to list all .wgsl files in examples/
      server.middlewares.use('/api/files', (req, res, next) => {
        // Resolve correctly from debugger/vite.config.ts -> ../examples
        const examplesDir = path.resolve(__dirname, '../examples');
        try {
          if (!fs.existsSync(examplesDir)) {
             res.statusCode = 404;
             res.end(JSON.stringify({ error: `Directory not found: ${examplesDir}` }));
             return;
          }

          const files = fs.readdirSync(examplesDir)
            .filter(f => f.endsWith('.wgsl'))
            .map(f => ({ name: f, path: `examples/${f}` }));
          
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(files));
        } catch (e) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: String(e) }));
        }
      });
      // API to read a specific shader file
      server.middlewares.use('/api/shader', (req, res, next) => {
        // Parse query param ?file=...
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const targetFile = url.searchParams.get('file');

        if (!targetFile) {
          // Fallback to env var if provided, or return 400
          if (process.env.TARGET_SHADER) {
            // keep existing logic for backward compat if needed, or just proceed
          } else {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: 'No file specified' }));
            return;
          }
        }

        const filePath = path.resolve(__dirname, '..', targetFile || process.env.TARGET_SHADER!);

        try {
          if (fs.existsSync(filePath)) {
            const content = fs.readFileSync(filePath, 'utf-8');
            const stats = fs.statSync(filePath);
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({
              content,
              lastModified: stats.mtimeMs,
              path: filePath
            }));
          } else {
            res.statusCode = 404;
            res.end(JSON.stringify({ error: 'File not found', path: filePath }));
          }
        } catch (e) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: String(e) }));
        }
      });
    }
  }
};

export default defineConfig({
  plugins: [shaderWatcherPlugin()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '../src'),
      'wgfx': path.resolve(__dirname, '../dist/wgfx.browser.esm.js')
    }
  },
  server: {
    fs: {
      // Allow serving files from one level up to the project root
      allow: ['..']
    }
  }
});
