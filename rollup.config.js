// rollup.config.js
import resolve from '@rollup/plugin-node-resolve';
import polyfillNode from 'rollup-plugin-polyfill-node';

export default {
    input: 'src/index.js', // 專案入口
    output: [
        {
            file: 'dist/wgfx.js',
            format: 'cjs',   // Node.js 使用
        },
        {
            file: 'dist/wgfx.esm.js',
            format: 'esm',   // ESM 模組
        },
        {
            file: 'dist/wgfx.umd.js',
            format: 'umd',   // 瀏覽器可用
            name: 'WGFX',    // 全域變數名稱
            globals: {},     // 若有 external 可在此對應
        },
    ],
    plugins: [
        resolve({browser: true}), // node_modules 模組解析，支援瀏覽器
        polyfillNode(),             // Node.js built-ins polyfill（fs、path 等）
    ],
    external: [],                 // 若有 external 模組，可在此列出
};
