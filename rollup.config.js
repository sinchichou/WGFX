import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import polyfillNode from 'rollup-plugin-polyfill-node';
import typescript from 'rollup-plugin-typescript2';

export default [
    // Browser bundle (ESM)
    {
        input: 'src/index.ts',
        output: {
            file: 'dist/wgfx.browser.esm.js',
            format: 'es',
            sourcemap: true
        },
        plugins: [
            resolve({
                browser: true,
                preferBuiltins: false
            }),
            commonjs(),
            polyfillNode(),
            typescript({
                compilerOptions: { target: "esnext" }
            })
        ]
    },
    // Browser bundle (UMD)
    {
        input: 'src/index.ts',
        output: {
            name: 'WGFX',
            file: 'dist/wgfx.browser.umd.js',
            format: 'umd',
            // 同 cjs：同時具名與預設匯出時明確標示 named，避免 .default 包裝警告
            exports: 'named',
            sourcemap: true
        },
        plugins: [
            resolve({
                browser: true,
                preferBuiltins: false
            }),
            commonjs(),
            polyfillNode(),
            typescript({
                compilerOptions: { target: "es2018" }
            })
        ]
    },
    // Node.js / Bundler (ESM & CJS)
    {
        input: 'src/index.ts',
        output: [
            {
                file: 'dist/wgfx.js',
                format: 'cjs',
                // 同時具名與預設匯出 → 明確標示為 named,避免 CJS 消費者拿到
                // 巢狀的 .default 包裝(rollup 的預設警告來源)。
                exports: 'named',
                sourcemap: true
            },
            {
                file: 'dist/wgfx.esm.js',
                format: 'es',
                sourcemap: true
            }
        ],
        plugins: [
            resolve(),
            commonjs(),
            typescript()
        ],
        external: ['webgpu', 'fs', 'path']
    }
];