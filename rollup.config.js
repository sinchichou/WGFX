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
    // Node.js / Bundler (ESM & CJS) - Core library
    {
        input: 'src/index.ts',
        output: [
            {
                file: 'dist/wgfx.js',
                format: 'cjs',
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
        external: ['webgpu', 'fs', 'path', 'glob', 'fast-glob']
    },
    // Node.js - Verification toolkit (includes node-only dependencies)
    {
        input: 'src/verification/entry.ts',
        output: [
            {
                file: 'dist/verification.js',
                format: 'cjs',
                exports: 'named',
                sourcemap: true
            },
            {
                file: 'dist/verification.esm.js',
                format: 'es',
                sourcemap: true
            }
        ],
        plugins: [
            resolve(),
            commonjs(),
            typescript()
        ],
        external: ['webgpu', 'fs', 'path', 'glob', 'fast-glob', 'pngjs']
    }
];