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