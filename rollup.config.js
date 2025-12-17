// rollup.config.js
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import polyfillNode from 'rollup-plugin-polyfill-node';
import peggy from 'rollup-plugin-peggy';

export default [
    // CLI / Node.js build for the main executable
    {
        input: 'src/index.js',
        output: {
            file: 'dist/wgfx.js',
            format: 'cjs',
        },
        plugins: [
            peggy({cache: true}),
            resolve({preferBuiltins: true}),
            commonjs(),
        ],
        external: ['fs', 'path', 'yargs'] // Externalize node built-ins and yargs
    },
    // Browser Library build
    {
        input: 'src/WGFX.js', // Entry point for the browser library
        output: [
            {
                file: 'dist/wgfx.browser.esm.js',
                format: 'esm',
                sourcemap: true
            },
            {
                file: 'dist/wgfx.browser.umd.js',
                format: 'umd',
                name: 'WGFX', // Global variable name for UMD build
                sourcemap: true
            },
        ],
        plugins: [
            peggy({cache: true}),
            resolve({browser: true}),
            commonjs(),
            polyfillNode(),
        ],
    }
];