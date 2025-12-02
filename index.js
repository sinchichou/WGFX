#!/usr/bin/env node

// index.js

/**
 * @fileoverview This is the main entry point for the WGFX package.
 * It serves as a router to either execute the command-line interface (CLI)
 * or to re-export the public API for consumption as a Node.js module.
 */

// Check if any command-line arguments are passed.
// process.argv contains the full command: ['node', 'path/to/script', 'arg1', 'arg2', ...]
const isCliCall = process.argv.slice(2).length > 0;

if (isCliCall) {
    // If arguments are present, treat it as a CLI call.
    // Dynamically import and run the CLI logic.
    import('./src/cli/wgfx-compile.js')
        .then(cliModule => {
            cliModule.mainCLI();
        })
        .catch(err => {
            console.error("Failed to load or run CLI module:", err);
            process.exit(1);
        });
} else {
    // If no arguments, it's being imported as a module.
    // Re-export the public API.
    // Note: This part will only be executed when this file is imported by another module.
    // The top-level 'if' statement's code will run first when the module is loaded.
}

export * from './src/index.js';
