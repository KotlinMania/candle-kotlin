// Entry point for the candle N-API native module
// Shared between Kotlin/JS (nodeMain) and Kotlin/WASM (wasmJsMain via Node.js)

try {
    module.exports = require('./build/Release/candle_native.node');
} catch (err) {
    try {
        module.exports = require('./build/Debug/candle_native.node');
    } catch (err2) {
        throw new Error(
            'Could not load candle_native addon. ' +
            'Run `npm install` or `node-gyp rebuild` in native/node-candle directory.\n' +
            'Original error: ' + err.message + '\n' +
            'Debug error: ' + err2.message
        );
    }
}
