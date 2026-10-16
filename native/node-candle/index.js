// Entry point for the candle N-API native module
// Shared between Kotlin/JS (nodeMain) and Kotlin/WASM (wasmJsMain via Node.js)

if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    const nodeRequire = eval('require');
    try {
        module.exports = nodeRequire(__dirname + '/build/Release/candle_native.node');
    } catch (err) {
        try {
            module.exports = nodeRequire(__dirname + '/build/Debug/candle_native.node');
        } catch (err2) {
            throw new Error(
                'Could not load candle_native addon. ' +
                'Run `npm install` or `node-gyp rebuild` in native/node-candle directory.\n' +
                'Original error: ' + err.message + '\n' +
                'Debug error: ' + err2.message
            );
        }
    }
} else {
    module.exports = {};
}
