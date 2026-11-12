@file:OptIn(kotlin.ExperimentalUnsignedTypes::class, kotlin.js.ExperimentalWasmJsInterop::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.candle.COpaquePointer
import io.github.kotlinmania.libc.CInt

@JsFun("() => { if (typeof globalThis.__km_napi_candle !== 'undefined') return globalThis.__km_napi_candle !== null; if (typeof process === 'undefined' || !process.versions || !process.versions.node) { globalThis.__km_napi_candle = null; return false; } var mod = process.getBuiltinModule ? process.getBuiltinModule('module') : (process.mainModule && process.mainModule.constructor); var req = (mod && mod.createRequire) ? mod.createRequire((process.cwd ? process.cwd() : '.') + '/package.json') : (process.mainModule ? process.mainModule.require : null); if (!req) { globalThis.__km_napi_candle = null; return false; } try { globalThis.__km_napi_candle = req('@kotlinmania/candle-native-bindings'); return true; } catch (_) {} var fs = req('fs'); var path = req('path'); var dir = process.cwd ? process.cwd() : '.'; var rels = ['candle_native.node', 'build/node-ffi/candle_native.node', 'build/node-ffi/release/candle_native.node', 'build/node-ffi/debug/candle_native.node', 'src/js/node-candle/build/Release/candle_native.node', 'src/js/node-candle/build/Debug/candle_native.node', 'src/js/node-candle/dist/build/Release/candle_native.node', 'src/js/node-candle/dist/build/Debug/candle_native.node']; for (var d = 0; d < 8; d++) { for (var i = 0; i < rels.length; i++) { var c = path.resolve(dir, rels[i]); if (fs.existsSync(c)) { globalThis.__km_napi_candle = req(c); return true; } } var parent = path.dirname(dir); if (parent === dir) break; dir = parent; } globalThis.__km_napi_candle = null; return false; }")
private external fun _initMklNapiAddon(): Boolean

private val _mklNapiAvailable: Boolean = _initMklNapiAddon()

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vsTanh(p0, p1, p2)")
private external fun _wasm_candle_mkl_vsTanh(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vdTanh(p0, p1, p2)")
private external fun _wasm_candle_mkl_vdTanh(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vsExp(p0, p1, p2)")
private external fun _wasm_candle_mkl_vsExp(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vdExp(p0, p1, p2)")
private external fun _wasm_candle_mkl_vdExp(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vsLn(p0, p1, p2)")
private external fun _wasm_candle_mkl_vsLn(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vdLn(p0, p1, p2)")
private external fun _wasm_candle_mkl_vdLn(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vsSin(p0, p1, p2)")
private external fun _wasm_candle_mkl_vsSin(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vdSin(p0, p1, p2)")
private external fun _wasm_candle_mkl_vdSin(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vsCos(p0, p1, p2)")
private external fun _wasm_candle_mkl_vsCos(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vdCos(p0, p1, p2)")
private external fun _wasm_candle_mkl_vdCos(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vsSqrt(p0, p1, p2)")
private external fun _wasm_candle_mkl_vsSqrt(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_mkl_vdSqrt(p0, p1, p2)")
private external fun _wasm_candle_mkl_vdSqrt(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vsAdd(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vsAdd(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vdAdd(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vdAdd(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vsSub(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vsSub(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vdSub(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vdSub(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vsMul(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vsMul(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vdMul(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vdMul(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vsDiv(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vsDiv(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vdDiv(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vdDiv(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vsFmax(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vsFmax(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vdFmax(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vdFmax(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vsFmin(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vsFmin(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candle_mkl_vdFmin(p0, p1, p2, p3)")
private external fun _wasm_candle_mkl_vdFmin(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candle_mkl_sgemm(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun _wasm_candle_mkl_sgemm(p0: String?, p1: String?, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candle_mkl_dgemm(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun _wasm_candle_mkl_dgemm(p0: String?, p1: String?, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candle_mkl_hgemm(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun _wasm_candle_mkl_hgemm(p0: String?, p1: String?, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

public actual fun vsTanh(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsTanh(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdTanh(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdTanh(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsExp(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsExp(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdExp(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdExp(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsLn(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsLn(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdLn(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdLn(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsSin(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsSin(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdSin(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdSin(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsCos(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsCos(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdCos(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdCos(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsSqrt(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsSqrt(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdSqrt(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdSqrt(n, (a?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsAdd(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsAdd(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdAdd(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdAdd(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsSub(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsSub(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdSub(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdSub(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsMul(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsMul(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdMul(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdMul(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsDiv(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsDiv(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdDiv(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdDiv(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsFmax(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsFmax(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdFmax(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdFmax(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsFmin(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vsFmin(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdFmin(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_vdFmin(n, (a?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (y?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun sgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_sgemm(transa, transb, (m?.value ?: 0L).toString(), (n?.value ?: 0L).toString(), (k?.value ?: 0L).toString(), (alpha?.value ?: 0L).toString(), (a?.value ?: 0L).toString(), (lda?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (ldb?.value ?: 0L).toString(), (beta?.value ?: 0L).toString(), (c?.value ?: 0L).toString(), (ldc?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun dgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_dgemm(transa, transb, (m?.value ?: 0L).toString(), (n?.value ?: 0L).toString(), (k?.value ?: 0L).toString(), (alpha?.value ?: 0L).toString(), (a?.value ?: 0L).toString(), (lda?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (ldb?.value ?: 0L).toString(), (beta?.value ?: 0L).toString(), (c?.value ?: 0L).toString(), (ldc?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun hgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    if (_mklNapiAvailable) {
        _wasm_candle_mkl_hgemm(transa, transb, (m?.value ?: 0L).toString(), (n?.value ?: 0L).toString(), (k?.value ?: 0L).toString(), (alpha?.value ?: 0L).toString(), (a?.value ?: 0L).toString(), (lda?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (ldb?.value ?: 0L).toString(), (beta?.value ?: 0L).toString(), (c?.value ?: 0L).toString(), (ldc?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}
