@file:OptIn(kotlin.ExperimentalUnsignedTypes::class, kotlin.js.ExperimentalWasmJsInterop::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.candle.COpaquePointer
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong

@JsFun("() => { if (typeof globalThis.__km_napi_candle !== 'undefined') return globalThis.__km_napi_candle !== null; if (typeof process === 'undefined' || !process.versions || !process.versions.node) { globalThis.__km_napi_candle = null; return false; } var mod = process.getBuiltinModule ? process.getBuiltinModule('module') : (process.mainModule && process.mainModule.constructor); var req = (mod && mod.createRequire) ? mod.createRequire((process.cwd ? process.cwd() : '.') + '/package.json') : (process.mainModule ? process.mainModule.require : null); if (!req) { globalThis.__km_napi_candle = null; return false; } try { globalThis.__km_napi_candle = req('@kotlinmania/candle-native-bindings'); return true; } catch (_) {} var fs = req('fs'); var path = req('path'); var dir = process.cwd ? process.cwd() : '.'; var rels = ['candle_native.node', 'build/node-ffi/candle_native.node', 'build/node-ffi/release/candle_native.node', 'build/node-ffi/debug/candle_native.node', 'src/js/node-candle/build/Release/candle_native.node', 'src/js/node-candle/build/Debug/candle_native.node', 'src/js/node-candle/dist/build/Release/candle_native.node', 'src/js/node-candle/dist/build/Debug/candle_native.node']; for (var d = 0; d < 8; d++) { for (var i = 0; i < rels.length; i++) { var c = path.resolve(dir, rels[i]); if (fs.existsSync(c)) { globalThis.__km_napi_candle = req(c); return true; } } var parent = path.dirname(dir); if (parent === dir) break; dir = parent; } globalThis.__km_napi_candle = null; return false; }")
private external fun _initAccelerateNapiAddon(): Boolean

private val _accelerateNapiAvailable: Boolean = _initAccelerateNapiAddon()

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candle_accelerate_sgemmFfi(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun _wasm_candle_accelerate_sgemmFfi(p0: String?, p1: String?, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candle_accelerate_dgemmFfi(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun _wasm_candle_accelerate_dgemmFfi(p0: String?, p1: String?, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvexpf(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvexpf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvexp(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvexp(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvsqrtf(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvsqrtf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvsqrt(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvsqrt(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvsinf(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvsinf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvsin(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvsin(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvcosf(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvcosf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvcos(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvcos(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvlogf(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvlogf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvlog(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvlog(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvtanhf(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvtanhf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candle_accelerate_vvtanh(p0, p1, p2)")
private external fun _wasm_candle_accelerate_vvtanh(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVaddD(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVaddD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVadd(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVadd(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVsubD(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVsubD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVsub(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVsub(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVmulD(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVmulD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVmul(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVmul(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVdivD(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVdivD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVdiv(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVdiv(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVminD(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVminD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVmin(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVmin(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVmaxD(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVmaxD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candle_accelerate_vDspVmax(p0, p1, p2, p3, p4, p5, p6)")
private external fun _wasm_candle_accelerate_vDspVmax(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

public actual fun sgemmFfi(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_sgemmFfi(transa, transb, (m?.value ?: 0L).toString(), (n?.value ?: 0L).toString(), (k?.value ?: 0L).toString(), (alpha?.value ?: 0L).toString(), (a?.value ?: 0L).toString(), (lda?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (ldb?.value ?: 0L).toString(), (beta?.value ?: 0L).toString(), (c?.value ?: 0L).toString(), (ldc?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun dgemmFfi(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_dgemmFfi(transa, transb, (m?.value ?: 0L).toString(), (n?.value ?: 0L).toString(), (k?.value ?: 0L).toString(), (alpha?.value ?: 0L).toString(), (a?.value ?: 0L).toString(), (lda?.value ?: 0L).toString(), (b?.value ?: 0L).toString(), (ldb?.value ?: 0L).toString(), (beta?.value ?: 0L).toString(), (c?.value ?: 0L).toString(), (ldc?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvexpf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvexpf((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvexp(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvexp((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsqrtf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvsqrtf((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsqrt(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvsqrt((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsinf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvsinf((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsin(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvsin((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvcosf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvcosf((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvcos(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvcos((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvlogf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvlogf((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvlog(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvlog((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvtanhf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvtanhf((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvtanh(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vvtanh((dst?.value ?: 0L).toString(), (src?.value ?: 0L).toString(), (len?.value ?: 0L).toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVaddD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVaddD((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVadd(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVadd((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVsubD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVsubD((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVsub(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVsub((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmulD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVmulD((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmul(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVmul((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVdivD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVdivD((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVdiv(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVdiv((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVminD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVminD((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmin(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVmin((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmaxD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVmaxD((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmax(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    if (_accelerateNapiAvailable) {
        _wasm_candle_accelerate_vDspVmax((arg0?.value ?: 0L).toString(), arg1.toString(), (arg2?.value ?: 0L).toString(), arg3.toString(), (arg4?.value ?: 0L).toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}
