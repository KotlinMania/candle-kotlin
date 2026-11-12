@file:OptIn(kotlin.ExperimentalUnsignedTypes::class, kotlin.js.ExperimentalWasmJsInterop::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

@JsFun("() => { if (typeof globalThis.__km_napi_candle !== 'undefined') return globalThis.__km_napi_candle !== null; if (typeof process === 'undefined' || !process.versions || !process.versions.node) { globalThis.__km_napi_candle = null; return false; } var mod = process.getBuiltinModule ? process.getBuiltinModule('module') : (process.mainModule && process.mainModule.constructor); var req = (mod && mod.createRequire) ? mod.createRequire((process.cwd ? process.cwd() : '.') + '/package.json') : (process.mainModule ? process.mainModule.require : null); if (!req) { globalThis.__km_napi_candle = null; return false; } try { globalThis.__km_napi_candle = req('@kotlinmania/candle-native-bindings'); return true; } catch (_) {} var fs = req('fs'); var path = req('path'); var dir = process.cwd ? process.cwd() : '.'; var rels = ['candle_native.node', 'build/node-ffi/candle_native.node', 'build/node-ffi/release/candle_native.node', 'build/node-ffi/debug/candle_native.node', 'src/js/node-candle/build/Release/candle_native.node', 'src/js/node-candle/build/Debug/candle_native.node', 'src/js/node-candle/dist/build/Release/candle_native.node', 'src/js/node-candle/dist/build/Debug/candle_native.node']; for (var d = 0; d < 8; d++) { for (var i = 0; i < rels.length; i++) { var c = path.resolve(dir, rels[i]); if (fs.existsSync(c)) { globalThis.__km_napi_candle = req(c); return true; } } var parent = path.dirname(dir); if (parent === dir) break; dir = parent; } globalThis.__km_napi_candle = null; return false; }")
private external fun initAccelerateNapiAddon(): Boolean

private val accelerateNapiAvailable: Boolean = initAccelerateNapiAddon()

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candleAccelerateSgemmFfi(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun wasmCandleAccelerateSgemmFfi(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candleAccelerateDgemmFfi(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun wasmCandleAccelerateDgemmFfi(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvexpf(p0, p1, p2)")
private external fun wasmCandleAccelerateVvexpf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvexp(p0, p1, p2)")
private external fun wasmCandleAccelerateVvexp(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvsqrtf(p0, p1, p2)")
private external fun wasmCandleAccelerateVvsqrtf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvsqrt(p0, p1, p2)")
private external fun wasmCandleAccelerateVvsqrt(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvsinf(p0, p1, p2)")
private external fun wasmCandleAccelerateVvsinf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvsin(p0, p1, p2)")
private external fun wasmCandleAccelerateVvsin(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvcosf(p0, p1, p2)")
private external fun wasmCandleAccelerateVvcosf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvcos(p0, p1, p2)")
private external fun wasmCandleAccelerateVvcos(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvlogf(p0, p1, p2)")
private external fun wasmCandleAccelerateVvlogf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvlog(p0, p1, p2)")
private external fun wasmCandleAccelerateVvlog(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvtanhf(p0, p1, p2)")
private external fun wasmCandleAccelerateVvtanhf(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleAccelerateVvtanh(p0, p1, p2)")
private external fun wasmCandleAccelerateVvtanh(p0: String, p1: String, p2: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVaddD(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVaddD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVadd(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVadd(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVsubD(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVsubD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVsub(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVsub(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVmulD(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVmulD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVmul(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVmul(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVdivD(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVdivD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVdiv(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVdiv(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVminD(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVminD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVmin(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVmin(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVmaxD(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVmaxD(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6) => globalThis.__km_napi_candle.candleAccelerateVDspVmax(p0, p1, p2, p3, p4, p5, p6)")
private external fun wasmCandleAccelerateVDspVmax(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String)

public actual fun sgemmFfi(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateSgemmFfi(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun dgemmFfi(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateDgemmFfi(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvexpf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvexpf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvexp(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvexp(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsqrtf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvsqrtf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsqrt(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvsqrt(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsinf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvsinf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsin(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvsin(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvcosf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvcosf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvcos(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvcos(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvlogf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvlogf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvlog(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvlog(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvtanhf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvtanhf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvtanh(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVvtanh(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVaddD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVaddD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVadd(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVadd(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVsubD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVsubD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVsub(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVsub(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmulD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVmulD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmul(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVmul(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVdivD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVdivD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVdiv(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVdiv(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVminD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVminD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmin(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVmin(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmaxD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVmaxD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmax(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAvailable) {
        wasmCandleAccelerateVDspVmax(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}
