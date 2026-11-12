@file:OptIn(kotlin.ExperimentalUnsignedTypes::class, kotlin.js.ExperimentalWasmJsInterop::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

@JsFun("() => { if (typeof globalThis.__km_napi_candle !== 'undefined') return globalThis.__km_napi_candle !== null; if (typeof process === 'undefined' || !process.versions || !process.versions.node) { globalThis.__km_napi_candle = null; return false; } var mod = process.getBuiltinModule ? process.getBuiltinModule('module') : (process.mainModule && process.mainModule.constructor); var req = (mod && mod.createRequire) ? mod.createRequire((process.cwd ? process.cwd() : '.') + '/package.json') : (process.mainModule ? process.mainModule.require : null); if (!req) { globalThis.__km_napi_candle = null; return false; } try { globalThis.__km_napi_candle = req('@kotlinmania/candle-native-bindings'); return true; } catch (_) {} var fs = req('fs'); var path = req('path'); var dir = process.cwd ? process.cwd() : '.'; var rels = ['candle_native.node', 'build/node-ffi/candle_native.node', 'build/node-ffi/release/candle_native.node', 'build/node-ffi/debug/candle_native.node', 'src/js/node-candle/build/Release/candle_native.node', 'src/js/node-candle/build/Debug/candle_native.node', 'src/js/node-candle/dist/build/Release/candle_native.node', 'src/js/node-candle/dist/build/Debug/candle_native.node']; for (var d = 0; d < 8; d++) { for (var i = 0; i < rels.length; i++) { var c = path.resolve(dir, rels[i]); if (fs.existsSync(c)) { globalThis.__km_napi_candle = req(c); return true; } } var parent = path.dirname(dir); if (parent === dir) break; dir = parent; } globalThis.__km_napi_candle = null; return false; }")
private external fun initMklNapiAddon(): Boolean

private val mklNapiAvailable: Boolean = initMklNapiAddon()

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVsTanh(p0, p1, p2)")
private external fun wasmCandleMklVsTanh(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVdTanh(p0, p1, p2)")
private external fun wasmCandleMklVdTanh(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVsExp(p0, p1, p2)")
private external fun wasmCandleMklVsExp(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVdExp(p0, p1, p2)")
private external fun wasmCandleMklVdExp(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVsLn(p0, p1, p2)")
private external fun wasmCandleMklVsLn(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVdLn(p0, p1, p2)")
private external fun wasmCandleMklVdLn(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVsSin(p0, p1, p2)")
private external fun wasmCandleMklVsSin(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVdSin(p0, p1, p2)")
private external fun wasmCandleMklVdSin(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVsCos(p0, p1, p2)")
private external fun wasmCandleMklVsCos(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVdCos(p0, p1, p2)")
private external fun wasmCandleMklVdCos(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVsSqrt(p0, p1, p2)")
private external fun wasmCandleMklVsSqrt(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2) => globalThis.__km_napi_candle.candleMklVdSqrt(p0, p1, p2)")
private external fun wasmCandleMklVdSqrt(p0: Int, p1: String, p2: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVsAdd(p0, p1, p2, p3)")
private external fun wasmCandleMklVsAdd(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVdAdd(p0, p1, p2, p3)")
private external fun wasmCandleMklVdAdd(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVsSub(p0, p1, p2, p3)")
private external fun wasmCandleMklVsSub(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVdSub(p0, p1, p2, p3)")
private external fun wasmCandleMklVdSub(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVsMul(p0, p1, p2, p3)")
private external fun wasmCandleMklVsMul(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVdMul(p0, p1, p2, p3)")
private external fun wasmCandleMklVdMul(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVsDiv(p0, p1, p2, p3)")
private external fun wasmCandleMklVsDiv(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVdDiv(p0, p1, p2, p3)")
private external fun wasmCandleMklVdDiv(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVsFmax(p0, p1, p2, p3)")
private external fun wasmCandleMklVsFmax(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVdFmax(p0, p1, p2, p3)")
private external fun wasmCandleMklVdFmax(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVsFmin(p0, p1, p2, p3)")
private external fun wasmCandleMklVsFmin(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3) => globalThis.__km_napi_candle.candleMklVdFmin(p0, p1, p2, p3)")
private external fun wasmCandleMklVdFmin(p0: Int, p1: String, p2: String, p3: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candleMklSgemm(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun wasmCandleMklSgemm(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candleMklDgemm(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun wasmCandleMklDgemm(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

@JsFun("(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) => globalThis.__km_napi_candle.candleMklHgemm(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)")
private external fun wasmCandleMklHgemm(p0: String, p1: String, p2: String, p3: String, p4: String, p5: String, p6: String, p7: String, p8: String, p9: String, p10: String, p11: String, p12: String)

public actual fun vsTanh(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsTanh(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdTanh(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdTanh(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsExp(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsExp(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdExp(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdExp(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsLn(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsLn(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdLn(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdLn(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsSin(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsSin(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdSin(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdSin(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsCos(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsCos(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdCos(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdCos(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsSqrt(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsSqrt(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdSqrt(n: CInt, a: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdSqrt(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsAdd(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsAdd(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdAdd(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdAdd(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsSub(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsSub(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdSub(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdSub(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsMul(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsMul(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdMul(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdMul(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsDiv(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsDiv(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdDiv(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdDiv(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsFmax(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsFmax(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdFmax(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdFmax(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsFmin(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVsFmin(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdFmin(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklVdFmin(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun sgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklSgemm(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun dgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklDgemm(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun hgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    if (mklNapiAvailable) {
        wasmCandleMklHgemm(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}
