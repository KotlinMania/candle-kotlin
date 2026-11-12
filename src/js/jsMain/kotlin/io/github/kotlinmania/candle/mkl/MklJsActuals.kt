@file:OptIn(kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.half.F16
import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

private val mklNapiAddon: dynamic =
    js("(function() { if (typeof globalThis !== 'undefined' && typeof globalThis.__km_napi_candle !== 'undefined') return globalThis.__km_napi_candle; if (typeof process === 'undefined' || !process.versions || !process.versions.node) return null; var mod = process.getBuiltinModule ? process.getBuiltinModule('module') : (process.mainModule && process.mainModule.constructor); var req = (mod && mod.createRequire) ? mod.createRequire((process.cwd ? process.cwd() : '.') + '/package.json') : (process.mainModule ? process.mainModule.require : null); if (!req) return null; try { var pkg = req('@kotlinmania/candle-native-bindings'); if (typeof globalThis !== 'undefined') globalThis.__km_napi_candle = pkg; return pkg; } catch (_) {} var fs = req('fs'); var path = req('path'); var dir = process.cwd ? process.cwd() : '.'; var rels = ['candle_native.node', 'build/node-ffi/candle_native.node', 'build/node-ffi/release/candle_native.node', 'build/node-ffi/debug/candle_native.node', 'src/js/node-candle/build/Release/candle_native.node', 'src/js/node-candle/build/Debug/candle_native.node', 'src/js/node-candle/dist/build/Release/candle_native.node', 'src/js/node-candle/dist/build/Debug/candle_native.node']; for (var d = 0; d < 8; d++) { for (var i = 0; i < rels.length; i++) { var c = path.resolve(dir, rels[i]); if (fs.existsSync(c)) { var loaded = req(c); if (typeof globalThis !== 'undefined') globalThis.__km_napi_candle = loaded; return loaded; } } var parent = path.dirname(dir); if (parent === dir) break; dir = parent; } if (typeof globalThis !== 'undefined') globalThis.__km_napi_candle = null; return null; })()")

public actual fun vsTanh(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsTanh(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdTanh(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdTanh(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsExp(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsExp(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdExp(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdExp(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsLn(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsLn(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdLn(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdLn(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsSin(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsSin(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdSin(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdSin(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsCos(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsCos(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdCos(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdCos(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsSqrt(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsSqrt(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdSqrt(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdSqrt(n, a.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsAdd(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsAdd(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdAdd(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdAdd(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsSub(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsSub(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdSub(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdSub(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsMul(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsMul(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdMul(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdMul(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsDiv(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsDiv(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdDiv(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdDiv(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsFmax(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsFmax(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdFmax(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdFmax(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vsFmin(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVsFmin(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vdFmin(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklVdFmin(n, a.rawAddress.toString(), b.rawAddress.toString(), y.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun sgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CFloat>, a: ConstPtr<CFloat>, lda: ConstPtr<CInt>, b: ConstPtr<CFloat>, ldb: ConstPtr<CInt>, beta: ConstPtr<CFloat>, c: MutPtr<CFloat>, ldc: ConstPtr<CInt>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklSgemm(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun dgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CDouble>, a: ConstPtr<CDouble>, lda: ConstPtr<CInt>, b: ConstPtr<CDouble>, ldb: ConstPtr<CInt>, beta: ConstPtr<CDouble>, c: MutPtr<CDouble>, ldc: ConstPtr<CInt>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklDgemm(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun hgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<F16>, a: ConstPtr<F16>, lda: ConstPtr<CInt>, b: ConstPtr<F16>, ldb: ConstPtr<CInt>, beta: ConstPtr<F16>, c: MutPtr<F16>, ldc: ConstPtr<CInt>) {
    if (mklNapiAddon != null) {
        mklNapiAddon.candleMklHgemm(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}
