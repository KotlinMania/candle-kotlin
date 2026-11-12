@file:OptIn(kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

private val accelerateNapiAddon: dynamic =
    js("(function() { if (typeof globalThis !== 'undefined' && typeof globalThis.__km_napi_candle !== 'undefined') return globalThis.__km_napi_candle; if (typeof process === 'undefined' || !process.versions || !process.versions.node) return null; var mod = process.getBuiltinModule ? process.getBuiltinModule('module') : (process.mainModule && process.mainModule.constructor); var req = (mod && mod.createRequire) ? mod.createRequire((process.cwd ? process.cwd() : '.') + '/package.json') : (process.mainModule ? process.mainModule.require : null); if (!req) return null; try { var pkg = req('@kotlinmania/candle-native-bindings'); if (typeof globalThis !== 'undefined') globalThis.__km_napi_candle = pkg; return pkg; } catch (_) {} var fs = req('fs'); var path = req('path'); var dir = process.cwd ? process.cwd() : '.'; var rels = ['candle_native.node', 'build/node-ffi/candle_native.node', 'build/node-ffi/release/candle_native.node', 'build/node-ffi/debug/candle_native.node', 'src/js/node-candle/build/Release/candle_native.node', 'src/js/node-candle/build/Debug/candle_native.node', 'src/js/node-candle/dist/build/Release/candle_native.node', 'src/js/node-candle/dist/build/Debug/candle_native.node']; for (var d = 0; d < 8; d++) { for (var i = 0; i < rels.length; i++) { var c = path.resolve(dir, rels[i]); if (fs.existsSync(c)) { var loaded = req(c); if (typeof globalThis !== 'undefined') globalThis.__km_napi_candle = loaded; return loaded; } } var parent = path.dirname(dir); if (parent === dir) break; dir = parent; } if (typeof globalThis !== 'undefined') globalThis.__km_napi_candle = null; return null; })()")

public actual fun sgemmFfi(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CFloat>, a: ConstPtr<CFloat>, lda: ConstPtr<CInt>, b: ConstPtr<CFloat>, ldb: ConstPtr<CInt>, beta: ConstPtr<CFloat>, c: MutPtr<CFloat>, ldc: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateSgemmFfi(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun dgemmFfi(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CDouble>, a: ConstPtr<CDouble>, lda: ConstPtr<CInt>, b: ConstPtr<CDouble>, ldb: ConstPtr<CInt>, beta: ConstPtr<CDouble>, c: MutPtr<CDouble>, ldc: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateDgemmFfi(transa.rawAddress.toString(), transb.rawAddress.toString(), m.rawAddress.toString(), n.rawAddress.toString(), k.rawAddress.toString(), alpha.rawAddress.toString(), a.rawAddress.toString(), lda.rawAddress.toString(), b.rawAddress.toString(), ldb.rawAddress.toString(), beta.rawAddress.toString(), c.rawAddress.toString(), ldc.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvexpf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvexpf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvexp(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvexp(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsqrtf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvsqrtf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsqrt(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvsqrt(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsinf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvsinf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvsin(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvsin(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvcosf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvcosf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvcos(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvcos(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvlogf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvlogf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvlog(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvlog(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvtanhf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvtanhf(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vvtanh(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVvtanh(dst.rawAddress.toString(), src.rawAddress.toString(), len.rawAddress.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVaddD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVaddD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVadd(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVadd(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVsubD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVsubD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVsub(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVsub(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmulD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVmulD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmul(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVmul(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVdivD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVdivD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVdiv(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVdiv(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVminD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVminD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmin(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVmin(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmaxD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVmaxD(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}

public actual fun vDspVmax(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    if (accelerateNapiAddon != null) {
        accelerateNapiAddon.candleAccelerateVDspVmax(arg0.rawAddress.toString(), arg1.toString(), arg2.rawAddress.toString(), arg3.toString(), arg4.rawAddress.toString(), arg5.toString(), arg6.toString())
    } else {
        throw UnsupportedOperationException("Native FFI addon is not available")
    }
}
