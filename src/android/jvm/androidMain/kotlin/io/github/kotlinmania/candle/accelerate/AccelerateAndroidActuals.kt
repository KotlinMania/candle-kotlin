@file:OptIn(kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

private object AccelerateJniBridge {
    init {
        System.loadLibrary("candle_jni")
    }

    @JvmStatic
    external fun candleAccelerateSgemmFfi(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candleAccelerateDgemmFfi(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candleAccelerateVvexpf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvexp(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvsqrtf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvsqrt(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvsinf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvsin(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvcosf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvcos(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvlogf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvlog(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvtanhf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvtanh(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVDspVaddD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVadd(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVsubD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVsub(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmulD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmul(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVdivD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVdiv(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVminD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmin(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmaxD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmax(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)
}

public actual fun sgemmFfi(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    AccelerateJniBridge.candleAccelerateSgemmFfi(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}

public actual fun dgemmFfi(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    AccelerateJniBridge.candleAccelerateDgemmFfi(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}

public actual fun vvexpf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvexpf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvexp(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvexp(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvsqrtf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvsqrtf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvsqrt(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvsqrt(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvsinf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvsinf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvsin(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvsin(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvcosf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvcosf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvcos(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvcos(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvlogf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvlogf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvlog(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvlog(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvtanhf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvtanhf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvtanh(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    AccelerateJniBridge.candleAccelerateVvtanh(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vDspVaddD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVaddD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVadd(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVadd(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVsubD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVsubD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVsub(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVsub(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmulD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmulD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmul(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmul(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVdivD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVdivD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVdiv(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVdiv(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVminD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVminD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmin(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmin(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmaxD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmaxD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmax(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmax(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}
