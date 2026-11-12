@file:OptIn(kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.candle.COpaquePointer
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong

private object AccelerateJniBridge {
    init {
        System.loadLibrary("candle_jni")
    }

    @JvmStatic
    external fun candle_accelerate_sgemmFfi(transa: String?, transb: String?, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candle_accelerate_dgemmFfi(transa: String?, transb: String?, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candle_accelerate_vvexpf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvexp(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvsqrtf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvsqrt(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvsinf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvsin(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvcosf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvcos(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvlogf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvlog(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvtanhf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vvtanh(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVaddD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVadd(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVsubD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVsub(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVmulD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVmul(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVdivD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVdiv(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVminD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVmin(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVmaxD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candle_accelerate_vDspVmax(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)
}

public actual fun sgemmFfi(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_sgemmFfi(transa, transb, (m?.value ?: 0L), (n?.value ?: 0L), (k?.value ?: 0L), (alpha?.value ?: 0L), (a?.value ?: 0L), (lda?.value ?: 0L), (b?.value ?: 0L), (ldb?.value ?: 0L), (beta?.value ?: 0L), (c?.value ?: 0L), (ldc?.value ?: 0L))
}

public actual fun dgemmFfi(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_dgemmFfi(transa, transb, (m?.value ?: 0L), (n?.value ?: 0L), (k?.value ?: 0L), (alpha?.value ?: 0L), (a?.value ?: 0L), (lda?.value ?: 0L), (b?.value ?: 0L), (ldb?.value ?: 0L), (beta?.value ?: 0L), (c?.value ?: 0L), (ldc?.value ?: 0L))
}

public actual fun vvexpf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvexpf((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvexp(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvexp((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvsqrtf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvsqrtf((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvsqrt(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvsqrt((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvsinf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvsinf((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvsin(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvsin((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvcosf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvcosf((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvcos(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvcos((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvlogf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvlogf((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvlog(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvlog((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvtanhf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvtanhf((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vvtanh(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    AccelerateJniBridge.candle_accelerate_vvtanh((dst?.value ?: 0L), (src?.value ?: 0L), (len?.value ?: 0L))
}

public actual fun vDspVaddD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVaddD((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVadd(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVadd((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVsubD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVsubD((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVsub(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVsub((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVmulD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVmulD((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVmul(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVmul((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVdivD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVdivD((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVdiv(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVdiv((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVminD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVminD((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVmin(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVmin((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVmaxD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVmaxD((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}

public actual fun vDspVmax(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candle_accelerate_vDspVmax((arg0?.value ?: 0L), arg1, (arg2?.value ?: 0L), arg3, (arg4?.value ?: 0L), arg5, arg6.toLong())
}
