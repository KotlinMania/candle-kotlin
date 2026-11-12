@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class, kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.candle.COpaquePointer
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_dgemmFfi
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_sgemmFfi
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVadd
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVaddD
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVdiv
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVdivD
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVmax
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVmaxD
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVmin
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVminD
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVmul
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVmulD
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVsub
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vDspVsubD
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvcos
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvcosf
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvexp
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvexpf
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvlog
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvlogf
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvsin
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvsinf
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvsqrt
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvsqrtf
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvtanh
import io.github.kotlinmania.candle.accelerate.cinterop.candle_accelerate_vvtanhf
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.toCPointer

public actual fun sgemmFfi(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    candle_accelerate_sgemmFfi(transa, transb, m?.value?.toCPointer<ByteVar>(), n?.value?.toCPointer<ByteVar>(), k?.value?.toCPointer<ByteVar>(), alpha?.value?.toCPointer<ByteVar>(), a?.value?.toCPointer<ByteVar>(), lda?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), ldb?.value?.toCPointer<ByteVar>(), beta?.value?.toCPointer<ByteVar>(), c?.value?.toCPointer<ByteVar>(), ldc?.value?.toCPointer<ByteVar>())
}

public actual fun dgemmFfi(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    candle_accelerate_dgemmFfi(transa, transb, m?.value?.toCPointer<ByteVar>(), n?.value?.toCPointer<ByteVar>(), k?.value?.toCPointer<ByteVar>(), alpha?.value?.toCPointer<ByteVar>(), a?.value?.toCPointer<ByteVar>(), lda?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), ldb?.value?.toCPointer<ByteVar>(), beta?.value?.toCPointer<ByteVar>(), c?.value?.toCPointer<ByteVar>(), ldc?.value?.toCPointer<ByteVar>())
}

public actual fun vvexpf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvexpf(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvexp(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvexp(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvsqrtf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvsqrtf(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvsqrt(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvsqrt(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvsinf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvsinf(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvsin(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvsin(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvcosf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvcosf(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvcos(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvcos(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvlogf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvlogf(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvlog(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvlog(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvtanhf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvtanhf(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vvtanh(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    candle_accelerate_vvtanh(dst?.value?.toCPointer<ByteVar>(), src?.value?.toCPointer<ByteVar>(), len?.value?.toCPointer<ByteVar>())
}

public actual fun vDspVaddD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVaddD(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVadd(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVadd(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVsubD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVsubD(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVsub(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVsub(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmulD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVmulD(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmul(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVmul(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVdivD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVdivD(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVdiv(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVdiv(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVminD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVminD(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmin(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVmin(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmaxD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVmaxD(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmax(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    candle_accelerate_vDspVmax(arg0?.value?.toCPointer<ByteVar>(), arg1, arg2?.value?.toCPointer<ByteVar>(), arg3, arg4?.value?.toCPointer<ByteVar>(), arg5, arg6)
}
