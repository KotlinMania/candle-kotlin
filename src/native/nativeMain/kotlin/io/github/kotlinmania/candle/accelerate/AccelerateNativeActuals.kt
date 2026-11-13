@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class, kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateDgemmFfi
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateSgemmFfi
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVadd
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVaddD
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVdiv
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVdivD
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVmax
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVmaxD
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVmin
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVminD
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVmul
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVmulD
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVsub
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVDspVsubD
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvcos
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvcosf
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvexp
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvexpf
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvlog
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvlogf
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvsin
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvsinf
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvsqrt
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvsqrtf
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvtanh
import io.github.kotlinmania.candle.accelerate.cinterop.candleAccelerateVvtanhf
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.toCPointer

public actual fun sgemmFfi(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    candleAccelerateSgemmFfi(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}

public actual fun dgemmFfi(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    candleAccelerateDgemmFfi(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvexpf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvexpf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvexp(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvexp(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvsqrtf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvsqrtf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvsqrt(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvsqrt(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvsinf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvsinf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvsin(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvsin(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvcosf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvcosf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvcos(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvcos(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvlogf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvlogf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvlog(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvlog(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvtanhf(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvtanhf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvtanh(dst: MutPtr, src: ConstPtr, len: ConstPtr) {
    candleAccelerateVvtanh(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vDspVaddD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVaddD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVadd(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVadd(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVsubD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVsubD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVsub(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVsub(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmulD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmulD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmul(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmul(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVdivD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVdivD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVdiv(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVdiv(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVminD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVminD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmin(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmin(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmaxD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmaxD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmax(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmax(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}
