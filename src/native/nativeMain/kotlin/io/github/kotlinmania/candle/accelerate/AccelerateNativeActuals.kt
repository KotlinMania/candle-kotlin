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
import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.toCPointer

public actual fun sgemmFfi(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CFloat>, a: ConstPtr<CFloat>, lda: ConstPtr<CInt>, b: ConstPtr<CFloat>, ldb: ConstPtr<CInt>, beta: ConstPtr<CFloat>, c: MutPtr<CFloat>, ldc: ConstPtr<CInt>) {
    candleAccelerateSgemmFfi(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}

public actual fun dgemmFfi(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CDouble>, a: ConstPtr<CDouble>, lda: ConstPtr<CInt>, b: ConstPtr<CDouble>, ldb: ConstPtr<CInt>, beta: ConstPtr<CDouble>, c: MutPtr<CDouble>, ldc: ConstPtr<CInt>) {
    candleAccelerateDgemmFfi(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvexpf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    candleAccelerateVvexpf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvexp(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    candleAccelerateVvexp(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvsqrtf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    candleAccelerateVvsqrtf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvsqrt(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    candleAccelerateVvsqrt(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvsinf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    candleAccelerateVvsinf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvsin(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    candleAccelerateVvsin(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvcosf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    candleAccelerateVvcosf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvcos(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    candleAccelerateVvcos(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvlogf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    candleAccelerateVvlogf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvlog(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    candleAccelerateVvlog(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvtanhf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    candleAccelerateVvtanhf(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vvtanh(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    candleAccelerateVvtanh(dst.rawAddress.toCPointer<ByteVar>(), src.rawAddress.toCPointer<ByteVar>(), len.rawAddress.toCPointer<ByteVar>())
}

public actual fun vDspVaddD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVaddD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVadd(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVadd(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVsubD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVsubD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVsub(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVsub(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmulD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmulD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmul(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmul(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVdivD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVdivD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVdiv(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVdiv(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVminD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVminD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmin(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmin(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmaxD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmaxD(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}

public actual fun vDspVmax(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    candleAccelerateVDspVmax(arg0.rawAddress.toCPointer<ByteVar>(), arg1, arg2.rawAddress.toCPointer<ByteVar>(), arg3, arg4.rawAddress.toCPointer<ByteVar>(), arg5, arg6)
}
