// port-lint: source accelerate.rs
package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

// It would be nice to be able to switch to the NEWLAPACK version of the function but this
// seems to trigger some link error. Available function names can be seen here:
// /Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate.tbd
public expect fun sgemmFfi(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CFloat>, a: ConstPtr<CFloat>, lda: ConstPtr<CInt>, b: ConstPtr<CFloat>, ldb: ConstPtr<CInt>, beta: ConstPtr<CFloat>, c: MutPtr<CFloat>, ldc: ConstPtr<CInt>)

public expect fun dgemmFfi(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CDouble>, a: ConstPtr<CDouble>, lda: ConstPtr<CInt>, b: ConstPtr<CDouble>, ldb: ConstPtr<CInt>, beta: ConstPtr<CDouble>, c: MutPtr<CDouble>, ldc: ConstPtr<CInt>)

public expect fun vvexpf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>)

public expect fun vvexp(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>)

public expect fun vvsqrtf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>)

public expect fun vvsqrt(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>)

public expect fun vvsinf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>)

public expect fun vvsin(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>)

public expect fun vvcosf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>)

public expect fun vvcos(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>)

public expect fun vvlogf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>)

public expect fun vvlog(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>)

public expect fun vvtanhf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>)

public expect fun vvtanh(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>)

public expect fun vDspVaddD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong)

public expect fun vDspVadd(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong)

public expect fun vDspVsubD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong)

public expect fun vDspVsub(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong)

public expect fun vDspVmulD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong)

public expect fun vDspVmul(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong)

public expect fun vDspVdivD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong)

public expect fun vDspVdiv(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong)

public expect fun vDspVminD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong)

public expect fun vDspVmin(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong)

public expect fun vDspVmaxD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong)

public expect fun vDspVmax(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong)

public fun sgemm(transa: UByte, transb: UByte, m: Int, n: Int, k: Int, alpha: Float, a: List<Float>, lda: Int, b: List<Float>, ldb: Int, beta: Float, c: List<Float>, ldc: Int) {
    sgemmFfi((transa.toByte()), (transb.toByte()), m, n, k, alpha, a.asPtr(), lda, b.asPtr(), ldb, beta, c.asMutPtr(), ldc)
}

public fun dgemm(transa: UByte, transb: UByte, m: Int, n: Int, k: Int, alpha: Double, a: List<Double>, lda: Int, b: List<Double>, ldb: Int, beta: Double, c: List<Double>, ldc: Int) {
    dgemmFfi((transa.toByte()), (transb.toByte()), m, n, k, alpha, a.asPtr(), lda, b.asPtr(), ldb, beta, c.asMutPtr(), ldc)
}

public fun vsExp(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvexpf(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vdExp(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvexp(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vsSqrt(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvsqrtf(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vdSqrt(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvsqrt(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vsSin(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvsinf(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vdSin(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvsin(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vsCos(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvcosf(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vdCos(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvcos(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vsTanh(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvtanhf(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vdTanh(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvtanh(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vsLn(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvlogf(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vdLn(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vvlog(y.asMutPtr(), a.asPtr(), (aLen))
}

public fun vsSqr(a: List<Float>, y: List<Float>) {
    val y1 = y
    val aLen = a.size
    val yLen = y1.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    (y1.zip(a)).forEach({ (y2, a1) -> y2 = a1 * a1 })
}

public fun vdSqr(a: List<Double>, y: List<Double>) {
    val y1 = y
    val aLen = a.size
    val yLen = y1.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    (y1.zip(a)).forEach({ (y2, a1) -> y2 = a1 * a1 })
}

public fun vsTanhInplace(y: List<Float>) {
    vvtanhf(y.asMutPtr(), y.asPtr(), (y.size))
}

public fun vdTanhInplace(y: List<Double>) {
    vvtanh(y.asMutPtr(), y.asPtr(), (y.size))
}

public fun vsExpInplace(y: List<Float>) {
    vvexpf(y.asMutPtr(), y.asPtr(), (y.size))
}

public fun vdExpInplace(y: List<Double>) {
    vvexp(y.asMutPtr(), y.asPtr(), (y.size))
}

public fun vsGelu(vs: List<Float>, ys: List<Float>) {
    for ((v, y) in (vs.zip(ys))) {
        y = (2.0f / Float.PI).sqrt() * v * (1.0 + 0.044715 * v * v)
    }
    vsTanhInplace(ys)
    for ((v1, y1) in (vs.zip(ys))) {
        y1 = 0.5 * v1 * (1.0 + y1)
    }
}

public fun vdGelu(vs: List<Double>, ys: List<Double>) {
    for ((v, y) in (vs.zip(ys))) {
        y = (2.0 / Double.PI).sqrt() * v * (1.0 + 0.044715 * v * v)
    }
    vdTanhInplace(ys)
    for ((v1, y1) in (vs.zip(ys))) {
        y1 = 0.5 * v1 * (1.0 + y1)
    }
}

public fun vsSilu(vs: List<Float>, ys: List<Float>) {
    for ((v, y) in (vs.zip(ys))) {
        y = -v
    }
    vsExpInplace(ys)
    for ((v1, y1) in (vs.zip(ys))) {
        y1 = v1 / (1.0 + y1)
    }
}

public fun vdSilu(vs: List<Double>, ys: List<Double>) {
    for ((v, y) in (vs.zip(ys))) {
        y = -v
    }
    vdExpInplace(ys)
    for ((v1, y1) in (vs.zip(ys))) {
        y1 = v1 / (1.0 + y1)
    }
}

public fun vsAdd(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_add").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVadd(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vdAdd(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_add").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVaddD(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vsSub(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_sub").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVsub(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vdSub(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_sub").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVsubD(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vsMul(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_mul").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVmul(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vdMul(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_mul").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVmulD(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vsDiv(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_div").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVdiv(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vdDiv(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_div").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVdivD(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vsMax(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_max").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVmax(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vdMax(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_max").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVmaxD(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vsMin(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_min").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVmin(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}

public fun vdMin(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_min").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vDspVminD(b.asPtr(), 1, a.asPtr(), 1, y.asMutPtr(), 1, aLen.toULong())
}
