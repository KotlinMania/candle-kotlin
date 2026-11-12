// port-lint: source mkl.rs
package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.half.F16
import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

public expect fun vsTanh(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdTanh(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsExp(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdExp(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsLn(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdLn(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsSin(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdSin(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsCos(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdCos(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsSqrt(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdSqrt(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsAdd(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdAdd(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsSub(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdSub(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsMul(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdMul(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsDiv(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdDiv(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsFmax(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdFmax(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun vsFmin(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>)

public expect fun vdFmin(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>)

public expect fun sgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CFloat>, a: ConstPtr<CFloat>, lda: ConstPtr<CInt>, b: ConstPtr<CFloat>, ldb: ConstPtr<CInt>, beta: ConstPtr<CFloat>, c: MutPtr<CFloat>, ldc: ConstPtr<CInt>)

public expect fun dgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CDouble>, a: ConstPtr<CDouble>, lda: ConstPtr<CInt>, b: ConstPtr<CDouble>, ldb: ConstPtr<CInt>, beta: ConstPtr<CDouble>, c: MutPtr<CDouble>, ldc: ConstPtr<CInt>)

public expect fun hgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<F16>, a: ConstPtr<F16>, lda: ConstPtr<CInt>, b: ConstPtr<F16>, ldb: ConstPtr<CInt>, beta: ConstPtr<F16>, c: MutPtr<F16>, ldc: ConstPtr<CInt>)

public fun sgemm(transa: UByte, transb: UByte, m: Int, n: Int, k: Int, alpha: Float, a: List<Float>, lda: Int, b: List<Float>, ldb: Int, beta: Float, c: List<Float>, ldc: Int) {
    sgemm((transa.toByte()), (transb.toByte()), m, n, k, alpha, a.asPtr(), lda, b.asPtr(), ldb, beta, c.asMutPtr(), ldc)
}

public fun dgemm(transa: UByte, transb: UByte, m: Int, n: Int, k: Int, alpha: Double, a: List<Double>, lda: Int, b: List<Double>, ldb: Int, beta: Double, c: List<Double>, ldc: Int) {
    dgemm((transa.toByte()), (transb.toByte()), m, n, k, alpha, a.asPtr(), lda, b.asPtr(), ldb, beta, c.asMutPtr(), ldc)
}

public fun hgemm(transa: UByte, transb: UByte, m: Int, n: Int, k: Int, alpha: F16, a: List<F16>, lda: Int, b: List<F16>, ldb: Int, beta: F16, c: List<F16>, ldc: Int) {
    hgemm((transa.toByte()), (transb.toByte()), m, n, k, alpha, a.asPtr(), lda, b.asPtr(), ldb, beta, c.asMutPtr(), ldc)
}

public fun vsExp(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vsExp(aLen, a.asPtr(), y.asMutPtr())
}

public fun vdExp(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vdExp(aLen, a.asPtr(), y.asMutPtr())
}

public fun vsLn(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vsLn(aLen, a.asPtr(), y.asMutPtr())
}

public fun vdLn(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vdLn(aLen, a.asPtr(), y.asMutPtr())
}

public fun vsSin(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vsSin(aLen, a.asPtr(), y.asMutPtr())
}

public fun vdSin(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vdSin(aLen, a.asPtr(), y.asMutPtr())
}

public fun vsCos(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vsCos(aLen, a.asPtr(), y.asMutPtr())
}

public fun vdCos(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vdCos(aLen, a.asPtr(), y.asMutPtr())
}

public fun vsSqrt(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vsSqrt(aLen, a.asPtr(), y.asMutPtr())
}

public fun vdSqrt(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vdSqrt(aLen, a.asPtr(), y.asMutPtr())
}

public fun vsSqr(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vsMul(aLen, a.asPtr(), a.asPtr(), y.asMutPtr())
}

public fun vdSqr(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vdMul(aLen, a.asPtr(), a.asPtr(), y.asMutPtr())
}

public fun vsTanh(a: List<Float>, y: List<Float>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vsTanh(aLen, a.asPtr(), y.asMutPtr())
}

public fun vdTanh(a: List<Double>, y: List<Double>) {
    val aLen = a.size
    val yLen = y.size
    if (aLen != yLen) {
        error("a and y have different lengths $aLen <> $yLen")
    }
    vdTanh(aLen, a.asPtr(), y.asMutPtr())
}

// The vector functions from mkl can be performed in place by using the same array for input and
// output.
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/vector-mathematical-functions.html
public fun vsTanhInplace(y: List<Float>) {
    vsTanh(y.size, y.asPtr(), y.asMutPtr())
}

public fun vdTanhInplace(y: List<Double>) {
    vdTanh(y.size, y.asPtr(), y.asMutPtr())
}

public fun vsExpInplace(y: List<Float>) {
    vsExp(y.size, y.asPtr(), y.asMutPtr())
}

public fun vdExpInplace(y: List<Double>) {
    vdExp(y.size, y.asPtr(), y.asMutPtr())
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
    vsAdd(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vdAdd(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_add").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vdAdd(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vsSub(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_sub").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vsSub(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vdSub(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_sub").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vdSub(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vsMul(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_mul").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vsMul(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vdMul(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_mul").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vdMul(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vsDiv(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_div").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vsDiv(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vdDiv(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_div").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vdDiv(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vsMax(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_max").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vsFmax(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vdMax(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_max").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vdFmax(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vsMin(a: List<Float>, b: List<Float>, y: List<Float>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vs_min").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vsFmin(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}

public fun vdMin(a: List<Double>, b: List<Double>, y: List<Double>) {
    val aLen = a.size
    val bLen = b.size
    val yLen = y.size
    if (aLen != yLen || bLen != yLen) {
        error(("vd_min").toString() + " a,b,y len mismatch " + (aLen).toString() + " " + (bLen).toString() + " " + (yLen).toString())
    }
    vdFmin(aLen, a.asPtr(), b.asPtr(), y.asMutPtr())
}
