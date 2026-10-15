// port-lint: source cuda_backend/error.rs
package io.github.kotlinmania.candle.cuda.error

import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Layout

sealed class CudaError(message: String) : Exception(message) {
    data class Cuda(val err: String) : CudaError(err)

    data class Compiler(val err: String) : CudaError(err)

    data class Cublas(val err: String) : CudaError(err)

    data class Curand(val err: String) : CudaError(err)

    data class MissingKernel(val moduleName: String) :
        CudaError("missing kernel `$moduleName`")

    data class UnsupportedDtype(val dtype: DType, val op: String) :
        CudaError("unsupported dtype $dtype for $op")

    data class InternalError(val msg: String) :
        CudaError("internal error `$msg`")

    data class MatMulNonContiguous(
        val lhsStride: Layout,
        val rhsStride: Layout,
        val mnk: Triple<Int, Int, Int>
    ) : CudaError("matmul is only supported for contiguous tensors lstride: $lhsStride rstride: $rhsStride mnk: $mnk")

    data class UnexpectedDType(
        val msg: String,
        val expected: DType,
        val got: DType
    ) : CudaError("$msg, expected: $expected, got: $got")

    data class Load(
        val cuda: String,
        val moduleName: String
    ) : CudaError("$cuda when loading $moduleName")
}

fun from(valErr: CudaError): CandleError {
    return CandleError.Cuda(valErr).bt()
}

interface WrapErr<O> {
    fun w(): CandleResult<O>
}

class ResultWrapErr<O>(private val inner: CandleResult<O>) : WrapErr<O> {
    override fun w(): CandleResult<O> {
        return when (inner) {
            is CandleResult.Ok -> CandleResult.Ok(inner.value)
            is CandleResult.Err -> CandleResult.Err(CandleError.Cuda(inner.error).bt())
        }
    }
}
