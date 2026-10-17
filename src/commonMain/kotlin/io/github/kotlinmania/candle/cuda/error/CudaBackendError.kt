// port-lint: source cuda_backend/error.rs
package io.github.kotlinmania.candle.cuda.error

import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.thiserror.StdError

/**
 * cudarc related errors
 */
sealed class CudaError(
    message: String,
) : Exception(message),
    StdError {
    data class Cuda(
        val err: String,
    ) : CudaError(err)

    data class Compiler(
        val err: String,
    ) : CudaError(err)

    data class Cublas(
        val err: String,
    ) : CudaError(err)

    data class Curand(
        val err: String,
    ) : CudaError(err)

    data class MissingKernel(
        val moduleName: String,
    ) : CudaError("missing kernel '$moduleName'")

    data class UnsupportedDtype(
        val dtype: DType,
        val op: String,
    ) : CudaError("unsupported dtype $dtype for $op")

    data class InternalError(
        val msg: String,
    ) : CudaError("internal error '$msg'")

    @ConsistentCopyVisibility
    data class MatMulNonContiguous internal constructor(
        val lhsStride: Layout,
        val rhsStride: Layout,
        val mnk: Triple<Int, Int, Int>,
    ) : CudaError("matmul is only supported for contiguous tensors lstride: $lhsStride rstride: $rhsStride mnk: $mnk")

    data class UnexpectedDType(
        val msg: String,
        val expected: DType,
        val got: DType,
    ) : CudaError("$msg, expected: $expected, got: $got")

    data class Load(
        val cuda: String,
        val moduleName: String,
    ) : CudaError("$cuda when loading $moduleName")

    fun into(): CudaError = this
}

fun from(v: CudaError): CandleError = io.github.kotlinmania.candle.CandleException.Cuda(v).bt()

interface WrapErr<O> {
    fun w(): Result<O>
}

class ResultWrapErr<O>(
    private val res: Result<O>,
) : WrapErr<O> {
    private val self: Result<O> get() = res

    private fun <R> Result<R>.mapErr(transform: (CudaError) -> Throwable): Result<R> =
        fold(
            onSuccess = { Result.success(it) },
            onFailure = { err ->
                val cudaErr = (err as? CudaError) ?: CudaError.Cuda(err.message.orEmpty())
                Result.failure(transform(cudaErr))
            },
        )

    override fun w(): Result<O> = self.mapErr { e -> io.github.kotlinmania.candle.CandleException.Cuda(e.into()).bt() }
}

fun <O> Result<O>.w(): Result<O> = ResultWrapErr(this).w()
