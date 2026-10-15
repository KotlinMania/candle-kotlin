// port-lint: source error.rs
package io.github.kotlinmania.candle

/**
 * Candle-specific Error and Result
 */

public data class MatMulUnexpectedStriding(
    val lhsL: Layout,
    val rhsL: Layout,
    val bmnk: List<Int>,
    val msg: String,
)

/**
 * Main library error type.
 */
public sealed class CandleException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause) {
    public interface Error
    public interface Result

    // === DType Errors ===
    public data class UnexpectedDType(
        val msg: String,
        val expected: DType,
        val got: DType,
    ) : CandleException("$msg, expected: $expected, got: $got")

    public data class DTypeMismatchBinaryOp(
        val lhs: DType,
        val rhs: DType,
        val op: String,
    ) : CandleException("dtype mismatch in $op, lhs: $lhs, rhs: $rhs")

    public data class UnsupportedDTypeForOp(
        val dtype: DType,
        val op: String,
    ) : CandleException("unsupported dtype $dtype for op $op")

    // === Dimension Index Errors ===
    public data class DimOutOfRange(
        val shape: Shape,
        val dim: Int,
        val op: String,
    ) : CandleException("$op: dimension index $dim out of range for shape $shape")

    public data class DuplicateDimIndex(
        val shape: Shape,
        val dims: List<Int>,
        val op: String,
    ) : CandleException("$op: duplicate dim index $dims for shape $shape")

    // === Shape Errors ===
    public data class UnexpectedNumberOfDims(
        val expected: Int,
        val got: Int,
        val shape: Shape,
    ) : CandleException("unexpected rank, expected: $expected, got: $got ($shape)")

    public data class UnexpectedShape(
        val msg: String,
        val expected: Shape,
        val got: Shape,
    ) : CandleException("$msg, expected: $expected, got: $got")

    public data class ShapeMismatch(
        val bufferSize: Int,
        val shape: Shape,
    ) : CandleException("Shape mismatch, got buffer of size $bufferSize which is compatible with shape $shape")

    public data class ShapeMismatchBinaryOp(
        val lhs: Shape,
        val rhs: Shape,
        val op: String,
    ) : CandleException("shape mismatch in $op, lhs: $lhs, rhs: $rhs")

    public data class ShapeMismatchCat(
        val dim: Int,
        val firstShape: Shape,
        val n: Int,
        val nthShape: Shape,
    ) : CandleException("shape mismatch in cat for dim $dim, shape for arg 1: $firstShape shape for arg $n: $nthShape")

    public data class ShapeMismatchSplit(
        val shape: Shape,
        val dim: Int,
        val nParts: Int,
    ) : CandleException("Cannot divide tensor of shape $shape equally along dim $dim into $nParts")

    public data class OnlySingleDimension(
        val op: String,
        val dims: List<Int>,
    ) : CandleException("$op can only be performed on a single dimension")

    public data class EmptyTensor(
        val op: String,
    ) : CandleException("empty tensor for $op")

    // === Device Errors ===
    public data class DeviceMismatchBinaryOp(
        val lhs: DeviceLocation,
        val rhs: DeviceLocation,
        val op: String,
    ) : CandleException("device mismatch in $op, lhs: $lhs, rhs: $rhs")

    // === Op Specific Errors ===
    public data class NarrowInvalidArgs(
        val shape: Shape,
        val dim: Int,
        val start: Int,
        val len: Int,
        val msg: String,
    ) : CandleException("narrow invalid args $msg: $shape, dim: $dim, start: $start, len:$len")

    public data class Conv1dInvalidArgs(
        val inpShape: Shape,
        val kShape: Shape,
        val padding: Int,
        val stride: Int,
        val msg: String,
    ) : CandleException("conv1d invalid args $msg: inp: $inpShape, k: $kShape, pad: $padding, stride: $stride")

    public data class InvalidIndex(
        val op: String,
        val index: Int,
        val size: Int,
    ) : CandleException("$op invalid index $index with dim size $size")

    public data class BroadcastIncompatibleShapes(
        val srcShape: Shape,
        val dstShape: Shape,
    ) : CandleException("cannot broadcast $srcShape to $dstShape")

    public data class CannotSetVar(
        val msg: String,
    ) : CandleException("cannot set variable $msg")

    // Indirection to avoid large variant.
    public data class MatMulUnexpectedStridingError(
        val details: MatMulUnexpectedStriding,
    ) : CandleException(details.toString())

    public data class RequiresContiguous(
        val op: String,
    ) : CandleException("$op only supports contiguous tensors")

    public data class OpRequiresAtLeastOneTensor(
        val op: String,
    ) : CandleException("$op expects at least one tensor")

    public data class OpRequiresAtLeastTwoTensors(
        val op: String,
    ) : CandleException("$op expects at least two tensors")

    public data class BackwardNotSupported(
        val op: String,
    ) : CandleException("backward is not supported for $op")

    // === Other Errors ===
    public object NotCompiledWithCudaSupport :
        CandleException("the candle crate has not been built with cuda support")

    public object NotCompiledWithMetalSupport :
        CandleException("the candle crate has not been built with metal support")

    public data class CannotFindTensor(
        val path: String,
    ) : CandleException("cannot find tensor $path")

    // === Wrapped Errors ===
    public data class Cuda(
        val details: String,
    ) : CandleException("cuda error: $details")

    public data class Metal(
        val details: String,
    ) : CandleException("Metal error $details")

    public data class Ug(
        val details: String,
    ) : CandleException("ug error: $details")

    public data class TryFromIntError(
        val details: String,
    ) : CandleException("try from int error: $details")

    public data class Npy(
        val details: String,
    ) : CandleException("npy/npz error $details")

    /**
     * Zip file format error.
     */
    public data class Zip(
        val details: String,
    ) : CandleException("zip error: $details")

    /**
     * Integer parse error.
     */
    public data class ParseInt(
        val details: String,
    ) : CandleException("parse int error: $details")

    /**
     * Utf8 parse error.
     */
    public data class FromUtf8(
        val details: String,
    ) : CandleException("utf8 parse error: $details")

    /**
     * I/O error.
     */
    public data class Io(
        val details: String,
    ) : CandleException("io error: $details")

    /**
     * SafeTensor error.
     */
    public data class SafeTensor(
        val details: String,
    ) : CandleException("safetensor error: $details")

    public data class UnsupportedSafeTensorDtype(
        val details: String,
    ) : CandleException("unsupported safetensor dtype $details")

    /**
     * Arbitrary errors wrapping.
     */
    public data class Wrapped(
        val details: String,
    ) : CandleException(details)

    /**
     * Arbitrary errors wrapping with context.
     */
    public data class WrappedContext(
        val details: String,
        val context: String,
    ) : CandleException("$details\n$context")

    public data class Context(
        val inner: CandleException,
        val context: String,
    ) : CandleException("$context\n${inner.message}", inner)

    /**
     * Adding path information to an error.
     */
    public data class WithPath(
        val inner: CandleException,
        val path: String,
    ) : CandleException("path: $path ${inner.message}", inner)

    public data class WithBacktrace(
        val inner: CandleException,
        val backtrace: String,
    ) : CandleException("${inner.message}\n$backtrace", inner)

    /**
     * User generated error message, typically created via `bail!`.
     */
    public data class Msg(
        val msg: String,
    ) : CandleException(msg)

    public object UnwrapNone :
        CandleException("unwrap none")

    public fun fmt(): String = message ?: ""

    public fun bt(): CandleException = this

    public fun withPath(p: String): CandleException = WithPath(this, p)

    public fun context(c: Any): CandleException = Context(this, c.toString())

    public companion object {
        public fun wrap(err: Any): CandleException = Wrapped(err.toString()).bt()

        public fun msg(err: Any): CandleException = Msg(err.toString()).bt()

        public fun debug(err: Any): CandleException = Msg(err.toString()).bt()
    }
}

public fun <T, U> zip(r1: Result<T>, r2: Result<U>): Result<Pair<T, U>> =
    when {
        r1.isSuccess && r2.isSuccess -> Result.success(Pair(r1.getOrThrow(), r2.getOrThrow()))
        r1.isFailure -> Result.failure(r1.exceptionOrNull()!!)
        else -> Result.failure(r2.exceptionOrNull()!!)
    }

public interface Sealed

/**
 * Attach more context to an error.
 *
 * Inspired by [`anyhow::Context`].
 */
public interface ErrorContext<T, E> : Sealed {
    /**
     * Wrap the error value with additional context.
     */
    public fun context(context: Any): Result<T>

    /**
     * Wrap the error value with additional context that is evaluated lazily
     * only once an error does occur.
     */
    public fun withContext(f: () -> Any): Result<T>

    public class ForResult<T>(
        private val result: Result<T>,
    ) : ErrorContext<T, Throwable> {
        override fun context(context: Any): Result<T> =
            result.fold(
                onSuccess = { Result.success(it) },
                onFailure = { error ->
                    Result.failure(
                        CandleException.WrappedContext(
                            details = error.toString(),
                            context = context.toString(),
                        ).bt(),
                    )
                },
            )

        override fun withContext(f: () -> Any): Result<T> =
            result.fold(
                onSuccess = { Result.success(it) },
                onFailure = { error ->
                    Result.failure(
                        CandleException.WrappedContext(
                            details = error.toString(),
                            context = f().toString(),
                        ).bt(),
                    )
                },
            )
    }

    public class ForOption<T>(
        private val value: T?,
    ) : ErrorContext<T, Nothing> {
        override fun context(context: Any): Result<T> =
            if (value != null) {
                Result.success(value)
            } else {
                Result.failure(CandleException.msg(context).bt())
            }

        override fun withContext(f: () -> Any): Result<T> =
            if (value != null) {
                Result.success(value)
            } else {
                Result.failure(CandleException.UnwrapNone.context(f()).bt())
            }
    }
}

public inline fun <T> candleResult(block: () -> T): Result<T> =
    try {
        Result.success(block())
    } catch (e: CandleException) {
        Result.failure(e)
    }

public fun bail(message: String): Nothing = throw CandleException.Msg(message)
