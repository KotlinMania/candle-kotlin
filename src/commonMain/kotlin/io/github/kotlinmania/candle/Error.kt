// port-lint: source candle-core/src/error.rs
package io.github.kotlinmania.candle

public data class MatMulUnexpectedStriding(
    val lhsL: Layout,
    val rhsL: Layout,
    val bmnk: List<Int>,
    val msg: String,
)

public sealed class CandleException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause) {
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

    public data class DeviceMismatchBinaryOp(
        val lhs: DeviceLocation,
        val rhs: DeviceLocation,
        val op: String,
    ) : CandleException("device mismatch in $op, lhs: $lhs, rhs: $rhs")

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

    public object NotCompiledWithCudaSupport :
        CandleException("the candle crate has not been built with cuda support")

    public object NotCompiledWithMetalSupport :
        CandleException("the candle crate has not been built with metal support")

    public data class CannotFindTensor(
        val path: String,
    ) : CandleException("cannot find tensor $path")

    public data class Cuda(
        val details: String,
    ) : CandleException("cuda error: $details")

    public data class Metal(
        val details: String,
    ) : CandleException("metal error: $details")

    public data class Ug(
        val details: String,
    ) : CandleException("ug error: $details")

    public data class Npy(
        val details: String,
    ) : CandleException("npy/npz error $details")

    public data class Zip(
        val details: String,
    ) : CandleException("zip error: $details")

    public data class ParseInt(
        val details: String,
    ) : CandleException("parse int error: $details")

    public data class FromUtf8(
        val details: String,
    ) : CandleException("utf8 parse error: $details")

    public data class Io(
        val details: String,
    ) : CandleException("io error: $details")

    public data class SafeTensor(
        val details: String,
    ) : CandleException("safetensor error: $details")

    public data class UnsupportedSafeTensorDtype(
        val details: String,
    ) : CandleException("unsupported safetensor dtype $details")

    public data class Wrapped(
        val details: String,
    ) : CandleException(details)

    public data class WrappedContext(
        val details: String,
        val context: String,
    ) : CandleException("$details\n$context")

    public data class Context(
        val inner: CandleException,
        val context: String,
    ) : CandleException("$context\n${inner.message}", inner)

    public data class WithPath(
        val inner: CandleException,
        val path: String,
    ) : CandleException("path: $path ${inner.message}", inner)

    public data class WithBacktrace(
        val inner: CandleException,
        val backtrace: String,
    ) : CandleException("${inner.message}\n$backtrace", inner)

    public data class Msg(
        val msg: String,
    ) : CandleException(msg)

    public object UnwrapNone :
        CandleException("unwrap none")

    public fun withPath(path: String): CandleException = WithPath(this, path)

    public fun context(context: String): CandleException = Context(this, context)

    public companion object {
        public fun wrap(err: Any): CandleException = Wrapped(err.toString())

        public fun msg(err: Any): CandleException = Msg(err.toString())

        public fun debug(err: Any): CandleException = Msg(err.toString())
    }
}

public inline fun <T> candleResult(block: () -> T): Result<T> =
    try {
        Result.success(block())
    } catch (e: CandleException) {
        Result.failure(e)
    }

public fun bail(message: String): Nothing = throw CandleException.Msg(message)
