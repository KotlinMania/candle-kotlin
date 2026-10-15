// port-lint: source error.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.thiserror.StdError
import io.github.kotlinmania.zip.ZipError

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
public enum class BacktraceStatus {
    Disabled,
    Unsupported,
    Captured,
}

public class Backtrace(
    private val statusVal: BacktraceStatus = BacktraceStatus.Disabled,
) {
    public val status: () -> BacktraceStatus = { statusVal }

    public companion object {
        public val capture: () -> Backtrace = { Backtrace(BacktraceStatus.Disabled) }
    }
}

public sealed class CandleException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause),
    StdError,
    CandleException.Error {
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
        val zipError: ZipError? = null,
    ) : CandleException("zip error: $details", zipError) {
        public constructor(err: ZipError) : this(err.message ?: err.toString(), err)
    }

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
        val wrapped: String,
        val context: String,
    ) : CandleException("$wrapped\n$context")

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

    public fun fmt(self: Error = this, f: Formatter = Formatter()): Result<Unit> {
        return write(f, "{self}")
    }

    public fun bt(self: CandleException = this): CandleException {
        val backtrace = std.backtrace.Backtrace.capture()
        return when (backtrace.status()) {
            std.backtrace.BacktraceStatus.Disabled,
            std.backtrace.BacktraceStatus.Unsupported,
            -> self
            else -> Self.WithBacktrace(
                inner = self,
                backtrace = backtrace,
            )
        }
    }

    public fun <P : Any> withPath(p: P, self: CandleException = this): CandleException {
        return Self.WithPath(
            inner = self,
            path = p.asRef().toPathBuf(),
        )
    }

    public fun context(c: Any, self: CandleException = this): CandleException {
        return Self.Context(
            inner = self,
            context = c,
        )
    }

    public companion object Self {
        public val Wrapped: (Any) -> CandleException = { err -> CandleException.Wrapped(err.toString()) }
        public val Msg: (String) -> CandleException = { err -> CandleException.Msg(err) }
        public val WithBacktrace: (CandleException, Any) -> CandleException = { inner, backtrace ->
            CandleException.WithBacktrace(inner = inner, backtrace = backtrace.toString())
        }
        public val WithPath: (CandleException, String) -> CandleException = { inner, path ->
            CandleException.WithPath(inner = inner, path = path)
        }
        public val Context: (CandleException, Any) -> CandleException = { inner, context ->
            CandleException.Context(inner = inner, context = context.toString())
        }

        public fun wrap(err: Any): CandleException {
            return Self.Wrapped(err).bt()
        }

        public fun msg(err: Any): CandleException {
            return Self.Msg(err.toString()).bt()
        }

        public fun debug(err: Any): CandleException {
            return Self.Msg(format("{err:?}", err)).bt()
        }
    }
}

private object std {
    object backtrace {
        val Backtrace = io.github.kotlinmania.candle.Backtrace
        val BacktraceStatus = io.github.kotlinmania.candle.BacktraceStatus
    }
}

public class Formatter {
    public val sb: StringBuilder = StringBuilder()
}

private val write: (Formatter, String) -> Result<Unit> = { f, s ->
    f.sb.append(s)
    Result.success(Unit)
}

private val format: (String, Any?) -> String = { _, arg -> arg.toString() }
private val Any.asRef: () -> Any get() = { this }
private val Any.toPathBuf: () -> String get() = { this.toString() }

private object Error {
    val UnwrapNone: CandleException = CandleException.UnwrapNone
    val msg: (Any) -> CandleException = { c -> CandleException.msg(c) }
    val WrappedContext: (Any, String) -> CandleException = { wrapped, context ->
        CandleException.WrappedContext(wrapped = wrapped.toString(), context = context)
    }
}

private val Err: (CandleException) -> Result<Nothing> = { e -> Result.failure(e) }

public fun <T, U> zip(r1: Result<T>, r2: Result<U>): Result<Pair<T, U>> {
    val Ok: (Pair<T, U>) -> Result<Pair<T, U>> = { v -> Result.success(v) }
    return when {
        r1.isSuccess && r2.isSuccess -> {
            val r1 = r1.getOrThrow()
            val r2 = r2.getOrThrow()
            Ok(Pair(r1, r2))
        }
        r1.isFailure -> {
            val e = r1.exceptionOrNull() as? CandleException ?: CandleException.wrap(r1.exceptionOrNull()!!)
            Err(e)
        }
        else -> {
            val e = r2.exceptionOrNull() as? CandleException ?: CandleException.wrap(r2.exceptionOrNull()!!)
            Err(e)
        }
    }
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
    public fun <C : Any> context(context: C): Result<T>

    /**
     * Wrap the error value with additional context that is evaluated lazily
     * only once an error does occur.
     */
    public fun <C : Any> withContext(f: () -> C): Result<T>

    public class ForResult<T, E : Throwable>(
        private val self: Result<T>,
    ) : ErrorContext<T, E> {
        private val Ok: (T) -> Result<T> = { v -> Result.success(v) }
        private val okVal: () -> T = { self.getOrThrow() }
        private val errVal: (Result<T>) -> Throwable? = { r -> r.exceptionOrNull() }

        override fun <C : Any> context(context: C): Result<T> {
            // Not using mapErr to save 2 useless frames off the captured backtrace
            // in extContext.
            val error = errVal(self)
            return when (error) {
                null -> {
                    val ok = okVal()
                    Ok(ok)
                }
                else -> Err(
                    Error.WrappedContext(
                        error,
                        context.toString(),
                    ).bt(),
                )
            }
        }

        override fun <C : Any> withContext(context: () -> C): Result<T> {
            val error = errVal(self)
            return when (error) {
                null -> {
                    val ok = okVal()
                    Ok(ok)
                }
                else -> Err(
                    Error.WrappedContext(
                        error,
                        context().toString(),
                    ).bt(),
                )
            }
        }
    }

    public class ForOption<T>(
        private val self: T?,
    ) : ErrorContext<T, Nothing> {
        private val Ok: (T) -> Result<T> = { v -> Result.success(v) }

        override fun <C : Any> context(context: C): Result<T> {
            // Not using okOrElse to save 2 useless frames off the captured
            // backtrace.
            return when (self) {
                null -> Err(Error.msg(context).bt())
                else -> {
                    val ok = self
                    Ok(ok)
                }
            }
        }

        override fun <C : Any> withContext(context: () -> C): Result<T> {
            return when (self) {
                null -> Err(Error.UnwrapNone.context(context()).bt())
                else -> {
                    val v = self
                    Ok(v)
                }
            }
        }
    }
}

public val bail: (String) -> Nothing = { msg -> throw CandleException.Msg(msg).bt() }
