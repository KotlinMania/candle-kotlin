// port-lint: source cuda_backend/error.rs
package io.github.kotlinmania.candle.cudabackend.error

import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.cudarc.cublas.result.CublasError
import io.github.kotlinmania.cudarc.curand.result.CurandError
import io.github.kotlinmania.cudarc.driver.DriverError
import io.github.kotlinmania.cudarc.nvrtc.CompileError

/**
 * cudarc related errors
 */
public sealed class CudaError {
    public data class Cuda(public var value: DriverError) : CudaError()
    public data class Compiler(public var value: CompileError) : CudaError()
    public data class Cublas(public var value: CublasError) : CudaError()
    public data class Curand(public var value: CurandError) : CudaError()
    public data class MissingKernel(public var moduleName: String) : CudaError()
    public data class UnsupportedDtype(public var dtype: DType, public var op: String) : CudaError()
    public data class InternalError(public var value: String) : CudaError()
    public data class MatMulNonContiguous(public var lhsStride: Layout, public var rhsStride: Layout, public var mnk: Triple<Int, Int, Int>) : CudaError()
    public data class UnexpectedDType(public var msg: String, public var expected: DType, public var got: DType) : CudaError()
    public data class Load(public var cuda: DriverError, public var moduleName: String) : CudaError()
}

public fun fromError(value: CudaError): Error = Error.Cuda(value).bt()

public interface WrapErr<O> {
    public fun w(): Result<O>
}

public fun <O> Result<O>.w(): Result<O> = this.mapErr({ e -> Error.Cuda(e.into()).bt() })
