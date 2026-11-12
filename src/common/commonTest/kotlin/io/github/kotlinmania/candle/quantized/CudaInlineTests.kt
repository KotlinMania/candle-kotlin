// port-lint: tests quantized/cuda.rs
package io.github.kotlinmania.candle.quantized.cuda

import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.dummycudabackend.CudaDevice
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.rustlang.core.result.fromResidual

@kotlin.test.Test
public fun cudaQuantizeQ81(): Result<Unit> {
    val dev =
        when (val tryResult = CudaDevice.new(0)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val el = 256
    val elPadded = pad(el, MATRIX_ROW_PADDING)
    val ySizeInBytes = elPadded * GgmlDType.Q8_1.typeSize() / GgmlDType.Q8_1.blockSize()
    val yQ81 =
        when (val tryResult = dev.alloc<UByte>(ySizeInBytes)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val vs: List<Float> = (0 until el).map({ v -> v.toFloat() }).collect()
    val y =
        when (val tryResult = dev.cloneHtod(vs)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    when (val tryResult = quantizeQ81(y.asView(), yQ81, el, 1, dev)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    return Ok(Unit)
}

@kotlin.test.Test
public fun cudaMmvQ81(): Result<Unit> {
    val dev =
        when (val tryResult = CudaDevice.new(0)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val ncols = 256
    val vs: List<Float> = (0 until ncols).map({ v -> v.toFloat() }).collect()
    val y =
        when (val tryResult = dev.cloneHtod(vs)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val xs =
        when (val tryResult = QCudaStorage.zeros(dev, ncols, GgmlDType.Q4_0)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    when (val tryResult = xs.quantize(CudaStorage.wrapCudaSlice(y.clone(), dev.clone()))) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    val cudaStorage =
        when (val tryResult = mulMatVecViaQ81(xs.data, y.asView(), GgmlDType.Q4_0, ncols, 1, 1, dev)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val vs1 =
        when (val tryResult = cudaStorage.asCudaSlice<Float>()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val vs2 =
        when (val tryResult = dev.cloneDtoh(vs1.asView())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    check(vs2.len() == 1)
    // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
    // Q8 means 1/256 precision.
    check(vs2[0] == 5561664.5)
    val cudaStorage1 =
        when (val tryResult = dequantizeMulMatVec(xs.data, y.asView(), GgmlDType.Q4_0, ncols, 1, dev)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val vs3 =
        when (val tryResult = cudaStorage1.asCudaSlice<Float>()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val vs4 =
        when (val tryResult = dev.cloneDtoh(vs3.asView())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    check(vs4.len() == 1)
    check(vs4[0] == 5561851.0)
    return Ok(Unit)
}

@kotlin.test.Test
public fun cudaMmQ81(): Result<Unit> {
    val dev =
        when (val tryResult = CudaDevice.new(0)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val ncols = 256
    val vs: List<Float> = (0 until ncols * 4).map({ v -> v.toFloat() / 4.0f }).collect()
    val y =
        when (val tryResult = dev.cloneHtod(vs)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val xs =
        when (val tryResult = QCudaStorage.zeros(dev, ncols * 4, GgmlDType.Q4_0)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    when (val tryResult = xs.quantize(CudaStorage.wrapCudaSlice(y.clone(), dev.clone()))) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    val cudaStorage =
        when (val tryResult = mulMatViaQ81(xs.data, y.asView(), GgmlDType.Q4_0, 4, ncols, ncols, 4, dev)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val vs1 =
        when (val tryResult = cudaStorage.asCudaSlice<Float>()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val vs2 =
        when (val tryResult = dev.cloneDtoh(vs1.asView())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    //
    // x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
    // x @ x.t() / 16
    // tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
    // [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
    // [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
    // [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
    //
    check(vs2.len() == 16)
    check(vs2[0] == 347604.0)
    check(vs2[1] == 888153.06)
    check(vs2[4] == 869780.7)
    check(vs2[5] == 2483145.0)
    check(vs2[11] == 9407368.0)
    check(vs2[14] == 9470856.0)
    check(vs2[15] == 13138824.0)
    return Ok(Unit)
}

// The following test used to fail under compute-sanitizer until #2526.
@kotlin.test.Test
public fun cudaMmQ81Pad(): Result<Unit> {
    val dev =
        when (val tryResult = CudaDevice.new(0)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val (xRows, ncols, yCols) = Triple(4, 16, 2048)
    val vs: List<Float> = (0 until ncols * yCols).map({ v -> v.toFloat() / 256.0f }).collect()
    val y =
        when (val tryResult = dev.cloneHtod(vs)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val xs =
        when (val tryResult = QCudaStorage.zeros(dev, ncols * xRows, GgmlDType.Q4_0)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    when (val tryResult = xs.quantize(CudaStorage.wrapCudaSlice(y.clone(), dev.clone()))) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    return Ok(Unit)
}
