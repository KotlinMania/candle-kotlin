package io.github.kotlinmania.candle.serializationtests

import io.github.kotlinmania.candlecore.DType
import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Result
import io.github.kotlinmania.candlecore.Tensor
import io.github.kotlinmania.candlecore.safetensors.SliceSafetensors
import io.github.kotlinmania.candlecore.safetensors.load

public class TmpFile(
    public var value: PathBuf = PathBuf(),
) : AsRef<Path> {
    public var field0: PathBuf
        get() = value
        set(v) {
            value = v
        }

    public fun asRef(): Path = this.value.asPath()

    public fun drop() {
        removeFile(this.value).unwrap()
    }

    public companion object {
        public fun create(base: String): TmpFile {
            val filename = tempDir().join("candle-$base-${id()}-${current().id()}")
            return TmpFile(filename)
        }
    }
}

@kotlin.test.Test
public fun npy(): Result<Unit> {
    val npy =
        when (val tryResult = Tensor.readNpy("tests/test.npy")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = npy.toDtype(DType.U8)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun npz(): Result<Unit> {
    val npz =
        when (val tryResult = Tensor.readNpz("tests/test.npz")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(npz.len() == 2)
    check(npz[0].first == "x")
    check(npz[1].first == "x_plus_one")
    check(
        when (
            val tryResult =
                when (val tryResult = npz[1].second.toDtype(DType.U8)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun safetensors(): Result<Unit> {
    val tmpFile = TmpFile.create("st")
    val t =
        when (val tryResult = Tensor.arange(0.0f, 24.0f, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = t.saveSafetensors("t", tmpFile)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    // Load from file.
    val st =
        when (val tryResult = load(tmpFile, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 = st.get("t")!!
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (t - t2)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0f)
    // Load from bytes.
    val bytes =
        when (val tryResult = read(tmpFile)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val st1 =
        when (val tryResult = SliceSafetensors.new(bytes)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t21 = st1.get("t")!!.load(Device.Cpu)
    val diff1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (t - t21)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff1 == 0.0f)
    return Result.Ok(Unit)
}
