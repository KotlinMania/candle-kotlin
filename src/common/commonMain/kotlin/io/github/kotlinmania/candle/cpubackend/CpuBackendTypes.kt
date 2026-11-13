package io.github.kotlinmania.candle.cpubackend

import io.github.kotlinmania.candle.backend.BackendStorage
import io.github.kotlinmania.candle.conv.ParamsConv1D
import io.github.kotlinmania.candle.conv.ParamsConv2D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose1D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose2D
import io.github.kotlinmania.candle.cpubackend.conv2d.Conv2D
import io.github.kotlinmania.candle.cpubackend.utils.binaryMap
import io.github.kotlinmania.candle.cpubackend.utils.binaryMapVec
import io.github.kotlinmania.candle.cpubackend.utils.binaryMapVecPar
import io.github.kotlinmania.candle.cpubackend.utils.unaryMap
import io.github.kotlinmania.candle.cpubackend.utils.unaryMapVecPar
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.op.BinaryOpT
import io.github.kotlinmania.candle.op.CmpOp
import io.github.kotlinmania.candle.op.ReduceOp
import io.github.kotlinmania.candle.op.UnaryOpT
import io.github.kotlinmania.candle.scalar.Scalar
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.stridedindex.StridedBlocks
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.numtraits.Float
import io.github.kotlinmania.rand.prelude.*
import io.github.kotlinmania.rayon.prelude.*
import io.github.kotlinmania.rustlang.core.result.fromResidual

public typealias Device = CpuDevice

// Note: Maybe we should not implement [Clone] here and instead have an explicit allocator +
// intercept the oom errors to avoid panicking and provide a proper error.
public sealed class CpuStorage : BackendStorage {
    public data class U8(
        public var value: List<UByte>,
    ) : CpuStorage()
    public data class U32(
        public var value: List<UInt>,
    ) : CpuStorage()
    public data class I16(
        public var value: List<Short>,
    ) : CpuStorage()
    public data class I32(
        public var value: List<Int>,
    ) : CpuStorage()
    public data class I64(
        public var value: List<Long>,
    ) : CpuStorage()
    public data class Bf16(
        public var value: List<Bf16>,
    ) : CpuStorage()
    public data class F16(
        public var value: List<F16>,
    ) : CpuStorage()
    public data class F32(
        public var value: List<Float>,
    ) : CpuStorage()
    public data class F64(
        public var value: List<Double>,
    ) : CpuStorage()
    public data class F8e4m3(
        public var value: List<F8e4m3>,
    ) : CpuStorage()

    // Dummy types that store raw bytes
    public data class F6e2m3(
        public var value: List<UByte>,
    ) : CpuStorage()
    public data class F6e3m2(
        public var value: List<UByte>,
    ) : CpuStorage()
    public data class F4(
        public var value: List<UByte>,
    ) : CpuStorage()
    public data class F8e8m0(
        public var value: List<UByte>,
    ) : CpuStorage()

    public fun <D : WithDType> asSlice(cpuStorageAsSliceDFn: (Any?) -> Any?): Result<List<D>> = cpuStorageAsSliceDFn(this)

    public fun dtype(): DType =
        when (this) {
            is U8 -> DType.U8
            is U32 -> DType.U32
            is I16 -> DType.I16
            is I32 -> DType.I32
            is I64 -> DType.I64
            is Bf16 -> DType.BF16
            is F16 -> DType.F16
            is F32 -> DType.F32
            is F64 -> DType.F64
            is F8e4m3 -> DType.F8E4M3
            is F6e2m3 -> DType.F6E2M3
            is F6e3m2 -> DType.F6E3M2
            is F4 -> DType.F4
            is F8e8m0 -> DType.F8E8M0
        }

    public fun toDtype(layout: Layout, dtype: DType): Result<CpuStorage> {
        // Note: find a way around the quadratic number of cases below.
        return run {
            val matchVal = Pair(this, dtype)
            when {
                matchVal.first is U8 && matchVal.second == DType.BF16 -> {
                    val (storage) = matchVal.first as U8
                    val data = unaryMap(storage, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.bf16(data))
                }
                matchVal.first is U32 && matchVal.second == DType.BF16 -> {
                    val (storage1) = matchVal.first as U32
                    val data1 = unaryMap(storage1, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.bf16(data1))
                }
                matchVal.first is I64 && matchVal.second == DType.BF16 -> {
                    val (storage2) = matchVal.first as I64
                    val data2 = unaryMap(storage2, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.bf16(data2))
                }
                matchVal.first is Bf16 && matchVal.second == DType.BF16 -> {
                    val (storage3) = matchVal.first as Bf16
                    val data3 = unaryMap(storage3, layout, { v -> v })
                    Ok(CpuStorage.bf16(data3))
                }
                matchVal.first is F16 && matchVal.second == DType.BF16 -> {
                    val (storage4) = matchVal.first as F16
                    val data4 = unaryMap(storage4, layout, { v -> fromF32(v.toF32()) })
                    Ok(CpuStorage.bf16(data4))
                }
                matchVal.first is F32 && matchVal.second == DType.BF16 -> {
                    val (storage5) = matchVal.first as F32
                    val data5 = unaryMap(storage5, layout, fromF32)
                    Ok(CpuStorage.bf16(data5))
                }
                matchVal.first is F64 && matchVal.second == DType.BF16 -> {
                    val (storage6) = matchVal.first as F64
                    val data6 = unaryMap(storage6, layout, fromF64)
                    Ok(CpuStorage.bf16(data6))
                }
                matchVal.first is U8 && matchVal.second == DType.F16 -> {
                    val (storage7) = matchVal.first as U8
                    val data7 = unaryMap(storage7, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.f16(data7))
                }
                matchVal.first is U32 && matchVal.second == DType.F16 -> {
                    val (storage8) = matchVal.first as U32
                    val data8 = unaryMap(storage8, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.f16(data8))
                }
                matchVal.first is I64 && matchVal.second == DType.F16 -> {
                    val (storage9) = matchVal.first as I64
                    val data9 = unaryMap(storage9, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.f16(data9))
                }
                matchVal.first is Bf16 && matchVal.second == DType.F16 -> {
                    val (storage10) = matchVal.first as Bf16
                    val data10 = unaryMap(storage10, layout, { v -> fromF32(v.toF32()) })
                    Ok(CpuStorage.f16(data10))
                }
                matchVal.first is F16 && matchVal.second == DType.F16 -> {
                    val (storage11) = matchVal.first as F16
                    val data11 = unaryMap(storage11, layout, { v -> v })
                    Ok(CpuStorage.f16(data11))
                }
                matchVal.first is F32 && matchVal.second == DType.F16 -> {
                    val (storage12) = matchVal.first as F32
                    val data12 = unaryMap(storage12, layout, fromF32)
                    Ok(CpuStorage.f16(data12))
                }
                matchVal.first is F64 && matchVal.second == DType.F16 -> {
                    val (storage13) = matchVal.first as F64
                    val data13 = unaryMap(storage13, layout, fromF64)
                    Ok(CpuStorage.f16(data13))
                }
                matchVal.first is U8 && matchVal.second == DType.F32 -> {
                    val (storage14) = matchVal.first as U8
                    val data14 = unaryMap(storage14, layout, { v -> v.toFloat() })
                    Ok(CpuStorage.f32(data14))
                }
                matchVal.first is U32 && matchVal.second == DType.F32 -> {
                    val (storage15) = matchVal.first as U32
                    val data15 = unaryMap(storage15, layout, { v -> v.toFloat() })
                    Ok(CpuStorage.f32(data15))
                }
                matchVal.first is I64 && matchVal.second == DType.F32 -> {
                    val (storage16) = matchVal.first as I64
                    val data16 = unaryMap(storage16, layout, { v -> v.toFloat() })
                    Ok(CpuStorage.f32(data16))
                }
                matchVal.first is Bf16 && matchVal.second == DType.F32 -> {
                    val (storage17) = matchVal.first as Bf16
                    val data17 = unaryMap(storage17, layout, { v -> v.toF32() })
                    Ok(CpuStorage.f32(data17))
                }
                matchVal.first is F16 && matchVal.second == DType.F32 -> {
                    val (storage18) = matchVal.first as F16
                    val data18 = unaryMap(storage18, layout, { v -> v.toF32() })
                    Ok(CpuStorage.f32(data18))
                }
                matchVal.first is F32 && matchVal.second == DType.F32 -> {
                    val (storage19) = matchVal.first as F32
                    val data19 = unaryMap(storage19, layout, { v -> v })
                    Ok(CpuStorage.f32(data19))
                }
                matchVal.first is F64 && matchVal.second == DType.F32 -> {
                    val (storage20) = matchVal.first as F64
                    val data20 = unaryMap(storage20, layout, { v -> v.toFloat() })
                    Ok(CpuStorage.f32(data20))
                }
                matchVal.first is U8 && matchVal.second == DType.U8 -> {
                    val (storage21) = matchVal.first as U8
                    val data21 = unaryMap(storage21, layout, { v -> v })
                    Ok(CpuStorage.u8(data21))
                }
                matchVal.first is Bf16 && matchVal.second == DType.U8 -> {
                    val (storage22) = matchVal.first as Bf16
                    val data22 = unaryMap(storage22, layout, { v -> v.toF32().toUByte() })
                    Ok(CpuStorage.u8(data22))
                }
                matchVal.first is F16 && matchVal.second == DType.U8 -> {
                    val (storage23) = matchVal.first as F16
                    val data23 = unaryMap(storage23, layout, { v -> v.toF32().toUByte() })
                    Ok(CpuStorage.u8(data23))
                }
                matchVal.first is F32 && matchVal.second == DType.U8 -> {
                    val (storage24) = matchVal.first as F32
                    val data24 = unaryMap(storage24, layout, { v -> v.toUByte() })
                    Ok(CpuStorage.u8(data24))
                }
                matchVal.first is F64 && matchVal.second == DType.U8 -> {
                    val (storage25) = matchVal.first as F64
                    val data25 = unaryMap(storage25, layout, { v -> v.toUByte() })
                    Ok(CpuStorage.u8(data25))
                }
                matchVal.first is U32 && matchVal.second == DType.U8 -> {
                    val (storage26) = matchVal.first as U32
                    val data26 = unaryMap(storage26, layout, { v -> v.toUByte() })
                    Ok(CpuStorage.u8(data26))
                }
                matchVal.first is I64 && matchVal.second == DType.U8 -> {
                    val (storage27) = matchVal.first as I64
                    val data27 = unaryMap(storage27, layout, { v -> v.toUByte() })
                    Ok(CpuStorage.u8(data27))
                }
                matchVal.first is U8 && matchVal.second == DType.U32 -> {
                    val (storage28) = matchVal.first as U8
                    val data28 = unaryMap(storage28, layout, { v -> v.toUInt() })
                    Ok(CpuStorage.u32(data28))
                }
                matchVal.first is U32 && matchVal.second == DType.U32 -> {
                    val (storage29) = matchVal.first as U32
                    val data29 = unaryMap(storage29, layout, { v -> v })
                    Ok(CpuStorage.u32(data29))
                }
                matchVal.first is I64 && matchVal.second == DType.U32 -> {
                    val (storage30) = matchVal.first as I64
                    val data30 = unaryMap(storage30, layout, { v -> v.toUInt() })
                    Ok(CpuStorage.u32(data30))
                }
                matchVal.first is Bf16 && matchVal.second == DType.U32 -> {
                    val (storage31) = matchVal.first as Bf16
                    val data31 = unaryMap(storage31, layout, { v -> v.toF32().toUInt() })
                    Ok(CpuStorage.u32(data31))
                }
                matchVal.first is F16 && matchVal.second == DType.U32 -> {
                    val (storage32) = matchVal.first as F16
                    val data32 = unaryMap(storage32, layout, { v -> v.toF32().toUInt() })
                    Ok(CpuStorage.u32(data32))
                }
                matchVal.first is F32 && matchVal.second == DType.U32 -> {
                    val (storage33) = matchVal.first as F32
                    val data33 = unaryMap(storage33, layout, { v -> v.toUInt() })
                    Ok(CpuStorage.u32(data33))
                }
                matchVal.first is F64 && matchVal.second == DType.U32 -> {
                    val (storage34) = matchVal.first as F64
                    val data34 = unaryMap(storage34, layout, { v -> v.toUInt() })
                    Ok(CpuStorage.u32(data34))
                }
                matchVal.first is U8 && matchVal.second == DType.I64 -> {
                    val (storage35) = matchVal.first as U8
                    val data35 = unaryMap(storage35, layout, { v -> v.toLong() })
                    Ok(CpuStorage.i64(data35))
                }
                matchVal.first is U32 && matchVal.second == DType.I64 -> {
                    val (storage36) = matchVal.first as U32
                    val data36 = unaryMap(storage36, layout, { v -> v.toLong() })
                    Ok(CpuStorage.i64(data36))
                }
                matchVal.first is I64 && matchVal.second == DType.I64 -> {
                    val (storage37) = matchVal.first as I64
                    val data37 = unaryMap(storage37, layout, { v -> v })
                    Ok(CpuStorage.i64(data37))
                }
                matchVal.first is Bf16 && matchVal.second == DType.I64 -> {
                    val (storage38) = matchVal.first as Bf16
                    val data38 = unaryMap(storage38, layout, { v -> v.toF32().toLong() })
                    Ok(CpuStorage.i64(data38))
                }
                matchVal.first is F16 && matchVal.second == DType.I64 -> {
                    val (storage39) = matchVal.first as F16
                    val data39 = unaryMap(storage39, layout, { v -> v.toF32().toLong() })
                    Ok(CpuStorage.i64(data39))
                }
                matchVal.first is F32 && matchVal.second == DType.I64 -> {
                    val (storage40) = matchVal.first as F32
                    val data40 = unaryMap(storage40, layout, { v -> v.toLong() })
                    Ok(CpuStorage.i64(data40))
                }
                matchVal.first is F64 && matchVal.second == DType.I64 -> {
                    val (storage41) = matchVal.first as F64
                    val data41 = unaryMap(storage41, layout, { v -> v.toLong() })
                    Ok(CpuStorage.i64(data41))
                }
                matchVal.first is U8 && matchVal.second == DType.F64 -> {
                    val (storage42) = matchVal.first as U8
                    val data42 = unaryMap(storage42, layout, { v -> v.toDouble() })
                    Ok(CpuStorage.f64(data42))
                }
                matchVal.first is U32 && matchVal.second == DType.F64 -> {
                    val (storage43) = matchVal.first as U32
                    val data43 = unaryMap(storage43, layout, { v -> v.toDouble() })
                    Ok(CpuStorage.f64(data43))
                }
                matchVal.first is I64 && matchVal.second == DType.F64 -> {
                    val (storage44) = matchVal.first as I64
                    val data44 = unaryMap(storage44, layout, { v -> v.toDouble() })
                    Ok(CpuStorage.f64(data44))
                }
                matchVal.first is Bf16 && matchVal.second == DType.F64 -> {
                    val (storage45) = matchVal.first as Bf16
                    val data45 = unaryMap(storage45, layout, { v -> v.toF64() })
                    Ok(CpuStorage.f64(data45))
                }
                matchVal.first is F16 && matchVal.second == DType.F64 -> {
                    val (storage46) = matchVal.first as F16
                    val data46 = unaryMap(storage46, layout, { v -> v.toF64() })
                    Ok(CpuStorage.f64(data46))
                }
                matchVal.first is F32 && matchVal.second == DType.F64 -> {
                    val (storage47) = matchVal.first as F32
                    val data47 = unaryMap(storage47, layout, { v -> v.toDouble() })
                    Ok(CpuStorage.f64(data47))
                }
                matchVal.first is F64 && matchVal.second == DType.F64 -> {
                    val (storage48) = matchVal.first as F64
                    val data48 = unaryMap(storage48, layout, { v -> v })
                    Ok(CpuStorage.f64(data48))
                }
                // Conversions to F8E4M3
                matchVal.first is U8 && matchVal.second == DType.F8E4M3 -> {
                    val (storage49) = matchVal.first as U8
                    val data49 = unaryMap(storage49, layout, { v -> F8e4m3.fromF32(v.toFloat()) })
                    Ok(CpuStorage.f8e4m3(data49))
                }
                matchVal.first is U32 && matchVal.second == DType.F8E4M3 -> {
                    val (storage50) = matchVal.first as U32
                    val data50 = unaryMap(storage50, layout, { v -> F8e4m3.fromF32(v.toFloat()) })
                    Ok(CpuStorage.f8e4m3(data50))
                }
                matchVal.first is I64 && matchVal.second == DType.F8E4M3 -> {
                    val (storage51) = matchVal.first as I64
                    val data51 = unaryMap(storage51, layout, { v -> F8e4m3.fromF32(v.toFloat()) })
                    Ok(CpuStorage.f8e4m3(data51))
                }
                matchVal.first is Bf16 && matchVal.second == DType.F8E4M3 -> {
                    val (storage52) = matchVal.first as Bf16
                    val data52 = unaryMap(storage52, layout, { v -> F8e4m3.fromF32(v.toF32()) })
                    Ok(CpuStorage.f8e4m3(data52))
                }
                matchVal.first is F16 && matchVal.second == DType.F8E4M3 -> {
                    val (storage53) = matchVal.first as F16
                    val data53 = unaryMap(storage53, layout, { v -> F8e4m3.fromF32(v.toF32()) })
                    Ok(CpuStorage.f8e4m3(data53))
                }
                matchVal.first is F32 && matchVal.second == DType.F8E4M3 -> {
                    val (storage54) = matchVal.first as F32
                    val data54 = unaryMap(storage54, layout, F8e4m3::fromF32)
                    Ok(CpuStorage.f8e4m3(data54))
                }
                matchVal.first is F64 && matchVal.second == DType.F8E4M3 -> {
                    val (storage55) = matchVal.first as F64
                    val data55 = unaryMap(storage55, layout, F8e4m3::fromF64)
                    Ok(CpuStorage.f8e4m3(data55))
                }
                matchVal.first is F8e4m3 && matchVal.second == DType.F8E4M3 -> {
                    val (storage56) = matchVal.first as F8e4m3
                    val data56 = unaryMap(storage56, layout, { v -> v })
                    Ok(CpuStorage.f8e4m3(data56))
                }
                // Conversions from F8E4M3
                matchVal.first is F8e4m3 && matchVal.second == DType.U8 -> {
                    val (storage57) = matchVal.first as F8e4m3
                    val data57 = unaryMap(storage57, layout, { v -> v.toF32().toUByte() })
                    Ok(CpuStorage.u8(data57))
                }
                matchVal.first is F8e4m3 && matchVal.second == DType.U32 -> {
                    val (storage58) = matchVal.first as F8e4m3
                    val data58 = unaryMap(storage58, layout, { v -> v.toF32().toUInt() })
                    Ok(CpuStorage.u32(data58))
                }
                matchVal.first is F8e4m3 && matchVal.second == DType.I64 -> {
                    val (storage59) = matchVal.first as F8e4m3
                    val data59 = unaryMap(storage59, layout, { v -> v.toF32().toLong() })
                    Ok(CpuStorage.i64(data59))
                }
                matchVal.first is F8e4m3 && matchVal.second == DType.BF16 -> {
                    val (storage60) = matchVal.first as F8e4m3
                    val data60 = unaryMap(storage60, layout, { v -> fromF32(v.toF32()) })
                    Ok(CpuStorage.bf16(data60))
                }
                matchVal.first is F8e4m3 && matchVal.second == DType.F16 -> {
                    val (storage61) = matchVal.first as F8e4m3
                    val data61 = unaryMap(storage61, layout, { v -> fromF32(v.toF32()) })
                    Ok(CpuStorage.f16(data61))
                }
                matchVal.first is F8e4m3 && matchVal.second == DType.F32 -> {
                    val (storage62) = matchVal.first as F8e4m3
                    val data62 = unaryMap(storage62, layout, { v -> v.toF32() })
                    Ok(CpuStorage.f32(data62))
                }
                matchVal.first is F8e4m3 && matchVal.second == DType.F64 -> {
                    val (storage63) = matchVal.first as F8e4m3
                    val data63 = unaryMap(storage63, layout, { v -> v.toF64() })
                    Ok(CpuStorage.f64(data63))
                }
                // Conversions to I16
                matchVal.first is U8 && matchVal.second == DType.I16 -> {
                    val (storage64) = matchVal.first as U8
                    val data64 = unaryMap(storage64, layout, { v -> v.toShort() })
                    Ok(CpuStorage.i16(data64))
                }
                matchVal.first is U32 && matchVal.second == DType.I16 -> {
                    val (storage65) = matchVal.first as U32
                    val data65 = unaryMap(storage65, layout, { v -> v.toShort() })
                    Ok(CpuStorage.i16(data65))
                }
                matchVal.first is I16 && matchVal.second == DType.I16 -> {
                    val (storage66) = matchVal.first as I16
                    val data66 = unaryMap(storage66, layout, { v -> v })
                    Ok(CpuStorage.i16(data66))
                }
                matchVal.first is I32 && matchVal.second == DType.I16 -> {
                    val (storage67) = matchVal.first as I32
                    val data67 = unaryMap(storage67, layout, { v -> v.toShort() })
                    Ok(CpuStorage.i16(data67))
                }
                matchVal.first is I64 && matchVal.second == DType.I16 -> {
                    val (storage68) = matchVal.first as I64
                    val data68 = unaryMap(storage68, layout, { v -> v.toShort() })
                    Ok(CpuStorage.i16(data68))
                }
                matchVal.first is Bf16 && matchVal.second == DType.I16 -> {
                    val (storage69) = matchVal.first as Bf16
                    val data69 = unaryMap(storage69, layout, { v -> v.toF32().toShort() })
                    Ok(CpuStorage.i16(data69))
                }
                matchVal.first is F16 && matchVal.second == DType.I16 -> {
                    val (storage70) = matchVal.first as F16
                    val data70 = unaryMap(storage70, layout, { v -> v.toF32().toShort() })
                    Ok(CpuStorage.i16(data70))
                }
                matchVal.first is F32 && matchVal.second == DType.I16 -> {
                    val (storage71) = matchVal.first as F32
                    val data71 = unaryMap(storage71, layout, { v -> v.toShort() })
                    Ok(CpuStorage.i16(data71))
                }
                matchVal.first is F64 && matchVal.second == DType.I16 -> {
                    val (storage72) = matchVal.first as F64
                    val data72 = unaryMap(storage72, layout, { v -> v.toShort() })
                    Ok(CpuStorage.i16(data72))
                }
                matchVal.first is F8e4m3 && matchVal.second == DType.I16 -> {
                    val (storage73) = matchVal.first as F8e4m3
                    val data73 = unaryMap(storage73, layout, { v -> v.toF32().toShort() })
                    Ok(CpuStorage.i16(data73))
                }
                // Conversions to I32
                matchVal.first is U8 && matchVal.second == DType.I32 -> {
                    val (storage74) = matchVal.first as U8
                    val data74 = unaryMap(storage74, layout, { v -> v.toInt() })
                    Ok(CpuStorage.i32(data74))
                }
                matchVal.first is U32 && matchVal.second == DType.I32 -> {
                    val (storage75) = matchVal.first as U32
                    val data75 = unaryMap(storage75, layout, { v -> v.toInt() })
                    Ok(CpuStorage.i32(data75))
                }
                matchVal.first is I16 && matchVal.second == DType.I32 -> {
                    val (storage76) = matchVal.first as I16
                    val data76 = unaryMap(storage76, layout, { v -> v.toInt() })
                    Ok(CpuStorage.i32(data76))
                }
                matchVal.first is I32 && matchVal.second == DType.I32 -> {
                    val (storage77) = matchVal.first as I32
                    val data77 = unaryMap(storage77, layout, { v -> v })
                    Ok(CpuStorage.i32(data77))
                }
                matchVal.first is I64 && matchVal.second == DType.I32 -> {
                    val (storage78) = matchVal.first as I64
                    val data78 = unaryMap(storage78, layout, { v -> v.toInt() })
                    Ok(CpuStorage.i32(data78))
                }
                matchVal.first is Bf16 && matchVal.second == DType.I32 -> {
                    val (storage79) = matchVal.first as Bf16
                    val data79 = unaryMap(storage79, layout, { v -> v.toF32().toInt() })
                    Ok(CpuStorage.i32(data79))
                }
                matchVal.first is F16 && matchVal.second == DType.I32 -> {
                    val (storage80) = matchVal.first as F16
                    val data80 = unaryMap(storage80, layout, { v -> v.toF32().toInt() })
                    Ok(CpuStorage.i32(data80))
                }
                matchVal.first is F32 && matchVal.second == DType.I32 -> {
                    val (storage81) = matchVal.first as F32
                    val data81 = unaryMap(storage81, layout, { v -> v.toInt() })
                    Ok(CpuStorage.i32(data81))
                }
                matchVal.first is F64 && matchVal.second == DType.I32 -> {
                    val (storage82) = matchVal.first as F64
                    val data82 = unaryMap(storage82, layout, { v -> v.toInt() })
                    Ok(CpuStorage.i32(data82))
                }
                matchVal.first is F8e4m3 && matchVal.second == DType.I32 -> {
                    val (storage83) = matchVal.first as F8e4m3
                    val data83 = unaryMap(storage83, layout, { v -> v.toF32().toInt() })
                    Ok(CpuStorage.i32(data83))
                }
                // Conversions from I16
                matchVal.first is I16 && matchVal.second == DType.U8 -> {
                    val (storage84) = matchVal.first as I16
                    val data84 = unaryMap(storage84, layout, { v -> v.toUByte() })
                    Ok(CpuStorage.u8(data84))
                }
                matchVal.first is I16 && matchVal.second == DType.U32 -> {
                    val (storage85) = matchVal.first as I16
                    val data85 = unaryMap(storage85, layout, { v -> v.toUInt() })
                    Ok(CpuStorage.u32(data85))
                }
                matchVal.first is I16 && matchVal.second == DType.I64 -> {
                    val (storage86) = matchVal.first as I16
                    val data86 = unaryMap(storage86, layout, { v -> v.toLong() })
                    Ok(CpuStorage.i64(data86))
                }
                matchVal.first is I16 && matchVal.second == DType.BF16 -> {
                    val (storage87) = matchVal.first as I16
                    val data87 = unaryMap(storage87, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.bf16(data87))
                }
                matchVal.first is I16 && matchVal.second == DType.F16 -> {
                    val (storage88) = matchVal.first as I16
                    val data88 = unaryMap(storage88, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.f16(data88))
                }
                matchVal.first is I16 && matchVal.second == DType.F32 -> {
                    val (storage89) = matchVal.first as I16
                    val data89 = unaryMap(storage89, layout, { v -> v.toFloat() })
                    Ok(CpuStorage.f32(data89))
                }
                matchVal.first is I16 && matchVal.second == DType.F64 -> {
                    val (storage90) = matchVal.first as I16
                    val data90 = unaryMap(storage90, layout, { v -> v.toDouble() })
                    Ok(CpuStorage.f64(data90))
                }
                matchVal.first is I16 && matchVal.second == DType.F8E4M3 -> {
                    val (storage91) = matchVal.first as I16
                    val data91 = unaryMap(storage91, layout, { v -> F8e4m3.fromF32(v.toFloat()) })
                    Ok(CpuStorage.f8e4m3(data91))
                }
                // Conversions from I32
                matchVal.first is I32 && matchVal.second == DType.U8 -> {
                    val (storage92) = matchVal.first as I32
                    val data92 = unaryMap(storage92, layout, { v -> v.toUByte() })
                    Ok(CpuStorage.u8(data92))
                }
                matchVal.first is I32 && matchVal.second == DType.U32 -> {
                    val (storage93) = matchVal.first as I32
                    val data93 = unaryMap(storage93, layout, { v -> v.toUInt() })
                    Ok(CpuStorage.u32(data93))
                }
                matchVal.first is I32 && matchVal.second == DType.I64 -> {
                    val (storage94) = matchVal.first as I32
                    val data94 = unaryMap(storage94, layout, { v -> v.toLong() })
                    Ok(CpuStorage.i64(data94))
                }
                matchVal.first is I32 && matchVal.second == DType.BF16 -> {
                    val (storage95) = matchVal.first as I32
                    val data95 = unaryMap(storage95, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.bf16(data95))
                }
                matchVal.first is I32 && matchVal.second == DType.F16 -> {
                    val (storage96) = matchVal.first as I32
                    val data96 = unaryMap(storage96, layout, { v -> fromF32(v.toFloat()) })
                    Ok(CpuStorage.f16(data96))
                }
                matchVal.first is I32 && matchVal.second == DType.F32 -> {
                    val (storage97) = matchVal.first as I32
                    val data97 = unaryMap(storage97, layout, { v -> v.toFloat() })
                    Ok(CpuStorage.f32(data97))
                }
                matchVal.first is I32 && matchVal.second == DType.F64 -> {
                    val (storage98) = matchVal.first as I32
                    val data98 = unaryMap(storage98, layout, { v -> v.toDouble() })
                    Ok(CpuStorage.f64(data98))
                }
                matchVal.first is I32 && matchVal.second == DType.F8E4M3 -> {
                    val (storage99) = matchVal.first as I32
                    val data99 = unaryMap(storage99, layout, { v -> F8e4m3.fromF32(v.toFloat()) })
                    Ok(CpuStorage.f8e4m3(data99))
                }
                // Dummy types - return error for all conversions to/from dummy types
                matchVal.second == DType.F6E2M3 -> {
                    Err(Error.UnsupportedDTypeForOp(dtype, "to_dtype").bt())
                }
                matchVal.second == DType.F6E3M2 -> {
                    Err(Error.UnsupportedDTypeForOp(dtype, "to_dtype").bt())
                }
                matchVal.second == DType.F4 -> {
                    Err(Error.UnsupportedDTypeForOp(dtype, "to_dtype").bt())
                }
                matchVal.second == DType.F8E8M0 -> {
                    Err(Error.UnsupportedDTypeForOp(dtype, "to_dtype").bt())
                }
                matchVal.first is F6e2m3 -> {
                    Err(Error.UnsupportedDTypeForOp(this.dtype(), "to_dtype").bt())
                }
                matchVal.first is F6e3m2 -> {
                    Err(Error.UnsupportedDTypeForOp(this.dtype(), "to_dtype").bt())
                }
                matchVal.first is F4 -> {
                    Err(Error.UnsupportedDTypeForOp(this.dtype(), "to_dtype").bt())
                }
                else -> {
                    Err(Error.UnsupportedDTypeForOp(this.dtype(), "to_dtype").bt())
                }
            }
        }
    }

    public fun reduceOp(op: ReduceOp, layout: Layout, reduceDims: List<Int>): Result<CpuStorage> {
        return when (op) {
            ReduceOp.Sum -> {
                val srcDims = layout.dims()
                val dstDims = srcDims.toVec()
                for (dim in reduceDims) {
                    dstDims[dim] = 1
                }
                val dstShape = Shape.from(dstDims)
                val reduceDims1 = reduceDims.toVec()
                // Sort the reduceDims as they have to be processed from left to right when converting the
                // indexes.
                reduceDims1.sort()
                val reduceDimsAndStride: Vec = reduceDims1.iter().map({ d -> Pair(srcDims[d], ((srcDims).slice(d + 1 until srcDims.size)).iter().product<Int>()) }).collect()
                ReduceSum(dstShape = dstShape, reduceDims = reduceDims1, reduceDimsAndStride = reduceDimsAndStride).map(this, layout)
            }
            ReduceOp.Min, ReduceOp.ArgMin, ReduceOp.Max, ReduceOp.ArgMax -> {
                val reduceDimIndex1 =
                    when {
                        reduceDims.size == 1 -> {
                            val reduceDimIndex = reduceDims[0]
                            reduceDimIndex
                        }
                        else -> {
                            val op1 =
                                when (op) {
                                    ReduceOp.Min -> "min"
                                    ReduceOp.ArgMin -> "argmin"
                                    ReduceOp.Max -> "max"
                                    ReduceOp.ArgMax -> "argmax"
                                    else -> error("unreachable")
                                }
                            val dims = reduceDims.toVec()
                            when (val tryResult = Err(Error.OnlySingleDimension(op = op1, dims = dims))) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        }
                    }
                val (useMin, returnIndex) =
                    when (op) {
                        ReduceOp.Min -> Pair(true, false)
                        ReduceOp.ArgMin -> Pair(true, true)
                        ReduceOp.Max -> Pair(false, false)
                        ReduceOp.ArgMax -> Pair(false, true)
                        else -> error("unreachable")
                    }
                ReduceIndex(reduceDimIndex = reduceDimIndex1, useMin = useMin, returnIndex = returnIndex).map(this, layout)
            }
        }
    }

    public fun cmp(op: CmpOp, rhs: CpuStorage, lhsL: Layout, rhsL: Layout): Result<CpuStorage> = Cmp(op).map(this, lhsL, rhs, rhsL)

    public fun affine(layout: Layout, mul: Double, add: Double): Result<CpuStorage> = Affine(mul, add).map(this, layout)

    public fun avgPool2d(layout: Layout, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): Result<CpuStorage> = AvgPool2D(kernelSize, stride).map(this, layout)

    public fun maxPool2d(layout: Layout, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): Result<CpuStorage> = MaxPool2D(kernelSize, stride).map(this, layout)

    public fun upsampleNearest1d(layout: Layout, sz: Int): Result<CpuStorage> = UpsampleNearest1D(sz).map(this, layout)

    public fun upsampleNearest2d(layout: Layout, h: Int, w: Int): Result<CpuStorage> = UpsampleNearest2D(h, w).map(this, layout)

    public fun upsampleBilinear2d(layout: Layout, h: Int, w: Int, alignCorners: Boolean, scaleH: Double?, scaleW: Double?): Result<CpuStorage> = UpsampleBilinear2D(targetH = h, targetW = w, alignCorners = alignCorners, scaleHFactor = scaleH, scaleWFactor = scaleW).map(this, layout)

    public fun powf(layout: Layout, e: Double): Result<CpuStorage> {
        // Note: Have some generic map for functions that apply on numTraits.Float elements.
        return when (this) {
            is Bf16 -> {
                val (storage) = this
                val data = unaryMap(storage, layout, { v -> v.powf(fromF64(e)) })
                Ok(CpuStorage.bf16(data))
            }
            is F16 -> {
                val (storage1) = this
                val data1 = unaryMap(storage1, layout, { v -> v.powf(fromF64(e)) })
                Ok(CpuStorage.f16(data1))
            }
            is F32 -> {
                val (storage2) = this
                val data2 = unaryMap(storage2, layout, { v -> v.powf(e.toFloat()) })
                Ok(CpuStorage.f32(data2))
            }
            is F64 -> {
                val (storage3) = this
                val data3 = unaryMap(storage3, layout, { v -> v.powf(e) })
                Ok(CpuStorage.f64(data3))
            }
            is F8e4m3 -> {
                val (storage4) = this
                val data4 = unaryMap(storage4, layout, { v -> v.powf(F8e4m3.fromF64(e)) })
                Ok(CpuStorage.f8e4m3(data4))
            }
            is U8 -> Err(Error.UnsupportedDTypeForOp(DType.U8, "powf").bt())
            is U32 -> Err(Error.UnsupportedDTypeForOp(DType.U32, "powf").bt())
            is I16 -> Err(Error.UnsupportedDTypeForOp(DType.I16, "powf").bt())
            is I32 -> Err(Error.UnsupportedDTypeForOp(DType.I32, "powf").bt())
            is I64 -> Err(Error.UnsupportedDTypeForOp(DType.I64, "powf").bt())
            is F6e2m3 -> Err(Error.UnsupportedDTypeForOp(DType.F6E2M3, "powf").bt())
            is F6e3m2 -> Err(Error.UnsupportedDTypeForOp(DType.F6E3M2, "powf").bt())
            is F4 -> Err(Error.UnsupportedDTypeForOp(DType.F4, "powf").bt())
            is F8e8m0 -> Err(Error.UnsupportedDTypeForOp(DType.F8E8M0, "powf").bt())
        }
    }

    public fun elu(layout: Layout, alpha: Double): Result<CpuStorage> {
        // Note: Have some generic map for functions that apply on numTraits.Float elements.
        return when (this) {
            is Bf16 -> {
                val (storage) = this
                val data = unaryMap(storage, layout, { v -> elu(v, fromF64(alpha)) })
                Ok(CpuStorage.bf16(data))
            }
            is F16 -> {
                val (storage1) = this
                val data1 = unaryMap(storage1, layout, { v -> elu(v, fromF64(alpha)) })
                Ok(CpuStorage.f16(data1))
            }
            is F32 -> {
                val (storage2) = this
                val data2 = unaryMap(storage2, layout, { v -> elu(v, Float.fromF64(alpha)) })
                Ok(CpuStorage.f32(data2))
            }
            is F64 -> {
                val (storage3) = this
                val data3 = unaryMap(storage3, layout, { v -> elu(v, alpha) })
                Ok(CpuStorage.f64(data3))
            }
            is F8e4m3 -> {
                val (storage4) = this
                val data4 = unaryMap(storage4, layout, { v -> elu(v, F8e4m3.fromF64(alpha)) })
                Ok(CpuStorage.f8e4m3(data4))
            }
            is U8 -> Err(Error.UnsupportedDTypeForOp(DType.U8, "elu").bt())
            is U32 -> Err(Error.UnsupportedDTypeForOp(DType.U32, "elu").bt())
            is I16 -> Err(Error.UnsupportedDTypeForOp(DType.I16, "elu").bt())
            is I32 -> Err(Error.UnsupportedDTypeForOp(DType.I32, "elu").bt())
            is I64 -> Err(Error.UnsupportedDTypeForOp(DType.I64, "elu").bt())
            is F6e2m3 -> Err(Error.UnsupportedDTypeForOp(DType.F6E2M3, "elu").bt())
            is F6e3m2 -> Err(Error.UnsupportedDTypeForOp(DType.F6E3M2, "elu").bt())
            is F4 -> Err(Error.UnsupportedDTypeForOp(DType.F4, "elu").bt())
            is F8e8m0 -> Err(Error.UnsupportedDTypeForOp(DType.F8E8M0, "elu").bt())
        }
    }

    public fun <B : UnaryOpT> unaryImpl(layout: Layout, bf16BFn: (Any?) -> Any?, bf16VecBFn: (Any?) -> Any?, f16BFn: (Any?) -> Any?, f16VecBFn: (Any?) -> Any?, f32BFn: (Any?) -> Any?, f32VecBFn: (Any?) -> Any?, f64BFn: (Any?) -> Any?, f64VecBFn: (Any?) -> Any?, u8BFn: (Any?) -> Any?, u32BFn: (Any?) -> Any?, i16BFn: (Any?) -> Any?, i32BFn: (Any?) -> Any?, i64BFn: (Any?) -> Any?, f8e4m3BFn: (Any?) -> Any?): Result<CpuStorage> =
        when (this) {
            is Bf16 -> {
                val (storage) = this
                val data = unaryMapVecPar(storage, layout, bf16BFn, bf16VecBFn)
                Ok(CpuStorage.bf16(data))
            }
            is F16 -> {
                val (storage1) = this
                val data1 = unaryMapVecPar(storage1, layout, f16BFn, f16VecBFn)
                Ok(CpuStorage.f16(data1))
            }
            is F32 -> {
                val (storage2) = this
                val data2 = unaryMapVecPar(storage2, layout, f32BFn, f32VecBFn)
                Ok(CpuStorage.f32(data2))
            }
            is F64 -> {
                val (storage3) = this
                val data3 = unaryMapVecPar(storage3, layout, f64BFn, f64VecBFn)
                Ok(CpuStorage.f64(data3))
            }
            is U8 -> {
                val (storage4) = this
                val data4 = unaryMap(storage4, layout, u8BFn)
                Ok(CpuStorage.u8(data4))
            }
            is U32 -> {
                val (storage5) = this
                val data5 = unaryMap(storage5, layout, u32BFn)
                Ok(CpuStorage.u32(data5))
            }
            is I16 -> {
                val (storage6) = this
                val data6 = unaryMap(storage6, layout, i16BFn)
                Ok(CpuStorage.i16(data6))
            }
            is I32 -> {
                val (storage7) = this
                val data7 = unaryMap(storage7, layout, i32BFn)
                Ok(CpuStorage.i32(data7))
            }
            is I64 -> {
                val (storage8) = this
                val data8 = unaryMap(storage8, layout, i64BFn)
                Ok(CpuStorage.i64(data8))
            }
            is F8e4m3 -> {
                val (storage9) = this
                val data9 = unaryMap(storage9, layout, f8e4m3BFn)
                Ok(CpuStorage.f8e4m3(data9))
            }
            is F6e2m3 -> Err(Error.UnsupportedDTypeForOp(DType.F6E2M3, "unary").bt())
            is F6e3m2 -> Err(Error.UnsupportedDTypeForOp(DType.F6E3M2, "unary").bt())
            is F4 -> Err(Error.UnsupportedDTypeForOp(DType.F4, "unary").bt())
            is F8e8m0 -> Err(Error.UnsupportedDTypeForOp(DType.F8E8M0, "unary").bt())
        }

    public fun <B : BinaryOpT> binaryImpl(rhs: CpuStorage, lhsL: Layout, rhsL: Layout, bf16BFn: (Any?) -> Any?, bf16VecBFn: (Any?) -> Any?, bf16ScalarVecBFn: (Any?) -> Any?, f16BFn: (Any?) -> Any?, f16VecBFn: (Any?) -> Any?, f16ScalarVecBFn: (Any?) -> Any?, f32BFn: (Any?) -> Any?, f32VecBFn: (Any?) -> Any?, f32ScalarVecBFn: (Any?) -> Any?, f64BFn: (Any?) -> Any?, f64VecBFn: (Any?) -> Any?, f64ScalarVecBFn: (Any?) -> Any?, u32BFn: (Any?) -> Any?, u32VecBFn: (Any?) -> Any?, u32ScalarVecBFn: (Any?) -> Any?, i16BFn: (Any?) -> Any?, i16VecBFn: (Any?) -> Any?, i16ScalarVecBFn: (Any?) -> Any?, i32BFn: (Any?) -> Any?, i32VecBFn: (Any?) -> Any?, i32ScalarVecBFn: (Any?) -> Any?, i64BFn: (Any?) -> Any?, i64VecBFn: (Any?) -> Any?, i64ScalarVecBFn: (Any?) -> Any?, u8BFn: (Any?) -> Any?, u8VecBFn: (Any?) -> Any?, u8ScalarVecBFn: (Any?) -> Any?, f8e4m3BFn: (Any?) -> Any?): Result<CpuStorage> =
        run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first is Bf16 && matchVal.second is Bf16 -> {
                    val (lhs) = matchVal.first as Bf16
                    val (rhs1) = matchVal.second as Bf16
                    val data = binaryMapVecPar(lhsL, rhsL, lhs, rhs1, bf16BFn, bf16VecBFn, bf16ScalarVecBFn)
                    Ok(CpuStorage.bf16(data))
                }
                matchVal.first is F16 && matchVal.second is F16 -> {
                    val (lhs1) = matchVal.first as F16
                    val (rhs2) = matchVal.second as F16
                    val data1 = binaryMapVecPar(lhsL, rhsL, lhs1, rhs2, f16BFn, f16VecBFn, f16ScalarVecBFn)
                    Ok(CpuStorage.f16(data1))
                }
                matchVal.first is F32 && matchVal.second is F32 -> {
                    val (lhs2) = matchVal.first as F32
                    val (rhs3) = matchVal.second as F32
                    val data2 = binaryMapVecPar(lhsL, rhsL, lhs2, rhs3, f32BFn, f32VecBFn, f32ScalarVecBFn)
                    Ok(CpuStorage.f32(data2))
                }
                matchVal.first is F64 && matchVal.second is F64 -> {
                    val (lhs3) = matchVal.first as F64
                    val (rhs4) = matchVal.second as F64
                    val data3 = binaryMapVecPar(lhsL, rhsL, lhs3, rhs4, f64BFn, f64VecBFn, f64ScalarVecBFn)
                    Ok(CpuStorage.f64(data3))
                }
                matchVal.first is U32 && matchVal.second is U32 -> {
                    val (lhs4) = matchVal.first as U32
                    val (rhs5) = matchVal.second as U32
                    val data4 = binaryMapVecPar(lhsL, rhsL, lhs4, rhs5, u32BFn, u32VecBFn, u32ScalarVecBFn)
                    Ok(CpuStorage.u32(data4))
                }
                matchVal.first is I16 && matchVal.second is I16 -> {
                    val (lhs5) = matchVal.first as I16
                    val (rhs6) = matchVal.second as I16
                    val data5 = binaryMapVecPar(lhsL, rhsL, lhs5, rhs6, i16BFn, i16VecBFn, i16ScalarVecBFn)
                    Ok(CpuStorage.i16(data5))
                }
                matchVal.first is I32 && matchVal.second is I32 -> {
                    val (lhs6) = matchVal.first as I32
                    val (rhs7) = matchVal.second as I32
                    val data6 = binaryMapVecPar(lhsL, rhsL, lhs6, rhs7, i32BFn, i32VecBFn, i32ScalarVecBFn)
                    Ok(CpuStorage.i32(data6))
                }
                matchVal.first is I64 && matchVal.second is I64 -> {
                    val (lhs7) = matchVal.first as I64
                    val (rhs8) = matchVal.second as I64
                    val data7 = binaryMapVecPar(lhsL, rhsL, lhs7, rhs8, i64BFn, i64VecBFn, i64ScalarVecBFn)
                    Ok(CpuStorage.i64(data7))
                }
                matchVal.first is U8 && matchVal.second is U8 -> {
                    val (lhs8) = matchVal.first as U8
                    val (rhs9) = matchVal.second as U8
                    val data8 = binaryMapVec(lhsL, rhsL, lhs8, rhs9, u8BFn, u8VecBFn, u8ScalarVecBFn)
                    Ok(CpuStorage.u8(data8))
                }
                matchVal.first is F8e4m3 && matchVal.second is F8e4m3 -> {
                    val (lhs9) = matchVal.first as F8e4m3
                    val (rhs10) = matchVal.second as F8e4m3
                    val data9 = binaryMap(lhsL, rhsL, lhs9, rhs10, f8e4m3BFn)
                    Ok(CpuStorage.f8e4m3(data9))
                }
                else -> {
                    // This should be covered by the dtype check above.
                    Err(Error.DTypeMismatchBinaryOp(lhs = this.dtype(), rhs = rhs.dtype(), op = B.NAME).bt())
                }
            }
        }

    public fun copy2d(dst: CpuStorage, d1: Int, d2: Int, srcS: Int, dstS: Int, srcO: Int, dstO: Int): Result<Unit> {
        run {
            val matchVal = Pair(this, dst)
            when {
                matchVal.first is U8 && matchVal.second is U8 -> {
                    val (src) = matchVal.first as U8
                    val (dst1) = matchVal.second as U8
                    copy2d(src, dst1, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is U32 && matchVal.second is U32 -> {
                    val (src1) = matchVal.first as U32
                    val (dst2) = matchVal.second as U32
                    copy2d(src1, dst2, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is I16 && matchVal.second is I16 -> {
                    val (src2) = matchVal.first as I16
                    val (dst3) = matchVal.second as I16
                    copy2d(src2, dst3, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is I32 && matchVal.second is I32 -> {
                    val (src3) = matchVal.first as I32
                    val (dst4) = matchVal.second as I32
                    copy2d(src3, dst4, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is I64 && matchVal.second is I64 -> {
                    val (src4) = matchVal.first as I64
                    val (dst5) = matchVal.second as I64
                    copy2d(src4, dst5, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is Bf16 && matchVal.second is Bf16 -> {
                    val (src5) = matchVal.first as Bf16
                    val (dst6) = matchVal.second as Bf16
                    copy2d(src5, dst6, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is F16 && matchVal.second is F16 -> {
                    val (src6) = matchVal.first as F16
                    val (dst7) = matchVal.second as F16
                    copy2d(src6, dst7, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is F32 && matchVal.second is F32 -> {
                    val (src7) = matchVal.first as F32
                    val (dst8) = matchVal.second as F32
                    copy2d(src7, dst8, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is F64 && matchVal.second is F64 -> {
                    val (src8) = matchVal.first as F64
                    val (dst9) = matchVal.second as F64
                    copy2d(src8, dst9, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is F8e4m3 && matchVal.second is F8e4m3 -> {
                    val (src9) = matchVal.first as F8e4m3
                    val (dst10) = matchVal.second as F8e4m3
                    copy2d(src9, dst10, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is F6e2m3 && matchVal.second is F6e2m3 -> {
                    val (src10) = matchVal.first as F6e2m3
                    val (dst11) = matchVal.second as F6e2m3
                    copy2d(src10, dst11, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is F6e3m2 && matchVal.second is F6e3m2 -> {
                    val (src11) = matchVal.first as F6e3m2
                    val (dst12) = matchVal.second as F6e3m2
                    copy2d(src11, dst12, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is F4 && matchVal.second is F4 -> {
                    val (src12) = matchVal.first as F4
                    val (dst13) = matchVal.second as F4
                    copy2d(src12, dst13, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is F8e8m0 && matchVal.second is F8e8m0 -> {
                    val (src13) = matchVal.first as F8e8m0
                    val (dst14) = matchVal.second as F8e8m0
                    copy2d(src13, dst14, d1, d2, srcS, dstS, srcO, dstO)
                }
                else -> {
                    val dst15 = matchVal.second
                    return Err(Error.DTypeMismatchBinaryOp(lhs = this.dtype(), rhs = dst15.dtype(), op = "copy2d").bt())
                }
            }
        }
        return Ok(Unit)
    }

    public fun copyStridedSrc(dst: CpuStorage, dstOffset: Int, srcL: Layout): Result<Unit> {
        run {
            val matchVal = Pair(this, dst)
            when {
                matchVal.first is U8 && matchVal.second is U8 -> {
                    val (src) = matchVal.first as U8
                    val (dst1) = matchVal.second as U8
                    copyStridedSrc(src, dst1, dstOffset, srcL)
                }
                matchVal.first is U32 && matchVal.second is U32 -> {
                    val (src1) = matchVal.first as U32
                    val (dst2) = matchVal.second as U32
                    copyStridedSrc(src1, dst2, dstOffset, srcL)
                }
                matchVal.first is I16 && matchVal.second is I16 -> {
                    val (src2) = matchVal.first as I16
                    val (dst3) = matchVal.second as I16
                    copyStridedSrc(src2, dst3, dstOffset, srcL)
                }
                matchVal.first is I32 && matchVal.second is I32 -> {
                    val (src3) = matchVal.first as I32
                    val (dst4) = matchVal.second as I32
                    copyStridedSrc(src3, dst4, dstOffset, srcL)
                }
                matchVal.first is I64 && matchVal.second is I64 -> {
                    val (src4) = matchVal.first as I64
                    val (dst5) = matchVal.second as I64
                    copyStridedSrc(src4, dst5, dstOffset, srcL)
                }
                matchVal.first is Bf16 && matchVal.second is Bf16 -> {
                    val (src5) = matchVal.first as Bf16
                    val (dst6) = matchVal.second as Bf16
                    copyStridedSrc(src5, dst6, dstOffset, srcL)
                }
                matchVal.first is F16 && matchVal.second is F16 -> {
                    val (src6) = matchVal.first as F16
                    val (dst7) = matchVal.second as F16
                    copyStridedSrc(src6, dst7, dstOffset, srcL)
                }
                matchVal.first is F32 && matchVal.second is F32 -> {
                    val (src7) = matchVal.first as F32
                    val (dst8) = matchVal.second as F32
                    copyStridedSrc(src7, dst8, dstOffset, srcL)
                }
                matchVal.first is F64 && matchVal.second is F64 -> {
                    val (src8) = matchVal.first as F64
                    val (dst9) = matchVal.second as F64
                    copyStridedSrc(src8, dst9, dstOffset, srcL)
                }
                matchVal.first is F8e4m3 && matchVal.second is F8e4m3 -> {
                    val (src9) = matchVal.first as F8e4m3
                    val (dst10) = matchVal.second as F8e4m3
                    copyStridedSrc(src9, dst10, dstOffset, srcL)
                }
                matchVal.first is F6e2m3 && matchVal.second is F6e2m3 -> {
                    val (src10) = matchVal.first as F6e2m3
                    val (dst11) = matchVal.second as F6e2m3
                    copyStridedSrc(src10, dst11, dstOffset, srcL)
                }
                matchVal.first is F6e3m2 && matchVal.second is F6e3m2 -> {
                    val (src11) = matchVal.first as F6e3m2
                    val (dst12) = matchVal.second as F6e3m2
                    copyStridedSrc(src11, dst12, dstOffset, srcL)
                }
                matchVal.first is F4 && matchVal.second is F4 -> {
                    val (src12) = matchVal.first as F4
                    val (dst13) = matchVal.second as F4
                    copyStridedSrc(src12, dst13, dstOffset, srcL)
                }
                matchVal.first is F8e8m0 && matchVal.second is F8e8m0 -> {
                    val (src13) = matchVal.first as F8e8m0
                    val (dst14) = matchVal.second as F8e8m0
                    copyStridedSrc(src13, dst14, dstOffset, srcL)
                }
                else -> {
                    val dst15 = matchVal.second
                    // This should be covered by the dtype check above.
                    return Err(Error.DTypeMismatchBinaryOp(lhs = this.dtype(), rhs = dst15.dtype(), op = "copy_strided").bt())
                }
            }
        }
        return Ok(Unit)
    }

    public fun whereCond(layout: Layout, t: CpuStorage, tL: Layout, f: CpuStorage, fL: Layout): Result<CpuStorage> =
        when (this) {
            is U8 -> {
                val (pred) = this
                WCond(pred, layout).map(t, tL, f, fL)
            }
            is U32 -> {
                val (pred1) = this
                WCond(pred1, layout).map(t, tL, f, fL)
            }
            is I16 -> {
                val (pred2) = this
                WCond(pred2, layout).map(t, tL, f, fL)
            }
            is I32 -> {
                val (pred3) = this
                WCond(pred3, layout).map(t, tL, f, fL)
            }
            is I64 -> {
                val (pred4) = this
                WCond(pred4, layout).map(t, tL, f, fL)
            }
            else -> Err(Error.UnsupportedDTypeForOp(this.dtype(), "where-cond"))
        }

    public fun conv1d(l: Layout, kernel: CpuStorage, kernelL: Layout, params: ParamsConv1D): Result<CpuStorage> {
        if (USE_IM2COL_CONV1D.inv()) {
            return Conv1D(params).map(this, l, kernel, kernelL)
        }
        val op = Im2Col1D(lK = params.kSize, padding = params.padding, stride = params.stride, dilation = params.dilation)
        val col =
            when (val tryResult = op.map(this, l)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val b = params.bSize
        val n = params.cOut
        val lOut = params.lOut()
        val k = op.lK * params.cIn
        val m = lOut
        val colL = Layout.contiguous(Triple(b, m, k))
        val res =
            if (kernelL.isContiguous()) {
                val kernelL1 =
                    when (
                        val tryResult =
                            when (val tryResult = Layout.contiguousWithOffset(Triple(1, n, k), kernelL.startOffset()).transpose(1, 2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.broadcastAs(Triple(b, k, n))
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = col.matmul(kernel, listOf(b, m, n, k), colL, kernelL1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            } else {
                // Make the kernel contiguous if not already the case.
                val kernelC =
                    when (val tryResult = this.device().allocUninit(kernelL.shape(), kernel.dtype())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = kernel.copyStridedSrc(kernelC, 0, kernelL)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
                val kernelL2 =
                    when (
                        val tryResult =
                            when (val tryResult = Layout.contiguousWithOffset(Triple(1, n, k), kernelL.startOffset()).transpose(1, 2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.broadcastAs(Triple(b, k, n))
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = col.matmul(kernel, listOf(b, m, n, k), colL, kernelL2)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        val resL =
            when (val tryResult = Layout.contiguous(Triple(b, lOut, params.cOut)).transpose(1, 2)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val resT =
            when (val tryResult = this.device().allocUninit(resL.shape(), res.dtype())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = res.copyStridedSrc(resT, 0, resL)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(resT)
    }

    public fun convTranspose1d(l: Layout, kernel: CpuStorage, kernelL: Layout, params: ParamsConvTranspose1D): Result<CpuStorage> {
        val canUseCol2im = kernelL.isContiguous() && params.dilation == 1 && params.padding == 0 && params.outputPadding == 0
        return if (USE_COL2IM_CONV1D_TR && canUseCol2im) {
            val (bSize, cIn, lIn) =
                when (val tryResult = l.shape().dims3()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val (cIn2, cOut, kSize) =
                when (val tryResult = kernelL.shape().dims3()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            if (!kernelL.isContiguous()) {
                bail("convtr1d: the second argument (kernel) has to be contiguous {kernel_l:?}")
            }
            if (cIn != cIn2) {
                bail("convtr1d: shape mismatch on c_in {:?} {:?}", l.shape(), kernelL.shape())
            }
            val col =
                run {
                    // This merges the last two dimensions of the kernel together.
                    val kernelLMm = Layout.new(Triple(bSize, cIn, kSize * cOut).into(), mutableListOf(0, kSize * cOut, 1), kernelL.startOffset())
                    when (
                        val tryResult =
                            this.matmul(
                                kernel,
                                listOf(bSize, lIn, cOut * kSize, cIn),
                                when (val tryResult = l.transpose(1, 2)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                },
                                kernelLMm,
                            )
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
            val colL = Layout.contiguous(listOf(bSize, lIn, cOut, kSize))
            Col2Im1D(stride = params.stride).map(col, colL)
        } else {
            ConvTranspose1D(params).map(this, l, kernel, kernelL)
        }
    }

    public fun conv2d(l: Layout, kernel: CpuStorage, kernelL: Layout, params: ParamsConv2D): Result<CpuStorage> = Conv2D(params).map(this, l, kernel, kernelL)

    public fun convTranspose2d(l: Layout, kernel: CpuStorage, kernelL: Layout, params: ParamsConvTranspose2D): Result<CpuStorage> = ConvTranspose2D(params).map(this, l, kernel, kernelL)

    public fun indexSelect(ids: CpuStorage, l: Layout, idsL: Layout, dim: Int): Result<CpuStorage> =
        when (ids) {
            is U8 -> {
                val (ids1) = ids
                IndexSelect(ids = ids1, idsL = idsL, dim = dim).map(this, l)
            }
            is U32 -> {
                val (ids2) = ids
                IndexSelect(ids = ids2, idsL = idsL, dim = dim).map(this, l)
            }
            is I64 -> {
                val (ids3) = ids
                IndexSelect(ids = ids3, idsL = idsL, dim = dim).map(this, l)
            }
            else -> Err(Error.UnsupportedDTypeForOp(this.dtype(), "index-select").bt())
        }

    public fun gather(l: Layout, ids: CpuStorage, idsL: Layout, dim: Int): Result<CpuStorage> =
        when (ids) {
            is U8 -> {
                val (ids1) = ids
                Gather(ids = ids1, idsL = idsL, dim = dim).map(this, l)
            }
            is U32 -> {
                val (ids2) = ids
                Gather(ids = ids2, idsL = idsL, dim = dim).map(this, l)
            }
            is I64 -> {
                val (ids3) = ids
                Gather(ids = ids3, idsL = idsL, dim = dim).map(this, l)
            }
            else -> Err(Error.UnsupportedDTypeForOp(this.dtype(), "gather").bt())
        }

    public fun scatterSet(l: Layout, ids: CpuStorage, idsL: Layout, src: CpuStorage, srcL: Layout, dim: Int): Result<Unit> =
        when (ids) {
            is U8 -> {
                val (ids1) = ids
                Scatter.new(ids1, idsL, dim).map(this, l, src, srcL)
            }
            is U32 -> {
                val (ids2) = ids
                Scatter.new(ids2, idsL, dim).map(this, l, src, srcL)
            }
            is I64 -> {
                val (ids3) = ids
                Scatter.new(ids3, idsL, dim).map(this, l, src, srcL)
            }
            else -> Err(Error.UnsupportedDTypeForOp(this.dtype(), "scatter").bt())
        }

    public fun scatterAddSet(l: Layout, ids: CpuStorage, idsL: Layout, src: CpuStorage, srcL: Layout, dim: Int): Result<Unit> =
        when (ids) {
            is U8 -> {
                val (ids1) = ids
                Scatter.new(ids1, idsL, dim).map(this, l, src, srcL)
            }
            is U32 -> {
                val (ids2) = ids
                Scatter.new(ids2, idsL, dim).map(this, l, src, srcL)
            }
            is I16 -> {
                val (ids3) = ids
                Scatter.new(ids3, idsL, dim).map(this, l, src, srcL)
            }
            is I32 -> {
                val (ids4) = ids
                Scatter.new(ids4, idsL, dim).map(this, l, src, srcL)
            }
            is I64 -> {
                val (ids5) = ids
                Scatter.new(ids5, idsL, dim).map(this, l, src, srcL)
            }
            else -> Err(Error.UnsupportedDTypeForOp(this.dtype(), "scatter-add").bt())
        }

    public fun indexAdd(l: Layout, ids: CpuStorage, idsL: Layout, src: CpuStorage, srcL: Layout, dim: Int): Result<CpuStorage> {
        return when (ids) {
            is U8 -> {
                val (ids1) = ids
                val ids2 =
                    when (val matchVal = idsL.contiguousOffsets()) {
                        null ->
                            when (val tryResult = Err(Error.RequiresContiguous(op = "index-add").bt())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        else -> {
                            val a = matchVal.first
                            val b = matchVal.second
                            ((ids1).slice(a until b))
                        }
                    }
                IndexAdd(ids = ids2, dim = dim).map(this, l, src, srcL)
            }
            is U32 -> {
                val (ids3) = ids
                val ids4 =
                    when (val matchVal1 = idsL.contiguousOffsets()) {
                        null ->
                            when (val tryResult = Err(Error.RequiresContiguous(op = "index-add").bt())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        else -> {
                            val a1 = matchVal1.first
                            val b1 = matchVal1.second
                            ((ids3).slice(a1 until b1))
                        }
                    }
                IndexAdd(ids = ids4, dim = dim).map(this, l, src, srcL)
            }
            is I16 -> {
                val (ids5) = ids
                val ids6 =
                    when (val matchVal2 = idsL.contiguousOffsets()) {
                        null ->
                            when (val tryResult = Err(Error.RequiresContiguous(op = "index-add").bt())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        else -> {
                            val a2 = matchVal2.first
                            val b2 = matchVal2.second
                            ((ids5).slice(a2 until b2))
                        }
                    }
                IndexAdd(ids = ids6, dim = dim).map(this, l, src, srcL)
            }
            is I32 -> {
                val (ids7) = ids
                val ids8 =
                    when (val matchVal3 = idsL.contiguousOffsets()) {
                        null ->
                            when (val tryResult = Err(Error.RequiresContiguous(op = "index-add").bt())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        else -> {
                            val a3 = matchVal3.first
                            val b3 = matchVal3.second
                            ((ids7).slice(a3 until b3))
                        }
                    }
                IndexAdd(ids = ids8, dim = dim).map(this, l, src, srcL)
            }
            is I64 -> {
                val (ids9) = ids
                val ids10 =
                    when (val matchVal4 = idsL.contiguousOffsets()) {
                        null ->
                            when (val tryResult = Err(Error.RequiresContiguous(op = "index-add").bt())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        else -> {
                            val a4 = matchVal4.first
                            val b4 = matchVal4.second
                            ((ids9).slice(a4 until b4))
                        }
                    }
                IndexAdd(ids = ids10, dim = dim).map(this, l, src, srcL)
            }
            else -> Err(Error.UnsupportedDTypeForOp(this.dtype(), "index-add").bt())
        }
    }

    public fun matmul(rhs: CpuStorage, bmnk: List<Any?>, lhsL: Layout, rhsL: Layout): Result<CpuStorage> {
        // no bf16 gemm kernel yet: widen to f32, multiply, narrow back
        if (Pair(this, rhs) != null) {
            val (lhsV) = Pair(this, rhs).first as Bf16
            val (rhsV) = Pair(this, rhs).second as Bf16
            val lhs32 = CpuStorage.f32(lhsV.toF32Vec())
            val rhs32 = CpuStorage.f32(rhsV.toF32Vec())
            val out =
                when (val tryResult = MatMul(bmnk).map(lhs32, lhsL, rhs32, rhsL)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val outV = out ?: bail("matmul dtype mismatch")
            return Ok(CpuStorage.bf16(Vec.fromF32Slice(outV)))
        }
        return MatMul(bmnk).map(this, lhsL, rhs, rhsL)
    }

    public fun device(): CpuDevice = CpuDevice()

    public fun tryClone(arg0: Layout): Result<CpuStorage> = Ok(this.clone())

    public fun toCpuStorage(): Result<CpuStorage> = Ok(this.clone())

    public fun constSet(s: Scalar, l: Layout): Result<Unit> {
        fun <T : WithDType> set(src: List<T>, l1: Layout, s1: T) {
            run {
                val matchVal = l1.stridedBlocks()
                when {
                    matchVal is StridedBlocks.SingleBlock -> {
                        val startOffset = matchVal.startOffset
                        val len = matchVal.len
                        ((src).slice(startOffset until startOffset + len)).fill(s1)
                    }
                    matchVal is StridedBlocks.UniformBlocks -> {
                        val startOffset1 = matchVal.startOffset
                        val blockLen = matchVal.blockLen
                        val count = matchVal.count
                        val srcStride = matchVal.srcStride
                        for (i in 0 until count) {
                            val start = startOffset1 + i * srcStride
                            ((src).slice(start until start + blockLen)).fill(s1)
                        }
                    }
                    matchVal is StridedBlocks.MultipleBlocks && matchVal.blockLen == 1 -> {
                        val blockStartIndex = matchVal.blockStartIndex
                        for (srcIndex in blockStartIndex) {
                            src[srcIndex] = s1
                        }
                    }
                    else -> {
                        val blockStartIndex1 = matchVal.blockStartIndex
                        val blockLen1 = matchVal.blockLen
                        for (srcIndex1 in blockStartIndex1) {
                            ((src).slice(srcIndex1 until srcIndex1 + blockLen1)).fill(s1)
                        }
                    }
                }
            }
        }
        run {
            val matchVal = Pair(this, s)
            when {
                matchVal.first is Bf16 && matchVal.second is Scalar.Bf16 -> {
                    val (storage) = matchVal.first as Bf16
                    val (v) = matchVal.second as Scalar.Bf16
                    Set(storage, l, v)
                }
                matchVal.first is F16 && matchVal.second is Scalar.F16 -> {
                    val (storage1) = matchVal.first as F16
                    val (v1) = matchVal.second as Scalar.F16
                    Set(storage1, l, v1)
                }
                matchVal.first is F32 && matchVal.second is Scalar.F32 -> {
                    val (storage2) = matchVal.first as F32
                    val (v2) = matchVal.second as Scalar.F32
                    Set(storage2, l, v2)
                }
                matchVal.first is F64 && matchVal.second is Scalar.F64 -> {
                    val (storage3) = matchVal.first as F64
                    val (v3) = matchVal.second as Scalar.F64
                    Set(storage3, l, v3)
                }
                matchVal.first is U8 && matchVal.second is Scalar.U8 -> {
                    val (storage4) = matchVal.first as U8
                    val (v4) = matchVal.second as Scalar.U8
                    Set(storage4, l, v4)
                }
                matchVal.first is U32 && matchVal.second is Scalar.U32 -> {
                    val (storage5) = matchVal.first as U32
                    val (v5) = matchVal.second as Scalar.U32
                    Set(storage5, l, v5)
                }
                matchVal.first is I16 && matchVal.second is Scalar.I16 -> {
                    val (storage6) = matchVal.first as I16
                    val (v6) = matchVal.second as Scalar.I16
                    Set(storage6, l, v6)
                }
                matchVal.first is I32 && matchVal.second is Scalar.I32 -> {
                    val (storage7) = matchVal.first as I32
                    val (v7) = matchVal.second as Scalar.I32
                    Set(storage7, l, v7)
                }
                matchVal.first is I64 && matchVal.second is Scalar.I64 -> {
                    val (storage8) = matchVal.first as I64
                    val (v8) = matchVal.second as Scalar.I64
                    Set(storage8, l, v8)
                }
                matchVal.first is F8e4m3 && matchVal.second is Scalar.F8e4m3 -> {
                    val (storage9) = matchVal.first as F8e4m3
                    val (v9) = matchVal.second as Scalar.F8e4m3
                    Set(storage9, l, v9)
                }
                // Dummy types don't support scalar operations
                matchVal.first is F6e2m3 -> {
                    bail("const_set not supported for dummy type F6E2M3")
                }
                matchVal.first is F6e3m2 -> {
                    bail("const_set not supported for dummy type F6E3M2")
                }
                matchVal.first is F4 -> {
                    bail("const_set not supported for dummy type F4")
                }
                matchVal.first is F8e8m0 -> {
                    bail("const_set not supported for dummy type F8E8M0")
                }
                else -> {
                    val st = matchVal.first
                    val s1 = matchVal.second
                    bail("const_set dtype mismatch, expected {:?} but got {:?}", st.dtype(), s1)
                }
            }
        }
        return Ok(Unit)
    }

    public companion object {
        public fun concat(storages: List<CpuStorage>): Result<CpuStorage> {
            val storage0 = storages[0]
            val s =
                when (storage0) {
                    is U8 -> {
                        val storages1 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is U8 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.u8(storages1)
                    }
                    is U32 -> {
                        val storages2 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is U32 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.u32(storages2)
                    }
                    is I16 -> {
                        val storages3 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is I16 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.i16(storages3)
                    }
                    is I32 -> {
                        val storages4 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is I32 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.i32(storages4)
                    }
                    is I64 -> {
                        val storages5 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is I64 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.i64(storages5)
                    }
                    is Bf16 -> {
                        val storages6 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is Bf16 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.bf16(storages6)
                    }
                    is F16 -> {
                        val storages7 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is F16 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.f16(storages7)
                    }
                    is F32 -> {
                        val storages8 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is F32 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.f32(storages8)
                    }
                    is F64 -> {
                        val storages9 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is F64 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.f64(storages9)
                    }
                    is F8e4m3 -> {
                        val storages10 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is F8e4m3 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.f8e4m3(storages10)
                    }
                    is F6e2m3 -> {
                        val storages11 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is F6e2m3 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.f6e2m3(storages11)
                    }
                    is F6e3m2 -> {
                        val storages12 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is F6e3m2 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.f6e3m2(storages12)
                    }
                    is F4 -> {
                        val storages13 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is F4 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.f4(storages13)
                    }
                    is F8e8m0 -> {
                        val storages14 =
                            when (
                                val tryResult =
                                    storages
                                        .map({ s ->
                                            when (s) {
                                                is F8e8m0 -> {
                                                    val (s1) = s
                                                    Ok(s1.asSlice())
                                                }
                                                else -> bail("dtype mismatch")
                                            }
                                        })
                                        .collect<Result<Vec>>()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.concat()
                        CpuStorage.f8e8m0(storages14)
                    }
                }
            return Ok(s)
        }
    }
}

public sealed class CpuStorageRef {
    public data class U8(
        public var value: List<UByte>,
    ) : CpuStorageRef()
    public data class U32(
        public var value: List<UInt>,
    ) : CpuStorageRef()
    public data class I16(
        public var value: List<Short>,
    ) : CpuStorageRef()
    public data class I32(
        public var value: List<Int>,
    ) : CpuStorageRef()
    public data class I64(
        public var value: List<Long>,
    ) : CpuStorageRef()
    public data class Bf16(
        public var value: List<Bf16>,
    ) : CpuStorageRef()
    public data class F16(
        public var value: List<F16>,
    ) : CpuStorageRef()
    public data class F32(
        public var value: List<Float>,
    ) : CpuStorageRef()
    public data class F64(
        public var value: List<Double>,
    ) : CpuStorageRef()
    public data class F8e4m3(
        public var value: List<F8e4m3>,
    ) : CpuStorageRef()

    // Dummy types that store raw bytes
    public data class F6e2m3(
        public var value: List<UByte>,
    ) : CpuStorageRef()
    public data class F6e3m2(
        public var value: List<UByte>,
    ) : CpuStorageRef()
    public data class F4(
        public var value: List<UByte>,
    ) : CpuStorageRef()
    public data class F8e8m0(
        public var value: List<UByte>,
    ) : CpuStorageRef()
}

public interface ElemUpdate {
    public fun <T : WithDType> f(dst: T, src: T)
}
