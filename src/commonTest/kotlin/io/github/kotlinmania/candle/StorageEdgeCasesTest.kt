package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.quantized.cuda.PaddedCudaSlice
import io.github.kotlinmania.float8.F8E4M3
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue
import io.github.kotlinmania.candle.quantized.cuda.QCudaStorage as KernelQCudaStorage

class StorageEdgeCasesTest {
    @Test
    fun scatterUnsignedSentinelsAndDimensionLengths() {
        val dstLayout = Layout.contiguous(Shape(2, 3))
        val srcLayout = Layout.contiguous(Shape(2, 2))
        val source = CpuStorage.F64(doubleArrayOf(1.0, 999.0, 2.0, 3.0))
        val indexes =
            listOf(
                CpuStorage.U8(ubyteArrayOf(2u, UByte.MAX_VALUE, 0u, 2u)),
                CpuStorage.U32(uintArrayOf(2u, UInt.MAX_VALUE, 0u, 2u)),
            )
        for (ids in indexes) {
            val dst = CpuStorage.F64(DoubleArray(6))
            dst.scatterSet(dstLayout, ids, srcLayout, source, srcLayout, 1)
            assertContentEquals(doubleArrayOf(0.0, 0.0, 1.0, 2.0, 0.0, 3.0), dst.data)
            dst.scatterAddSet(dstLayout, ids, srcLayout, source, srcLayout, 1)
            assertContentEquals(doubleArrayOf(0.0, 0.0, 2.0, 4.0, 0.0, 6.0), dst.data)
        }
    }

    @Test
    fun scatterAddSupportsSignedSmallIndexes() {
        val layout = Layout.contiguous(Shape(2, 2))
        val source = CpuStorage.I32(intArrayOf(1, 999, 2, 3))
        for (ids in listOf(
            CpuStorage.I16(shortArrayOf(1, Short.MAX_VALUE, 0, 1)),
            CpuStorage.I32(intArrayOf(1, Int.MAX_VALUE, 0, 1)),
        )) {
            val dst = CpuStorage.I32(IntArray(4))
            dst.scatterAddSet(layout, ids, layout, source, layout, 1)
            assertContentEquals(intArrayOf(0, 1, 2, 3), dst.data)
            assertFailsWith<CandleException.UnsupportedDTypeForOp> { dst.scatterSet(layout, ids, layout, source, layout, 1) }
        }
    }

    @Test
    fun indexAddUsesVectorIndexesAndBothDimensions() {
        val dst = Tensor.zeros(Shape(2, 2), DType.F32, Device.Cpu)
        val ids = Tensor.new(longArrayOf(1, 0, 1), Shape(3), Device.Cpu)
        val values = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val rows = Tensor.new(values, Shape(3, 2), Device.Cpu)
        val columns = Tensor.new(values, Shape(2, 3), Device.Cpu)
        assertContentEquals(floatArrayOf(3f, 4f, 6f, 8f), dst.indexAdd(ids, rows, 0).toVec1F32())
        assertContentEquals(floatArrayOf(2f, 4f, 5f, 10f), dst.indexAdd(ids, columns, 1).toVec1F32())
    }

    @Test
    fun indexAddPreservesLongsAndCopiesStridedDestination() {
        val large = 9007199254740993L
        val dst = CpuStorage.I64(longArrayOf(-1, 10, 20, 30, 40, -1))
        val dstLayout = Layout(Shape(2, 2), listOf(1, 2), 1)
        val src = CpuStorage.I64(longArrayOf(-9, large, 2, 3, 4, 5, 6))
        val srcLayout = Layout.contiguousWithOffset(Shape(2, 3), 1)
        val ids = CpuStorage.I64(longArrayOf(-9, 1, 0, 1))
        val idsLayout = Layout.contiguousWithOffset(Shape(3), 1)
        val result = dst.indexAdd(dstLayout, ids, idsLayout, src, srcLayout, 1) as CpuStorage.I64
        assertContentEquals(longArrayOf(12, large + 33, 25, 50), result.data)
        assertContentEquals(longArrayOf(-1, 10, 20, 30, 40, -1), dst.data)
    }

    @Test
    fun reductionsPreserveUnsignedAndSignedIntegers() {
        val layout = Layout.contiguous(Shape(2))
        val signed = CpuStorage.I32(intArrayOf(16777217, 2))
        val unsigned = CpuStorage.U32(uintArrayOf(4000000001u, 1u))
        assertContentEquals(intArrayOf(16777219), (signed.reduceOp(ReduceOp.Sum, layout, listOf(0)) as CpuStorage.I32).data)
        assertContentEquals(uintArrayOf(4000000002u), (unsigned.reduceOp(ReduceOp.Sum, layout, listOf(0)) as CpuStorage.U32).data)
        assertContentEquals(uintArrayOf(4000000001u), (unsigned.reduceOp(ReduceOp.Max, layout, listOf(0)) as CpuStorage.U32).data)
        val endpoints = CpuStorage.I64(longArrayOf(Long.MIN_VALUE, Long.MAX_VALUE))
        assertContentEquals(longArrayOf(Long.MIN_VALUE), (endpoints.reduceOp(ReduceOp.Min, layout, listOf(0)) as CpuStorage.I64).data)
        assertContentEquals(longArrayOf(Long.MAX_VALUE), (endpoints.reduceOp(ReduceOp.Max, layout, listOf(0)) as CpuStorage.I64).data)
    }

    @Test
    fun reductionsAcrossNonadjacentDimensionsAndBroadcastLayout() {
        val source = CpuStorage.I64(LongArray(12) { 16777217L + it })
        val layout = Layout.contiguous(Shape(2, 3, 2))
        assertContentEquals(longArrayOf(67108882, 67108890, 67108898), (source.reduceOp(ReduceOp.Sum, layout, listOf(2, 0)) as CpuStorage.I64).data)
        val broadcast = Layout(Shape(3, 2), listOf(0, 1), 1)
        assertContentEquals(longArrayOf(50331654, 50331657), (source.reduceOp(ReduceOp.Sum, broadcast, listOf(0)) as CpuStorage.I64).data)
        assertFailsWith<CandleException.DuplicateDimIndex> { source.reduceOp(ReduceOp.Sum, layout, listOf(1, 1)) }
        assertFailsWith<CandleException.DimOutOfRange> { source.reduceOp(ReduceOp.Sum, layout, listOf(3)) }
    }

    @Test
    fun extremaKeepFirstNanAndSignedZero() {
        val layout = Layout.contiguous(Shape(2))
        val nan = CpuStorage.F64(doubleArrayOf(Double.NaN, 4.0))
        assertTrue((nan.reduceOp(ReduceOp.Max, layout, listOf(0)) as CpuStorage.F64).data[0].isNaN())
        assertContentEquals(uintArrayOf(0u), (nan.reduceOp(ReduceOp.ArgMin, layout, listOf(0)) as CpuStorage.U32).data)
        val zeros = CpuStorage.F64(doubleArrayOf(-0.0, 0.0))
        assertEquals((-0.0).toBits(), (zeros.reduceOp(ReduceOp.Min, layout, listOf(0)) as CpuStorage.F64).data[0].toBits())
    }

    @Test
    fun encodedFloatReductionsAndScatterUseNumericValues() {
        val layout = Layout.contiguous(Shape(2))
        val ids = CpuStorage.I64(longArrayOf(0, 0))
        val cases =
            listOf(
                CpuStorage.F16(shortArrayOf(Float16Utils.floatToF16Bits(1f), Float16Utils.floatToF16Bits(2f))) to CpuStorage.F16(shortArrayOf(Float16Utils.floatToF16Bits(3f))),
                CpuStorage.BF16(shortArrayOf(Float16Utils.floatToBF16Bits(1f), Float16Utils.floatToBF16Bits(2f))) to CpuStorage.BF16(shortArrayOf(Float16Utils.floatToBF16Bits(3f))),
                CpuStorage.F8E4M3(byteArrayOf(F8E4M3.fromF32(1f).bits, F8E4M3.fromF32(2f).bits)) to CpuStorage.F8E4M3(byteArrayOf(F8E4M3.fromF32(3f).bits)),
            )
        for ((src, expected) in cases) {
            val sum = src.reduceOp(ReduceOp.Sum, layout, listOf(0))
            assertContentEquals(expected.asSlice(), sum.asSlice())
            assertContentEquals(uintArrayOf(1u), (src.reduceOp(ReduceOp.ArgMax, layout, listOf(0)) as CpuStorage.U32).data)
            val dst = CpuDevice.zerosImpl(Shape(1), src.dtype())
            dst.scatterAddSet(Layout.contiguous(Shape(1)), ids, layout, src, layout, 0)
            assertContentEquals(expected.asSlice(), dst.asSlice())
        }
    }

    @Test
    fun packedPlaceholderDtypesFailClearly() {
        val layout = Layout.contiguous(Shape(1))
        for (storage in listOf(CpuStorage.F6E2M3(byteArrayOf(1)), CpuStorage.F6E3M2(byteArrayOf(1)), CpuStorage.F4(byteArrayOf(1)), CpuStorage.F8E8M0(byteArrayOf(1)))) {
            assertFailsWith<CandleException.UnsupportedDTypeForOp> { storage.reduceOp(ReduceOp.Sum, layout, listOf(0)) }
            assertFailsWith<CandleException.UnsupportedDTypeForOp> { storage.scatterSet(layout, CpuStorage.I64(longArrayOf(0)), layout, storage, layout, 0) }
        }
    }

    @Test
    fun quantizedCudaAllocationTransfersAndPointersFailClearly() {
        val device = CudaDevice(0)
        assertSame(CandleException.NotCompiledWithCudaSupport, assertFailsWith<CandleException> { KernelQCudaStorage.zeros(device, 32, GgmlDType.Q4_0) })
        val storage = KernelQCudaStorage(PaddedCudaSlice(byteArrayOf(1, 2, 3, 4), 4), GgmlDType.F32, device)
        assertSame(CandleException.NotCompiledWithCudaSupport, assertFailsWith<CandleException> { storage.data() })
        assertSame(CandleException.NotCompiledWithCudaSupport, assertFailsWith<CandleException> { storage.devicePtr() })
        assertSame(CandleException.NotCompiledWithCudaSupport, assertFailsWith<CandleException> { storage.devicePtrWithGuard(0) })
        assertSame(CandleException.NotCompiledWithCudaSupport, assertFailsWith<CandleException> { storage.dequantize(1) })
    }
}
