package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import io.github.kotlinmania.candle.quantized.cuda.loadQuantized as loadCudaQuantized

class StorageCorrectnessTest {
    @Test
    fun scatterUsesEveryDimension() {
        val shape = Shape(2, 3, 2)
        val layout = Layout.contiguous(shape)
        val values = FloatArray(12) { (it + 1).toFloat() }
        for (dim in 0..2) {
            val stride = shape.dims.drop(dim + 1).fold(1, Int::times)
            val ids = LongArray(12) { (shape.dims[dim] - 1 - (it / stride) % shape.dims[dim]).toLong() }
            val expected = FloatArray(12)
            for (i in values.indices) {
                val coord = (i / stride) % shape.dims[dim]
                expected[i + (ids[i].toInt() - coord) * stride] = values[i]
            }
            val dst = CpuStorage.F32(FloatArray(12))
            dst.scatterSet(layout, CpuStorage.I64(ids), layout, CpuStorage.F32(values), layout, dim)
            assertContentEquals(expected, dst.data, "scatter dimension $dim")
            val added = CpuStorage.F32(FloatArray(12) { 10f })
            added.scatterAddSet(layout, CpuStorage.I64(ids), layout, CpuStorage.F32(values), layout, dim)
            assertContentEquals(expected.map { it + 10f }.toFloatArray(), added.data, "scatter-add dimension $dim")
        }
    }

    @Test
    fun tensorScatterMatchesAuditExample() {
        val shape = Shape(2, 2)
        val dst = Tensor.zeros(shape, DType.F32, Device.Cpu)
        val ids = Tensor.new(longArrayOf(1, 0, 0, 1), shape, Device.Cpu)
        val src = Tensor.new(floatArrayOf(1f, 2f, 3f, 4f), shape, Device.Cpu)
        assertContentEquals(floatArrayOf(2f, 1f, 3f, 4f), dst.scatter(ids, src, 1).toVec1F32())
        assertContentEquals(floatArrayOf(2f, 1f, 3f, 4f), dst.scatterAdd(ids, src, 1).toVec1F32())
        assertContentEquals(floatArrayOf(0f, 0f, 0f, 0f), dst.toVec1F32())
    }

    @Test
    fun scatterOffsetsDuplicatesAndSentinelsPreserveLongs() {
        val dstLayout = Layout.contiguousWithOffset(Shape(2, 2), 1)
        val srcLayout = Layout.contiguousWithOffset(Shape(2, 3), 1)
        val idsLayout = Layout.contiguousWithOffset(Shape(2, 3), 2)
        val large = 9007199254740993L
        val ids = CpuStorage.I64(longArrayOf(-9, -9, 1, 1, Long.MAX_VALUE, 0, 0, 1))
        val src = CpuStorage.I64(longArrayOf(-9, large, 2, 999, 3, 4, 5))
        val set = CpuStorage.I64(longArrayOf(-7, 10, 10, 10, 10, -7))
        set.scatterSet(dstLayout, ids, idsLayout, src, srcLayout, 1)
        assertContentEquals(longArrayOf(-7, 10, 2, 4, 5, -7), set.data)
        val add = CpuStorage.I64(longArrayOf(-7, 10, 10, 10, 10, -7))
        add.scatterAddSet(dstLayout, ids, idsLayout, src, srcLayout, 1)
        assertContentEquals(longArrayOf(-7, 10, large + 12, 17, 15, -7), add.data)
    }

    @Test
    fun scatterRejectsInvalidIndexesAndTypes() {
        val layout = Layout.contiguous(Shape(2, 2))
        val src = CpuStorage.F32(floatArrayOf(1f, 2f, 3f, 4f))
        for (invalid in listOf(-1L, 2L, 4294967296L)) {
            val dst = CpuStorage.F32(FloatArray(4))
            assertFailsWith<CandleException.InvalidIndex> {
                dst.scatterSet(layout, CpuStorage.I64(longArrayOf(invalid, 0, 0, 0)), layout, src, layout, 1)
            }
        }
        assertFailsWith<CandleException.UnsupportedDTypeForOp> {
            src.scatterSet(layout, CpuStorage.F32(FloatArray(4)), layout, src, layout, 1)
        }
        assertFailsWith<CandleException.DTypeMismatchBinaryOp> {
            src.scatterSet(layout, CpuStorage.I64(LongArray(4)), layout, CpuStorage.F64(DoubleArray(4)), layout, 1)
        }
    }

    @Test
    fun scatterRequiresContiguousInputs() {
        val layout = Layout.contiguous(Shape(2, 2))
        val transposed = Layout(Shape(2, 2), listOf(1, 2))
        val dst = CpuStorage.F32(FloatArray(4))
        val ids = CpuStorage.I64(LongArray(4))
        val src = CpuStorage.F32(FloatArray(4))
        for (layouts in listOf(listOf(transposed, layout, layout), listOf(layout, transposed, layout), listOf(layout, layout, transposed))) {
            assertFailsWith<CandleException.RequiresContiguous> {
                dst.scatterSet(layouts[0], ids, layouts[1], src, layouts[2], 1)
            }
        }
    }

    @Test
    fun reductionsPreserveF64Precision() {
        val layout = Layout.contiguous(Shape(2, 2))
        val src = CpuStorage.F64(doubleArrayOf(16777216.0, 1.0, 16777217.0, 2.0))
        assertContentEquals(doubleArrayOf(16777217.0, 16777219.0), (src.reduceOp(ReduceOp.Sum, layout, listOf(1)) as CpuStorage.F64).data)
        assertContentEquals(doubleArrayOf(33554436.0), (src.reduceOp(ReduceOp.Sum, layout, listOf(0, 1)) as CpuStorage.F64).data)
        assertContentEquals(doubleArrayOf(16777217.0, 2.0), (src.reduceOp(ReduceOp.Max, layout, listOf(0)) as CpuStorage.F64).data)
        assertContentEquals(uintArrayOf(1u, 1u), (src.reduceOp(ReduceOp.ArgMax, layout, listOf(0)) as CpuStorage.U32).data)
    }

    @Test
    fun reductionsPreserveAllLongBitsAndArgIndexes() {
        val large = 9007199254740993L
        val layout = Layout.contiguous(Shape(2, 2))
        val src = CpuStorage.I64(longArrayOf(large, large + 1, -large, -large - 1))
        assertContentEquals(longArrayOf(large + large + 1, -large - large - 1), (src.reduceOp(ReduceOp.Sum, layout, listOf(1)) as CpuStorage.I64).data)
        assertContentEquals(longArrayOf(large, -large - 1), (src.reduceOp(ReduceOp.Min, layout, listOf(1)) as CpuStorage.I64).data)
        assertContentEquals(longArrayOf(large + 1, -large), (src.reduceOp(ReduceOp.Max, layout, listOf(1)) as CpuStorage.I64).data)
        assertContentEquals(uintArrayOf(0u, 1u), (src.reduceOp(ReduceOp.ArgMin, layout, listOf(1)) as CpuStorage.U32).data)
        assertContentEquals(uintArrayOf(1u, 0u), (src.reduceOp(ReduceOp.ArgMax, layout, listOf(1)) as CpuStorage.U32).data)
    }

    @Test
    fun reductionsHonorStridesOffsetsEmptyInputsAndFirstTies() {
        val src = CpuStorage.I64(longArrayOf(-99, 16777217, 16777218, 16777219, 16777220))
        val layout = Layout(Shape(2, 2), listOf(1, 2), 1)
        assertContentEquals(longArrayOf(33554436, 33554438), (src.reduceOp(ReduceOp.Sum, layout, listOf(1)) as CpuStorage.I64).data)
        assertContentEquals(uintArrayOf(1u, 1u), (src.reduceOp(ReduceOp.ArgMax, layout, listOf(1)) as CpuStorage.U32).data)
        val ties = CpuStorage.F64(doubleArrayOf(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY))
        assertContentEquals(uintArrayOf(0u), (ties.reduceOp(ReduceOp.ArgMin, Layout.contiguous(Shape(2)), listOf(0)) as CpuStorage.U32).data)
        val empty = CpuStorage.F64(doubleArrayOf())
        val emptyLayout = Layout.contiguous(Shape(2, 0))
        assertContentEquals(doubleArrayOf(0.0, 0.0), (empty.reduceOp(ReduceOp.Sum, emptyLayout, listOf(1)) as CpuStorage.F64).data)
        assertFailsWith<CandleException.EmptyTensor> { empty.reduceOp(ReduceOp.Min, emptyLayout, listOf(1)) }
        assertFailsWith<CandleException.OnlySingleDimension> { src.reduceOp(ReduceOp.ArgMax, layout, listOf(0, 1)) }
    }

    @Test
    fun tensorAllocationHonorsCpuAndRejectsUnavailableBackends() {
        val shape = Shape(2)
        assertSame(Device.Cpu, Tensor.zeros(shape, DType.F64, Device.Cpu).device())
        for (shapeToTest in listOf(shape, Shape(0))) {
            val cuda = Device.Cuda(CudaDevice(3))
            val metal = Device.Metal(MetalDevice(2))
            assertSame(CandleException.NotCompiledWithCudaSupport, assertFailsWith<CandleException> { Tensor.zeros(shapeToTest, DType.F32, cuda) })
            assertSame(CandleException.NotCompiledWithCudaSupport, assertFailsWith<CandleException> { Tensor.ones(shapeToTest, DType.F32, cuda) })
            assertSame(CandleException.NotCompiledWithMetalSupport, assertFailsWith<CandleException> { Tensor.zeros(shapeToTest, DType.F32, metal) })
            assertSame(CandleException.NotCompiledWithMetalSupport, assertFailsWith<CandleException> { Tensor.ones(shapeToTest, DType.F32, metal) })
        }
    }

    @Test
    fun tensorArrayConstructorsHonorDeviceAndEmptyDtype() {
        for (device in listOf(Device.Cuda(CudaDevice(1)), Device.Metal(MetalDevice(1)))) {
            val expected = if (device is Device.Cuda) CandleException.NotCompiledWithCudaSupport else CandleException.NotCompiledWithMetalSupport
            val shape = Shape(1)
            assertSame(expected, assertFailsWith<CandleException> { Tensor.new(floatArrayOf(1f), shape, device) })
            assertSame(expected, assertFailsWith<CandleException> { Tensor.new(doubleArrayOf(1.0), shape, device) })
            assertSame(expected, assertFailsWith<CandleException> { Tensor.new(intArrayOf(1), shape, device) })
            assertSame(expected, assertFailsWith<CandleException> { Tensor.new(longArrayOf(1), shape, device) })
            assertSame(expected, assertFailsWith<CandleException> { Tensor.new(uintArrayOf(1u), shape, device) })
            assertSame(expected, assertFailsWith<CandleException> { Tensor.new(ubyteArrayOf(1u), shape, device) })
            assertSame(expected, assertFailsWith<CandleException> { Tensor.new(doubleArrayOf(), Shape(0), device) })
        }
        assertEquals(DType.F64, Tensor.new(doubleArrayOf(), Shape(0), Device.Cpu).dtype())
        assertEquals(DType.I64, Tensor.new(longArrayOf(), Shape(0), Device.Cpu).dtype())
    }

    @Test
    fun quantizedCudaLoadingFailsInsteadOfReturningDummyStorage() {
        val device = CudaDevice(0)
        for (values in listOf(listOf(1f, -2f, 3f), emptyList())) {
            assertSame(CandleException.NotCompiledWithCudaSupport, assertFailsWith<CandleException> { loadCudaQuantized(device, values) })
            assertSame(CandleException.NotCompiledWithCudaSupport, (loadQuantized(device, values) as CandleResult.Err).error)
        }
    }
}
