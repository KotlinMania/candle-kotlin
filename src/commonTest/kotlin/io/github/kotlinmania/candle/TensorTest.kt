// port-lint: tests candle-core/tests/tensor_tests.rs
package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TensorTest {
    @Test
    fun testZerosAndOnes() {
        val zeros = Tensor.zeros(Shape(listOf(2, 3)), DType.F32, Device.Cpu)
        assertEquals(listOf(2, 3), zeros.dims())
        assertEquals(DType.F32, zeros.dtype())
        val zData = zeros.toVec1F32()
        assertEquals(6, zData.size)
        assertTrue(zData.all { it == 0.0f })

        val ones = Tensor.ones(Shape(listOf(2, 3)), DType.F32, Device.Cpu)
        val oData = ones.toVec1F32()
        assertEquals(6, oData.size)
        assertTrue(oData.all { it == 1.0f })
    }

    @Test
    fun testArithmetic() {
        val a = Tensor.new(floatArrayOf(1f, 2f, 3f, 4f), Shape(listOf(2, 2)), Device.Cpu)
        val b = Tensor.new(floatArrayOf(10f, 20f, 30f, 40f), Shape(listOf(2, 2)), Device.Cpu)

        val c = a + b
        val cData = c.toVec1F32()
        assertEquals(11f, cData[0])
        assertEquals(22f, cData[1])
        assertEquals(33f, cData[2])
        assertEquals(44f, cData[3])

        val d = b - a
        val dData = d.toVec1F32()
        assertEquals(9f, dData[0])
        assertEquals(18f, dData[1])
        assertEquals(27f, dData[2])
        assertEquals(36f, dData[3])
    }

    @Test
    fun testMatmul() {
        val a = Tensor.new(floatArrayOf(1f, 2f, 3f, 4f), Shape(listOf(2, 2)), Device.Cpu)
        val b = Tensor.new(floatArrayOf(2f, 0f, 1f, 2f), Shape(listOf(2, 2)), Device.Cpu)

        val c = a.matmul(b)
        assertEquals(listOf(2, 2), c.dims())
        val cData = c.toVec1F32()
        // [1*2 + 2*1, 1*0 + 2*2] = [4, 4]
        // [3*2 + 4*1, 3*0 + 4*2] = [10, 8]
        assertEquals(4f, cData[0])
        assertEquals(4f, cData[1])
        assertEquals(10f, cData[2])
        assertEquals(8f, cData[3])
    }

    @Test
    fun testReductions() {
        val a = Tensor.new(floatArrayOf(1f, 2f, 3f, 4f), Shape(listOf(2, 2)), Device.Cpu)
        val sumAll = a.sumAll().toScalarF32()
        assertEquals(10f, sumAll)

        val sum0 = a.sum(0).toVec1F32()
        assertEquals(4f, sum0[0])
        assertEquals(6f, sum0[1])

        val sum1 = a.sum(1).toVec1F32()
        assertEquals(3f, sum1[0])
        assertEquals(7f, sum1[1])
    }

    @Test
    fun testIndexing() {
        val a = Tensor.arange(0f, 6f, Device.Cpu).reshape(2, 3)
        val row1 = a.i(1)
        assertEquals(listOf(3), row1.dims())
        val rData = row1.toVec1F32()
        assertEquals(3f, rData[0])
        assertEquals(4f, rData[1])
        assertEquals(5f, rData[2])
    }

    @Test
    fun testVariables() {
        val t = Tensor.zeros(Shape(listOf(2, 2)), DType.F32, Device.Cpu)
        val v = Var.fromTensor(t)
        val src = Tensor.ones(Shape(listOf(2, 2)), DType.F32, Device.Cpu)
        v.set(src)
        val data = v.asTensor().toVec1F32()
        assertTrue(data.all { it == 1.0f })
    }
}
