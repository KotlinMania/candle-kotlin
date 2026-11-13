package consumer

import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class PublishedCandleTest {
    @Test
    fun publishedAndroidApiCreatesTensor() {
        val tensor = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
        assertNotNull(tensor)
        assertEquals(6, tensor.elemCount())
        assertEquals(Shape(2, 3), tensor.shape())
        assertEquals(DType.F32, tensor.dtype())
    }
}
