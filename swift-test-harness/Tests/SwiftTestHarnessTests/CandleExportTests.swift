import Testing
import Candle

@Suite
struct CandleExportTests {
    @Test
    func testSwiftModuleLoads() {
        let shape = Shape(dims: [2, 3, 4])
        #expect(shape.rank() == 3)
        #expect(shape.elemCount() == 24)
    }
}
