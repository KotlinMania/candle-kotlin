import Testing
import Candle

@Suite
struct CandleExportTests {
    @Test
    func testSwiftModuleLoads() {
        _ = CandleCore.shared.VERSION
        let shape = Shape(dims: [2, 3, 4])
        #expect(shape.rank == 3)
    }
}
