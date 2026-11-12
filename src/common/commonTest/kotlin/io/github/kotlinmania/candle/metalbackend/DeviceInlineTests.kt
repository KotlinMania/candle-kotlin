// port-lint: tests metal_backend/device.rs
package io.github.kotlinmania.candle.metalbackend.device

@kotlin.test.Test
public fun testBufSizeExactPowersOfTwo() {
    check(bufSize(1) == 1)
    check(bufSize(2) == 2)
    check(bufSize(4) == 4)
    check(bufSize(8) == 8)
    check(bufSize(16) == 16)
    check(bufSize(1024) == 1024)
}

@kotlin.test.Test
public fun testBufSizeRoundsUp() {
    check(bufSize(3) == 4)
    check(bufSize(5) == 8)
    check(bufSize(6) == 8)
    check(bufSize(7) == 8)
    check(bufSize(9) == 16)
    check(bufSize(1000) == 1024)
    check(bufSize(1025) == 2048)
}

@kotlin.test.Test
public fun testBufSizeBf16F16Scalar() {
    // BF16 and F16 are 2 bytes per element. A scalar tensor requests
    // a 2-byte buffer. This must not be rounded down to 1.
    check(bufSize(2) == 2)
}
