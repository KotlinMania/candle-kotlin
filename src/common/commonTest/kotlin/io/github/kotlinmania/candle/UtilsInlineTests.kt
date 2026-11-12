// port-lint: tests utils.rs
package io.github.kotlinmania.candle.utils

// Useful functions for checking features.

@kotlin.test.Test
public fun parsesCpuList() {
    check(linuxParseCpuList("5-9,15,17-19") == mutableListOf(5, 6, 7, 8, 9, 15, 17, 18, 19))
    check(linuxParseCpuList(" 3 , 1-2,2 ") == mutableListOf(1, 2, 3))
}

@kotlin.test.Test
public fun rejectsInvalidCpuList() {
    check(linuxParseCpuList("") == null)
    check(linuxParseCpuList("9-5") == null)
    check(linuxParseCpuList("5,a") == null)
}
