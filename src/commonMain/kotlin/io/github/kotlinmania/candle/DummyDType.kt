// port-lint: source dummy_dtype.rs
package io.github.kotlinmania.candle

/**
 * 6-bit float with 2 exponent bits and 3 mantissa bits (MX6 format)
 * This is a dummy type.
 */
public class F6E2M3 {
    public fun fmt(): String = "F6E2M3"

    override fun toString(): String = fmt()

    override fun equals(other: Any?): Boolean = other is F6E2M3

    override fun hashCode(): Int = "F6E2M3".hashCode()
}

/**
 * 6-bit float with 3 exponent bits and 2 mantissa bits (MX6 format)
 * This is a dummy type.
 */
public class F6E3M2 {
    public fun fmt(): String = "F6E3M2"

    override fun toString(): String = fmt()

    override fun equals(other: Any?): Boolean = other is F6E3M2

    override fun hashCode(): Int = "F6E3M2".hashCode()
}

/**
 * 4-bit float (MX4 format)
 * This is a dummy type.
 */
public class F4 {
    public fun fmt(): String = "F4"

    override fun toString(): String = fmt()

    override fun equals(other: Any?): Boolean = other is F4

    override fun hashCode(): Int = "F4".hashCode()
}

/**
 * 8-bit float with 8 exponent bits and 0 mantissa bits
 * This is a dummy type.
 */
public class F8E8M0 {
    public fun fmt(): String = "F8E8M0"

    override fun toString(): String = fmt()

    override fun equals(other: Any?): Boolean = other is F8E8M0

    override fun hashCode(): Int = "F8E8M0".hashCode()
}
