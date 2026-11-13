// port-lint: tests cpu_backend/utils.rs
package io.github.kotlinmania.candle.cpubackend.utils

import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.shape.Shape

@kotlin.test.Test
public fun binaryMapVecParChunksNonContiguousBroadcasts() {
    val ROWS: Int = 2
    val COLS: Int = 2 * PAR_ELEMWISE_CHUNK
    val LHS_ROW_STRIDE: Int = COLS + 1
    val lhs = mutableListOf(Long.MAX_VALUE, ROWS * LHS_ROW_STRIDE)
    for (row in 0 until ROWS) {
        for (col in 0 until COLS) {
            lhs[row * LHS_ROW_STRIDE + col] = (row * COLS + col).toLong()
        }
    }
    val rhs = listOf(11L, 29)
    val shape = Shape.from(Pair(ROWS, COLS))
    val lhsL = Layout.new(shape.clone(), mutableListOf(LHS_ROW_STRIDE, 1), 0)
    val rhsL = Layout.new(shape, mutableListOf(1, 0), 0)
    val calls = AtomicUsize.new(0)
    val elements = AtomicUsize.new(0)
    val result =
        binaryMapVecPar(lhsL, rhsL, lhs, rhs, { lhs1, rhs1 -> lhs1 - rhs1 }, { error("broadcast layout must not use the two-vector kernel") }, { scalar, xs, ys ->
            check(xs.len() <= PAR_ELEMWISE_CHUNK)
            calls.fetchAdd(1, Ordering.Relaxed)
            elements.fetchAdd(xs.len(), Ordering.Relaxed)
            for ((x, y) in (xs.iter().zip(ys))) {
                y = x - scalar
            }
        })
    val expected: Vec = (0 until ROWS).flatMap({ row1 -> (0 until COLS).map({ col1 -> (row1 * COLS + col1).toLong() - rhs[row1] }) }).collect()
    check(result == expected)
    check(calls.load(Ordering.Relaxed) == 4)
    check(elements.load(Ordering.Relaxed) == ROWS * COLS)
}

@kotlin.test.Test
public fun binaryMapVecParHandlesTwoStridedInputs() {
    val ROWS: Int = 2
    val COLS: Int = 2 * PAR_ELEMWISE_CHUNK
    val LHS_ROW_STRIDE: Int = COLS + 1
    val RHS_ROW_STRIDE: Int = COLS + 2
    val lhs = mutableListOf(0L, ROWS * LHS_ROW_STRIDE)
    val rhs = mutableListOf(0L, ROWS * RHS_ROW_STRIDE)
    for (row in 0 until ROWS) {
        for (col in 0 until COLS) {
            lhs[row * LHS_ROW_STRIDE + col] = (row * COLS + col).toLong()
            rhs[row * RHS_ROW_STRIDE + col] = (2 * row * COLS + col).toLong()
        }
    }
    val shape = Shape.from(Pair(COLS, ROWS))
    val lhsL = Layout.new(shape.clone(), mutableListOf(1, LHS_ROW_STRIDE), 0)
    val rhsL = Layout.new(shape, mutableListOf(1, RHS_ROW_STRIDE), 0)
    val result = binaryMapVecPar(lhsL, rhsL, lhs, rhs, { lhs1, rhs1 -> lhs1 - rhs1 }, { error("strided layout must not use the two-vector kernel") }, { error("strided layout must not use the scalar-vector kernel") })
    val lhs1 = lhs
    val rhs1 = rhs
    val expected: Vec =
        (0 until COLS)
            .flatMap({ col1 ->
                (0 until ROWS).map({ row1 ->
                    lhs1[row1 * LHS_ROW_STRIDE + col1] - rhs1[row1 * RHS_ROW_STRIDE + col1]
                })
            })
            .collect()
    check(result == expected)
}
