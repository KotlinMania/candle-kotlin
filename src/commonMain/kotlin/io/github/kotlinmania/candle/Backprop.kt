// port-lint: source backprop.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.envie.getenv
import io.github.kotlinmania.threadlocal.ThreadLocal

private fun broadcastBack(arg: Tensor, node: Tensor, reducedDims: List<Int>): Tensor =
    if (arg.rank() == node.rank()) {
        node.broadcastAs(arg.shape())
    } else {
        node.reshape(Shape(reducedDims)).broadcastAs(arg.shape())
    }

private val CANDLE_GRAD_DO_NOT_DETACH: ThreadLocal<Boolean> = ThreadLocal()

/**
 * A store for gradients, associating a tensor id to the corresponding gradient tensor, used for back propagation.
 */
public class GradStore(
    private val grads: MutableMap<TensorId, Tensor> = mutableMapOf(),
) {
    /**
     * Get the gradient tensor corresponding to the given tensor id
     */
    public fun getId(id: TensorId): Tensor? = grads[id]

    /**
     * Get the gradient tensor associated with the given tensor
     */
    public fun get(tensor: Tensor): Tensor? = grads[tensor.id()]

    /**
     * Remove the gradient tensor associated with the given tensor, returning it if it exists
     */
    public fun remove(tensor: Tensor): Tensor? = grads.remove(tensor.id())

    /**
     * Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
     */
    public fun insert(tensor: Tensor, grad: Tensor): Tensor? = grads.put(tensor.id(), grad)

    /**
     * Insert a gradient tensor associated with the given tensor id, returning the previous gradient tensor if it existed
     */
    public fun insertId(id: TensorId, grad: Tensor): Tensor? = grads.put(id, grad)

    /**
     * Get the gradient tensor associated with the given tensor, or, if it does not exist,
     * insert a tensor of zeroes, with the same shape and type as the given tensors and return it
     */
    public fun orInsert(tensor: Tensor): Tensor =
        grads.getOrPut(tensor.id()) { tensor.zerosLike() }

    /**
     * Extend this gradient store with the contents of another.
     * If an entry is already occupied, updates and ensures new tensor follows correct detach semantics.
     * Otherwise simply inserts.
     */
    public fun extend(other: GradStore) {
        for ((id, grad) in other.grads) {
            val existing = grads[id]
            if (existing != null) {
                val newGrad = existing.add(grad)
                grads[id] = newGrad
            } else {
                grads[id] = grad
            }
        }
    }

    /**
     * Get the tensor ids of the stored gradient tensors
     */
    public fun getIds(): Set<TensorId> = grads.keys

    public companion object {
        /**
         * Create a new gradient store
         */
        public fun new(): GradStore = GradStore()
    }
}

/**
 * Return all the nodes that lead to this value in a topologically sorted vec, the first
 * elements having dependencies on the latter ones, e.g. the first element if any is the
 * argument.
 * This assumes that the op graph is a DAG.
 */
public fun Tensor.sortedNodes(): List<Tensor> {
    val alreadySeen = mutableMapOf<TensorId, Boolean>()

    fun walk(node: Tensor, nodes: MutableList<Tensor>): Pair<Boolean, MutableList<Tensor>> {
        val tg = alreadySeen[node.id()]
        if (tg != null) {
            return Pair(tg, nodes)
        }
        var trackGrad = false
        var currentNodes = if (node.isVariable()) {
            trackGrad = true
            nodes
        } else if (node.dtype().isInt()) {
            nodes
        } else {
            val op = node.op()
            if (op != null) {
                when (op) {
                    is Op.IndexAdd -> {
                        val (tg1, n1) = walk(op.init, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.indexes, n1)
                        trackGrad = trackGrad or tg2
                        val (tg3, n3) = walk(op.src, n2)
                        trackGrad = trackGrad or tg3
                        n3
                    }
                    is Op.Scatter -> {
                        val (tg1, n1) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.index, n1)
                        trackGrad = trackGrad or tg2
                        val (tg3, n3) = walk(op.src, n2)
                        trackGrad = trackGrad or tg3
                        n3
                    }
                    is Op.ScatterAdd -> {
                        val (tg1, n1) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.index, n1)
                        trackGrad = trackGrad or tg2
                        val (tg3, n3) = walk(op.src, n2)
                        trackGrad = trackGrad or tg3
                        n3
                    }
                    is Op.CustomOp3 -> {
                        val (tg1, n1) = walk(op.t1, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.t2, n1)
                        trackGrad = trackGrad or tg2
                        val (tg3, n3) = walk(op.t3, n2)
                        trackGrad = trackGrad or tg3
                        n3
                    }
                    is Op.WhereCond -> {
                        val (tg1, n1) = walk(op.cond, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.onTrue, n1)
                        trackGrad = trackGrad or tg2
                        val (tg3, n3) = walk(op.onFalse, n2)
                        trackGrad = trackGrad or tg3
                        n3
                    }
                    is Op.Conv1D -> {
                        val (tg1, n1) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.kernel, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.ConvTranspose1D -> {
                        val (tg1, n1) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.kernel, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.Conv2D -> {
                        val (tg1, n1) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.kernel, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.ConvTranspose2D -> {
                        val (tg1, n1) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.kernel, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.CustomOp2 -> {
                        val (tg1, n1) = walk(op.lhs, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.rhs, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.Binary -> {
                        val (tg1, n1) = walk(op.lhs, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.rhs, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.Gather -> {
                        val (tg1, n1) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.index, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.IndexSelect -> {
                        val (tg1, n1) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.index, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.Matmul -> {
                        val (tg1, n1) = walk(op.lhs, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.rhs, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.SliceScatter0 -> {
                        val (tg1, n1) = walk(op.lhs, nodes)
                        trackGrad = trackGrad or tg1
                        val (tg2, n2) = walk(op.rhs, n1)
                        trackGrad = trackGrad or tg2
                        n2
                    }
                    is Op.Cat -> {
                        var cur = nodes
                        for (arg in op.args) {
                            val (tg, n) = walk(arg, cur)
                            trackGrad = trackGrad or tg
                            cur = n
                        }
                        cur
                    }
                    is Op.Affine -> {
                        if (op.mul == 0.0) {
                            nodes
                        } else {
                            val (tg, n) = walk(op.arg, nodes)
                            trackGrad = trackGrad or tg
                            n
                        }
                    }
                    is Op.Unary -> {
                        when (op.op) {
                            UnaryOp.Ceil, UnaryOp.Floor, UnaryOp.Round, UnaryOp.Sign -> nodes
                            else -> {
                                val (tg, n) = walk(op.arg, nodes)
                                trackGrad = trackGrad or tg
                                n
                            }
                        }
                    }
                    is Op.Reshape -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.UpsampleNearest1D -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.UpsampleNearest2D -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.UpsampleBilinear2D -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.AvgPool2D -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.MaxPool2D -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.Copy -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.Broadcast -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.Cmp -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.Reduce -> {
                        when (op.op) {
                            ReduceOp.Min, ReduceOp.Sum, ReduceOp.Max -> {
                                val (tg, n) = walk(op.arg, nodes)
                                trackGrad = trackGrad or tg
                                n
                            }
                            ReduceOp.ArgMin, ReduceOp.ArgMax -> nodes
                        }
                    }
                    is Op.ToDevice -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.Transpose -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.Permute -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.Narrow -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.Elu -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.Powf -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.CustomOp1 -> {
                        val (tg, n) = walk(op.arg, nodes)
                        trackGrad = trackGrad or tg
                        n
                    }
                    is Op.ToDType -> {
                        if (op.arg.dtype().isFloat()) {
                            val (tg, n) = walk(op.arg, nodes)
                            trackGrad = trackGrad or tg
                            n
                        } else {
                            nodes
                        }
                    }
                }
            } else {
                nodes
            }
        }
        alreadySeen[node.id()] = trackGrad
        if (trackGrad) {
            currentNodes.add(node)
        }
        return Pair(trackGrad, currentNodes)
    }

    val (_, nodes) = walk(this, mutableListOf())
    nodes.reverse()
    return nodes
}

public fun Tensor.backward(): GradStore {
    val sortedNodes = sortedNodes()
    val grads = GradStore.new()
    grads.insert(this, onesLike().contiguous())
    for (node in sortedNodes) {
        if (node.isVariable()) {
            continue
        }
        val grad = grads.remove(node) ?: bail("candle internal error - grad not populated")
        val doNotDetach = CANDLE_GRAD_DO_NOT_DETACH.getOr {
            val s = getenv("CANDLE_GRAD_DO_NOT_DETACH")
            s != null && s.isNotEmpty() && s != "0"
        }
        val detachedGrad = if (doNotDetach) grad else grad.detach()
        val op = node.op() ?: continue
        when (op) {
            is Op.Binary -> {
                when (op.op) {
                    BinaryOp.Add -> {
                        val lhsSumGrad = grads.orInsert(op.lhs)
                        grads.insert(op.lhs, lhsSumGrad.add(detachedGrad))
                        val rhsSumGrad = grads.orInsert(op.rhs)
                        grads.insert(op.rhs, rhsSumGrad.add(detachedGrad))
                    }
                    BinaryOp.Sub -> {
                        val lhsSumGrad = grads.orInsert(op.lhs)
                        grads.insert(op.lhs, lhsSumGrad.add(detachedGrad))
                        val rhsSumGrad = grads.orInsert(op.rhs)
                        grads.insert(op.rhs, rhsSumGrad.sub(detachedGrad))
                    }
                    BinaryOp.Mul -> {
                        val lhsGrad = detachedGrad.mul(op.rhs)
                        val lhsSumGrad = grads.orInsert(op.lhs)
                        grads.insert(op.lhs, lhsSumGrad.add(lhsGrad))
                        val rhsGrad = detachedGrad.mul(op.lhs)
                        val rhsSumGrad = grads.orInsert(op.rhs)
                        grads.insert(op.rhs, rhsSumGrad.add(rhsGrad))
                    }
                    BinaryOp.Div -> {
                        val lhsGrad = detachedGrad.div(op.rhs)
                        val lhsSumGrad = grads.orInsert(op.lhs)
                        grads.insert(op.lhs, lhsSumGrad.add(lhsGrad))
                        val rhsGrad = detachedGrad.mul(op.lhs).div(op.rhs.sqr())
                        val rhsSumGrad = grads.orInsert(op.rhs)
                        grads.insert(op.rhs, rhsSumGrad.sub(rhsGrad))
                    }
                    BinaryOp.Minimum, BinaryOp.Maximum -> {
                        val maskLhs = node.eq(op.lhs).toDtype(detachedGrad.dtype())
                        val maskRhs = node.eq(op.rhs).toDtype(detachedGrad.dtype())
                        val lhsGrad = maskLhs.mul(detachedGrad).div(maskRhs.affine(1.0, 1.0))
                        val lhsSumGrad = grads.orInsert(op.lhs)
                        grads.insert(op.lhs, lhsSumGrad.add(lhsGrad))
                        val rhsGrad = maskRhs.mul(detachedGrad).div(maskLhs.affine(1.0, 1.0))
                        val rhsSumGrad = grads.orInsert(op.rhs)
                        grads.insert(op.rhs, rhsSumGrad.add(rhsGrad))
                    }
                }
            }
            is Op.WhereCond -> {
                val zeros = detachedGrad.zerosLike()
                val tSumGrad = grads.orInsert(op.onTrue)
                val tGrad = op.cond.whereCond(detachedGrad, zeros)
                grads.insert(op.onTrue, tSumGrad.add(tGrad))
                val fSumGrad = grads.orInsert(op.onFalse)
                val fGrad = op.cond.whereCond(zeros, detachedGrad)
                grads.insert(op.onFalse, fSumGrad.add(fGrad))
            }
            is Op.Matmul -> {
                val lhsGrad = detachedGrad.matmul(op.rhs.t())
                val lhsSumGrad = grads.orInsert(op.lhs)
                grads.insert(op.lhs, lhsSumGrad.add(lhsGrad))
                val rhsGrad = op.lhs.t().matmul(detachedGrad)
                val rhsSumGrad = grads.orInsert(op.rhs)
                grads.insert(op.rhs, rhsSumGrad.add(rhsGrad))
            }
            is Op.Cat -> {
                var startIdx = 0
                for (arg in op.args) {
                    val len = arg.dims()[op.dim]
                    val argGrad = detachedGrad.narrow(op.dim, startIdx, len)
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(argGrad))
                    startIdx += len
                }
            }
            is Op.Reduce -> {
                when (op.op) {
                    ReduceOp.Sum -> {
                        val g = broadcastBack(op.arg, detachedGrad, op.dims)
                        val sumGrad = grads.orInsert(op.arg)
                        grads.insert(op.arg, sumGrad.add(g))
                    }
                    ReduceOp.Max, ReduceOp.Min -> {
                        val nodeB = broadcastBack(op.arg, node, op.dims)
                        val g = broadcastBack(op.arg, detachedGrad, op.dims)
                        val mask = nodeB.eq(op.arg).toDtype(g.dtype()).mul(g)
                        val sumGrad = grads.orInsert(op.arg)
                        grads.insert(op.arg, sumGrad.add(mask.broadcastAs(sumGrad.shape())))
                    }
                    ReduceOp.ArgMin, ReduceOp.ArgMax -> {}
                }
            }
            is Op.ToDType -> {
                val sumGrad = grads.orInsert(op.arg)
                grads.insert(op.arg, sumGrad.add(detachedGrad.toDtype(op.arg.dtype())))
            }
            is Op.Copy -> {
                val sumGrad = grads.orInsert(op.arg)
                grads.insert(op.arg, sumGrad.add(detachedGrad))
            }
            is Op.Affine -> {
                val argGrad = detachedGrad.affine(op.mul, 0.0)
                val sumGrad = grads.orInsert(op.arg)
                grads.insert(op.arg, sumGrad.add(argGrad))
            }
            is Op.Unary -> {
                when (op.op) {
                    UnaryOp.Log -> {
                        val sumGrad = grads.orInsert(op.arg)
                        grads.insert(op.arg, sumGrad.add(detachedGrad.div(op.arg)))
                    }
                    UnaryOp.Sin -> {
                        val sumGrad = grads.orInsert(op.arg)
                        grads.insert(op.arg, sumGrad.add(detachedGrad.mul(op.arg.cos())))
                    }
                    UnaryOp.Cos -> {
                        val sumGrad = grads.orInsert(op.arg)
                        grads.insert(op.arg, sumGrad.sub(detachedGrad.mul(op.arg.sin())))
                    }
                    UnaryOp.Tanh -> {
                        val sumGrad = grads.orInsert(op.arg)
                        val minusDtanh = node.sqr().affine(1.0, -1.0)
                        grads.insert(op.arg, sumGrad.sub(detachedGrad.mul(minusDtanh)))
                    }
                    UnaryOp.Abs -> {
                        val sumGrad = grads.orInsert(op.arg)
                        val ones = op.arg.onesLike()
                        val absGrad = op.arg.ge(op.arg.zerosLike()).whereCond(ones, ones.neg())
                        grads.insert(op.arg, sumGrad.add(detachedGrad.mul(absGrad)))
                    }
                    UnaryOp.Exp -> {
                        val sumGrad = grads.orInsert(op.arg)
                        grads.insert(op.arg, sumGrad.add(detachedGrad.mul(node)))
                    }
                    UnaryOp.Neg -> {
                        val sumGrad = grads.orInsert(op.arg)
                        grads.insert(op.arg, sumGrad.sub(detachedGrad))
                    }
                    UnaryOp.Recip -> {
                        val sumGrad = grads.orInsert(op.arg)
                        val g = detachedGrad.div(op.arg.sqr())
                        grads.insert(op.arg, sumGrad.sub(g))
                    }
                    UnaryOp.Sqr -> {
                        val argGrad = op.arg.mul(detachedGrad).affine(2.0, 0.0)
                        val sumGrad = grads.orInsert(op.arg)
                        grads.insert(op.arg, sumGrad.add(argGrad))
                    }
                    UnaryOp.Sqrt -> {
                        val argGrad = detachedGrad.div(node).affine(0.5, 0.0)
                        val sumGrad = grads.orInsert(op.arg)
                        grads.insert(op.arg, sumGrad.add(argGrad))
                    }
                    UnaryOp.Relu -> {
                        val sumGrad = grads.orInsert(op.arg)
                        val reluGrad = op.arg.ge(op.arg.zerosLike()).toDtype(op.arg.dtype())
                        grads.insert(op.arg, sumGrad.add(detachedGrad.mul(reluGrad)))
                    }
                    UnaryOp.Silu -> {
                        val sumGrad = grads.orInsert(op.arg)
                        val sigmoidArg = op.arg.neg().exp().affine(1.0, 1.0).recip()
                        val siluGrad = sigmoidArg.mul(node.affine(-1.0, 1.0)).add(node)
                        grads.insert(op.arg, sumGrad.add(detachedGrad.mul(siluGrad)))
                    }
                    UnaryOp.Ceil -> throw CandleException.BackwardNotSupported("ceil")
                    UnaryOp.Floor, UnaryOp.Round, UnaryOp.Sign -> {}
                    else -> {}
                }
            }
            is Op.Narrow -> {
                val argDims = op.arg.dims()
                val leftPad = if (op.start == 0) {
                    null
                } else {
                    val dims = argDims.toMutableList()
                    dims[op.dim] = op.start
                    Tensor.zeros(Shape(dims), detachedGrad.dtype(), detachedGrad.device())
                }
                val rightPadLen = argDims[op.dim] - op.start - op.len
                val rightPad = if (rightPadLen == 0) {
                    null
                } else {
                    val dims = argDims.toMutableList()
                    dims[op.dim] = rightPadLen
                    Tensor.zeros(Shape(dims), detachedGrad.dtype(), detachedGrad.device())
                }
                val argGrad = when {
                    leftPad == null && rightPad == null -> detachedGrad
                    leftPad != null && rightPad == null -> Tensor.cat(listOf(leftPad, detachedGrad), op.dim)
                    leftPad == null && rightPad != null -> Tensor.cat(listOf(detachedGrad, rightPad), op.dim)
                    else -> Tensor.cat(listOf(leftPad!!, detachedGrad, rightPad!!), op.dim)
                }
                val sumGrad = grads.orInsert(op.arg)
                grads.insert(op.arg, sumGrad.add(argGrad))
            }
            is Op.Reshape -> {
                val argGrad = detachedGrad.reshape(op.arg.shape())
                val sumGrad = grads.orInsert(op.arg)
                grads.insert(op.arg, sumGrad.add(argGrad))
            }
            is Op.Powf -> {
                val argGrad = detachedGrad.mul(op.arg.powf(op.e - 1.0)).affine(op.e, 0.0)
                val sumGrad = grads.orInsert(op.arg)
                grads.insert(op.arg, sumGrad.add(argGrad))
            }
            is Op.CustomOp1 -> {
                val argGrad = op.op.bwd(op.arg, node, detachedGrad)
                if (argGrad != null) {
                    val sumGrad = grads.orInsert(op.arg)
                    grads.insert(op.arg, sumGrad.add(argGrad))
                }
            }
            is Op.CustomOp2 -> {
                val (argGrad1, argGrad2) = op.op.bwd(op.lhs, op.rhs, node, detachedGrad)
                if (argGrad1 != null) {
                    val sumGrad = grads.orInsert(op.lhs)
                    grads.insert(op.lhs, sumGrad.add(argGrad1))
                }
                if (argGrad2 != null) {
                    val sumGrad = grads.orInsert(op.rhs)
                    grads.insert(op.rhs, sumGrad.add(argGrad2))
                }
            }
            is Op.CustomOp3 -> {
                val (argGrad1, argGrad2, argGrad3) = op.op.bwd(op.t1, op.t2, op.t3, node, detachedGrad)
                if (argGrad1 != null) {
                    val sumGrad = grads.orInsert(op.t1)
                    grads.insert(op.t1, sumGrad.add(argGrad1))
                }
                if (argGrad2 != null) {
                    val sumGrad = grads.orInsert(op.t2)
                    grads.insert(op.t2, sumGrad.add(argGrad2))
                }
                if (argGrad3 != null) {
                    val sumGrad = grads.orInsert(op.t3)
                    grads.insert(op.t3, sumGrad.add(argGrad3))
                }
            }
            is Op.ToDevice -> {
                val sumGrad = grads.orInsert(op.arg)
                val argGrad = detachedGrad.toDevice(sumGrad.device())
                grads.insert(op.arg, sumGrad.add(argGrad))
            }
            is Op.Transpose -> {
                val argGrad = detachedGrad.transpose(op.dim1, op.dim2)
                val sumGrad = grads.orInsert(op.arg)
                grads.insert(op.arg, sumGrad.add(argGrad))
            }
            is Op.Permute -> {
                val invDims = IntArray(op.dims.size)
                for ((i, dimIdx) in op.dims.withIndex()) {
                    invDims[dimIdx] = i
                }
                val argGrad = detachedGrad.permute(*invDims)
                val sumGrad = grads.orInsert(op.arg)
                grads.insert(op.arg, sumGrad.add(argGrad))
            }
            else -> {}
        }
    }
    return grads
}
