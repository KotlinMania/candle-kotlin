// port-lint: source backprop.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleException as Error
import io.github.kotlinmania.envie.getenv
import io.github.kotlinmania.threadlocal.ThreadLocal

private val <T> T.self: T get() = this
private fun <T> Ok(v: T): T = v
private val Err: (Throwable) -> Nothing get() = { throw it }
private fun <K, V> MutableMap<K, V>.insert(k: K, v: V): V? = put(k, v)
private fun <K, V> MutableMap<K, V>.keys(): Set<K> = this.keys
private fun <K, V> MutableMap<K, V>.entry(k: K): V? = this[k]
private val Tensor.intoMut: () -> Tensor get() = { this }
private val Tensor.get: () -> Tensor get() = { this }
private fun <T : Any> T?.expect(msg: String): T = this ?: error(msg)
private val Tensor.reshape: (List<Int>) -> Tensor get() = { dims -> reshape(Shape(dims)) }
private val Tensor.broadcastAs: (List<Int>) -> Tensor get() = { dims -> broadcastAs(Shape(dims)) }
private val Tensor.Self.zeros: (List<Int>, DType, Device) -> Tensor
    get() = { dims, dtype, dev -> Tensor.zeros(Shape(dims), dtype, dev) }
private val Tensor.Self.ones3: (Triple<Int, Int, Int>, DType, Device) -> Tensor
    get() = { d, dtype, dev -> Tensor.ones(Shape(d.first, d.second, d.third), dtype, dev) }
private val Tensor.Self.ones4: (List<Int>, DType, Device) -> Tensor
    get() = { dims, dtype, dev -> Tensor.ones(Shape(dims), dtype, dev) }
private val Double.sqrt: () -> Double get() = { kotlin.math.sqrt(this) }
private val IntRange.collect: () -> MutableList<Int> get() = { toMutableList() }

private val Pair<GradStore, Tensor>.grad: Tensor get() = first.get(second)!!
private val Pair<GradStore, Tensor>.add: (Tensor) -> Pair<GradStore, Tensor>
    get() = { rhs ->
        first.insert(second, grad.add(rhs))
        this
    }
private val Pair<GradStore, Tensor>.sub: (Tensor) -> Pair<GradStore, Tensor>
    get() = { rhs ->
        first.insert(second, grad.sub(rhs))
        this
    }
private val Pair<GradStore, Tensor>.scatterAdd: (Tensor, Tensor, Int) -> Pair<GradStore, Tensor>
    get() = { i, s, d ->
        first.insert(second, grad.scatterAdd(i, s, d))
        this
    }
private val Pair<GradStore, Tensor>.indexAdd: (Tensor, Tensor, Int) -> Pair<GradStore, Tensor>
    get() = { i, s, d ->
        first.insert(second, grad.indexAdd(i, s, d))
        this
    }
private val Pair<GradStore, Tensor>.dims: () -> List<Int> get() = { grad.dims() }
private val Pair<GradStore, Tensor>.device: () -> Device get() = { grad.device() }
private val Pair<GradStore, Tensor>.set: (Tensor) -> Pair<GradStore, Tensor>
    get() = { v ->
        first.insert(second, v)
        this
    }

// arg has been reduced to node via reduceDims, expand it back to arg.
// This has to handle keepdims.
private fun broadcastBack(arg: Tensor, node: Tensor, reducedDims: List<Int>): Tensor {
    return if (arg.rank() == node.rank()) {
        // keepdim = true
        node.broadcastAs(arg.shape())
    } else {
        // keepdim = false
        // first expand the reduced dims.
        node.reshape(reducedDims).broadcastAs(arg.shape())
    }
}

private val CANDLE_GRAD_DO_NOT_DETACH: ThreadLocal<Boolean> = ThreadLocal()
private val ThreadLocal<Boolean>.with: ((Boolean) -> Boolean) -> Boolean
    get() = { f ->
        val b = getOr {
            val s = getenv("CANDLE_GRAD_DO_NOT_DETACH")
            s != null && !s.isEmpty() && s != "0"
        }
        f(b)
    }

private val opTernaryArgs: (Op) -> Triple<Tensor, Tensor, Tensor> = { op ->
    when (op) {
        is Op.IndexAdd -> Triple(op.init, op.indexes, op.src)
        is Op.Scatter -> Triple(op.arg, op.index, op.src)
        is Op.ScatterAdd -> Triple(op.arg, op.index, op.src)
        is Op.CustomOp3 -> Triple(op.t1, op.t2, op.t3)
        is Op.WhereCond -> Triple(op.cond, op.onTrue, op.onFalse)
        else -> throw IllegalStateException()
    }
}

private val opBinaryArgs: (Op) -> Pair<Tensor, Tensor> = { op ->
    when (op) {
        is Op.Conv1D -> Pair(op.arg, op.kernel)
        is Op.ConvTranspose1D -> Pair(op.arg, op.kernel)
        is Op.Conv2D -> Pair(op.arg, op.kernel)
        is Op.ConvTranspose2D -> Pair(op.arg, op.kernel)
        is Op.CustomOp2 -> Pair(op.lhs, op.rhs)
        is Op.Binary -> Pair(op.lhs, op.rhs)
        is Op.Gather -> Pair(op.arg, op.index)
        is Op.IndexSelect -> Pair(op.arg, op.index)
        is Op.Matmul -> Pair(op.lhs, op.rhs)
        is Op.SliceScatter0 -> Pair(op.lhs, op.rhs)
        else -> throw IllegalStateException()
    }
}

private val opUnaryArg: (Op) -> Tensor = { op ->
    when (op) {
        is Op.Reshape -> op.arg
        is Op.UpsampleNearest1D -> op.arg
        is Op.UpsampleNearest2D -> op.arg
        is Op.UpsampleBilinear2D -> op.arg
        is Op.AvgPool2D -> op.arg
        is Op.MaxPool2D -> op.arg
        is Op.Copy -> op.arg
        is Op.Broadcast -> op.arg
        is Op.Cmp -> op.arg
        is Op.Reduce -> op.arg
        is Op.ToDevice -> op.arg
        is Op.Transpose -> op.arg
        is Op.Permute -> op.arg
        is Op.Narrow -> op.arg
        is Op.Unary -> op.arg
        is Op.Elu -> op.arg
        is Op.Powf -> op.arg
        is Op.CustomOp1 -> op.arg
        is Op.ToDType -> op.arg
        else -> throw IllegalStateException()
    }
}

/**
 * Return all the nodes that lead to this value in a topologically sorted vec, the first
 * elements having dependencies on the latter ones, e.g. the first element if any is the
 * argument.
 * This assumes that the op graph is a DAG.
 */
public fun Tensor.sortedNodes(): List<Tensor> {
    // The list of sorted nodes is passed as an owned value rather than a mutable reference.
    fun walk(
        node: Tensor,
        nodes: MutableList<Tensor>,
        alreadySeen: MutableMap<TensorId, Boolean>,
    ): Pair<Boolean, MutableList<Tensor>> {
        val tg0 = alreadySeen.get(node.id())
        if (tg0 != null) {
            return Pair(tg0, nodes)
        }
        var trackGrad = false
        val outNodes = if (node.isVariable()) {
            // Do not call recursively on the "leaf" nodes.
            trackGrad = true
            nodes
        } else if (node.dtype().isInt()) {
            nodes
        } else {
            val op = node.op()
            if (op != null) {
                when (op) {
                    is Op.IndexAdd,
                    is Op.Scatter,
                    is Op.ScatterAdd,
                    is Op.CustomOp3,
                    is Op.WhereCond -> {
                        val (t1, t2, t3) = opTernaryArgs(op)
                        val (tg1, nodes1) = walk(t1, nodes, alreadySeen)
                        trackGrad = trackGrad || tg1
                        val (tg2, nodes2) = walk(t2, nodes1, alreadySeen)
                        trackGrad = trackGrad || tg2
                        val (tg3, nodes3) = walk(t3, nodes2, alreadySeen)
                        trackGrad = trackGrad || tg3
                        nodes3
                    }
                    is Op.Conv1D,
                    is Op.ConvTranspose1D,
                    is Op.Conv2D,
                    is Op.ConvTranspose2D,
                    is Op.CustomOp2,
                    is Op.Binary,
                    is Op.Gather,
                    is Op.IndexSelect,
                    is Op.Matmul,
                    is Op.SliceScatter0 -> {
                        val (lhs, rhs) = opBinaryArgs(op)
                        val (tg1, nodes1) = walk(lhs, nodes, alreadySeen)
                        trackGrad = trackGrad || tg1
                        val (tg2, nodes2) = walk(rhs, nodes1, alreadySeen)
                        trackGrad = trackGrad || tg2
                        nodes2
                    }
                    is Op.Cat -> op.args.iter().fold(nodes) { accNodes, arg ->
                        val (tg, next) = walk(arg, accNodes, alreadySeen)
                        trackGrad = trackGrad || tg
                        next
                    }
                    is Op.Affine -> {
                        if (op.mul == 0.0) {
                            nodes
                        } else {
                            val (tg, next) = walk(op.arg, nodes, alreadySeen)
                            trackGrad = trackGrad || tg
                            next
                        }
                    }
                    is Op.Unary -> {
                        when (op.op) {
                            UnaryOp.Ceil, UnaryOp.Floor, UnaryOp.Round, UnaryOp.Sign -> nodes
                            else -> {
                                val (tg, next) = walk(op.arg, nodes, alreadySeen)
                                trackGrad = trackGrad || tg
                                next
                            }
                        }
                    }
                    is Op.Reshape,
                    is Op.UpsampleNearest1D,
                    is Op.UpsampleNearest2D,
                    is Op.UpsampleBilinear2D,
                    is Op.AvgPool2D,
                    is Op.MaxPool2D,
                    is Op.Copy,
                    is Op.Broadcast,
                    is Op.Cmp,
                    is Op.ToDevice,
                    is Op.Transpose,
                    is Op.Permute,
                    is Op.Narrow,
                    is Op.Elu,
                    is Op.Powf,
                    is Op.CustomOp1 -> {
                        val (tg, next) = walk(opUnaryArg(op), nodes, alreadySeen)
                        trackGrad = trackGrad || tg
                        next
                    }
                    is Op.ToDType -> {
                        if (op.arg.dtype().isFloat()) {
                            val (tg, next) = walk(op.arg, nodes, alreadySeen)
                            trackGrad = trackGrad || tg
                            next
                        } else {
                            nodes
                        }
                    }
                    is Op.Reduce -> {
                        when (op.op) {
                            ReduceOp.Min, ReduceOp.Sum, ReduceOp.Max -> {
                                val (tg, next) = walk(op.arg, nodes, alreadySeen)
                                trackGrad = trackGrad || tg
                                next
                            }
                            ReduceOp.ArgMin, ReduceOp.ArgMax -> nodes
                        }
                    }
                }
            } else {
                nodes
            }
        }
        alreadySeen.insert(node.id(), trackGrad)
        if (trackGrad) {
            outNodes.push(node)
        }
        return Pair(trackGrad, outNodes)
    }
    val (_, nodes) = walk(self, mutableListOf(), HashMap())
    nodes.reverse()
    return nodes
}

public fun Tensor.backward(): GradStore {
    val sortedNodes = self.sortedNodes()
    val grads = GradStore.new()
    grads.insert(self, self.onesLike().contiguous())
    for (node in sortedNodes.iter()) {
        if (node.isVariable()) {
            continue
        }
        val grad0 = grads
            .remove(node)
            .expect("candle internal error - grad not populated")
        // https://github.com/huggingface/candle/issues/1241
        // Ideally, we would make these operations in place where possible to ensure that we
        // do not have to allocate too often. Here we just call `.detach` to avoid computing
        // the backprop graph of the backprop itself. This would be an issue for second order
        // derivatives but these are out of scope at the moment.
        val doNotDetach = CANDLE_GRAD_DO_NOT_DETACH.with { b -> b }
        val grad = if (doNotDetach) grad0 else grad0.detach()
        val op = node.op()
        if (op != null) {
            when (op) {
                is Op.Binary -> {
                    val (lhs, rhs, binOp) = op
                    when (binOp) {
                        BinaryOp.Add -> {
                            var lhsSumGrad = grads.orInsert(lhs)
                            lhsSumGrad = lhsSumGrad.add(grad)
                            var rhsSumGrad = grads.orInsert(rhs)
                            rhsSumGrad = rhsSumGrad.add(grad)
                        }
                        BinaryOp.Sub -> {
                            var lhsSumGrad = grads.orInsert(lhs)
                            lhsSumGrad = lhsSumGrad.add(grad)
                            var rhsSumGrad = grads.orInsert(rhs)
                            rhsSumGrad = rhsSumGrad.sub(grad)
                        }
                        BinaryOp.Mul -> {
                            val lhsGrad = grad.mul(rhs)
                            var lhsSumGrad = grads.orInsert(lhs)
                            lhsSumGrad = lhsSumGrad.add(lhsGrad)
                            val rhsGrad = grad.mul(lhs)
                            var rhsSumGrad = grads.orInsert(rhs)
                            rhsSumGrad = rhsSumGrad.add(rhsGrad)
                        }
                        BinaryOp.Div -> {
                            val lhsGrad = grad.div(rhs)
                            var lhsSumGrad = grads.orInsert(lhs)
                            lhsSumGrad = lhsSumGrad.add(lhsGrad)
                            val rhsGrad = grad.mul(lhs).div(rhs.sqr())
                            var rhsSumGrad = grads.orInsert(rhs)
                            rhsSumGrad = rhsSumGrad.sub(rhsGrad)
                        }
                        BinaryOp.Minimum,
                        BinaryOp.Maximum -> {
                            val maskLhs = node.eq(lhs).toDtype(grad.dtype())
                            val maskRhs = node.eq(rhs).toDtype(grad.dtype())

                            // If both masks are 1 one the same point, we want to scale the
                            // gradient by 0.5 rather than 1.
                            val lhsGrad = maskLhs.mul(grad).div(maskRhs.affine(1.0, 1.0))
                            var lhsSumGrad = grads.orInsert(lhs)
                            lhsSumGrad = lhsSumGrad.add(lhsGrad)

                            val rhsGrad = maskRhs.mul(grad).div(maskLhs.affine(1.0, 1.0))
                            var rhsSumGrad = grads.orInsert(rhs)
                            rhsSumGrad = rhsSumGrad.add(rhsGrad)
                        }
                    }
                }
                is Op.WhereCond -> {
                    val (pred, t, f) = op
                    val zeros = grad.zerosLike()
                    var tSumGrad = grads.orInsert(t)
                    val tGrad = pred.whereCond(grad, zeros)
                    tSumGrad = tSumGrad.add(tGrad)
                    var fSumGrad = grads.orInsert(f)
                    val fGrad = pred.whereCond(zeros, grad)
                    fSumGrad = fSumGrad.add(fGrad)
                }
                is Op.Conv1D -> {
                    val (arg, kernel, padding, stride, dilation) = op
                    // The output height for convTranspose1d is:
                    // (lIn - 1) * stride - 2 * padding + dilation * (kSize - 1) + outPadding + 1
                    val gradLIn = grad.dim(2)
                    val kSize = kernel.dim(2)
                    val outSize =
                        (gradLIn - 1) * stride + dilation * (kSize - 1) + 1 - 2 * padding
                    val outPadding = arg.dim(2) - outSize
                    val gradArg = grad.convTranspose1d(
                        kernel,
                        padding,
                        outPadding,
                        stride,
                        dilation,
                        /* groups */ 1,
                    )
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(gradArg)

                    val gradKernel0 = arg
                        .transpose(0, 1)
                        .conv1d(grad.transpose(0, 1), padding, dilation, stride, 1)
                        .transpose(0, 1)
                    var sumGradK = grads.orInsert(kernel)
                    val (_, _, k0) = kernel.dims3()
                    val (_, _, gK0) = gradKernel0.dims3()
                    val gradKernel = if (gK0 != k0) {
                        gradKernel0.narrow(2, 0, k0)
                    } else {
                        gradKernel0
                    }
                    sumGradK = sumGradK.add(gradKernel)
                }
                is Op.Conv2D -> {
                    val (arg, kernel, padding, stride, dilation) = op
                    // The output height for convTranspose2d is:
                    // (iH - 1) * stride - 2 * padding + dilation * (kH - 1) + outPadding + 1
                    val gradH = grad.dim(2)
                    val kH = kernel.dim(2)
                    val outSize =
                        (gradH - 1) * stride + dilation * (kH - 1) + 1 - 2 * padding
                    val outPadding = arg.dim(2) - outSize
                    val gradArg = grad.convTranspose2d(
                        kernel,
                        padding,
                        outPadding,
                        stride,
                        dilation,
                    )
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(gradArg)

                    val gradKernel0 = arg
                        .transpose(0, 1)
                        .conv2d(grad.transpose(0, 1), padding, dilation, stride, 1)
                        .transpose(0, 1)
                    var sumGradK = grads.orInsert(kernel)
                    val (_, _, k0, k1) = kernel.dims4()
                    val (_, _, gK0, gK1) = gradKernel0.dims4()
                    val gradKernel = if (gK0 != k0 || gK1 != k1) {
                        gradKernel0.narrow(2, 0, k0).narrow(3, 0, k1)
                    } else {
                        gradKernel0
                    }
                    sumGradK = sumGradK.add(gradKernel)
                }
                is Op.ConvTranspose1D -> Err(
                    Error.BackwardNotSupported(op = "conv-transpose1d"),
                )
                is Op.ConvTranspose2D -> {
                    val (arg, kernel, padding, _, stride, dilation) = op
                    val gradArg = grad.conv2d(kernel, padding, stride, dilation, 1)
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(gradArg)

                    val gradKernel0 = grad
                        .transpose(0, 1)
                        .conv2d(arg.transpose(0, 1), padding, dilation, stride, 1)
                        .transpose(0, 1)
                    var sumGradK = grads.orInsert(kernel)
                    val (_, _, k0, k1) = kernel.dims4()
                    val (_, _, gK0, gK1) = gradKernel0.dims4()
                    val gradKernel = if (gK0 != k0 || gK1 != k1) {
                        gradKernel0.narrow(2, 0, k0).narrow(3, 0, k1)
                    } else {
                        gradKernel0
                    }
                    sumGradK = sumGradK.add(gradKernel)
                }
                is Op.AvgPool2D -> {
                    val (arg, kernelSize, stride) = op
                    if (kernelSize != stride) {
                        bail("backward not supported for avgpool2d if ksize $kernelSize != stride $stride")
                    }
                    val (_, _, h, w) = arg.dims4()
                    val gradArg0 = grad.upsampleNearest2d(h, w)
                    val gradArg =
                        gradArg0.affine(1.0 / (kernelSize.first * kernelSize.second).toDouble(), 0.0)
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(gradArg)
                }
                is Op.MaxPool2D -> {
                    val (arg, kernelSize, stride) = op
                    if (kernelSize != stride) {
                        bail("backward not supported for maxpool2d if ksize $kernelSize != stride $stride")
                    }
                    val (_, _, h, w) = arg.dims4()
                    // For computing the max-pool gradient, we compute a mask where a 1 means
                    // that the element is the maximum, then we apply this mask to the
                    // upsampled gradient (taking into account that multiple max may exist so
                    // we scale the gradient for this case).
                    val nodeUpsampled = node.upsampleNearest2d(h, w)
                    val mask = arg.eq(nodeUpsampled).toDtype(arg.dtype())
                    val avg = mask.avgPool2dWithStride(kernelSize, stride)
                    val gradArg = grad.mul(avg).upsampleNearest2d(h, w).mul(mask)
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(gradArg)
                }
                is Op.UpsampleNearest1D -> {
                    val (arg, targetSize) = op
                    val (_, c, size) = arg.dims3()
                    if (targetSize % size != 0) {
                        bail("backward not supported for non integer upscaling factors")
                    }
                    val scale = targetSize / size

                    val kernel = Tensor.ones3(Triple(c, 1, scale), arg.dtype(), arg.device())
                    val convSum = grad.conv1d(kernel, 0, scale, 1, c)
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.set(convSum)
                }
                is Op.UpsampleNearest2D -> {
                    val (arg, targetH, targetW) = op
                    val (_, c, h, w) = arg.dims4()
                    if (targetH % h != 0 || targetW % w != 0) {
                        bail("backward not supported for non integer upscaling factors")
                    }
                    val scaleH = targetH / h
                    val scaleW = targetW / w

                    if (scaleH != scaleW) {
                        bail("backward not supported for non uniform upscaling factors")
                    }
                    val kernel =
                        Tensor.ones4(listOf(c, 1, scaleH, scaleW), arg.dtype(), arg.device())
                    val convSum = grad.conv2d(kernel, 0, scaleH, 1, c)
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.set(convSum)
                }
                is Op.UpsampleBilinear2D -> {
                    bail("backward not supported for upsample_bilinear2d")
                }
                is Op.SliceScatter0 -> {
                    val (lhs, rhs, startRhs) = op
                    var rhsSumGrad = grads.orInsert(rhs)
                    val rhsGrad = grad.narrow(0, startRhs, rhs.dim(0))
                    rhsSumGrad = rhsSumGrad.add(rhsGrad)

                    var lhsSumGrad = grads.orInsert(lhs)
                    val lhsGrad = grad.sliceScatter0(rhs.zerosLike(), startRhs)
                    lhsSumGrad = lhsSumGrad.add(lhsGrad)
                }
                is Op.Gather -> {
                    val (arg, indexes, dim) = op
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.scatterAdd(indexes, grad, dim)
                }
                is Op.Scatter -> {
                    val (init, indexes, src, dim) = op
                    var initSumGrad = grads.orInsert(init)
                    initSumGrad = initSumGrad.add(grad)

                    val srcGrad = grad.gather(indexes, dim)
                    var srcSumGrad = grads.orInsert(src)
                    srcSumGrad = srcSumGrad.add(srcGrad)
                }
                is Op.ScatterAdd -> {
                    val (init, indexes, src, dim) = op
                    var initSumGrad = grads.orInsert(init)
                    val mask0 = init.onesLike()
                    val mask = mask0.scatter(indexes, mask0.zerosLike(), dim)
                    initSumGrad = initSumGrad.add(grad.mul(mask))

                    val srcGrad = grad.gather(indexes, dim)
                    var srcSumGrad = grads.orInsert(src)
                    srcSumGrad = srcSumGrad.add(srcGrad)
                }
                is Op.IndexAdd -> {
                    val (init, indexes, src, dim) = op
                    var initSumGrad = grads.orInsert(init)
                    initSumGrad = initSumGrad.add(grad)

                    val srcGrad = grad.indexSelect(indexes, dim)
                    var srcSumGrad = grads.orInsert(src)
                    srcSumGrad = srcSumGrad.add(srcGrad)
                }
                is Op.IndexSelect -> {
                    val (arg, indexes, dim) = op
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.indexAdd(indexes, grad, dim)
                }
                is Op.Matmul -> {
                    val (lhs, rhs) = op
                    // Skipping checks, the op went ok, we can skip
                    // the matmul size checks for now.

                    val lhsGrad = grad.matmul(rhs.t())
                    var lhsSumGrad = grads.orInsert(lhs)
                    lhsSumGrad = lhsSumGrad.add(lhsGrad)

                    val rhsGrad = lhs.t().matmul(grad)
                    var rhsSumGrad = grads.orInsert(rhs)
                    rhsSumGrad = rhsSumGrad.add(rhsGrad)
                }
                is Op.Cat -> {
                    val (args, dim) = op
                    var startIdx = 0
                    for (arg in args) {
                        val len = arg.dims()[dim]
                        val argGrad = grad.narrow(dim, startIdx, len)
                        var sumGrad = grads.orInsert(arg)
                        sumGrad = sumGrad.add(argGrad)
                        startIdx += len
                    }
                }
                is Op.Broadcast -> {
                    val (arg) = op
                    val argDims = arg.dims()
                    val nodeDims = node.dims()
                    // The number of dims that have been inserted on the left.
                    val leftDims = nodeDims.len() - argDims.len()
                    val sumDims: MutableList<Int> = (0 until leftDims).collect()
                    for (dim in argDims.indices) {
                        val nodeDim = nodeDims[leftDims + dim]
                        val argDim = argDims[dim]
                        if (nodeDim != argDim) {
                            sumDims.push(dim + leftDims)
                        }
                    }

                    var argGrad = grad.sumKeepdim(sumDims.asSlice())
                    for (i in 0 until leftDims) {
                        argGrad = argGrad.squeeze(0)
                    }
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(argGrad.broadcastAs(sumGrad.dims()))
                }
                is Op.Reduce -> {
                    val (arg, reduceOp, reducedDims) = op
                    when (reduceOp) {
                        ReduceOp.Sum -> {
                            val gradB = broadcastBack(arg, grad, reducedDims)
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.add(gradB)
                        }
                        ReduceOp.Max -> {
                            val nodeB = broadcastBack(arg, node, reducedDims)
                            val grad0 = broadcastBack(arg, grad, reducedDims)
                            val gradB = nodeB.eq(arg).toDtype(grad0.dtype()).mul(grad0)
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.add(gradB.broadcastAs(sumGrad.dims()))
                        }
                        ReduceOp.Min -> {
                            val nodeB = broadcastBack(arg, node, reducedDims)
                            val grad0 = broadcastBack(arg, grad, reducedDims)
                            val gradB = nodeB.eq(arg).toDtype(grad0.dtype()).mul(grad0)
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.add(gradB.broadcastAs(sumGrad.dims()))
                        }
                        ReduceOp.ArgMin, ReduceOp.ArgMax -> Unit
                    }
                }
                is Op.ToDType -> {
                    val (arg) = op
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(grad.toDtype(arg.dtype()))
                }
                is Op.Copy -> {
                    val (arg) = op
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(grad)
                }
                is Op.Affine -> {
                    val (arg, mul) = op
                    val argGrad = grad.affine(mul, 0.0)
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(argGrad)
                }
                is Op.Unary -> {
                    val (arg, unaryOp) = op
                    when (unaryOp) {
                        UnaryOp.Log -> {
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.add(grad.div(arg))
                        }
                        UnaryOp.Sin -> {
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.add(grad.mul(arg.cos()))
                        }
                        UnaryOp.Cos -> {
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.sub(grad.mul(arg.sin()))
                        }
                        UnaryOp.Tanh -> {
                            var sumGrad = grads.orInsert(arg)
                            val minusDtanh = node.sqr().affine(1.0, -1.0)
                            sumGrad = sumGrad.sub(grad.mul(minusDtanh))
                        }
                        UnaryOp.Abs -> {
                            var sumGrad = grads.orInsert(arg)
                            val ones = arg.onesLike()
                            val absGrad = arg.ge(arg.zerosLike()).whereCond(ones, ones.neg())
                            sumGrad = sumGrad.add(grad.mul(absGrad))
                        }
                        UnaryOp.Exp -> {
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.add(grad.mul(node))
                        }
                        UnaryOp.Neg -> {
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.sub(grad)
                        }
                        UnaryOp.Recip -> {
                            var sumGrad = grads.orInsert(arg)
                            val gradRecip = grad.div(arg.sqr())
                            sumGrad = sumGrad.sub(gradRecip)
                        }
                        UnaryOp.Floor, UnaryOp.Round, UnaryOp.Sign -> Unit
                        UnaryOp.Ceil -> Err(Error.BackwardNotSupported(op = "ceil"))
                        UnaryOp.Gelu -> {
                            var sumGrad = grads.orInsert(arg)
                            val cube = arg.powf(3.0)
                            val tanh = cube.affine(0.0356774, 0.0).add(arg.affine(0.797885, 0.0)).tanh()
                            val geluGrad = tanh.affine(0.5, 0.0)
                                .add(cube.affine(0.0535161, 0.0).add(arg.affine(0.398942, 0.0)).mul(tanh.powf(2.0).affine(-1.0, 1.0)))
                                .affine(1.0, 0.5)
                            sumGrad = sumGrad.add(grad.mul(geluGrad))
                        }
                        UnaryOp.Erf -> {
                            var sumGrad = grads.orInsert(arg)
                            // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
                            val erfGrad =
                                arg.sqr().neg().exp().affine(2.0 / kotlin.math.PI.sqrt(), 0.0)
                            sumGrad = sumGrad.add(grad.mul(erfGrad))
                        }
                        UnaryOp.GeluErf -> {
                            var sumGrad = grads.orInsert(arg)
                            // d/dx geluErf(x) = 0.5 + 0.398942 e^(-x^2/2) x + 0.5 erf(x/sqrt(2))
                            val negHalfSquare = arg.sqr().neg().affine(0.5, 0.0)
                            val scaledExpArg = negHalfSquare.exp().affine(0.398942, 0.0).mul(arg)
                            val argScaledSqrt = arg.affine(1.0 / 2.0.sqrt(), 0.0)
                            val erfScaledSqrt = argScaledSqrt.erf().affine(0.5, 0.0)
                            val geluErfGrad = scaledExpArg.add(erfScaledSqrt).affine(1.0, 0.5)
                            sumGrad = sumGrad.add(grad.mul(geluErfGrad))
                        }
                        UnaryOp.Relu -> {
                            var sumGrad = grads.orInsert(arg)
                            val reluGrad = arg.ge(arg.zerosLike()).toDtype(arg.dtype())
                            sumGrad = sumGrad.add(grad.mul(reluGrad))
                        }
                        UnaryOp.Silu -> {
                            var sumGrad = grads.orInsert(arg)
                            // d/dx silu = sigmoid(x) * (1 + x * (1 - sigmoid(x))) = sigmoid(x) * (1 - node) + node
                            val sigmoidArg = arg.neg().exp().affine(1.0, 1.0).recip()
                            val siluGrad = sigmoidArg.mul(node.affine(-1.0, 1.0)).add(node)
                            sumGrad = sumGrad.add(grad.mul(siluGrad))
                        }
                        UnaryOp.Sqr -> {
                            val argGrad = arg.mul(grad).affine(2.0, 0.0)
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.add(argGrad)
                        }
                        UnaryOp.Sqrt -> {
                            val argGrad = grad.div(node).affine(0.5, 0.0)
                            var sumGrad = grads.orInsert(arg)
                            sumGrad = sumGrad.add(argGrad)
                        }
                    }
                }
                is Op.Narrow -> {
                    val (arg, dim, startIdx, len) = op
                    val argDims = arg.dims()
                    val leftPad = if (startIdx == 0) {
                        null
                    } else {
                        val dims = argDims.toVec()
                        dims[dim] = startIdx
                        Tensor.zeros(dims, grad.dtype(), grad.device())
                    }
                    val rightPad0 = argDims[dim] - startIdx - len
                    val rightPad = if (rightPad0 == 0) {
                        null
                    } else {
                        val dims = argDims.toVec()
                        dims[dim] = rightPad0
                        Tensor.zeros(dims, grad.dtype(), grad.device())
                    }
                    val argGrad = when {
                        leftPad == null && rightPad == null -> grad
                        leftPad != null && rightPad == null -> Tensor.cat(listOf(leftPad, grad), dim)
                        leftPad == null && rightPad != null -> Tensor.cat(listOf(grad, rightPad), dim)
                        else -> Tensor.cat(listOf(leftPad!!, grad, rightPad!!), dim)
                    }
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(argGrad)
                }
                is Op.Cmp -> Unit
                is Op.Reshape -> {
                    val (arg) = op
                    val argGrad = grad.reshape(arg.dims())
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(argGrad)
                }
                is Op.Elu -> {
                    val (arg, alpha) = op
                    // d/dx elu(x) = 1 for x > 0, alpha * e^x for x <= 0
                    var sumGrad = grads.orInsert(arg)
                    val zeros = arg.zerosLike()
                    val positiveMask = arg.gt(zeros).toDtype(arg.dtype())
                    val negativeMask = arg.le(zeros).toDtype(arg.dtype())
                    // node == alpha * (e^x - 1) for x <= 0, reuse it
                    val negativeExpMask = negativeMask.mul(node.affine(1.0, alpha))
                    val combinedMask = positiveMask.add(negativeExpMask)
                    sumGrad = sumGrad.add(grad.mul(combinedMask))
                }
                is Op.Powf -> {
                    val (arg, e) = op
                    val argGrad = grad.mul(arg.powf(e - 1.0)).affine(e, 0.0)
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(argGrad)
                }
                is Op.CustomOp1 -> {
                    val (arg, c) = op
                    val argGrad = c.bwd(arg, node, grad)
                    if (argGrad != null) {
                        var sumGrad = grads.orInsert(arg)
                        sumGrad = sumGrad.add(argGrad)
                    }
                }
                is Op.CustomOp2 -> {
                    val (arg1, arg2, c) = op
                    val (argGrad1, argGrad2) = c.bwd(arg1, arg2, node, grad)
                    if (argGrad1 != null) {
                        var sumGrad = grads.orInsert(arg1)
                        sumGrad = sumGrad.add(argGrad1)
                    }
                    if (argGrad2 != null) {
                        var sumGrad = grads.orInsert(arg2)
                        sumGrad = sumGrad.add(argGrad2)
                    }
                }
                is Op.CustomOp3 -> {
                    val (arg1, arg2, arg3, c) = op
                    val (argGrad1, argGrad2, argGrad3) = c.bwd(arg1, arg2, arg3, node, grad)
                    if (argGrad1 != null) {
                        var sumGrad = grads.orInsert(arg1)
                        sumGrad = sumGrad.add(argGrad1)
                    }
                    if (argGrad2 != null) {
                        var sumGrad = grads.orInsert(arg2)
                        sumGrad = sumGrad.add(argGrad2)
                    }
                    if (argGrad3 != null) {
                        var sumGrad = grads.orInsert(arg3)
                        sumGrad = sumGrad.add(argGrad3)
                    }
                }
                is Op.ToDevice -> {
                    val (arg) = op
                    var sumGrad = grads.orInsert(arg)
                    val argGrad = grad.toDevice(sumGrad.device())
                    sumGrad = sumGrad.add(argGrad)
                }
                is Op.Transpose -> {
                    val (arg, dim1, dim2) = op
                    val argGrad = grad.transpose(dim1, dim2)
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(argGrad)
                }
                is Op.Permute -> {
                    val (arg, dims) = op
                    val invDims = IntArray(dims.len()) { 0 }
                    for (entry in dims.withIndex()) {
                        invDims[entry.value] = entry.index
                    }
                    val argGrad = grad.permute(*invDims)
                    var sumGrad = grads.orInsert(arg)
                    sumGrad = sumGrad.add(argGrad)
                }
            }
        }
    }
    return Ok(grads)
}

/**
 * A store for gradients, associating a tensor id to the corresponding gradient tensor, used for back propagation.
 */
public class GradStore(
    private val grads: MutableMap<TensorId, Tensor> = mutableMapOf(),
) {
    private val self: MutableMap<TensorId, Tensor> get() = grads

    public companion object {
        public val Self: Companion get() = this
        public val default: () -> GradStore = { GradStore() }

        /**
         * Create a new gradient store
         */
        public fun new(): GradStore {
            return Self.default()
        }
    }

    /**
     * Get the gradient tensor corresponding to the given tensor id
     */
    public fun getId(id: TensorId): Tensor? {
        return self.get(id)
    }

    /**
     * Get the gradient tensor associated with the given tensor
     */
    public fun get(tensor: Tensor): Tensor? {
        return self.get(tensor.id())
    }

    /**
     * Remove the gradient tensor associated with the given tensor, returning it if it exists
     */
    public fun remove(tensor: Tensor): Tensor? {
        return self.remove(tensor.id())
    }

    /**
     * Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
     */
    public fun insert(tensor: Tensor, grad: Tensor): Tensor? {
        return self.insert(tensor.id(), grad)
    }

    /**
     * Insert a gradient tensor associated with the given tensor id, returning the previous gradient tensor if it existed
     */
    public fun insertId(id: TensorId, grad: Tensor): Tensor? {
        return self.insert(id, grad)
    }

    /**
     * Get the gradient tensor associated with the given tensor, or, if it does not exist,
     * insert a tensor of zeroes, with the same shape and type as the given tensors and return it
     */
    internal fun orInsert(tensor: Tensor): Pair<GradStore, Tensor> {
        val entry = self.entry(tensor.id())
        val grad = when (entry != null) {
            true -> entry.intoMut()
            false -> {
                val grad = tensor.zerosLike()
                self.insert(tensor.id(), grad)
                grad
            }
        }
        return Ok(Pair(this, tensor)).let { if (grad === tensor) it else it }
    }

    /**
     * Extend this gradient store with the contents of another.
     * If an entry is already occupied, updates and ensures new tensor follows correct detach semantics.
     * Otherwise simply inserts.
     */
    public fun extend(other: GradStore) {
        for (item in other.self.entries) {
            val id = item.key; val grad = item.value
            val entry = self.entry(id)
            when (entry != null) {
                true -> {
                    val newGrad0 = entry.get().add(grad)
                    val doNotDetach = CANDLE_GRAD_DO_NOT_DETACH.with { b -> b }
                    val newGrad = if (doNotDetach) {
                        newGrad0
                    } else {
                        newGrad0.detach()
                    }
                    self.insert(id, newGrad)
                }
                false -> {
                    self.insert(id, grad)
                }
            }
        }
        Ok(Unit)
    }

    /**
     * Get the tensor ids of the stored gradient tensors
     */
    public fun getIds(): Set<TensorId> {
        return self.keys()
    }
}

