// #include /nn.kojo
clearOutput()

// NDArrays or (Tensors) are the core data-structure for neural networks 
// and deep learning
// The encode the inputs and outputs of neural networks

// Tensors are allocated outside Kojo/JVM's garbage-collected heap memory
// So they need to be closed when they are not needed any more
// An ndScope does this - it closes any tensors created inside it

ndScoped { use =>
    val nm = use(ndMaker)
    val t1 = nm.create(3f)
    val t2 = nm.create(5f)
    val t3 = t1.mul(t2)
    println(t3.getFloat())
}

println("\n***\n")

// Now let's do some interesting calulations with tensors

// Here's our function of interest that we want to play with using tensors
// y = 3 x^2
// dy/dx = 6 x
// at x = 4, y = 48, dy/dx = 24

// Let's define the function in Scala
def y(x: NDArray): NDArray = {
    x.pow(2).mul(3)
}

// Let's define another function to calculate the numerical gradient
// (which is an approximation of the mathematical gradient)
// of the above function at a given x
def numericalYGradient(atx: NDArray): NDArray = {
    val h = 0.001f
    val y0 = y(atx)
    val y1 = y(atx.add(h))
    y1.sub(y0).div(h)
}

ndScoped { use =>
    val nm = use(ndMaker)
    
    // The gradient collector's job is to compute gradients for any
    // mathematical expression involving tensors
    val gc = use(gradientCollector)

    // create an x tensor at which we want to determine our
    // function's gradient
    val x1 = nm.create(4f)
    x1.setRequiresGradient(true)

    // calculate the corresponding y for the given x
    val y1 = y(x1)

    // Now calculate the gradient using the gradient collector
    gc.backward(y1)
    
    println(s"At x=${x1.getFloat()}, y=${y1.getFloat()}, dy/dx = ${x1.getGradient.getFloat()}")
    println("---")

    // Finally, calculate the numerical gradient and compare with the 
    // gradient calculated by the gradient-collector
    val ng = numericalYGradient(x1)
    println(s"At x=${x1.getFloat()}, numerical dy/dx = ${ng.getFloat()}")
}
