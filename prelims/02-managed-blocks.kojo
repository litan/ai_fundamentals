// #include /nn.kojo

// Image that Resource1 and Resource2 are heavy resources that 
// need to be closed when they are done
// so that system resources that they use can be freed up
class Resource1 extends AutoCloseable {
    def work() {
        println("Resource 1 did some work")
    }

    def problem() {
        throw new RuntimeException("Resource 1 problem")
    }

    def close() {
        println("Closing resource 1")
    }
}

class Resource2 extends AutoCloseable {
    def work() {
        println("Resource 2 did some work")
    }

    def problem() {
        throw new RuntimeException("Resource 2 problem")
    }

    def close() {
        println("Closing resource 2")
    }
}

// managed blocks provide managed `use`
// anything acquired via `use` is guaranteed to get closed

// r1 exception occurs before r2 usage, 
// r1 anb r2 still get closed
// resources acquired later via `use` are closed first
managed { use =>
    val r1 = use(new Resource1)
    val r2 = use(new Resource2)
    r1.work()
    r1.problem()
    r2.work()
}
