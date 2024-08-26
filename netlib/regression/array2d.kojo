// #include neural-net.kojo

cleari()
clearOutput()

val data = Array(
    Array(1f, 2f, 3f),
    Array(4f, 5f, 6f),
)

val nm = NDManager.newBaseManager()

val nData = nm.create(data)

nData.close()
nm.close()