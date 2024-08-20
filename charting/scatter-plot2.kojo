// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("boston-housing.csv"))

val rooms = df.doubleColumn("RM")
val price = df.doubleColumn("MEDV")
val scatter = scatterChart("Correlation between Rooms and Price", "Rooms", "Price", rooms.asDoubleSeq, price.asDoubleSeq)
drawChart(scatter)
