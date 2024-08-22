// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("xy.csv"))

val x = df.intColumn("X")
val y = df.doubleColumn("Y")
val scatter = scatterChart("X vs Y", "X", "Y", x.asDoubleSeq, y.asDoubleSeq)
drawChart(scatter)
