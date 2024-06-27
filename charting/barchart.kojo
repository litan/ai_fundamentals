// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("student-marks-info.csv"))

val mathMarks = df.intColumn("Math")
val names = df.stringColumn("Name")
val bar = barChart("Math Marks", "Names", "Marks", names.asStringSeq, mathMarks.asIntSeq)
bar.getStyler.setAxisTickLabelsFont(Font("Mono", 12))
bar.getStyler.setXAxisLabelRotation(60)
drawChart(bar)
