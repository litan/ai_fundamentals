// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("student-marks-info.csv"))


val mathMarks = df.intColumn("Math")
val hist = histogram("Math Marks", "Marks", "# Students", mathMarks.asDoubleSeq, 4)
drawChart(hist)