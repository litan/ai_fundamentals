// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("student-marks-info.csv"))

val area = df.stringColumn("Area")
val areaCounts = area.countByCategory
val areas = areaCounts.stringColumn("Category")
val counts = areaCounts.intColumn("Count")

val pie = pieChart("Locality Distribution", areas.asStringSeq, counts.asIntSeq)
drawChart(pie)
