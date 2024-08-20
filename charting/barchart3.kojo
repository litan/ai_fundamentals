// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("student-hobbies.csv"))

val area = df.stringColumn("Hobby")
val areaCounts = area.countByCategory
val areas = areaCounts.stringColumn("Category")
val counts = areaCounts.intColumn("Count")

val bar = barChart("Interests", "Hobby", "Count", areas.asStringSeq, counts.asIntSeq)
drawChart(bar)
