// integers
1 + 7

// double precision floats
1.1 + 7.5

// single precision floats, commonly used in AI and gaming
1.1f + 7.5f

// named values
val x1 = 9 // the rhs can be any expression
// x1 = 7 -> not allowed, as vals are immutable (cannot change)

// vars
var x2 = 9
x2 = 7

// defining a new command/procedure
def printSum(n1: Int, n2: Int) {
    val ans = n1 + n2
    println(ans)
}

// calling the new command
printSum(2, 3)

// defining a new function
def sum(n1: Int, n2: Int): Int = {
    val ans = n1 + n2
    ans
}

// calling the new function, getting a result, and printing it out
val ans = sum(2, 3)
println(ans)

// fixed size sequence
Array(1, 2, 3)

// sequence that can grow in size
val ab = ArrayBuffer.empty[Int]
ab.append(1); ab.append(2); ab.append(3)

// iterate through a sequence
for (n <- ab) {
    // do whatever with current element n
    println(n)
}
// or
ab.foreach { n =>
    println(n)
}

// convert one sequence into another
for (n <- ab) yield (n * 2)
// or
ab.map(n => n * 2)

// creating a new type for storing useful information
case class Point(x: Double, y: Double) {
    def distanceTo(other: Point): Double = {
        // Pythagoras theorem
        math.sqrt(
            math.pow(x - other.x, 2) + math.pow(y - other.y, 2)
        )
    }
}

val p1 = Point(0, 0)
val p2 = Point(3, 4)
p1.distanceTo(p2)

// create a new type to represent an active entity
class GameEntity {
    def update() { /* do what needs to be done */ }
    def draw() { /* do what needs to be done */ }
}

val allEntities = ArrayBuffer.empty[GameEntity]
// add entities to allEntities as needed

def gameLoop() {
    for (e <- allEntities) {
        e.update()
        e.draw()
    }
}

