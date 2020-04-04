import java.util.*
import kotlin.Comparator
import kotlin.collections.HashSet

fun main() {
    val solution = Solution()
//    println(solution.evalRPN(arrayOf("10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+")))
//    println(solution.evalRPN(arrayOf("4", "13", "5", "/", "+")))
//    println(solution.hasGroupsSizeX(intArrayOf(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)))
//    println(solution.hasGroupsSizeX(intArrayOf(1, 1, 1, 1, 2, 2, 3, 3)))
//    println(solution.gcd(27, 15))
//    println(solution.reverseWords(" hello world!"))
//    print(solution.minimumLengthEncoding(arrayOf("time", "me")))
//    print(solution.maxProduct(intArrayOf(2, 3, -2, 4)))
//    print(solution.findMin(intArrayOf(3, 1, 1)))
//    print(solution.maxDistance(arrayOf(intArrayOf(1, 0, 0), intArrayOf(0, 0, 0), intArrayOf(0, 0, 0))))
//    val minStack = MinStack()
//    minStack.push(-2)
//    minStack.push(0)
//    minStack.push(-3)
//    print(minStack.getMin())
//    minStack.pop()
//    print(minStack.top())
//    print(minStack.getMin())
//    print(solution.isMatch("aa", ".a"))
//    print('0'.toInt())
    val board = arrayOf(
        charArrayOf('5','3','.','.','7','.','.','.','.'),
        charArrayOf('6','.','.','1','9','5','.','.','.'),
        charArrayOf('.','9','8','.','.','.','.','6','.'),
        charArrayOf('8','.','.','.','6','.','.','.','3'),
        charArrayOf('4','.','.','8','.','3','.','.','1'),
        charArrayOf('7','.','.','.','2','.','.','.','6'),
        charArrayOf('.','6','.','.','.','.','2','8','.'),
        charArrayOf('.','.','.','4','1','9','.','.','5'),
        charArrayOf('.','.','.','.','8','.','.','7','9')
    )
    solution.solveSudoku(board)
    for (i in 0..8) {
        for (j in 0..8) {
            print(board[i][j])
            print(' ')
        }
        println()
    }
}

class Solution {
    fun evalRPN(tokens: Array<String>): Int {
        // https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/
        val stack = Stack<Int>()
        for (item in tokens) {
            val num = str2Int(item)
            if (num != null) {
                stack.push(num)
            } else {
                val b = stack.pop()
                val a = stack.pop()
                if (a == null || b == null) {
                    return 0
                }
                stack.push(
                    when (item) {
                        "+" -> a + b
                        "-" -> a - b
                        "*" -> a * b
                        "/" -> a / b
                        else -> null
                    }
                )
            }
        }
        return if (stack.size == 1 && stack.peek() != null) {
            stack.peek()
        } else {
            0
        }
    }

    private fun str2Int(arg: String) = try {
        arg.toInt()
    } catch (e: Exception) {
        null
    }

    fun hasGroupsSizeX(deck: IntArray): Boolean {
        // https://leetcode-cn.com/problems/x-of-a-kind-in-a-deck-of-cards/
        if (deck.size <= 1) {
            return false
        }
        deck.sort()
        var count = 0
        var start = 0
        var i = 1
        while (i < deck.size) {
            while (i < deck.size && deck[i] == deck[i - 1]) {
                i++
            }
            count = if (count == 0) {
                i - start
            } else {
                val gcd = gcd(count, i - start)
                if (gcd == 1) {
                    return false
                } else {
                    gcd
                }
            }
            start = i
            i++
        }
        return true
    }

    fun gcd(a: Int, b: Int): Int {
        val l = if (a > b) a else b
        val s = if (a > b) b else a
        return if (l % s == 0) {
            s
        } else {
            gcd(s, l % s)
        }
    }

    fun reverseWords(s: String): String {
        // https://leetcode-cn.com/problems/reverse-words-in-a-string/
        return s.split(' ').filter { it != "" }.reversed().joinToString(separator = " ")
    }

    fun maxProduct(nums: IntArray): Int {
        // https://leetcode-cn.com/problems/maximum-product-subarray/
        var max = Int.MIN_VALUE
        var imax = 1
        var imin = 1
        for (i in nums.indices) {
            if (nums[i] < 0) {
                var temp = imax
                imax = imin
                imin = temp
            }
            imax = (imax * nums[i]).coerceAtLeast(nums[i])
            imin = (imin * nums[i]).coerceAtMost(nums[i])
            max = max.coerceAtLeast(imax)
        }
        return max
    }

    fun minimumLengthEncoding(words: Array<String>): Int {
        // https://leetcode-cn.com/problems/short-encoding-of-words/
        val list = words.sortedWith(Comparator { o1, o2 ->
            if (o1.length > o2.length) -1 else if (o1.length == o2.length) 0 else 1
        })
        val set = HashSet<String>()
        var ans = 0
        for (item in list) {
            if (item in set) {
                continue
            }
            ans += item.length + 1
            for (i in 0 until item.length - 1) {
                set.add(item.substring(i))
            }
        }
        return ans
    }

    fun findMin(nums: IntArray): Int {
        // https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/
        var left = 0
        var right = nums.size - 1
        while (left < right) {
            val mid = (left + right) / 2
            if (nums[left] <= nums[mid] && nums[mid] <= nums[right]) {
                return nums[left]
            }
            if (nums[left] <= nums[mid] && nums[mid] >= nums[right]) {
                left = mid + 1
            } else if (nums[left] >= nums[mid] && nums[mid] <= nums[right]) {
                if (nums[mid] <= nums[mid - 1] && nums[mid] <= nums[mid + 1]) {
                    return nums[mid]
                }
                right = mid - 1
            }
        }
        return nums[right]
    }

    fun findMinII(nums: IntArray): Int {
        var left = 0
        var right = nums.size - 1
        while (left < right) {
            val mid = (left + right) / 2
            when {
                nums[mid] > nums[right] -> left = mid + 1
                nums[mid] < nums[right] -> right = mid
                else -> right -= 1
            }
        }
        return nums[left]
    }

    fun maxDistance(grid: Array<IntArray>): Int {
        // https://leetcode-cn.com/problems/as-far-from-land-as-possible/
        if (grid.isEmpty() || grid[0].isEmpty()) {
            return 0
        }
        var ans = 0
        val queue = mutableListOf<Pair<Int, Int>>()
        for (i in grid.indices) {
            for (j in grid[0].indices) {
                if (grid[i][j] == 1) {
                    queue.add(Pair(i, j))
                }
            }
        }
        while (queue.isNotEmpty()) {
            val a = queue[0].first
            val b = queue[0].second
            ans = maxOf(nextStep(queue, grid, a, b, a - 1, b), ans)
            ans = maxOf(nextStep(queue, grid, a, b, a + 1, b), ans)
            ans = maxOf(nextStep(queue, grid, a, b, a, b - 1), ans)
            ans = maxOf(nextStep(queue, grid, a, b, a, b + 1), ans)
            queue.removeAt(0)
        }
        return ans - 1
    }

    fun nextStep(
        queue: MutableList<Pair<Int, Int>>, grid: Array<IntArray>,
        x1: Int, y1: Int,
        x2: Int, y2: Int
    ): Int {
        if (x2 < 0 || x2 >= grid.size || y2 < 0 || y2 >= grid[0].size || grid[x2][y2] != 0) {
            return -1
        }
        val res = grid[x1][y1] + 1
        grid[x2][y2] = res
        queue.add(Pair(x2, y2))
        return res
    }

    fun isMatch(s: String, p: String): Boolean {
        if (p.isEmpty()) {
            return s.isEmpty()
        }
        val firstMatch = s.isNotEmpty() && (p[0] == s[0] || p[0] == '.')
        return if (p.length >= 2 && p[1] == '*') {
            isMatch(s, p.substring(2)) ||
                    firstMatch && isMatch(s.substring(1), p)
        } else {
            firstMatch && isMatch(s.substring(1), p.substring(1))
        }
    }

    // box size
    private val boxSize = 3
    // row size
    private val rowSize = boxSize * boxSize
    private val rows = Array(rowSize) { IntArray(rowSize + 1) }
    private val columns = Array(rowSize) { IntArray(rowSize + 1) }
    private val boxes = Array(rowSize) { IntArray(rowSize + 1) }
    private var board: Array<CharArray> = Array(0){ CharArray(0) }
    private var sudokuSolved = false

    private fun couldPlace(d: Int, row: Int, col: Int): Boolean {
        val idx = (row / boxSize) * boxSize + col / boxSize
        return rows[row][d] + columns[col][d] + boxes[idx][d] == 0
    }

    private fun placeNumber(d: Int, row: Int, col: Int) {
        val idx = (row / boxSize) * boxSize + col / boxSize
        rows[row][d]++
        columns[col][d]++
        boxes[idx][d]++
        board[row][col] = (d + 48).toChar()
    }

    private fun removeNumber(d: Int, row: Int, col: Int) {
        val idx = (row / boxSize) * boxSize + col / boxSize
        rows[row][d]--
        columns[col][d]--
        boxes[idx][d]--
        board[row][col] = '.'
    }

    private fun placeNextNumbers(row: Int, col: Int) {
        if (col == rowSize - 1 && row == rowSize - 1) {
            sudokuSolved = true
        } else {
            if (col == rowSize - 1) {
                backtrace(row + 1, 0)
            } else {
                backtrace(row, col + 1)
            }
        }
    }

    private fun backtrace(row: Int, col: Int) {
        if (board[row][col] == '.') {
            for (d in 1..9) {
                if (couldPlace(d, row, col)) {
                    placeNumber(d, row, col)
                    placeNextNumbers(row, col)
                    if (!sudokuSolved) {
                        removeNumber(d, row, col)
                    }
                }
            }
        } else {
            placeNextNumbers(row, col)
        }
    }

    fun solveSudoku(board: Array<CharArray>) {
        this.board = board
        for (i in 0 until rowSize) {
            for (j in 0 until rowSize) {
                val num = board[i][j]
                if (num != '.') {
                    val d = num.toInt() - 48
                    placeNumber(d, i, j)
                }
            }
        }
        backtrace(0, 0)
    }
}

class MinStack() {
    // https://leetcode-cn.com/problems/min-stack/submissions/
    /** initialize your data structure here. */
    private val stack = mutableListOf<Int>()
    private val minHeap = PriorityQueue<Int>()

    fun push(x: Int) {
        stack.add(x)
        minHeap.add(x)
    }

    fun pop() {
        val item = stack.last()
        stack.removeAt(stack.lastIndex)
        minHeap.remove(item)
    }

    fun top(): Int {
        return stack[stack.lastIndex]
    }

    fun getMin(): Int {
        return minHeap.peek()
    }

}