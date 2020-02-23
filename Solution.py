import bisect
import collections
from typing import List


# noinspection PyPep8Naming
class ListNode:

    def __init__(self, x, nextNode=None):
        self.val = x
        self.next = nextNode


class TreeNode:

    def __init__(self, x, left_node=None, right_node=None):
        self.val = x
        self.left = left_node
        self.right = right_node


class Solution:

    def two_sum(self, nums: List[int], target: int) -> List[int]:
        """https://leetcode-cn.com/problems/two-sum/"""
        nums_hash = {}
        for i in range(0, len(nums)):
            nums_hash[nums[i]] = i

        for i in range(0, len(nums)):
            second_index = nums_hash.get(target - nums[i])
            if second_index is not None and second_index != i:
                return [i, second_index]

        return [0, 0]

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """https://leetcode-cn.com/problems/add-two-numbers/"""
        i = 0
        head = None
        temp = None
        pre = None
        while l1 is not None or l2 is not None or i != 0:
            a = 0
            if l1 is not None:
                a = l1.val

            b = 0
            if l2 is not None:
                b = l2.val
            sum = a + b + i
            i = sum // 10
            temp = ListNode(sum % 10)
            if head is None:
                head = temp
            else:
                pre.next = temp
            pre = temp
            if l1 is not None:
                l1 = l1.next
            if l2 is not None:
                l2 = l2.next
        return head

    def printListNode(self, l: ListNode):
        print('(', end='')
        while l is not None:
            print(l.val, end='')
            if l.next is not None:
                print('->', end='')
            l = l.next
        print(')')

    def createLinkList(self, a):
        head = None
        temp = None
        for i in range(len(a)):
            if head is None:
                head = ListNode(a[i])
                temp = head
            else:
                temp.next = ListNode(a[i])
                temp = temp.next
        return head

    def createBinaryTree(self, a):
        b = []
        idx = -1
        for i in range(len(a)):
            if a[i] is not None:
                node = TreeNode(a[i])
                b.append(node)
                if idx == -1:
                    idx += 1
                    continue
                n = i - 1
                if n % 2 == 0:
                    b[n // 2].right = node
                else:
                    b[n // 2].left = node
        return b[0]

    def printBinaryTree(self, root: TreeNode):
        if root is None:
            return
        stack = [root]
        ans = []
        while stack:
            temp = stack[-1]
            ans.append(temp.val)
            if temp.left is not None:
                stack.append(temp.left)
                temp.left = None
            elif temp.right is not None:
                stack.append(temp.right)
                temp.right = None
            else:
                while temp.left is None and temp.right is None:
                    stack.pop()
                    if stack:
                        temp = stack[-1]
                    else:
                        print(ans)
                        return
                if temp.right is not None:
                    stack.append(temp.right)
                    temp.right = None
        print(ans)

    def majorityElement(self, nums: List[int]) -> int:
        """https://leetcode-cn.com/problems/majority-element/"""
        current = None
        size = 0
        for index, item in enumerate(nums):
            if size == 0:
                current = item
                size = size + 1
            elif current == item:
                size = size + 1
            else:
                size = size - 1
        return current

    def lengthOfLongestSubstring(self, s: str) -> int:
        """https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/"""
        # current_str = {}
        # i, answer = -1, 0
        # for j in range(len(s)):
        #     if s[j] in current_str:
        #         i = max(current_str[s[j]], i)
        #     answer = max(ans, j - i)
        #     current_str[s[j]] = j
        # return answer
        st = s[0:1]
        i, ans = 0, 0
        k, ind = 0, 0
        for j in range(1, len(s)):
            k = j + 1
            ind = st.find(s[j]) + 1
            if ind > 0:
                i = ind + i
            st = s[i:k]
            ans = max(ans, k - i)
        return ans

    def longestPalindrome(self, s: str) -> str:
        """https://leetcode-cn.com/problems/longest-palindromic-substring/"""
        len_of_s = len(s)
        logs = [([0] * len_of_s) for i in range(len_of_s)]
        ans, ans_i, ans_j = 0, 0, 0
        for i in range(len_of_s):
            for j in range(i + 1):
                update = False
                if i == j:
                    update = True
                elif s[i] == s[j] and (i - j == 1 or logs[i - 1][j + 1] == 1):
                    update = True
                if update:
                    logs[i][j] = 1
                    curr = i - j + 1
                    if curr > ans:
                        ans = curr
                        ans_i = i
                        ans_j = j
        return s[ans_j:ans_i + 1]

    def convert(self, s: str, numRows: int) -> str:
        """https://leetcode-cn.com/problems/zigzag-conversion/"""
        # print(s + " " + numRows.__str__() + ":")
        if numRows == 1:
            return s
        ans = ""
        length = len(s)
        for rows in range(numRows):
            index = rows
            ans = ans + s[index]
            first = True
            step1 = 2 * (numRows - rows - 1)
            step2 = 2 * rows
            while index < length:
                if first:
                    index = index + step1
                    if step1 != 0 and index < length:
                        ans = ans + s[index]
                    first = False
                else:
                    index = index + step2
                    if step2 != 0 and index < length:
                        ans = ans + s[index]
                    first = True
        return ans

    def reverse(self, x: int) -> int:
        """https://leetcode-cn.com/problems/reverse-integer/"""
        if x == 0:
            return x
        ans = ""
        pre = ""
        if x < 0:
            pre = "-"
            x = abs(x)
        while x != 0:
            item = x % 10
            x = x // 10
            if len(ans) == 0 and item == 0:
                continue
            ans = ans + item.__str__()
        ans = pre + ans
        result = int(ans)
        if result > 2147483647 or result < -2147483648:
            return 0
        return result

    def myAtoi(self, s: str) -> int:
        """https://leetcode-cn.com/problems/string-to-integer-atoi/"""
        int_max = 2147483647
        int_min = -2147483648
        int_set = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        s = s.strip()
        end = -1
        length = len(s)
        if length == 0:
            return 0
        if s[0] not in int_set and s[0] != '-' and s[0] != '+':
            return 0
        if (s[0] == '-' or s[0] == '+') and (length <= 1 or s[1] not in int_set):
            return 0
        for i in range(1, length):
            item = s[i]
            if item not in int_set:
                end = i
                break
        if end == -1:
            end = length
        result = 0
        if end != 0:
            result = int(s[0: end])
        if result > int_max:
            result = int_max
        elif result < int_min:
            result = int_min
        return result

    def isPalindrome(self, x: int) -> bool:
        """https://leetcode-cn.com/problems/palindrome-number/"""
        # type 1:
        start = x
        result = 0
        if x < 0:
            return False
        while x > 0:
            result = result * 10 + x % 10
            x = x // 10
        return result == start

    def maxArea(self, height: List[int]) -> int:
        """https://leetcode-cn.com/problems/container-with-most-water/"""
        i, j = 0, len(height) - 1
        result = 0
        while i < j:
            width = j - i
            if height[i] < height[j]:
                temp = height[i] * width
                i = i + 1
            else:
                temp = height[j] * width
                j = j - 1
            result = max(result, temp)
        return result

    def intToRoman(self, num: int) -> str:
        """https://leetcode-cn.com/problems/integer-to-roman/"""
        one_digits = ['I', 'X', 'C', 'M']
        five_digits = ['V', 'L', 'D']
        ans = ""
        bit = 0
        while num > 0:
            temp = num % 10
            num = num // 10
            if temp < 4:
                ans = one_digits[bit] * temp + ans
            elif temp == 4:
                ans = one_digits[bit] + five_digits[bit] + ans
            elif temp == 5:
                ans = five_digits[bit] + ans
            elif temp < 9:
                ans = five_digits[bit] + one_digits[bit] * (temp - 5) + ans
            else:
                ans = one_digits[bit] + one_digits[bit + 1] + ans
            bit = bit + 1
        return ans

    def romanToInt(self, s: str) -> int:
        """https://leetcode-cn.com/problems/roman-to-integer/"""
        digits = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        romans = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        result = 0
        for i in range(13):
            if len(romans[i]) == 1:
                while len(s) > 0 and s[0] == romans[i]:
                    result = result + digits[i]
                    s = s[1:]
            elif len(romans[i]) == 2:
                while len(s) > 1 and s[0:2] == romans[i]:
                    result = result + digits[i]
                    s = s[2:]
        return result

    def longestCommonPrefix(self, strs: List[str]) -> str:
        """https://leetcode-cn.com/problems/longest-common-prefix/"""
        ans = ''
        i = 0
        length = len(strs)
        search = True
        while search:
            temp = ''
            for j in range(length):
                if i >= len(strs[j]):
                    search = False
                    break
                if temp != '':
                    temp = strs[j][i]
                elif temp != strs[j][i]:
                    search = False
                    break
            if search:
                ans = ans + temp
            i = i + 1
        return ans

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """https://leetcode-cn.com/problems/3sum/"""
        ans = []
        counts = {}
        for i in nums:
            counts[i] = counts.get(i, 0) + 1

        nums = sorted(counts.keys())

        for i, num in enumerate(nums):
            if counts[num] > 1:
                if num == 0:
                    if counts[num] > 2:
                        ans.append([0, 0, 0])
                else:
                    if -num * 2 in counts:
                        ans.append([num, num, -2 * num])
            if num < 0:
                two_sum = -num
                left = bisect.bisect_left(nums, (two_sum - nums[-1]), i + 1)
                for i in nums[left: bisect.bisect_right(nums, (two_sum // 2), left)]:
                    j = two_sum - i
                    if j in counts and j != i:
                        ans.append([num, i, j])

        return ans

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """https://leetcode-cn.com/problems/3sum-closest/"""
        n = len(nums)
        nums = sorted(nums)
        result = None
        different = None
        for i in range(n):
            l = i + 1
            r = n - 1
            while l < r:
                temp_result = nums[i] + nums[l] + nums[r]
                temp_diff = abs(temp_result - target)
                if different is None or different > temp_diff:
                    result = temp_result
                    different = temp_diff
                if temp_result < target:
                    while l < r and nums[l] == nums[l + 1]:
                        l = l + 1
                    l = l + 1
                elif temp_result > target:
                    while l < r and nums[r] == nums[r - 1]:
                        r = r - 1
                    r = r - 1
                else:
                    return result
        return result

    def letterCombinations(self, digits: str) -> List[str]:
        """https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/"""
        nums_map = {'2': 'abc', '3': 'def', '4': 'ghi',
                    '5': 'jkl', '6': 'mno', '7': 'pqrs',
                    '8': 'tuv', '9': 'wxyz'}
        digits = digits.replace('1', '')
        n = len(digits)
        result = []
        if n == 0:
            return result
        count = [0] * n
        max = []
        for i in range(n):
            max = max + [len(nums_map[digits[i]])]
        temp = "#" * n
        top = 0
        while top >= 0:
            if top >= n:
                result = result + [temp]
                top = top - 1
                continue
            if count[top] >= max[top]:
                count[top] = 0
                top = top - 1
                continue
            item = nums_map[digits[top]][count[top]]
            if top == 0:
                temp = item
            else:
                temp = temp[0: top] + item
            count[top] = count[top] + 1
            top = top + 1
        return result

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """https://leetcode-cn.com/problems/4sum/"""
        ans = []
        n = len(nums)
        if n < 4:
            return ans
        nums = sorted(nums)
        max_sum3 = nums[n - 1] + nums[n - 2] + nums[n - 3]
        max_sum2 = nums[n - 1] + nums[n - 2]
        for i in range(n - 3):
            if 4 * nums[i] > target: break
            if i > 0 and nums[i] == nums[i - 1]: continue
            if nums[i] + max_sum3 < target: continue
            for j in range(i + 1, n - 2):
                if 2 * (nums[i] + nums[j]) > target: break
                if j > i + 1 and nums[j - 1] == nums[j]: continue
                if nums[i] + nums[j] + max_sum2 < target: continue
                item = target - nums[i] - nums[j]
                l = j + 1
                r = n - 1
                while l < r:
                    k = nums[l] + nums[r]
                    if k > item:
                        while l < r and nums[r] == nums[r - 1]:
                            r = r - 1
                        r = r - 1
                    elif k < item:
                        while l < r and nums[l] == nums[l + 1]:
                            l = l + 1
                        l = l + 1
                    else:
                        ans.append([nums[i], nums[j], nums[l], nums[r]])
                        while l < r and nums[l] == nums[l + 1]:
                            l = l + 1
                        l = l + 1
                        while l < r and nums[r] == nums[r - 1]:
                            r = r - 1
                        r = r - 1
        return ans

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        """https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/"""
        delete_pre = head
        index_delete = head
        index_tail = head
        i = 1
        while index_tail.next is not None:
            if i < n:
                i = i + 1
            else:
                if delete_pre == index_delete:
                    index_delete = index_delete.next
                else:
                    delete_pre = delete_pre.next
                    index_delete = index_delete.next
            index_tail = index_tail.next
        if i == n:
            if head != index_delete:
                delete_pre.next = index_delete.next
                index_delete.next = None
            else:
                head = head.next
                index_delete.next = None
        return head

    def isValid(self, s: str) -> bool:
        """https://leetcode-cn.com/problems/valid-parentheses/"""
        n = len(s)
        if n == 0:
            return True
        if n % 2 == 1:
            return False
        valid = {'(': [')', '[', '{', '('],
                 '[': [']', '(', '{', '['],
                 '{': ['}', '(', '[', '{']}
        bingo = {'(': ')', '[': ']', '{': '}'}
        if s[0] not in valid.keys():
            return False
        temp = ""
        length_temp = 0
        for i in range(n):
            if length_temp == 0:
                temp = temp + s[i]
                length_temp = length_temp + 1
                continue
            pre = temp[length_temp - 1]
            if pre not in valid.keys():
                return False
            elif s[i] not in valid[pre]:
                return False
            elif s[i] in bingo[pre]:
                temp = temp[:length_temp - 1]
                length_temp = length_temp - 1
            else:
                temp = temp + s[i]
                length_temp = length_temp + 1
        return temp == ""

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        """https://leetcode-cn.com/problems/merge-two-sorted-lists/"""
        result = None
        tail = None
        while l1 is not None and l2 is not None:
            if l1.val < l2.val:
                max = l1
                l1 = l1.next
            else:
                max = l2
                l2 = l2.next
            if result is None:
                result = tail = max
            else:
                tail.next = max
                tail = max
        if l1 is not None:
            if result is None:
                return l1
            else:
                tail.next = l1
        else:
            if result is None:
                return l2
            else:
                tail.next = l2
        return result

    def myPow(self, x: float, n: int) -> float:
        """https://leetcode-cn.com/problems/powx-n/"""
        if n == 0:
            return 1
        if n == 1:
            return x
        if n == -1:
            return 1 / x
        half = self.myPow(x, n // 2)
        rest = self.myPow(x, n % 2)
        return rest * half * half

    def generateParenthesis(self, n: int) -> List[str]:
        """https://leetcode-cn.com/problems/generate-parentheses/"""
        # queue = Queue()
        # left = Queue()
        # right = Queue()
        # result = []
        # queue.put('(')
        # left.put(1)
        # right.put(0)
        # while not queue.empty():
        #     temp = queue.get()
        #     temp_l = left.get()
        #     temp_r = right.get()
        #     if temp_l == n:
        #         temp = temp + ')' * (n - temp_r)
        #         temp_r = n
        #     elif temp_l == temp_r:
        #         temp = temp + '('
        #         temp_l = temp_l + 1
        #     else:
        #         queue.put(temp + '(')
        #         left.put(temp_l + 1)
        #         right.put(temp_r)
        #         temp = temp + ')'
        #         temp_r = temp_r + 1
        #     if temp_l == n and temp_r == n:
        #         result = result + [temp]
        #         continue
        #     queue.put(temp)
        #     left.put(temp_l)
        #     right.put(temp_r)
        # return result
        ans = []

        def backtrack(s='', left=0, right=0):
            if len(s) == 2 * n:
                ans.append(s)
                return
            if left < n:
                backtrack(s + '(', left + 1, right)
            if left > right:
                backtrack(s + ')', left, right + 1)

        backtrack()
        return ans

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        """https://leetcode-cn.com/problems/merge-k-sorted-lists/"""
        import heapq
        min_heap = []
        head = None
        tail = None
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(min_heap, (lists[i].val, i))
                lists[i] = lists[i].next
        while min_heap:
            val, idx = heapq.heappop(min_heap)
            if not head:
                head = ListNode(val)
                tail = head
            else:
                tail.next = ListNode(val)
                tail = tail.next
            if lists[idx]:
                heapq.heappush(min_heap, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return head

    def swapPairs(self, head: ListNode) -> ListNode:
        """https://leetcode-cn.com/problems/swap-nodes-in-pairs/"""
        if not head:
            return head
        pre = None
        first = head
        second = first.next
        while first and second:
            first.next = second.next
            second.next = first
            if pre:
                pre.next = second
                pre = first
            else:
                pre = first
                head = second
            first = first.next
            if first:
                second = first.next
        return head

    def removeDuplicates(self, nums: List[int]) -> int:
        """https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/"""
        result = 0
        n = len(nums)
        idx = 1
        for i in range(1, n):
            if nums[i] == nums[i - 1]:
                result = result + 1
            else:
                nums[idx] = nums[i]
                idx = idx + 1
        return n - result

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        """https://leetcode-cn.com/problems/reverse-nodes-in-k-group/"""
        n = 0
        temp = head
        while temp:
            temp = temp.next
            n = n + 1
        if k > n or k <= 1:
            return head

        def reverse(first: ListNode, l_in: int) -> (ListNode, ListNode):
            if not first or l_in <= 1:
                return first, first
            last = first
            second = first.next
            temp_in = None
            for i_in in range(l_in - 1):
                temp_in = second.next
                second.next = first
                first = second
                second = temp_in
            last.next = temp_in
            return first, last

        start = head
        result = None
        temp = None
        for i in range(n // k):
            start, end = reverse(start, k)
            if end:
                if temp:
                    temp.next = start
                else:
                    result = start
                temp = end
                start = end.next
        return result

    def removeElement(self, nums: List[int], val: int) -> int:
        """https://leetcode-cn.com/problems/remove-element/"""
        j = 0
        for i in range(len(nums)):
            if nums[i] == val:
                continue
            else:
                nums[j] = nums[i]
                j = j + 1
        return j

    def strStr(self, haystack: str, needle: str) -> int:
        """https://leetcode-cn.com/problems/implement-strstr/comments/"""
        return haystack.find(needle)

    def divide(self, dividend: int, divisor: int) -> int:
        """https://leetcode-cn.com/problems/divide-two-integers/submissions/"""
        int_max = 2147483647
        int_min = -2147483648
        if dividend == 0:
            return 0
        negative = (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0)
        dividend = abs(dividend)
        divisor = abs(divisor)

        def constraintResult(result_im: int) -> int:
            if negative:
                result_im = -result_im
                if result_im < int_min:
                    result_im = int_min
                return result_im
            else:
                if result_im > int_max:
                    result_im = int_max
                return result_im

        if dividend < divisor:
            return 0
        elif divisor == 1:
            return constraintResult(dividend)
        result = 1
        temp = divisor
        while temp < dividend:
            pre = result
            temp_pre = temp
            temp = temp + temp
            result = result + result
            if temp > dividend:
                result = result + abs(self.divide(dividend - temp_pre, divisor)) - pre
            elif temp == dividend:
                break
        return constraintResult(result)

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        """https://leetcode-cn.com/problems/substring-with-concatenation-of-all-words/"""
        # from collections import Counter
        # if not s or not words: return []
        # one_word = len(words[0])
        # all_len = len(words) * one_word
        # n = len(s)
        # words = Counter(words)
        # res = []
        # for i in range(0, n - all_len + 1):
        #     tmp = s[i:i + all_len]
        #     c_tmp = []
        #     for j in range(0, all_len, one_word):
        #         c_tmp.append(tmp[j:j + one_word])
        #     if Counter(c_tmp) == words:
        #         res.append(i)
        # return res
        res = []
        n = len(s)
        word_count = len(words)
        if word_count == 0:
            return res
        one_word = len(words[0])
        all_len = word_count * one_word
        if n < all_len:
            return res

        def updateMatch(temp_in: str, match_count_in: int) -> int:
            if temp_in in real_count.keys():
                if real_count[temp_in] == count_map[temp_in]:
                    match_map[temp_in] = True
                    return match_count_in + 1
                else:
                    match_map[temp_in] = False
                    return match_count_in - 1
            return match_count_in

        count_map = dict()
        real_count = dict()
        match_map = dict()
        for i, item in enumerate(words):
            if item in real_count.keys():
                real_count[item] = real_count[item] + 1
            else:
                real_count[item] = 1
        match_count = 0
        match_destiny = len(real_count.keys())
        for l in range(0, one_word):
            count_map.clear()
            match_count = 0
            match_map.clear()
            for i in range(l, n - all_len + 1, one_word):
                if not count_map:
                    for j in range(0, all_len, one_word):
                        item = s[i + j: i + j + one_word]
                        if item in count_map.keys():
                            count_map[item] = count_map[item] + 1
                        else:
                            count_map[item] = 1
                    for item in count_map.keys():
                        if item in real_count.keys() and real_count[item] == count_map[item]:
                            match_map[item] = True
                            match_count = match_count + 1
                else:
                    last = s[i - one_word: i]
                    count_map[last] = count_map[last] - 1
                    match_count = updateMatch(last, match_count)
                    temp = s[i + all_len - one_word: i + all_len]
                    if temp in count_map.keys():
                        count_map[temp] = count_map[temp] + 1
                    else:
                        count_map[temp] = 1
                    match_count = updateMatch(temp, match_count)
                if match_count == match_destiny:
                    res.append(i)
        return res

    def nextPermutation(self, nums: List[int]) -> None:
        """https://leetcode-cn.com/problems/next-permutation/"""
        n = len(nums)
        if n <= 1:
            return
        max_num = None
        change_idx = None
        change_item = None
        for i in range(n - 1, -1, -1):
            item = nums[i]
            if max_num:
                if max_num < item:
                    max_num = item
                elif max_num > item:
                    change_item = item
                    change_idx = i
                    break
            else:
                max_num = item
                change_item = item
        if change_idx is None:
            sort = sorted(nums)
            for i in range(n):
                nums[i] = sort[i]
            return
        change_idx_2 = change_idx + 1
        for i in range(n - 1, change_idx, -1):
            item = nums[i]
            if item > change_item:
                change_idx_2 = i
                break
        temp = nums[change_idx]
        nums[change_idx] = nums[change_idx_2]
        nums[change_idx_2] = temp
        sort = sorted(nums[change_idx + 1:])
        for i in range(0, n - change_idx - 1):
            nums[i + change_idx + 1] = sort[i]

    def longestValidParentheses(self, s: str) -> int:
        """https://leetcode-cn.com/problems/longest-valid-parentheses/"""
        stack = []
        stack.append(-1)
        max_num = 0
        for i in len(s):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if stack:
                    max_num = max(max_num, i - stack[len(stack) - 1])
                else:
                    stack.append(i)
        return max_num

    def search(self, nums: List[int], target: int) -> int:
        """https://leetcode-cn.com/problems/search-in-rotated-sorted-array/"""
        l = 0
        r = len(nums) - 1
        while l <= r:
            middle = (l + r) // 2
            item = nums[middle]
            left = nums[l]
            right = nums[r]
            if item == target:
                return middle
            elif left < right:
                # 排好序
                if item > target:
                    r = middle - 1
                else:
                    l = middle + 1
            elif item > right:
                # middle 在反转的左边
                if target > item or target <= right:
                    l = middle + 1
                else:
                    r = middle - 1
            else:
                # middle 在反转的右边
                if target < item or target > right:
                    r = middle - 1
                else:
                    l = middle + 1
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/"""

        def binary_search_left(nums_left: List[int], target_left: int) -> int:
            l_left = 0
            r_left = len(nums_left) - 1
            result_left = -1
            while l_left <= r_left:
                middle_left = (l_left + r_left) // 2
                item_left = nums_left[middle_left]
                if item_left == target_left:
                    result_left = middle_left
                    r_left = r_left - 1
                elif target_left < item_left:
                    r_left = middle_left - 1
                elif target_left > item_left:
                    l_left = middle_left + 1
            return result_left

        def binary_search_right(nums_right: List[int], target_right: int) -> int:
            l_right = 0
            result_right = -1
            r_right = len(nums_right) - 1
            while l_right <= r_right:
                middle_right = (l_right + r_right) // 2
                item_right = nums_right[middle_right]
                if item_right == target_right:
                    result_right = middle_right
                    l_right = middle_right + 1
                elif target_right < item_right:
                    r_right = middle_right - 1
                elif target_right > item_right:
                    l_right = middle_right + 1
            return result_right

        return [binary_search_left(nums, target), binary_search_right(nums, target)]

    def searchInsert(self, nums: List[int], target: int) -> int:
        """https://leetcode-cn.com/problems/search-insert-position/"""

        l = 0
        r = len(nums) - 1
        while l <= r:
            middle = (l + r) // 2
            item = nums[middle]
            if item == target:
                return middle
            elif target > item:
                l = middle + 1
            else:
                r = middle - 1
        return max(l, r)

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        """https://leetcode-cn.com/problems/valid-sudoku/"""
        nums = [0] * 9
        for i in range(9):
            for j in range(9):
                nums[j] = 0
            for j in range(9):
                if board[i][j] == '.':
                    continue
                num = int(board[i][j])
                idx = num - 1
                nums[idx] = nums[idx] + 1
                if nums[idx] > 1:
                    return False

        for i in range(9):
            for j in range(9):
                nums[j] = 0
            for j in range(9):
                if board[j][i] == '.':
                    continue
                num = int(board[j][i])
                idx = num - 1
                nums[idx] = nums[idx] + 1
                if nums[idx] > 1:
                    return False

        for i in range(9):
            for j in range(9):
                nums[j] = 0
            for j in range(9):
                item = board[i // 3 * 3 + j // 3][(i % 3) * 3 + (j % 3)]
                if item == '.':
                    continue
                num = int(item)
                idx = num - 1
                nums[idx] = nums[idx] + 1
                if nums[idx] > 1:
                    return False
        return True

    def countAndSay(self, n: int) -> str:
        """https://leetcode-cn.com/problems/count-and-say/"""
        pre = "1"
        result = pre
        for i in range(n - 1):
            count = 0
            result = ""
            for j in range(len(pre)):
                if j == 0:
                    count = 1
                elif pre[j] == pre[j - 1]:
                    count = count + 1
                else:
                    result = result + str(count) + pre[j - 1]
                    count = 1
            if count != 0:
                result = result + str(count) + pre[j]
            pre = result
        return result

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """https://leetcode-cn.com/problems/combination-sum/"""

        def combinationSumRecursive(candidates_in: List[int], target_in: int, current_result: List[int],
                                    result: List[List[int]]):
            n = len(candidates_in)
            for i in range(n):
                current = candidates_in[i]
                temp = target_in - current
                current_result.append(current)
                if temp == 0:
                    result.append(current_result.copy())
                    del current_result[len(current_result) - 1]
                    continue
                elif temp < 0:
                    del current_result[len(current_result) - 1]
                    continue
                else:
                    combinationSumRecursive(candidates_in[i: n], temp, current_result, result)
                    del current_result[len(current_result) - 1]

        result = []
        combinationSumRecursive(candidates, target, [], result)
        return result

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """https://leetcode-cn.com/problems/combination-sum-ii/"""

        def combinationSumRecursive(candidates_in: List[int], target_in: int, current_result: List[int],
                                    result: List[List[int]]):
            n = len(candidates_in)
            i = 0
            while i < n:
                current = candidates_in[i]
                temp = target_in - current
                current_result.append(current)
                if temp == 0:
                    result.append(current_result.copy())
                    del current_result[len(current_result) - 1]
                    return
                elif temp < 0:
                    del current_result[len(current_result) - 1]
                    while i < n - 1 and candidates_in[i] == candidates_in[i + 1]:
                        i = i + 1
                    i = i + 1
                    continue
                else:
                    combinationSumRecursive(candidates_in[i + 1: n], temp, current_result, result)
                    del current_result[len(current_result) - 1]
                while i < n - 1 and candidates_in[i] == candidates_in[i + 1]:
                    i = i + 1
                i = i + 1

        result = []
        combinationSumRecursive(sorted(candidates), target, [], result)
        return result

    def firstMissingPositive(self, nums: List[int]) -> int:
        """https://leetcode-cn.com/problems/first-missing-positive/"""
        n = len(nums)
        if n == 0:
            return 1
        has_one = False
        for i in range(n):
            if not has_one and nums[i] == 1:
                has_one = True
            elif nums[i] <= 0 or nums[i] > n:
                nums[i] = 1

        if not has_one:
            return 1
        for i in range(n):
            if nums[i] < 0:
                idx = -nums[i] - 1
            else:
                idx = nums[i] - 1
            if nums[idx] > 0:
                nums[idx] = -nums[idx]
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        return n + 1

    def multiply(self, num1: str, num2: str) -> str:
        """https://leetcode-cn.com/problems/multiply-strings/"""
        if num1 == "0" or num2 == "0":
            return "0"
        if num1 == "1":
            return num2
        if num2 == "1":
            return num1

        m, n = len(num1), len(num2)
        result = [0 for _ in range(m + n)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                x, y = ord(num1[i]) - ord('0'), ord(num2[j]) - ord('0')
                result[i + j + 1] += x * y

        for i in range(m + n - 1, 0, -1):
            carry = result[i] // 10
            result[i] = result[i] % 10
            result[i - 1] += carry

        return ''.join([str(x) for x in result]).lstrip('0')

    def permute(self, nums: List[int]) -> List[List[int]]:
        """https://leetcode-cn.com/problems/permutations/"""
        ans = []
        res = []

        def premute_in(ans_in: List[List[int]], res_in: List[int], nums_in: List[int]):
            n = len(nums_in)
            if n == 0:
                return
            if n == 1:
                res_in.append(nums_in[0])
                ans_in.append(res_in.copy())
                del res_in[len(res_in) - 1]
                return
            for i in range(n):
                res_in.append(nums_in[i])
                premute_in(ans_in, res_in, nums_in[0: i] + nums_in[i + 1: n])
                del res_in[len(res_in) - 1]

        premute_in(ans, res, nums)
        return ans

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """https://leetcode-cn.com/problems/permutations-ii/"""
        ans = []
        res = []

        def premute_in(ans_in: List[List[int]], res_in: List[int], nums_in: List[int]):
            n = len(nums_in)
            if n == 0:
                return
            if n == 1:
                res_in.append(nums_in[0])
                ans_in.append(res_in.copy())
                del res_in[len(res_in) - 1]
                return
            i = 0
            while i < n:
                res_in.append(nums_in[i])
                premute_in(ans_in, res_in, nums_in[0: i] + nums_in[i + 1: n])
                del res_in[len(res_in) - 1]
                while i < n - 1 and nums_in[i] == nums_in[i + 1]:
                    i += 1
                i += 1

        nums = sorted(nums)
        premute_in(ans, res, nums)
        return ans

    def rotate(self, matrix: List[List[int]]) -> None:
        """https://leetcode-cn.com/problems/rotate-image/"""
        n = len(matrix)
        for i in range(n // 2 + n % 2):
            for j in range(n // 2):
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """https://leetcode-cn.com/problems/group-anagrams/"""
        # map_list = []
        # res_list = []
        # for i in range(len(strs)):
        #     item_map = {}
        #     for j in range(len(strs[i])):
        #         item = strs[i][j]
        #         if item in item_map.keys():
        #             item_map[item] += 1
        #         else:
        #             item_map[item] = 1
        #     if item_map in map_list:
        #         res_list[map_list.index(item_map)].append(strs[i])
        #     else:
        #         map_list.append(item_map)
        #         res_list.append([strs[i]])
        # return res_list
        res = []
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        for i in ans.values():
            res.append(i)
        return res

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """https://leetcode-cn.com/problems/median-of-two-sorted-arrays/"""
        m, n = len(nums1), len(nums2)
        if m > n:
            nums1, nums2, m, n = nums2, nums1, n, m
        if n == 0:
            return 0

        imin, imax, half_len = 0, m, (m + n + 1) // 2
        i = 0
        j = 0
        while imin <= imax:
            i = (imin + imax) // 2
            j = half_len - i
            if i < m and nums2[j - 1] > nums1[i]:
                # i is too small
                imin = i + 1
            elif i > 0 and nums1[i - 1] > nums2[j]:
                # i is too big
                imax = i - 1
            else:
                # i is perfect
                break

        if i == 0:
            max_of_left = nums2[j - 1]
        elif j == 0:
            max_of_left = nums1[i - 1]
        else:
            max_of_left = max(nums1[i - 1], nums2[j - 1])

        if (m + n) % 2 == 1:
            return max_of_left

        if i == m:
            min_of_right = nums2[j]
        elif j == n:
            min_of_right = nums1[i]
        else:
            min_of_right = min(nums1[i], nums2[j])

        return (max_of_left + min_of_right) / 2

    def maxSubArray(self, nums: List[int]) -> int:
        """https://leetcode-cn.com/problems/maximum-subarray/"""
        ans = nums[0]
        for i in range(1, len(nums)):
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
            ans = max(nums[i], ans)
        return ans

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """https://leetcode-cn.com/problems/spiral-matrix/"""
        # res = []
        # while matrix:
        #     res += matrix.pop(0)
        #     matrix = list(map(list, zip(*matrix)))[::-1]
        # return res
        r, i, j, di, dj = [], 0, 0, 0, 1
        if matrix != []:
            for _ in range(len(matrix) * len(matrix[0])):
                r.append(matrix[i][j])
                matrix[i][j] = 0
                if matrix[(i + di) % len(matrix)][(j + dj) % len(matrix[0])] == 0:
                    di, dj = dj, -di
                i += di
                j += dj
        return r

    def canJump(self, nums: List[int]) -> bool:
        """https://leetcode-cn.com/problems/jump-game/"""
        max_n = len(nums) - 1
        for i in range(max_n, -1, -1):
            if i + nums[i] >= max_n:
                max_n = i
        return max_n == 0

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """https://leetcode-cn.com/problems/merge-intervals/"""
        intervals = sorted(intervals)
        last = None
        ans = []
        for i in range(len(intervals)):
            if not last:
                last = intervals[i]
                continue
            if last[1] >= intervals[i][0]:
                last[1] = intervals[i][1]
            else:
                ans.append(last)
                last = intervals[i]
        if not last:
            ans.append(last)
        return ans

    def lengthOfLastWord(self, s: str) -> int:
        """https://leetcode-cn.com/problems/length-of-last-word/"""
        splited = s.split(' ')
        n = len(splited)
        i = 0
        ans = 0
        while i < n:
            if splited[i] == '':
                del splited[i]
                n = n - 1
                continue
            ans = len(splited[i])
            i = i + 1
        return ans

    def generateMatrix(self, n: int) -> List[List[int]]:
        """https://leetcode-cn.com/problems/spiral-matrix-ii/"""
        matrix, i, j, di, dj = [], 0, 0, 0, 1
        if n == 0:
            return matrix
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for num in range(1, n ** 2 + 1):
            matrix[i][j] = num
            if i + di >= n or i + di < 0 or j + dj >= n or j + dj < 0 or matrix[i + di][j + dj] != 0:
                di, dj = dj, -di
            i += di
            j += dj
        return matrix

    def getPermutation(self, n: int, k: int) -> str:
        """https://leetcode-cn.com/problems/permutation-sequence/"""
        num = 1
        ans = ""
        nums = []
        for i in range(n):
            num *= i + 1
            nums.append(str(i + 1))
        for i in range(n, 0, -1):
            num /= i
            index = int(k // num) - 1
            k = k % num
            if k > 0:
                index += 1
            ans += nums[index]
            del nums[index]
        return ans

    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        """https://leetcode-cn.com/problems/rotate-list/"""
        if not head:
            return head
        temp = head
        n = 0
        while temp.next is not None:
            n += 1
            temp = temp.next
        n += 1
        tail = temp
        k %= n
        temp = head
        pre = head
        count = 0
        while temp.next is not None:
            if count < k:
                count += 1
            else:
                pre = pre.next
            temp = temp.next
        tail.next = head
        head = pre.next
        pre.next = None
        return head

    def uniquePaths(self, m: int, n: int) -> int:
        """https://leetcode-cn.com/problems/unique-paths/"""
        if m > n:
            m, n = n, m
        ans = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if ans[i][j] != 0:
                    continue
                if i == 0 or j == 0:
                    ans[i][j] = 1
                else:
                    item = ans[i][j - 1] + ans[i - 1][j]
                    ans[i][j] = item
                    if i != j and i < m and j < n:
                        ans[j][i] = item
        return ans[n - 1][m - 1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """https://leetcode-cn.com/problems/unique-paths-ii/"""
        ans = obstacleGrid
        if ans[0][0] == 1:
            return 0
        m, n = len(ans[0]), len(ans)
        for i in range(n):
            for j in range(m):
                if ans[i][j] == 1:
                    ans[i][j] = None
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if ans[i][j] != 0:
                    continue
                if i == n - 1 and j == m - 1:
                    if ans[i][j] is not None:
                        ans[i][j] = 1
                elif i == n - 1 and ans[i][j + 1] is not None:
                    ans[i][j] = 1
                elif i == n - 1 and ans[i][j + 1] is None:
                    ans[i][j] = None
                elif j == m - 1 and ans[i + 1][j] is not None:
                    ans[i][j] = 1
                elif j == m - 1 and ans[i + 1][j] is None:
                    ans[i][j] = None
                else:
                    if ans[i][j + 1] is not None and ans[i + 1][j] is not None:
                        item = ans[i][j + 1] + ans[i + 1][j]
                    elif ans[i][j + 1] is not None:
                        item = ans[i][j + 1]
                    elif ans[i + 1][j] is not None:
                        item = ans[i + 1][j]
                    else:
                        item = None
                    ans[i][j] = item
        if ans[0][0] is not None:
            return ans[0][0]
        else:
            return 0

    def minPathSum(self, grid: List[List[int]]) -> int:
        """https://leetcode-cn.com/problems/minimum-path-sum/"""
        n, m = len(grid), len(grid[0])
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    continue
                if i == 0:
                    grid[i][j] += grid[i][j - 1]
                elif j == 0:
                    grid[i][j] += grid[i - 1][j]
                else:
                    grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[n - 1][m - 1]

    def plusOne(self, digits: List[int]) -> List[int]:
        """https://leetcode-cn.com/problems/plus-one/"""
        n = len(digits)
        i = n - 1
        digits[i] += 1
        ten = 0
        while i >= 0:
            ten = digits[i] // 10
            digits[i] %= 10
            if ten < 1:
                break
            if i > 0:
                digits[i - 1] += ten
            i = i - 1
        if ten != 0:
            digits = [ten] + digits
        return digits

    def addBinary(self, a: str, b: str) -> str:
        """https://leetcode-cn.com/problems/add-binary/"""

        def add(add_list: List[str]) -> (str, str):
            add_one_count = 0
            for add_i in range(len(add_list)):
                if add_list[add_i] == '1':
                    add_one_count += 1
            if add_one_count == 0:
                add_ten = '0'
                add_one = '0'
            elif add_one_count == 1:
                add_ten = '0'
                add_one = '1'
            elif add_one_count == 2:
                add_ten = '1'
                add_one = '0'
            else:
                add_ten = '1'
                add_one = '1'
            return add_ten, add_one

        n, m = len(a), len(b)
        ans, i, j, pre = '', n - 1, m - 1, '0'
        while i >= 0 or j >= 0:
            if i >= 0 and j >= 0:
                pre, one = add([a[i], b[j], pre])
                ans = one + ans
            elif i >= 0:
                pre, one = add([a[i], pre])
                ans = one + ans
            else:
                pre, one = add([b[j], pre])
                ans = one + ans
            i -= 1
            j -= 1
        if pre == '1':
            ans = pre + ans
        return ans

    def mySqrt(self, x: int) -> int:
        """https://leetcode-cn.com/problems/sqrtx/"""
        import math
        from math import log
        left = int(math.e ** (0.5 * log(x)))
        right = left + 1
        return left if right * right > x else right

    def climbStairs(self, n: int) -> int:
        """https://leetcode-cn.com/problems/climbing-stairs/"""
        # if n == 1:
        #     return 1
        # if n == 2:
        #     return 2
        # ans = [1 for _ in range(n)]
        # ans[2] = 2
        # for i in range(2, n):
        #     ans[i] = ans[i - 1] + ans[i - 2]
        # return ans[n - 1]
        from math import sqrt
        s5 = sqrt(5)
        return int(1 / s5 * (((1 + s5) / 2) ** (n + 1) - ((1 - s5) / 2) ** (n + 1)))

    def simplifyPath(self, path: str) -> str:
        """https://leetcode-cn.com/problems/simplify-path/"""
        path_list = path.split('/')
        ans = []
        for i in range(len(path_list)):
            if path_list[i] == '' or path_list[i] == '.':
                continue
            if path_list[i] == '..':
                if ans:
                    ans.pop()
            else:
                ans.append('/' + path_list[i])
        if not ans:
            return '/'
        return ''.join(ans)

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """https://leetcode-cn.com/problems/insert-interval/"""
        n = len(intervals)
        idx = -1
        i = 0
        while i < n:
            if intervals[i][0] < newInterval[0] and intervals[i][1] < newInterval[0]:
                i += 1
                idx = i
                continue
            if intervals[i][0] > newInterval[1]:
                idx = i
                break
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            del intervals[i]
            n -= 1
        return intervals[0: idx] + [newInterval] + intervals[idx: n]

    def setZeroes(self, matrix: List[List[int]]) -> None:
        """https://leetcode-cn.com/problems/set-matrix-zeroes/"""
        if not matrix:
            return
        n, m = len(matrix), len(matrix[0])
        idx = []
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    idx.append([i, j])
        for i in range(len(idx)):
            for j in range(n):
                matrix[j][idx[i][1]] = 0
            for j in range(m):
                matrix[idx[i][0]][j] = 0

    def trap(self, height: List[int]) -> int:
        """https://leetcode-cn.com/problems/trapping-rain-water/"""
        n = len(height)
        if n == 0:
            return 0
        ans = 0
        max_top = 0
        max_idx = 0
        for i in range(n):
            if max_top <= height[i]:
                max_top = height[i]
                max_idx = i
            else:
                ans += max_top - height[i]
        max_current = 0
        for i in range(n - 1, max_idx, -1):
            if max_current < height[i]:
                max_current = height[i]
            ans -= max_top - max_current
        return ans

    def jump(self, nums: List[int]) -> int:
        """https://leetcode-cn.com/problems/jump-game-ii/"""
        n = len(nums)
        ans = [0 for _ in range(n)]
        idx = 1
        for i in range(n):
            end = min(i + 1 + nums[i], n)
            if end > idx:
                for j in range(idx, end):
                    if ans[j] == 0:
                        ans[j] = ans[i] + 1
                    else:
                        ans[j] = min(ans[j], ans[i] + 1)
                idx = end
                if idx == n:
                    break
        return ans[n - 1]

    def isNumber(self, s: str) -> bool:
        """https://leetcode-cn.com/problems/valid-number/"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """https://leetcode-cn.com/problems/search-a-2d-matrix/"""
        n = len(matrix)
        if n == 0:
            return False
        m = len(matrix[0])
        if m == 0:
            return False
        i, j = 0, n - 1
        r = -1
        while i <= j:
            middle = (i + j) // 2
            if matrix[middle][0] <= target <= matrix[middle][m - 1]:
                r = middle
                break
            elif matrix[middle][0] > target:
                j = middle - 1
            else:
                i = middle + 1
        i, j = 0, m - 1
        while i <= j:
            middle = (i + j) // 2
            if matrix[r][middle] == target:
                return True
            elif matrix[r][middle] > target:
                j = middle - 1
            else:
                i = middle + 1
        return False

    def sortColors(self, nums: List[int]) -> None:
        "https://leetcode-cn.com/problems/sort-colors/"
        n = len(nums)
        if n == 0:
            return
        # red white blue
        color = [0, 0, 0]
        for i in range(n):
            color[nums[i]] += 1
        for i in range(n):
            if i < color[0]:
                nums[i] = 0
            elif color[0] <= i < color[0] + color[1]:
                nums[i] = 1
            else:
                nums[i] = 2

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        """https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/"""
        pre, temp = head, head
        while temp is not None:
            if pre == temp:
                temp = temp.next
                continue
            elif pre.val == temp.val:
                pre.next = temp.next
                temp.next = None
                temp = pre.next
            else:
                pre = temp
                temp = temp.next
        return head

    def deleteDuplicates_ii(self, head: ListNode) -> ListNode:
        """https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/"""
        pre, temp = head, head
        while temp is not None:
            if temp.next is not None and temp.val == temp.next.val:
                item = temp.next
                while item is not None and item.val == temp.val:
                    item = item.next
                if pre != temp:
                    pre.next = item
                    temp = item
                else:
                    head = item
                    pre = item
                    temp = item
            elif pre != temp:
                pre = temp
                temp = temp.next
            else:
                temp = temp.next
        return head

    def largestRectangleArea(self, heights: List[int]) -> int:
        """https://leetcode-cn.com/problems/largest-rectangle-in-histogram/"""
        # n = len(heights)
        # if n == 0:
        #     return 0
        # min_height = [[None for _ in range(i)] for i in range(1, n + 1)]
        # ans = 0
        # for i in range(n):
        #     for j in range(i, -1, -1):
        #         if i == j:
        #             min_height[i][j] = heights[i]
        #         else:
        #             min_height[i][j] = min(min_height[i][j + 1], heights[j])
        #         ans = max(min_height[i][j] * (i - j + 1), ans)
        # return ans
        heights.append(0)
        stack = [-1]
        ans = 0
        for i in range(0, len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                ans = max(ans, heights[stack.pop()] * (i - stack[-1] - 1))
            stack.append(i)
        return ans

    def partition(self, head: ListNode, x: int) -> ListNode:
        """https://leetcode-cn.com/problems/partition-list/"""
        # if head is None:
        #     return head
        # left, temp, item = head, head, head
        # while temp is not None and temp.val < x:
        #     temp = temp.next
        # if temp is None:
        #     return head
        # right = temp
        # if temp == head or left.val > x:
        #     item = temp
        #     pre = item
        #     while item is not None and item.val >= x:
        #         if pre != item:
        #             pre = pre.next
        #         item = item.next
        #     if item is not None:
        #         pre.next = item.next
        #         item.next = head
        #         left = head = item
        #     else:
        #         return head
        # else:
        #     while left.next != temp and left.next.val <= x:
        #         left = left.next
        # while left.next != temp or right.next is not None:
        #     if left.next != temp and left.next.val > x:
        #         item = left.next
        #         left.next = item.next
        #         item.next = right.next
        #         right.next = item
        #         right = item
        #     elif right.next is not None and right.next.val < x:
        #         item = right.next
        #         right.next = item.next
        #         item.next = left.next
        #         left.next = item
        #         left = item
        #     elif left.next != temp:
        #         left = left.next
        #     else:
        #         right = right.next
        # return head
        temp = head
        smaller, bigger = [], []
        while temp is not None:
            if temp.val < x:
                smaller.append(temp)
            else:
                bigger.append(temp)
            temp = temp.next
        ans = None
        temp = None
        tail = None
        for i, item in enumerate(smaller):
            tail = item
            if ans is None:
                ans = temp = item
            else:
                temp.next = item
                temp = item
        for i, item in enumerate(bigger):
            tail = item
            if ans is None:
                ans = temp = item
            else:
                temp.next = item
                temp = item
        if tail:
            tail.next = None
        return ans

    def mergeSortedArray(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """https://leetcode-cn.com/problems/merge-sorted-array/"""
        j = 0
        for i in range(m + n):
            if j < n and nums2[j] < nums1[i]:
                for k in range(m + j, i, -1):
                    nums1[k] = nums1[k - 1]
                nums1[i] = nums2[j]
                j += 1
            elif i - j >= m:
                nums1[i] = nums2[j]
                j += 1

    def combine(self, n: int, k: int) -> List[List[int]]:
        """https://leetcode-cn.com/problems/combinations/"""
        if n == 0 or k == 0:
            return []
        nums = [0 for _ in range(n)]
        for i in range(n):
            nums[i] = i + 1

        def insert_num(insert_nums, current_nums, all_nums, count, max_count):
            if count == max_count:
                result = current_nums.copy()
                insert_nums.append(result)
            else:
                nums_len = len(all_nums)
                for idx in range(nums_len):
                    current_nums[count] = all_nums[idx]
                    insert_num(insert_nums, current_nums, all_nums[idx + 1: nums_len], count + 1, max_count)

        ans = []
        cache = [0 for _ in range(k)]
        insert_num(ans, cache, nums, 0, k)
        return ans

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """https://leetcode-cn.com/problems/subsets/"""
        # n = len(nums)
        # ans = []
        #
        # def insert_num(insert_nums, current_nums, all_nums, count, max_count):
        #     if count == max_count:
        #         result = current_nums.copy()
        #         insert_nums.append(result)
        #     else:
        #         nums_len = len(all_nums)
        #         for idx in range(nums_len):
        #             current_nums[count] = all_nums[idx]
        #             insert_num(insert_nums, current_nums, all_nums[idx + 1: nums_len], count + 1, max_count)
        #
        # for i in range(0, n + 1):
        #     insert_num(ans, [0 for _ in range(i)], nums, 0, i)
        # return ans
        n = len(nums)
        ans = [[]]
        for i in range(n):
            len_ans = len(ans)
            for j in range(len_ans):
                item = ans[j].copy()
                item.append(nums[i])
                ans.append(item)
        return ans

    def exist(self, board: List[List[str]], word: str) -> bool:
        """https://leetcode-cn.com/problems/word-search/"""
        n = len(board)
        if n == 0:
            return False
        m = len(board[0])
        if m == 0:
            return False
        len_word = len(word)
        if len_word == 0:
            return True
        start_num = []
        for i in range(n):
            for j in range(m):
                if board[i][j] == word[0]:
                    start_num.append([i, j])
        if not start_num:
            return False
        for k in range(len(start_num)):
            idx, step, board_cp = 0, [0 for _ in range(len_word)], [[board[i][j] for j in range(m)] for i in range(n)]
            i, j = start_num[k][0], start_num[k][1]
            while -1 <= i <= n and -1 <= j <= m:
                if 0 <= i < n and 0 <= j < m and board_cp[i][j] == word[idx] and step[idx] < 4:
                    if idx == len_word - 1:
                        return True
                    board_cp[i][j] = None
                    if step[idx] == 0:
                        # 下
                        i += 1
                    elif step[idx] == 1:
                        # 右
                        j += 1
                    elif step[idx] == 2:
                        # 上
                        i -= 1
                    elif step[idx] == 3:
                        # 左
                        j -= 1
                    else:
                        break
                    idx += 1
                else:
                    if step[idx] > 3:
                        step[idx] = 0
                        idx -= 1
                        continue
                    idx -= 1
                    if idx < 0:
                        break
                    elif step[idx] == 0:
                        # 上一步下，这步右
                        step[idx] += 1
                        i -= 1
                        j += 1
                        idx += 1
                    elif step[idx] == 1:
                        # 上一步右，这步上
                        step[idx] += 1
                        j -= 1
                        i -= 1
                        idx += 1
                    elif step[idx] == 2:
                        # 上一步上，这步左
                        step[idx] += 1
                        i += 1
                        j -= 1
                        idx += 1
                    elif step[idx] == 3:
                        # 上一步左，这步跳出
                        step[idx] += 1
                        j += 1
                        board_cp[i][j] = word[idx]
        return False

    def removeDuplicates_o1(self, nums: List[int]) -> int:
        """https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/"""
        n = len(nums)
        if n <= 2:
            return n
        j, pre, count = 1, 0, 0
        for i in range(1, n):
            if nums[i] != nums[pre]:
                count = 0
                nums[j] = nums[i]
                j += 1
            else:
                count += 1
                if count < 2:
                    nums[j] = nums[i]
                    j += 1
            pre = i
        return j

    def search_ii(self, nums: List[int], target: int) -> bool:
        """https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/"""
        return target in nums

    def inorderTraversal(self, root: TreeNode) -> List[int]:
        """https://leetcode-cn.com/problems/binary-tree-inorder-traversal/"""
        if root is None:
            return []
        stack = [root]
        ans = []
        while stack:
            temp = stack[-1]
            if temp.left is not None:
                stack.append(temp.left)
                temp.left = None
            elif temp.right is not None:
                ans.append(temp.val)
                stack.pop()
                stack.append(temp.right)
                temp.right = None
            else:
                ans.append(temp.val)
                stack.pop()
        return ans

    def postorderTraversal(self, root: TreeNode) -> List[int]:
        """https://leetcode-cn.com/problems/binary-tree-postorder-traversal/"""
        if root is None:
            return []
        stack = [root]
        ans = []
        while stack:
            temp = stack[-1]
            if temp.left is not None:
                stack.append(temp.left)
                temp.left = None
            elif temp.right is not None:
                stack.append(temp.right)
                temp.right = None
            else:
                ans.append(temp.val)
                stack.pop()
        return ans

    def preorderTraversal(self, root: TreeNode) -> List[int]:
        """https://leetcode-cn.com/problems/binary-tree-preorder-traversal/"""
        if root is None:
            return []
        stack = [root]
        ans = []
        while stack:
            temp = stack[-1]
            ans.append(temp.val)
            if temp.left is not None:
                stack.append(temp.left)
                temp.left = None
            elif temp.right is not None:
                stack.append(temp.right)
                temp.right = None
            else:
                while temp.left is None and temp.right is None:
                    stack.pop()
                    if stack:
                        temp = stack[-1]
                    else:
                        return ans
                if temp.right is not None:
                    stack.append(temp.right)
                    temp.right = None
        return ans

    def numTrees(self, n: int) -> int:
        """https://leetcode-cn.com/problems/unique-binary-search-trees/"""
        if n == 0:
            return 0
        ans = [0 for _ in range(n + 1)]
        ans[0] = 1
        for i in range(n + 1):
            for j in range(i):
                ans[i] += ans[j] * ans[i - j - 1]
        return ans[n]

    def isValidBST(self, root: TreeNode) -> bool:
        """https://leetcode-cn.com/problems/validate-binary-search-tree/"""
        if root is None:
            return True
        stack = [root]
        ans = []
        while stack:
            temp = stack[-1]
            if temp.left is not None:
                stack.append(temp.left)
                temp.left = None
            elif temp.right is not None:
                ans.append(temp.val)
                stack.pop()
                stack.append(temp.right)
                temp.right = None
            else:
                ans.append(temp.val)
                stack.pop()
        for i in range(len(ans) - 1):
            if ans[i] >= ans[i + 1]:
                return False
        return True

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        """https://leetcode-cn.com/problems/binary-tree-level-order-traversal/"""
        ans = []
        if root is None:
            return ans
        pre = [root]
        current = []
        while True:
            item = []
            for i in range(len(pre)):
                item.append(pre[i].val)
                if pre[i].left is not None:
                    current.append(pre[i].left)
                if pre[i].right is not None:
                    current.append(pre[i].right)
            ans.append(item)
            if current:
                pre = current
                current = []
            else:
                break
        return ans

    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        """https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/"""
        ans = []
        if root is None:
            return ans
        pre = [root]
        current = []
        while True:
            item = []
            for i in range(len(pre)):
                item.append(pre[i].val)
                if pre[i].left is not None:
                    current.append(pre[i].left)
                if pre[i].right is not None:
                    current.append(pre[i].right)
            ans.append(item)
            if current:
                pre = current
                current = []
            else:
                break
        for i in range(1, len(ans), 2):
            n = len(ans[i])
            ans[i] = ans[i][::-1]
        return ans

    def isSymmetric(self, root: TreeNode) -> bool:
        """https://leetcode-cn.com/problems/symmetric-tree/"""
        # if root is None:
        #     return True
        # pre = [root]
        # current = []
        # while True:
        #     item = []
        #     allNone = True
        #     for i in range(len(pre)):
        #         if pre[i] is not None:
        #             allNone = False
        #             item.append(pre[i].val)
        #             current.append(pre[i].left)
        #             current.append(pre[i].right)
        #         else:
        #             item.append(None)
        #             current.append(None)
        #             current.append(None)
        #     if allNone:
        #         break
        #     n = len(item)
        #     for i in range(n // 2):
        #         if item[i] != item[n - i - 1]:
        #             return False
        #     if current:
        #         pre = current
        #         current = []
        #     else:
        #         break
        # return True
        if root is None:
            return True

        def symmetirc(left: TreeNode, right: TreeNode) -> bool:
            if left is None and right is None:
                return True
            if left is None or right is None:
                return False
            return left.val == right.val and symmetirc(left.right, right.left) and symmetirc(left.left, right.right)

        return symmetirc(root.left, root.right)

    def maxDepth(self, root: TreeNode) -> int:
        """https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/"""
        if root is None:
            return 0
        return max(self.maxDepth(root.left) + 1, self.maxDepth(root.right) + 1)

    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        """https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/"""
        ans = []
        if root is None:
            return ans
        pre = [root]
        current = []
        while True:
            item = []
            for i in range(len(pre)):
                item.append(pre[i].val)
                if pre[i].left is not None:
                    current.append(pre[i].left)
                if pre[i].right is not None:
                    current.append(pre[i].right)
            ans.append(item)
            if current:
                pre = current
                current = []
            else:
                break
        return ans[::-1]

    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        """https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/"""
        if not nums:
            return None
        idx = len(nums) // 2
        node = TreeNode(nums[idx])
        node.left = self.sortedArrayToBST(nums[: idx])
        node.right = self.sortedArrayToBST(nums[idx + 1:])
        return node

    def sortedListToBST(self, head: ListNode) -> TreeNode:
        """https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/"""
        if not head:
            return None
        if head.next is None:
            return TreeNode(head.val)
        pre, temp, middle = head, head, head
        while temp is not None and temp.next is not None:
            temp = temp.next.next
            if pre != middle:
                pre = pre.next
            middle = middle.next
        node = TreeNode(middle.val)
        pre.next = None
        node.left = self.sortedListToBST(head)
        node.right = self.sortedListToBST(middle.next)
        return node

    def grayCode(self, n: int) -> List[int]:
        """https://leetcode-cn.com/problems/gray-code/"""
        # if n == 1:
        #     return [0, 1]
        # else:
        #     ans = self.grayCode(n - 1)
        #     value = 2 ** (n - 1)
        #     for i in range(len(ans) - 1, -1, -1):
        #         ans.append(value + ans[i])
        #     return ans
        # i, ans = 0, []
        # while i < 1 << n:
        #     ans.append(i ^ i >> 1)
        #     i += 1
        # return ans
        return [i ^ (i >> 1) for i in range(1 << n)]

    def minWindow(self, s: str, t: str) -> str:
        """https://leetcode-cn.com/problems/minimum-window-substring/"""
        if not t:
            return t
        if not s:
            return s
        char_map = {}
        count_map = {}
        match_map = {}
        for i in range(len(t)):
            if t[i] in char_map.keys():
                char_map[t[i]] += 1
            else:
                char_map[t[i]] = 1
                count_map[t[i]] = 0
                match_map[t[i]] = False
        i, j, n, right, ans, ans_i, ans_j, m = 0, 0, len(s), True, float('inf'), None, None, len(char_map.keys())

        def all_match(all_match_map):
            for i, item in enumerate(all_match_map.values()):
                if not item:
                    return False
            return True

        while i < n and j < n:
            if right:
                if s[j] in char_map.keys():
                    count_map[s[j]] += 1
                    match_map[s[j]] = count_map[s[j]] >= char_map[s[j]]
                    if match_map[s[j]]:
                        if all_match(match_map):
                            right = False
                            if ans > j - i + 1:
                                ans = j - i + 1
                                ans_i = i
                                ans_j = j
                            continue
            else:
                if s[i] in char_map.keys():
                    count_map[s[i]] -= 1
                    match_map[s[i]] = count_map[s[i]] >= char_map[s[i]]
                    if not match_map[s[i]]:
                        match_map[s[i]] = True
                        count_map[s[i]] += 1
                        right = True
                        j += 1
                        continue
                    else:
                        if ans > j - i:
                            ans = j - i
                            ans_i = i + 1
                            ans_j = j
                else:
                    if ans > j - i:
                        ans = j - i
                        ans_i = i + 1
                        ans_j = j
            if right:
                j += 1
            else:
                i += 1
        if ans_i is None or ans_j is None:
            return ""
        return s[ans_i:ans_j + 1]

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """https://leetcode-cn.com/problems/subsets-ii/"""
        nums = sorted(nums)
        n = len(nums)
        ans = [[]]
        for i in range(n):
            len_ans = len(ans)
            for j in range(len_ans):
                item = ans[j].copy()
                item.append(nums[i])
                if item not in ans:
                    ans.append(item)
        return ans

    def numDecodings(self, s: str) -> int:
        """https://leetcode-cn.com/problems/decode-ways/"""
        n = len(s)
        if n == 0:
            return 0
        if n == 1 and s[0] == '0':
            return 0
        short_dp = [1]
        long_dp = [0]
        for i in range(1, n):
            if s[i] == '0':
                if s[i - 1] == '1' or s[i - 1] == '2':
                    short_dp.append(0)
                    long_dp.append(short_dp[i - 1])
                else:
                    return 0
            elif s[i - 1] == '1' or (s[i - 1] == '2' and s[i] <= '6'):
                short_dp.append(short_dp[i - 1] + long_dp[i - 1])
                long_dp.append(short_dp[i - 1])
            else:
                short_dp.append(short_dp[i - 1] + long_dp[i - 1])
                long_dp.append(0)
        return long_dp[n - 1] + short_dp[n - 1]

    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        """https://leetcode-cn.com/problems/reverse-linked-list-ii/"""
        if head is None:
            return head

        def reverse(first: ListNode, l_in: int) -> (ListNode, ListNode):
            if not first or l_in <= 1:
                return first, first
            last = first
            second = first.next
            temp_in = None
            for i_in in range(l_in - 1):
                temp_in = second.next
                second.next = first
                first = second
                second = temp_in
            last.next = temp_in
            return first, last

        if m <= 1:
            first_n, last = reverse(head, n - m + 1)
            return first_n
        temp, start_count = head, 1
        while temp is not None:
            if start_count < m:
                start_count += 1
                if start_count == m:
                    first_n, last = reverse(temp.next, n - m + 1)
                    temp.next = first_n
                    return head
            temp = temp.next
        return head

    def restoreIpAddresses(self, s: str) -> List[str]:
        """https://leetcode-cn.com/problems/restore-ip-addresses/"""

        def tryIpAddress(all_ips: List[str], cur_rel: str, current_ips: str, about_position: int):
            if about_position == 0 or len(current_ips) == 0:
                return
            if about_position == 1:
                if int(current_ips) <= 255 and ((len(current_ips) == 1) or (len(current_ips) and current_ips[0] != '0')):
                    all_ips.append(cur_rel + current_ips)
                return
            for i in range(1, 4):
                if int(current_ips[:i]) <= 255 and ((i == 1) or (i > 1 and current_ips[0] != '0')):
                    tryIpAddress(all_ips, cur_rel + current_ips[:i] + '.', current_ips[i: len(current_ips)], about_position - 1)

        ans = []
        tryIpAddress(ans, '', s, 4)
        return ans

    def generateTrees(self, n: int) -> List[TreeNode]:
        """https://leetcode-cn.com/problems/unique-binary-search-trees-ii/"""

        # def generate_trees(start, end):
        #     if start > end:
        #         return [None, ]
        #
        #     all_trees = []
        #     for i in range(start, end + 1):  # pick up a root
        #         # all possible left subtrees if i is choosen to be a root
        #         left_trees = generate_trees(start, i - 1)
        #
        #         # all possible right subtrees if i is choosen to be a root
        #         right_trees = generate_trees(i + 1, end)
        #
        #         # connect left and right subtrees to the root i
        #         for l in left_trees:
        #             for r in right_trees:
        #                 current_tree = TreeNode(i)
        #                 current_tree.left = l
        #                 current_tree.right = r
        #                 all_trees.append(current_tree)
        #
        #     return all_trees
        #
        # return generate_trees(1, n) if n else []
        from functools import reduce
        from operator import add
        if n:
            situ = {0: dict.fromkeys(range(1, n + 2), [None])}  # 外层的键表示当前的树由多少个数字组成 内层的键表示root的val
            for i in range(1, n + 1):
                situ[i] = dict()
                for j in range(1, n + 2 - i):  # 表示左端其实的索引
                    situ[i][j] = list()
                    for step in range(i):
                        for le in situ[step][j]:
                            for ri in situ[i - step - 1][j + step + 1]:
                                new = TreeNode(step + j)
                                new.left, new.right = le, ri
                                situ[i][j].append(new)
            return reduce(add, situ[n].values())
        return []

    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        """https://leetcode-cn.com/problems/same-tree/"""
        if p is None or q is None:
            return p == q
        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False

    def buildTree_pre_in(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        """https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/"""
        if not preorder or not inorder:
            return None

        if preorder[0] not in inorder:
            return None
        val = preorder.pop(0)
        root = TreeNode(val)
        i = inorder.index(val)
        root.left = self.buildTree_pre_in(preorder, inorder[0: i])
        root.right = self.buildTree_pre_in(preorder, inorder[i + 1:])
        return root

    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        """https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/"""
        if not postorder or not inorder:
            return None

        if postorder[-1] not in inorder:
            return None
        val = postorder.pop()
        root = TreeNode(val)
        i = inorder.index(val)
        root.right = self.buildTree(inorder[i + 1:], postorder)
        root.left = self.buildTree(inorder[0: i], postorder)
        return root

    def isBalanced(self, root: TreeNode) -> bool:
        """https://leetcode-cn.com/problems/balanced-binary-tree/"""
        if root is None:
            return True

        def getHeight(tree_node: TreeNode):
            if tree_node is None:
                return 0
            return max(getHeight(tree_node.left), getHeight(tree_node.right)) + 1

        return abs(getHeight(root.left) - getHeight(root.right)) > 1


