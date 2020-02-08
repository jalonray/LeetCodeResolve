import bisect
import collections
from typing import List


class ListNode:

    def __init__(self, x, nextNode=None):
        self.val = x
        self.next = nextNode


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

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        len1 = len(nums1)
        len2 = len(nums2)
        return 1

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
