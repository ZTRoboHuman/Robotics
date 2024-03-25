
# string solution collection
class Solution:
    # 3/22/2024
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 3
        start = 0
        visited=set()
        res=0
        for i in range(len(s)):
            while s[i] in visited:
                visited.remove(s[start])
                start+=1
            visited.add(s[i])
            res=max(res,i-start+1)
        return res

    def longestPalindrome(self, s: str) -> str:
        # 5
        if len(s) <=1:
            return s
        res_str=""
        slen=len(s)
        # abasabbab
        for i in range(slen):
            l=0
            # i-l should be >=0, not >0, be careful about the conner case
            while i-l>=0 and i+l<slen and s[i-l]==s[i+l]:
                l+=1
            l -= 1  # Adjusting l to get the correct length of the palindrome
            # s[i-l: i+l+1] represents a substring of s starting from index i-l (inclusive) and ending at index i+l+1 (exclusive).
            if len(res_str) < 2 * l + 1:
                res_str=s[i-l : i+l+1]
            l = 0
            while i-l>=0 and i+l+1<slen and s[i-l]==s[i+l+1]:
                l+=1
            l -= 1
            if len(res_str) < 2 * l + 2:
                res_str=s[i-l : i+l+2]

        return res_str

    def convert(self, s: str, numRows: int) -> str:
        # 6 Zigzag Conversion
        # Initialize the result string
        res = ""

        # Handle edge case
        if numRows == 1 or numRows >= len(s):
            return s

        strlist=[""]*numRows
        pos=0
        direction=1
        for c in s:
            strlist[pos] += c
            pos += direction
            if pos==0 or pos==numRows-1:
                direction = -direction
        res="".join(strlist)
        return res
