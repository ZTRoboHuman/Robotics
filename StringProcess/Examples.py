
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
        res_str=""

        return res_str