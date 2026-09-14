---
comments: true
difficulty: 简单
---

<!-- problem:start -->

# [面试题 03. 数组中重复的数字](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

## 题目描述

<!-- description:start -->

<p>找出数组中重复的数字。</p>

<p><strong>示例 1：</strong></p>

<pre><strong>输入：</strong>
[2, 3, 1, 0, 2, 5, 3]
<strong>输出：</strong>2 或 3
</pre>

<!-- description:end -->

## 解法

<!-- solution:start -->

### 方法一：排序

<!-- tabs:start -->

#### Python3

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        for a, b in pairwise(sorted(nums)):
            if a == b:
                return a
```

<!-- tabs:end -->

<!-- solution:end -->

<!-- problem:end -->
