---
comments: true
difficulty: Easy
tags:
    - Database
---

<!-- problem:start -->

# [627. Swap Sex of Employees 🔒](https://leetcode.com/problems/swap-sex-of-employees)

[中文文档](/solution/0600-0699/0627.Swap%20Sex%20of%20Employees/README.md)

## Description

<!-- description:start -->

<p>Premium-only problem, but a community solution exists below, so no data is actually missing.</p>

<!-- description:end -->

## Solutions

<!-- solution:start -->

### Solution 1

<!-- tabs:start -->

#### MySQL

```sql
UPDATE salary
SET sex = IF(sex = 'm', 'f', 'm');
```

<!-- tabs:end -->

<!-- solution:end -->

<!-- problem:end -->
