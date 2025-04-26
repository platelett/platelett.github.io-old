---
title: Many Slimes | abc 140f
date: 2021-03-13 23:10:49
updated: 2021-03-13 23:10:49
tags: [构造,贪心]
categories: ABC
---
> [题目链接](https://atcoder.jp/contests/abc140/tasks/abc140_f)
>
> 每个细菌都有一个 $s$ 值。
>
> 最初只有一个细菌。每一天原先的细菌都会生一个孩子，且孩子的 $s$ 值小于母体的 $s$ 值。
>
> 给定 $n$ 和 $2^n$ 个数。
>
> 判断是否存在一种情况，经过 $n$ 天后所有细菌的 $s$ 值为给定的数。
>
> $n \le 10^5,s_i \le 10^9$

首先最大的数一定是原始细菌的 $s$ 值。然后考虑第 $i$ 天的细菌，它们所选择的 $s$ 一定要尽可能的大，如果选小了会使它们的后代选择变少。

然后对于每一天产生的每个细菌都按顺序贪心地选择一个尽可能大的值。

同一天选择的顺序是**无关紧要**的，下面给出证明：

> 考虑第 $i$ 天有两个选择的顺序上**相邻**的细菌  $a, b$，记它们父亲的 s 分别为 $A, B (A < B)$。
>
> 如果让 $a$ 先选，且两者的 $s$ 分别为 $x，y(x < y)$。
>
> 若 $y < A$，那么交换顺序后两者的 $s$ 就为 $y，x$。
>
> 由于 $a, b$ 在同一天产生，它们在决定后续方案时的**地位**相同，那么选择的顺序也就无关紧要了。
>
> 若 $y > A$，那么交换顺序后两者的 s 不变，选择的顺序连方案都不会影响。

为了方便，把父亲编号为 $j(j<2^i)$ 的细菌编号为 $j+2^i$。

然后直接按编号顺序确定 $s$ 就行了，为了支持删除和查找前驱，可以用 ```multiset``` 来维护没被选择的 $s$。

代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)

using namespace std;

int main() {
    cin >> n, m = 1 << n;
    multiset <int, greater <int> > s;
    For(i, 0, m) s.insert(read());
    vector <int> c;
    c.push_back(*s.begin()), s.erase(s.begin());
    For(i, 0, m) {
        For(j, 0, 1 << i) {
            auto it = s.upper_bound(c[j]);
            if(it != s.end()) c.push_back(*it), s.erase(it);
            else return puts("No") < 0;
        }
    }
    return puts("Yes") < 0;
    return 0;
}
```

