---
title: 一道题2
date: 2021-03-13 22:33:00
updated: 2021-03-13 22:33:00
tags: [数论,线性基]
categories: 考试
---
> 给定一个长度为 $n$ 的序列 $A$，把 $A$ 划分为两个非空集合，划分的价值为两个集合的异或和加起来，求可能的最大价值。
>
> $n \le 10^5,A_i \le 10^{18}$

设总异或和为 $sum$ ，若两个集合的异或和分别为 $S_1,S_2$，由 $S_1 \oplus S_2 = sum$ 知道，若 $sum$ 第 $i$ 位为 $1$，不管怎么划分，第 $i$ 位一定产生一次贡献。若 $sum$ 第 $i$ 位为 $0$，那么 $S_1,S_2$ 在第 $i$ 位上相同。

考虑 $sum$ 为 $0$ 的位，由于 $S_1,S_2$ 在这些位上是相同的，只需要最大化 $S_1$ 就行了，由于 $S_1$ 为全集和空集都不会成为最优解，问题就转化为： 

给定n个整数（数字可能重复），求在这些数中选取任意个，使得他们的异或和最大。 

用线性基解决。

代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
using namespace std;

int n;
long long a[100005], bas[70], su, ans;
int main() {
    cin >> n;
    rep(i, 1, n) scanf("%lld", &a[i]), su ^= a[i];
    rep(i, 1, n) a[i] &= ~su;
    rep(i, 1, n) per(j, 59, 0) if(a[i] >> j & 1) {
        if(bas[j]) a[i] ^= bas[j];
        else {
            per(k, j - 1, 0) if(a[i] >> k & 1) a[i] ^= bas[k];
            rep(k, j + 1, 59) if(bas[k] >> j & 1) bas[k] ^= a[i];
            bas[j] = a[i]; break;
        }
    }
    rep(i, 1, 59) ans ^= bas[i];
    cout << su + ans * 2;
    return 0;
}
```