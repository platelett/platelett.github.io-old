---
title: 不条理狂诗曲 | 洛谷 P7482
date: 2021-04-12 15:24:46
updated: 2021-04-12 15:24:46
tags: [分治,双指针]
categories: 洛谷月赛
---
> [题目链接](https://www.luogu.com.cn/problem/P7482)
>
> 给定一个长度为 $n$ 的序列 $A$，定义 $f(l,r)$ 表示从区间 $[l,r]$ 中选择若干不相邻的数的和的最大值。
>
> 求 $\sum\limits_{l=1}^n\sum\limits_{r=l}^nf(l,r) \bmod 10^9 + 7$。
>
> $n \le 10^5,0 \le a_i \le 10^9$

$f(l,r)$ 显然可以通过 ```DP``` 在 $O(r-l+1)$ 的时间内求出，枚举左端点或右端点来计算答案都不太行得通，因为端点移动一步后难以快速维护。

但所有包含同一位置 $p$ 的区间（$p$ 是端点不算）的 $f$ 值之和却可以快速计算，因为枚举了 $p$ 选不选后，$p$ 左右的部分是**独立的**。

对于一个包含 $p$ 的区间 $[l,r]$，有
$$
f(l,r)=\max\{f(l,p-1)+f(p+1,r),f(l,p-2)+f(p+2,r)+A_p\}
$$
设
$$
x_i=\begin{cases}f(i,p-1)&(i < p)\\\\f(p+1,i)&(i>p)\\\\0&(i=p)\end{cases},y_i=\begin{cases}f(i,p-2)&(i<p-1)\\\\f(p+2,i)&(i>p+1)\\\\0&(p-1 \le i \le p+1)\end{cases}
$$
$x,y$ 数组可以通过 ```DP``` 在 $O(n)$ 的时间内求出。
$$
f(l,r)=\max\{x_l+x_r,y_l+y_r+A_p\}\\\\
x_l+x_r \ge y_l + y_r + A_p \iff (x_l - y_l) + (x_r - y_r) \ge A_p
$$
把所有除 $p$ 以外的位置按 $x_i-y_i$ 为关键字升序排序，再用双指针扫描一遍即可求出所有包含 $p$ （$p$ 是端点不算）的 $f$ 值之和，单次的复杂度为 $O(n\log n)$。

接下来就是套路了，考虑分治，定义函数 $solve(L,R)$ 表示 $\sum\limits_{l=L}^R\sum\limits_{r=l}^Rf(l,r)$。

令 $mid = \lfloor \frac {L + R}2 \rfloor$，那么 $solve(L, R) = solve(L, mid) + solve(mid, R) - A_{mid} + \sum\limits_{l=L}^{mid-1}\sum\limits_{r=mid+1}^Rf(l,r)$。

最后一项用上述方法计算，总复杂度为 $O(n \log^2 n)$。

代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back

using namespace std;
const int N = 1e5 + 5;
typedef long long ll;
const ll P = 1e9 + 7, Inf = 1e18;
int n; ll a[N], f[N][2][2], A[N], B[N], as;
void solve(int l, int r) {
    if(l == r) { (as += a[l]) %= P; return; }
    if(r == l + 1) { (as += a[l] + a[r] + max(a[l], a[r])) %= P; return; }
    int mid = (l + r) / 2;
    rep(i, 0, 1) f[mid][i][i] = 0, f[mid][i][!i] = -Inf;
    rep(i, mid + 1, r) rep(j, 0, 1) {
        f[i][j][0] = max(f[i - 1][j][0], f[i - 1][j][1]);
        f[i][j][1] = f[i - 1][j][0] + a[i];
    }
    per(i, mid - 1, l) rep(j, 0, 1) {
        f[i][j][0] = max(f[i + 1][j][0], f[i + 1][j][1]);
        f[i][j][1] = f[i + 1][j][0] + a[i];
    }
    vector <int> v;
    rep(i, l, r) if(i ^ mid) {
        v.pb(i);
        A[i] = max(f[i][0][0], f[i][0][1]);
        B[i] = max(f[i][1][0], f[i][1][1]);
    }
    sort(v.begin(), v.end(), [](int i, int j){ return A[i] - B[i] < A[j] - B[j]; });
    int j = v.size() - 1;
    ll sua = 0, sub = 0; int cna = 0, cnb = 0;
    for(int i : v) if(i > mid) (sub += B[i]) %= P, cnb++;
    for(int i : v) if(i < mid) {
        while(j >= 0 && (v[j] < mid || A[i] - B[i] + A[v[j]] - B[v[j]] > a[mid])) {
            if(v[j] > mid) (sua += A[v[j]]) %= P, (sub -= B[v[j]]) %= P, cna++, cnb--;
            j--;
        }
        (as += (a[mid] + B[i]) * cnb + A[i] * cna + sua + sub) %= P;
    }
    (as -= a[mid]) %= P, solve(l, mid), solve(mid, r);
}
int main() {
    cin >> n;
    rep(i, 1, n) scanf("%lld", &a[i]);
    solve(1, n);
    cout << (as + P) % P;
    return 0;
}
```