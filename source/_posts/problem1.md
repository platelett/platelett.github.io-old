---
title: 一道题1
date: 2021-03-13 22:29:46
updated: 2021-03-13 22:29:46
tags: [数据结构,分块,斜率优化]
categories: 考试
---
> 给定一个长度为 $n$ 的序列 $a$。有 $q$ 次询问，每次给出区间 $[L,R]$ ,
>
> 求 $\max\limits_{L\le l<r\le R}\dfrac{\sum_{i=l}^ra_i}{r-l}$
>
> $n\le 10^5,q\le 3\cdot 10^4,|a_i|\le 10^6$

先把原问题转化成斜率最大值。

记 $s_i=\sum_{j=1}^ia_j,\text{点}A_i(i,s_{i-1}),\text{点}B_i(i,s_i)$

$\text{原式}=\max\limits_{L\le i < j\le R}\text{Slope}_{A_i,B_j}$

考虑 $L=1,R=n$ 时怎么做。

- 从左往右依次扫描，同时维护 $A$ 类点的下凸包，每次二分求出 $B_i$ 过当前凸包上的切点。

  时间复杂度 $O(n\log n)$。

- **线性做法**：单调队列维护 $A$ 类点的下凸包，一旦发现队首不是当前 $B_i$ 过当前凸包上的切点，就将队首弹出。

  这样做可能会导致之后的一些 $B_i$ 的切点被弹出而失去这些 $B_i$ 的最优解，但可以证明这样做不会错过**全局最优解**。

  **如图**，扫描到 $E$ 时切点已经不是 $A$ 了，弹出 $A$ 后会导致扫到 $F$ 时失去最优解。

  但因为 $F$ 的切线是 $AF$, 所以 $\text{Slope}_{AF}<\text{Slope}_{AB}<\text{Slope}_{BE}$。

  故 $AF$ 劣于 $BE$, $A$ 已经不可能更新最优解。

  

  ![](https://i.loli.net/2020/11/18/ncWZHmCtw2bokxX.png)

考虑分块，令 $sz=\sqrt n$，每次查询分两类：

- 当 $A_i,B_j$ 都在边角（在同一个块或不同块）中时，直接对所有边角使用线性做法求出这类最优解。

- 当 $A_i,B_j$ 至少有一个在大块中时。

  预处理 $pre_{i,j}=\max\limits_{sz\cdot i\le I< J \le j}\text{Slope}_{A_I,B_J},suf_{i,j}=\max\limits_{j\le I<J\le sz\cdot i}\text{Slope}_{A_I,B_J}$

  预处理方式也使用线性做法。

  用 $pre_{\lceil\frac L{sz}\rceil,R},suf_{\lfloor\frac R{sz}\rfloor,L}$  更新答案。

综上，时间复杂度 $O(n\sqrt n)$。

代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)

using namespace std;
const int N = 100005, M = 125;
typedef long long ll;
int read() {
    int c = getchar(), r = 0, f = 1; 
    while(c < 48) { if(c == 45) f = -1; c = getchar(); }
    while(c > 47) r = r * 10 + c - 48, c = getchar();
    return r * f;
}
int n, Q, sz; ll a[N], maU[M][N]; int maD[M][N];// U 后缀表示分子，D 后缀表示分母
ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }
int l, r, x[N], D; ll y[N], U;// U/D 是答案
void init() { l = 1, r = 0, U = -1e9, D = 1; }// 清空凸包
void upd(ll u, int d) {// 更新答案
    if(d < 0) u = -u, d = -d;
    if(u * D > U * d) U = u, D = d;
}
void ins(int X, ll Y) {// X 递增时维护的是下凸包，递减时维护的是上凸包
    while(l < r && (y[r] - y[r-1]) * (X - x[r]) >= (Y - y[r]) * (x[r] - x[r-1])) r--;
    x[++r] = X, y[r] = Y;
}
void qry(int X, ll Y) {// 过该点作切线并更新答案
    if(l > r) return;
    while(l < r && (Y - y[l]) * (X - x[l+1]) <= (Y - y[l+1]) * (X - x[l])) l++;
    upd(Y - y[l], X - x[l]);
}
int main() {
    cin >> n >> Q, sz = sqrt(n * 7);
    For(i, 0, n) a[i] = read() + a[i-1];
    per(i, (n - 1) / sz, 0) { // 为了压空间，pre, suf 数组合并成 ma 数组 
        init(); For(j, i * sz, n) qry(j, a[j]), maU[i][j] = U, maD[i][j] = D, ins(j, a[j-1]);
        init(); per(j, i * sz, 0) qry(j, a[j-1]), maU[i][j] = U, maD[i][j] = D, ins(j, a[j]);
    }
    int L, R;
    while(Q--) {
        L = read() - 1, R = read() - 1, init();
        if(L / sz ^ R / sz) {
            For(i, L, L / sz * sz + sz) qry(i, a[i]), ins(i, a[i-1]);
            rep(i, R / sz * sz, R) qry(i, a[i]), ins(i, a[i-1]);
            upd(maU[(L+sz-1)/sz][R], maD[(L+sz-1)/sz][R]);
            upd(maU[R/sz][L], maD[R/sz][L]);
        } else rep(i, L, R) qry(i, a[i]), ins(i, a[i-1]);
        ll d = gcd(llabs(U), D);
        printf("%lld/%lld\n", U / d, D / d);
    }
    return 0;
}
```