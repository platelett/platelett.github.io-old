---
title: 一道题3
date: 2021-04-12 19:59:27
updated: 2021-04-12 19:59:27
tags: [动态规划,概率,数论]
categories: 考试
---
> $n$ 个数 $p_i$ 构成一个随机排列。
>
> 当你手上的数是 $p_i$ 时，你并不知道 $p_i$ 为多少，但你会获知是否 $p_i = \min\{p_1,p_2,\cdots,p_i\}$，然后做出以下两种行为之一：
>
> - 如果 $i < n$，用 $p_{i+1}$ 换掉 $p_i$。
> - 带走 $p_i$，并结束。
>
> 现在你手上的数是 $p_1$，你想带走 $1$，求在最优决策下你带走 $1$ 的概率。
>
> 共 $T$ 组数据。
>
> $n \le 10^6,T \le 10^5$

设 $f_i$ 表示你手上的数是 $p_i$ 时，在最优决策下你带走 $1$ 的概率。

有 $\frac 1i$ 的概率你获知了 $p_i = \min\{p_1,p_2,\cdots,p_i\}$，此时一定采取两种行为中带走 $1$ 的概率较大的一种。

采取第一种行为时带走 $1$ 的概率为 $f_{i+1}$，采取第二种行为时带走 $1$ 的概率为 $\frac in$。

否则 $p_i \ne \min\{p_1,p_2,\cdots,p_i\}$，因为 $p_i$ 一定不为 $1$，所以一定会采取​第一种行为。

即
$$
f_i = \dfrac {\max\{f_{i+1},\frac in\} + (i-1)f_{i+1}}i
$$
初始化 $f_{n+1} = 0$，答案即为 $f_1$。

至此得到一个 $O(Tn)$ 的做法。

注意到 $f_i$ 是单调不增的，而 $\frac in$ 是单调递增的，并且 $f_1 > \frac 1n,f_{n+1} < \frac nn$。

不难证明存在一个 $d$，满足
$$
\begin{cases}
f_i \ge \frac in &(i \le d)\\\\
f_i < \frac in &(i > d)
\end{cases}
$$
故
$$
f_i = \begin{cases}
f_{i+1} &(i < d)\\\\
\dfrac {\frac in+(i-1)f_{i+1}}i &(i \ge d)
\end{cases}
$$
考虑当 $i \ge d$ 时
$$
f_i=\frac {i-1}if_{i+1}+\frac 1n\\\\
\frac 1{i-1}f_i=\frac 1if_{i+1}+\frac 1{n(i-1)}
$$
设 $g_i = \frac 1{i-1}f_i$，则
$$
\begin{aligned}
&g_i=g_{i+1}+\frac 1{n(i-1)}\\\\
&=\sum_{j=i-1}^n\frac 1{nj}\\\\
&=\frac 1n\sum_{j=i-1}^n\frac 1j
\end{aligned}
\Rightarrow
f_i=\frac {i-1}n\sum_{j=i-1}^n\frac 1j
$$
预处理调和数列前缀和后，对于每个 $n$，可以二分出 $d$，答案即为
$$
\frac {d-1}n\sum_{i=d-1}^n\frac 1i
$$
复杂度 $O(n_{\max}+T)$。

代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back

using namespace std;

const int N = 1e6 + 5;
typedef long long ll;
const ll P = 998244353;
const double e = exp(1);
int T, n; ll inv[N], s[N];
double S[N];
int main() {
    inv[0] = inv[1] = 1;
    rep(i, 2, 1000000) inv[i] = P / i * -inv[P % i] % P;
    rep(i, 1, 1000000) S[i] = S[i - 1] + 1. / i, (s[i] += s[i - 1] + inv[i]) %= P;
    for(cin >> T; T--;) {
        scanf("%d", &n);
        if(n == 1) puts("1");
        else {
            int k = upper_bound(S + 1, S + n + 1, S[n - 1] - 1) - S;
            printf("%lld\n", ((s[n - 1] - s[k - 1]) * k % P * inv[n] % P + P) % P);
        }
    }
    return 0;
}
```