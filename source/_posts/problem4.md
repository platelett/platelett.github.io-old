---
title: 一道题 4
date: 2021-09-04 16:01:14
updated: 2021-09-04 16:01:14
tags: []
categories: 考试
---
> 有 $n$ 个数，每个数都是不超过 $m$ 的质数，问有多少种情况它们的异或和等于 $0$。
>
> 答案对给定的模数 $p$ 取模。
>
> $n \le 10^9, m \le 3 \cdot 10^6$，保证 $p$ 是奇数。

定义一个位向量 $A$，满足
$$
A_i =
\begin{cases}
1 &(i \le m \land i \in \text{Prime})\\\\
0 &(otherwise)
\end{cases}
$$
定义乘法为异或卷积，那么 $(A^n)_0$ 就是答案。

用 ```FWT``` + 快速幂可以做到 $O(m \log m\log n)$。

但如果先 ```FWT```，再把每个位置上的值变成它的 $n$ 次方，最后 ```IFWT```，复杂度 $O(m(\log n + \log m))$。

但还不足以通过此题，因为模数是变量，快速幂极慢，调用 $O(m)$ 次会 ```TLE```。

考虑 ```FWT``` 的定义：
$$
FWT(A)_i = \sum_{2\ |\ pop(j \& i)}A_j - \sum_{2\ \not|\ pop(j \& i)}A_j
$$
可以知道 $|FWT(A)_i| \le \sum_iA_i$，以及 $FWT(A)_0 = \sum_iA_i$。

$\sum_iA_i$ 即 $m$ 以内质数个数，是 $O(\frac m {\log m})$ 级别的，可以预处理这 $O(\frac m {\log m})$ 个值的 $n$ 次方。

这样就只需要做 $O(\frac m {\log m})$ 次快速幂。

代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back

using namespace std;

typedef long long ll;
const int N = 1 << 22;

int n, m; ll P;
int pid, f[N], g[N], prm[216900], b, lim = 1;

ll Pow(ll a, int n, ll r = 1) {
    for(; n; n /= 2, a = a * a % P)
    if(n & 1) r = r * a % P;
    return r;
}
void FWT(int a[]) {
    For(i, 0, b) For(S, 0, lim) if(S >> i & 1) {
        int& x = a[S ^ 1 << i], y = a[S];
        a[S] = x - y, x += y;
    }
}

int main() {
    cin >> n >> m >> P;
    if(P == 1) puts("0"), exit(0);
    while(lim <= m) lim *= 2, b++;
    rep(i, 2, m) {
        if(!f[i]) prm[++pid] = i;
        for(int j = 1; i * prm[j] <= m; j++) {
            f[i * prm[j]] = 1;
            if(i % prm[j] == 0) break;
        }
    }
    mem(f, 0);
    rep(i, 1, pid) f[prm[i]]++, g[i] = Pow(i, n);
    FWT(f);
    int as = 0;
    For(S, 0, lim) (as += (f[S] < 0 && n % 2 ? -1 : 1) * g[abs(f[S])]) %= P;
    cout << (as + P) * Pow(P / 2 + 1, b) % P;
    return 0;
}
```
