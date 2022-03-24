---
title: 杜教筛 & Min-25 筛 & Powerful Number 筛
date: 2021-03-13 23:13:29
updated: 2021-03-13 23:13:29
tags: [知识总结,数论,杜教筛,Min-25 筛,Powerful Number 筛]
categories: 算法
---
> 给定积性函数 $f(x)$，求 $\sum\limits_{i=1}^Nf(i)$。

### 杜教筛

找到积性函数 $g(x)$，设 $S(x) = \sum\limits_{i=1}^x f(i)$，那么答案是 $S(N)$。

**核心**：
$$
g(1)S(n) = \sum\limits_{i=1}^n (f*g)(i) - \sum\limits_{i=2}^n g(i)S(\big\lfloor \dfrac n i \big\rfloor)
$$
如果 $g(x)$ 和 $(f*g)(x)$ 的前缀和很好求，就可以代这个公式递归。

**需要记忆化**。

先欧拉筛出 $S(1) - S(N^{\frac 2 3})$，对于其他会递归到的 $S(x)$，$x$ 只可能是 $\big\lfloor\dfrac N i \big \rfloor$，故可以用 $F_i$ 存 $S(x)$。

时空复杂度均为 $O(n^{\frac 2 3})$。

### Min-25 筛

$p_i$ 表示第 $i$ 个质数，$d_i$ 表示 $i$ 的最小质因子。

$S(n, i) = \sum\limits_{j = 2}^n [d_j > p_i]f(j)$

那么答案是 $S(N, 0)$。

有递推 $S(n, i) = \sum\limits_{j = i+1}^{p_j \le n} f(p_j) + \sum\limits_{j = i+1}^{p_j^2 \le n}\sum\limits_{k=1}^{p_j^k \le n} f(p_j^k)(S(\big\lfloor \dfrac n{p_j^k} \big\rfloor, j) + [k>1])$。

如果 $f(p^k)$ 和第一项很好求，那就可以递归到 $0$，而且**无需记忆化**。

所以问题是求第一项。

找一个完全积性函数 $h(x)$ 满足 $h(p) = f(p)$。

设 $g(n, i) = \sum\limits_{j=2}^n [j \in P \lor d_j > p_i]h(j)$

有递推 $g(n, i) = g(n, i - 1) - h(p_i)(g(\big\lfloor\dfrac n {p_i} \big\rfloor, i - 1) - \sum\limits_{j=1}^{i-1}h(p_j))$。

减后面那项是因为 $p_ip_j(j < i)$ 的最小质因子不是 $p_i$。

当 $n < p_i^2$ 时，$g(n,i)=g(n,i-1)$，即不需要转移。

注意到 $n$ 会取到的值一定是 $\big\lfloor\dfrac N i \big \rfloor$，且 $j \le \sqrt N$。

初始化 $g(n, 0) = \sum\limits_{i=2}^n h(i)$，**注意 $1$ 不算**。

然后从小到大枚举 $j$，开滚动数组存另一维，所以另一维需从大到小枚举计算。

> 求 1 - n 的质数数量。
>
> $n \le 10^{11}$

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)

using namespace std;
constexpr int N = sqrt(1e11) + 5;
typedef long long ll;
ll n, f1[N], f2[N];
int main() {
    cin >> n;
    int lim = sqrt(n);
    rep(i, 1, lim) f1[i] = i - 1, f2[i] = n / i - 1;

    rep(p, 2, lim) if (f1[p] ^ f1[p - 1]) {
        int w1 = lim / p;
        ll x = f1[p - 1], w3 = (ll)p * p, w2 = min((ll)lim, n / w3), d = n / p;
        rep(i, 1, w1) f2[i] -= f2[i * p] - x;
        rep(i, w1 + 1, w2) f2[i] -= f1[d / i] - x;
        per(i, lim, w3) f1[i] -= f1[i / p] - x;
    }

    cout << f2[1];
    return 0;
}
```

#### 常数优化

由于 ```double``` 乘法快于 ```long long``` 除法，可以预处理 $1-\sqrt n$ 的倒数。

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)

using namespace std;
constexpr int N = sqrt(1e11) + 5;
const double Eps = 1e-7;
typedef long long ll;
ll n, f1[N], f2[N];
double inv[N];
int main() {
    cin >> n;
    int lim = sqrt(n);
    rep(i, 1, lim) f1[i] = i - 1, f2[i] = n / i - 1, inv[i] = 1.0 / i;

    rep(p, 2, lim) if (f1[p] ^ f1[p - 1]) {
        int w1 = lim / p;
        ll x = f1[p - 1], w3 = (ll)p * p, w2 = min((ll)lim, n / w3), d = n / p;
        rep(i, 1, w1) f2[i] -= f2[i * p] - x;
        rep(i, w1 + 1, w2) f2[i] -= f1[int(d * inv[i] + Eps)] - x;
        per(i, lim, w3) f1[i] -= f1[int(i * inv[p] + Eps)] - x;
    }

    cout << f2[1];
    return 0;
}
```

### Powerful Number 筛

定义：所有质因子的指数都大于 $1$ 的数叫 ```Powerful Number```。

性质：可以表示为 $a^2b^3$。

> 引理：$n$ 以内的 ```Powerful Number``` 的数量为 $O(\sqrt n)$ 级别。
>
> 证明：根据性质，枚举 $a$，可以算出数量的上界：
> $$
> \sum_{i=1}^{\sqrt n}\sqrt[3] {\frac n{a^2}} < \int_0^{\sqrt n - 1}\sqrt[3] {\frac n{a^2}}da = 3n^\frac 13(\sqrt n - 1)^\frac 13 < 3\sqrt n
> $$

一个想法：如果一个积性函数仅在 ```Powerful Number``` 处取值不为 $0$，那么与之相关的计算可能做到较低的复杂度。

找到一个**积性函数** $g(x)$ 满足 $g(x)=f(x) (x \in Prime)$，设 $f(x)=\sum\limits_{ab=x}g(a)h(b)$，可知 $h(x)$ 仅在 ```Powerful Number``` 处取值不为 $0$。

首先预处理出 $h(x)$ 在 $p^k(p \in Prime,k > 1)$ 处的取值，这里可以做到 $O(\sqrt N)$（无论 $O(1)$ 算 $h(p^k)$ 还是 $O(k)$ 算都是这个复杂度）。

于是
$$
\sum_{i=1}^Nf(i)=\sum_{ab \le N}g(a)h(b)=\sum_{b=1}^Nh(b)\sum_{a=1}^{\lfloor \frac Na \rfloor}g(a)
$$
最后要求 $g(x)$ 在 $\lfloor \frac Na \rfloor$ 的前缀和能快速求。