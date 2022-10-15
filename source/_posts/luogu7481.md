---
title: 梦现时刻 | 洛谷 P7481
date: 2021-04-12 15:25:13
updated: 2021-04-12 15:25:13
tags: [数论,生成函数,递推]
categories: 洛谷月赛
---
> [题目链接](https://www.luogu.com.cn/problem/P7481)
>
> 给定 $n,m$，保证 $m \le n$，设 $f(a,b)=\sum\limits_{i=0}^b\binom bi\binom {n-i}a$。
>
> 求 $\bigoplus\limits_{a=1}^m\bigoplus\limits_{b=1}^m\left(f(a,b) \bmod 998244353\right)$。
>
> $n \le 10^9,m \le 5000$

### 引理

$$
\sum_{a=0}^n f(a,b)x^a=(x+2)^b(x+1)^{n-b}
$$

#### 证明一：

$$
\begin{aligned}
&\sum_{a=0}^n f(a,b)x^a\\\\
&=\sum_{a=0}^n\sum\limits_{i=0}^b\binom bi\binom {n-i}ax^a\\\\
&=\sum_{i=0}^b\binom bi\sum_{a=0}^n\binom {n-i}ax^a\\\\
&=\sum_{i=0}^b\binom bi(x+1)^{n-i}\\\\
&=\sum_{i=0}^b\binom bi(x+1)^{b-i}(x+1)^{n-b}\\\\
&=(x+2)^b(x+1)^{n-b}
\end{aligned}
$$

#### 证明二

考虑 $f(a,b)$ 的组合意义：有 $n$ 个人，其中 $b$ 个人比较强，要选出两批人：

- 先从比较强的 $b$ 个人中选出 $i$ 个人作为第一批。
- 再从剩下的 $n-i$ 个人中选出 $a$ 个人作为第二批。

因为要对 $i \in [0,b]$ 求和，所以 $f(a,b)$ 表示第一批的人数任意，第二批的人数为 $a$ 的方案数。

换一个角度看这个方案数：设第二批的 $a$ 个人中有 $x$ 个人比较强，$y$ 个人不强（$a=x+y$）。

- 对于一个比较强的人，如果他入选第二批，方案数为 $1$，否则他既可能入选第一批，也可能落选，方案数为 $2$。

  因此比较强的 $b$ 个人入选的情况数的生成函数为 $(x+2)^b$。

- 对于一个不强的人，如果他入选第二批，方案数为 $1$，否则他落选了，方案数为 $1$。

  因此不强的 $n-b$ 个人入选的情况数的生成函数为 $(x+1)^{n-b}$。

综上，得到 $\sum\limits_{a=0}^n f(a,b)x^a=(x+2)^b(x+1)^{n-b}$。

### 两种解法

#### 类 01 背包 ```DP```

根据引理，有
$$
\sum_{a=0}^n f(a,b)x^a=\frac {x+2}{x+1}\sum_{a=0}^nf(a,b-1)x^a
$$
乘 $x+2$ 相当于添加一个物品，除以 $x+1$ 相当于删除一个物品，后者用可撤销背包解决。

复杂度 $O(m^2)$。

#### 递推

 考虑
$$
\begin{aligned}
&[(x+2)^b(x+1)^{n-b}]'\\\\
&=b(x+2)^{b-1}(x+1)^{n-b}+(x+2)^b(n-b)(x+1)^{n-b-1}\\\\
&=\dfrac{b(x+1)+(n-b)(x+2)}{(x+2)(x+1)}(x+2)^b(x+1)^{n-b}\\\\
&=\dfrac{nx+2n-b}{(x+2)(x+1)}(x+2)^b(x+1)^{n-b}\\\\
\end{aligned}
$$
两边同乘 $(x+2)(x+1)$ 得：
$$
(x+2)(x+1)[(x+2)^b(x+1)^{n-b}]'=(nx+2n-b)(x+2)^b(x+1)^{n-b}
$$
提取 $x^a$ 系数：
$$
\begin{aligned}
&(a-1)f(a-1,b)+3af(a,b)+2(a+1)f(a+1,b)\\\\
&=nf(a-1,b)+(2n-b)f(a,b)
\end{aligned}
$$
整理得：
$$
f(a+1,b)=\frac {(2n-b-3a)f(a,b)+(n-a+1)f(a-1,b)}{2(a+1)}
$$
进一步：
$$
f(a,b)=\frac {(2n-b-3a+3)f(a-1,b)+(n-a+2)f(a-2,b)}{2a}
$$
对于 $b \in [1,m]$ 都递推一遍即可。

复杂度 $O(m^2)$。

代码（类 01 背包 ```DP```）：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back

using namespace std;
const int P = 998244353;

int n, m;
long long inv[5005], f[5005];

int main() {
    cin >> n >> m;
    f[0] = inv[1] = 1;
    rep(i, 2, m) inv[i] = (P - P / i) * inv[P % i] % P;
    rep(i, 1, m) f[i] = f[i - 1] * (n - i + 1) % P * inv[i] % P;
    int as = 0;
    rep(i, 1, m) {
        per(j, m, 0) f[j] = (f[j] * 2 + f[j - 1]) % P;
        rep(j, 1, m) as ^= f[j] = (f[j] + P - f[j - 1]) % P;
    }
    cout << as;
    return 0;
}
```

代码（递推）：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back

using namespace std;
const int P = 998244353;

int n, m;
long long inv[10005], f[5005];

int main() {
    cin >> n >> m;
    inv[1] = 1;
    rep(i, 2, m * 2) inv[i] = (P - P / i) * inv[P % i] % P;
    int as = 0;
    f[0] = 1;
    rep(i, 1, m) {
        (f[0] *= 2) %= P;
        rep(j, 1, m) as ^= f[j] = (((n * 2 - i - j * 3 + 3) * f[j - 1] + (n - j + 2) * f[j - 2]) % P * inv[j * 2] % P + P) % P;
    }
    cout << as;
    return 0;
}
```