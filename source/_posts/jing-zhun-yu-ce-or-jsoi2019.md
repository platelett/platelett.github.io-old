---
title: 精准预测 | JSOI2019
date: 2021-09-04 15:58:47
updated: 2021-09-04 15:58:47
tags: []
categories: 省选
---
>[题目链接](https://loj.ac/p/3101)
>
>有 $n$ 个人和 $m$ 个限制，以及 $T +1$ 个时刻。限制分两种：
>
>- 如果 $x$ 在 $t$ 时刻是死亡状态，则 $y$ 在 $t+1$ 时刻是死亡状态。
>- 如果 $x$ 在 $t$ 时刻是生存状态，则 $y$ 在 $t$ 时刻是死亡状态。
>
>对于每个人 $i$，求出其他人中有多少个人 $j$ 满足 **$i$ 与 $j$ 能够同时存活到最后**。
>
>$n \le 5 \cdot 10^4,m \le 10^5,t \le T \le 10^6$

对于每个人每个时刻，建生死两个点。

对于第一种限制，连有向边 $Dead(x,t) \rightarrow Dead(y,t+1),Alive(y,t+1) \rightarrow Alive(x,t)$。

对于第二种限制，连有向边 $Alive(x,t) \rightarrow Dead(y,t),Alive(y,t) \rightarrow Dead(x,t)$。

另外，由于每个人不能复活，连有向边 $Dead(x,t) \rightarrow Dead(x,t+1),Alive(x,t+1) \rightarrow Alive(x,t)$。

首先有一些 $x$ 满足 $Alive(x,T+1)$ 能到 $Dead(x, T+1)$，删除 $Alive(x,T+1)$ 和 $Dead(x, T+1)$。

如果从 $Dead(x,T+1)$ 不能到达 $Dead(y,T+1)$，则 $y$ 对 $x$ 造成贡献。

建图时其实只有所有的 $(x,t)$ 和所有 $(x,T+1)$ 有用，$(x,t)$ 向 $(y,t')$ 连边，其中 $(y,t')$ 是横坐标为 $y$ 的点中第一个纵坐标大于（等于）的点，取决于是哪种限制，然后 $Dead(x,t)$ 向后继连边，$Alive(x,t)$ 向前驱连边。

求 $Dead(x,T+1)$ 能到达多少 $Dead(y,T+1)$ 可以用 ```bitset``` 优化 ```DP```。

但空间开不下，```bitset``` 只能开 $10000$，因此需要每 $10000$ 个点做一次 ```DP```。

代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)

using namespace std;
const int N = 1e5 + 5;
int T, n, K, c[N], t[N], x[N], y[N];
map<int, int> m[N];
vector<int> G[N * 3];
int ed[N], vis[N * 3], as[N];
bitset<10000> ban, s[N * 3];
void add(int c, int u, int v) {
    G[u + c].push_back(v), G[v + 1].push_back(u + !c);
}
void dfs(int u) {
    if(vis[u]) return;
    vis[u] = 1;
    for(int v : G[u]) dfs(v), s[u] |= s[v];
}
int main() {
    cin >> T >> n >> K;
    int p = 0;
    rep(i, 1, K) scanf("%d%d%d%d", &c[i], &t[i], &x[i], &y[i]), m[x[i]][t[i]] = p += 2;
    rep(i, 1, n) ed[i] = m[i][T + 1] = p += 2;
    rep(i, 1, K) add(c[i], m[x[i]][t[i]], m[y[i]].lower_bound(t[i] + !c[i]) -> second);
    rep(i, 1, n) {
        int lst = 0;
        for(auto [t, j] : m[i]) { if(lst) add(0, lst, j); lst = j; }
    }
    for(int l = 1, r; l <= n; l += 10000) {
        r = min(l + 9999, n), ban.reset();
        rep(i, 2, p + 1) s[i].reset(), vis[i] = 0;
        rep(i, l, r) s[ed[i]][i - l] = 1;
        rep(i, 2, p + 1) dfs(i);
        rep(i, l, r) if(s[ed[i] + 1][i - l]) ban[i - l] = 1, as[i] = -1e9;
        rep(i, 1, n) as[i] += r - l + 1 - (s[ed[i] + 1] | ban).count();
    }
    rep(i, 1, n) printf("%d ", max(as[i] - 1, 0));
    return 0;
}
```