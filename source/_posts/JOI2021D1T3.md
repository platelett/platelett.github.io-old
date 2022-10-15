---
title: Food Court | JOISC2021 D1T3
date: 2021-03-22 20:16:07
updated: 2021-03-22 20:16:07
tags: [数据结构]
categories: JOISC
---
> 有 $N$ 个商店，在每个商店的前面，都有顾客排队，顾客分 $M$ 类。
>
> 一共发生了 $Q$ 个事件，每个事件是以下之一。
>
> - 在第 $L$ 到 $R$ 的每个商店前的队列末尾加入 $K$ 个 $C$ 类顾客。
> - 对于第 $L$ 到 $R$ 的每个商店，如果队列中的人数不少于 $K$，则队列最前面的 $K$ 个顾客离开，否则队列中的顾客将全部离开。
> - 询问第 $A$ 个商店前的队列中从前往后数的第 $B$ 个人属于哪一类，不存在输出 $0$
>
> $N, M, Q \le 2.5 \cdot 10^5$

### 先考虑第 $A$ 个商店前的队列中的人数怎么求。

对于第一种操作，就是区间加上一个正数；对于第二种操作，就是区间减去一个正数，再和 $0$ 取 ```max```。

简化一下就是：区间加上一个整数，再和 $0$ 取 ```max```。

对于一个位置，假设它依次进行操作的数分别为 $a_1,a_2,\cdots,a_k$，那么此时它的值为
$$
\max\{0,\max\{0,\max\{0,\cdots\max\{0,a_1\}\cdots+a_{k-1}\}+a_k\}\\\\
=\max\{0,a_k,a_{k-1}+a_k,a_{k-2}+a_{k-1}+a_k,\cdots,a_1+a_2+\cdots+a_k\}\\\\
=\sum_{i=1}^ka_i-\min\{0,a_1,a_1+a_2,\cdots,a_1+a_2+\cdots+a_k\}
$$
即在不考虑再和 $0$ 取 ```max``` 的情况下，当前值减**历史**最小值。

做法是建立线段树，对每个结点维护加的懒标记 $tag_i$ 和**仅考虑包含此区间的操作**时的历史最小值 $minTag_i$。

区间加 $b$ 时就 ```tag[i] += b, minTag[i] = min(minTag[i], tag[i])```。

下放懒标记时就 ```tag[son] += tag[fa], minTag[son] = min(minTag[son], tag[son] + minTag[fa])```。

### 再考虑原题

假设第 $A$ 个商店当前有 $K$ 个人，这个商店总共来过 $S$ 个人，则当前第 $B$ 个人就是所有来过的人中第 $B+S-K$ 个人。

$S$ 很好算，而 $K$ 用上述方法即可求得。问题是转化为求所有来过的人中第 $K$ 个人。

离线下所有第一种操作，依次扫描第 $1$ 个到第 $n$ 个商店，维护一个时间轴线段树，每个结点的值这段时间内来过多少人。

当扫到一个区间的左端点 $L$ 时在对应的时间 $t$ 上加人数 $B$。扫**过**一个区间的右端点 $R$ （即扫到 $R+1$）时在对应的时间 $t$ 上减人数 $B$。

询问就是查询最小的 $T$ 满足 $[1,T]$ 内加入的人数不少于要求的 $K$ ，答案是第 $T$ 个操作对于的 $C$，线段树上二分一下即可。

时间复杂度 $O(n\log n)$。

代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define mid ((l + r) / 2)
#define lch l, mid, o * 2
#define rch mid + 1, r, o * 2 + 1

using namespace std;
const int N = 2.5e5 + 5;
typedef long long ll;
int n, m, q;
ll c[N];
void add(int i, int v) { for(; i <= n; i += i & -i) c[i] += v; }
ll qry(int i, ll r = 0) { for(; i; i &= i - 1) r += c[i]; return r; }
namespace seg1 {
    ll ta[N * 4], mi[N * 4];
    void put(int o, ll k, ll kk) {
        mi[o] = min(mi[o], ta[o] + kk), ta[o] += k;
    }
    void pd(int o) {
        put(o * 2, ta[o], mi[o]);
        put(o * 2 + 1, ta[o], mi[o]);
        ta[o] = mi[o] = 0;
    }
    void upd(int L, int R, int k, int l, int r, int o) {
        if(L <= l && r <= R) return put(o, k, k);
        pd(o);
        if(L <= mid) upd(L, R, k, lch);
        if(R > mid) upd(L, R, k, rch);
    }
    ll qry(int p, int l, int r, int o) {
        if(l == r) return ta[o] - mi[o];
        pd(o);
        return p <= mid ? qry(p, lch) : qry(p, rch);
    }
};
namespace seg2 {
    ll c[N * 4];
    void upd(int p, int v, int l, int r, int o) {
        c[o] += v; if(p > mid) upd(p, v, rch); else if(l < r) upd(p, v, lch);
    }
    ll qry(ll k, int l, int r, int o) {
        return l < r ? k <= c[o * 2] ? qry(k, lch) : qry(k - c[o * 2], rch) : l;
    }
};
vector <pair <int, int>> Add[N];
vector <pair <ll, int>> Qry[N];
int id[N], ans[N];
int main() {
    cin >> n >> m >> q;
    int op, l, r, a, b; ll c;
    rep(i, 1, q) {
        scanf("%d", &op);
        if(op == 1) {
            scanf("%d%d%d%d", &l, &r, &id[i], &b);
            seg1::upd(l, r, b, 1, n, 1);
            Add[l].push_back({ b, i }), add(l, b);
            Add[r + 1].push_back({ -b, i }), add(r + 1, -b);
        }
        if(op == 2) {
            scanf("%d%d%d", &l, &r, &b);
            seg1::upd(l, r, -b, 1, n, 1);
        }
        if(op == 3) {
            scanf("%d%lld", &a, &c);
            ll K = seg1::qry(a, 1, n, 1);
            if(K < c) ans[i] = 0;
            else Qry[a].push_back({ c + qry(a) - K, i });
        } else ans[i] = -1;
    }
    rep(i, 1, n) {
        for(auto [v, t] : Add[i]) seg2::upd(t, v, 1, q, 1);
        for(auto [k, t] : Qry[i]) ans[t] = id[seg2::qry(k, 1, q, 1)];
    }
    rep(i, 1, q) if(~ans[i]) printf("%d\n", ans[i]);
    return 0;
}
```

