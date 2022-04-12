---
title: 模板（持续更新）
date: 2021-04-16 16:45:41
tags: [模板,持续更新]
hidden: true
---
**左手栏有目录。**

出现的宏：

```cpp
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back

using ll = long long;
using lf = double;
```

# 有用的模板

## Fast IO

#### 非负整数版

##### 函数

```cpp
int read() {
    const int M = 1e6;
    static streambuf* in = cin.rdbuf();
    #define gc (p1 == p2 && (p2 = (p1 = buf) + in -> sgetn(buf, M), p1 == p2) ? -1 : *p1++)
    static char buf[M], *p1, *p2;
    int c = gc, r = 0;
    while(c < 48) c = gc;
    while(c > 47) r = r * 10 + (c & 15), c = gc;
    return r;
}
void wrt(int x) {
    static streambuf* out = cout.rdbuf();
    #define pc out -> sputc
    static char c[11]; int sz = 0;
    do c[++sz] = x % 10, x /= 10; while(x);
    while(sz) pc(c[sz--] + 48);
    pc(10);
}
```

##### 预处理

```cpp
cin.tie()->sync_with_stdio(0);
```

#### 整数版

```cpp
int read() {
    const int M = 1e6;
    static streambuf* in = cin.rdbuf();
    #define gc (p1 == p2 && (p2 = (p1 = buf) + in -> sgetn(buf, M), p1 == p2) ? -1 : *p1++)
    static char buf[M], *p1, *p2;
    int c = gc, r = 0, f = 1;
    for(; c < 48; c = gc) if(c == 45) f = -1;
    while(c > 47) r = r * 10 + (c & 15), c = gc;
    return r * f;
}
void wrt(int x) {
    static streambuf* out = cout.rdbuf();
    #define pc out -> sputc
    static char c[11]; int sz = 0;
    if(x < 0) pc(45), x = -x;
    do c[++sz] = x % 10, x /= 10; while(x);
    while(sz) pc(c[sz--] + 48);
    pc(10);
}
```

**注：输出 ```long long``` 时 ```wrt``` 函数中的 ```c``` 数组大小要开到 $20$。**

## AtCoder-modint

```cpp
#include <bits/stdc++.h>

using namespace std;
using ll = long long;
using uint = unsigned;
using ull = unsigned ll;

constexpr ll safe_mod(ll x, ll m) {
    return x %= m, x < 0 ? x + m : x;
}
constexpr ll pow_mod_constexpr(ll x, ll n, int m) {
    if(m == 1) return 0;
    uint _m = m;
    ull r = 1, _x = safe_mod(x, m);
    for(; n; n >>= 1, _x = _x * _x % _m)
    if(n & 1) r = r * _x % m;
    return r;
}
constexpr bool is_prime_constexpr(int n) {
    if(n <= 1) return false;
    if(n == 2 || n == 7 || n == 61) return true;
    if(n % 2 == 0) return false;
    ll d = n - 1;
    while(~d & 1) d >>= 1;
    for(ll a : {2, 7, 61}) {
        ll t = d, y = pow_mod_constexpr(a, t, n);
        while(t != n - 1 && y != 1 && y != n - 1)
            y = y * y % n, t <<= 1;
        if(y != n - 1 && t % 2 == 0) return false;
    }
    return true;
}
constexpr pair<ll, ll> inv_gcd(ll a, ll b) {
    a = safe_mod(a, b);
    if(a == 0) return {b, 0};
    ll s = b, t = a, m0 = 0, m1 = 1;
    while(t) {
        ll u = s / t;
        s -= t * u, m0 -= m1 * u;
        ll tmp = s;
        s = t, t = tmp, tmp = m0, m0 = m1, m1 = tmp;
    }
    if(m0 < 0) m0 += b / s;
    return {s, m0};
}
struct barrett {
    uint m; ull im;
    barrett(uint m) :m(m), im(~0ull / m + 1) {}
    uint mul(uint a, uint b) const {
        ull z = (ull)a * b;
        ull x = (__uint128_t)z * im >> 64;
        uint v = z - x * m;
        return m <= v ? v + m : v;
    }
};
template<int m> struct static_modint {
    using mint = static_modint;
  public:
    static mint raw(int v) {
        mint x;
        return x._v = v, x;
    }
    static_modint() : _v(0) {}
    template<class T> static_modint(T v) {
        ll x = v % m;
        _v = x < 0 ? x + m : x;
    }
    uint val() const { return _v; }
    mint& operator++() {
        if(++_v == m) _v = 0;
        return *this;
    }
    mint& operator--() {
        if(!_v--) _v = m - 1;
        return *this;
    }
    mint operator++(int) {
        mint res = *this;
        ++*this;
        return res;
    }
    mint operator--(int) {
        mint res = *this;
        --*this;
        return res;
    }
    mint& operator+=(const mint& rhs) {
        _v += rhs._v;
        if(_v >= m) _v -= m;
        return *this;
    }
    mint& operator-=(const mint& rhs) {
        _v -= rhs._v;
        if(_v >= m) _v += m;
        return *this;
    }
    mint& operator*=(const mint& rhs) {
        ull z = _v;
        z *= rhs._v, _v = z % m;
        return *this;
    }
    mint& operator/=(const mint& rhs) { return *this = *this * rhs.inv(); }
    mint operator+() const { return *this; }
    mint operator-() const { return mint() - *this; }

    mint pow(ll n) const {
        assert(0 <= n);
        mint x = *this, r = 1;
        for(; n; n >>= 1, x *= x) if(n & 1) r *= x;
        return r;
    }
    mint inv() const {
        if(prime) {
            assert(_v);
            return pow(m - 2);
        } else {
            auto eg = inv_gcd(_v, m);
            assert(eg.first == 1);
            return eg.second;
        }
    }

    friend mint operator+(const mint& lhs, const mint& rhs) {
        return mint(lhs) += rhs;
    }
    friend mint operator-(const mint& lhs, const mint& rhs) {
        return mint(lhs) -= rhs;
    }
    friend mint operator*(const mint& lhs, const mint& rhs) {
        return mint(lhs) *= rhs;
    }
    friend mint operator/(const mint& lhs, const mint& rhs) {
        return mint(lhs) /= rhs;
    }
    friend bool operator==(const mint& lhs, const mint& rhs) {
        return lhs._v == rhs._v;
    }
    friend bool operator!=(const mint& lhs, const mint& rhs) {
        return lhs._v != rhs._v;
    }

  private:
    uint _v;
    static constexpr bool prime = is_prime_constexpr(m);
};

template<int id> struct dynamic_modint {
    using mint = dynamic_modint;

  public:
    static void set_mod(int m) {
        assert(1 <= m), bt = barrett(m);
    }
    static mint raw(int v) {
        mint x;
        return x._v = v, x;
    }

    dynamic_modint() : _v(0) {}
    template<class T> dynamic_modint(T v) {
        ll x = v % (int)bt.m;
        _v = x < 0 ? x + bt.m : x;
    }

    uint val() const { return _v; }

    mint& operator++() {
        if(++_v == bt.m) _v = 0;
        return *this;
    }
    mint& operator--() {
        if(!_v--) _v = bt.m - 1;
        return *this;
    }
    mint operator++(int) {
        mint res = *this;
        ++*this;
        return res;
    }
    mint operator--(int) {
        mint res = *this;
        --*this;
        return res;
    }

    mint& operator+=(const mint& rhs) {
        _v += rhs._v;
        if(_v >= bt.m) _v -= bt.m;
        return *this;
    }
    mint& operator-=(const mint& rhs) {
        _v += bt.m - rhs._v;
        if(_v >= bt.m) _v -= bt.m;
        return *this;
    }
    mint& operator*=(const mint& rhs) {
        _v = bt.mul(_v, rhs._v);
        return *this;
    }
    mint& operator/=(const mint& rhs) { return *this = *this * rhs.inv(); }

    mint operator+() const { return *this; }
    mint operator-() const { return mint() - *this; }

    mint pow(ll n) const {
        assert(0 <= n);
        mint x = *this, r = 1;
        for(; n; n >>= 1, x *= x) if(n & 1) r *= x;
        return r;
    }
    mint inv() const {
        auto eg = inv_gcd(_v, bt.m);
        assert(eg.first == 1);
        return eg.second;
    }

    friend mint operator+(const mint& lhs, const mint& rhs) {
        return mint(lhs) += rhs;
    }
    friend mint operator-(const mint& lhs, const mint& rhs) {
        return mint(lhs) -= rhs;
    }
    friend mint operator*(const mint& lhs, const mint& rhs) {
        return mint(lhs) *= rhs;
    }
    friend mint operator/(const mint& lhs, const mint& rhs) {
        return mint(lhs) /= rhs;
    }
    friend bool operator==(const mint& lhs, const mint& rhs) {
        return lhs._v == rhs._v;
    }
    friend bool operator!=(const mint& lhs, const mint& rhs) {
        return lhs._v != rhs._v;
    }

  private:
    uint _v;
    static barrett bt;
};
template<int id> barrett dynamic_modint<id>::bt = 998244353;
```

## 打比赛专用模板

```cpp
#include <bits/stdc++.h>

using namespace std;
using ll = long long;
using uint = unsigned;
using ull = uint64_t;

constexpr ll safe_mod(ll x, ll m) { return x %= m, x < 0 ? x + m : x; }
constexpr ll pow_mod_constexpr(ll x, ll n, int m) {
    if(m == 1) return 0;
    uint _m = m; ull r = 1, _x = safe_mod(x, m);
    for(; n; n >>= 1, _x = _x * _x % _m) if(n & 1) r = r * _x % m;
    return r;
}
constexpr bool is_prime_constexpr(int n) {
    if(n <= 1) return false;
    if(n == 2 || n == 7 || n == 61) return true;
    if(n % 2 == 0) return false;
    ll d = n - 1; while(~d & 1) d /= 2;
    for(ll a : {2, 7, 61}) {
        ll t = d, y = pow_mod_constexpr(a, t, n);
        while(t != n - 1 && y != 1 && y != n - 1) y = y * y % n, t <<= 1;
        if(y != n - 1 && t % 2 == 0) return false;
    }
    return true;
}
constexpr pair<ll, ll> inv_gcd(ll a, ll b) {
    a = safe_mod(a, b);
    if(a == 0) return {b, 0};
    ll s = b, t = a, m0 = 0, m1 = 1;
    while(t) {
        ll u = s / t; s -= t * u, m0 -= m1 * u;
        ll tmp = s; s = t, t = tmp, tmp = m0, m0 = m1, m1 = tmp;
    }
    if(m0 < 0) m0 += b / s;
    return {s, m0};
}
struct barrett {
    uint m; ull im;
    barrett(uint m) :m(m), im(~0ull / m + 1) {}
    uint mul(uint a, uint b) const {
        ull z = (ull)a * b, x = (__uint128_t)z * im >> 64; uint v = z - x * m;
        return m <= v ? v + m : v;
    }
};
template<int m> struct static_modint {
    using mint = static_modint;
  public:
    static mint raw(int v) { mint x; return x._v = v, x; }
    static_modint() :_v(0) {}
    template<class T> static_modint(T v) { ll x = v % m; _v = x < 0 ? x + m : x; }
    uint val()const { return _v; }
    mint& operator++() { if(++_v == m) _v = 0; return *this; }
    mint& operator--() { if(!_v--) _v = m - 1; return *this; }
    mint operator++(int) { mint res = *this; ++*this; return res; }
    mint operator--(int) { mint res = *this; --*this; return res; }
    mint& operator+=(const mint& rhs) { _v += rhs._v; if(_v >= m) _v -= m; return *this; }
    mint& operator-=(const mint& rhs) { _v -= rhs._v; if(_v >= m) _v += m; return *this; }
    mint& operator*=(const mint& rhs) { ull z = _v; z *= rhs._v, _v = z % m; return *this; }
    mint& operator/=(const mint& rhs) { return *this = *this * rhs.inv(); }
    mint operator+()const { return *this; }
    mint operator-()const { return mint() - *this; }
    mint pow(ll n)const { assert(0 <= n); mint x = *this, r = 1; for(; n; n >>= 1, x *= x) if(n & 1) r *= x; return r; }
    mint inv() const{ if(prime) { assert(_v); return pow(m - 2); } else { auto eg = inv_gcd(_v, m); assert(eg.first == 1); return eg.second; } }
    friend mint operator+(const mint& lhs, const mint& rhs) { return mint(lhs) += rhs; }
    friend mint operator-(const mint& lhs, const mint& rhs) { return mint(lhs) -= rhs; }
    friend mint operator*(const mint& lhs, const mint& rhs) { return mint(lhs) *= rhs; }
    friend mint operator/(const mint& lhs, const mint& rhs) { return mint(lhs) /= rhs; }
    friend bool operator==(const mint& lhs, const mint& rhs) { return lhs._v == rhs._v; }
    friend bool operator!=(const mint& lhs, const mint& rhs) { return lhs._v != rhs._v; }
  private:
    uint _v;
    static constexpr bool prime = is_prime_constexpr(m);
};
struct dynamic_modint {
    using mint = dynamic_modint;
  public:
    static void set_mod(int m) { assert(1 <= m), bt = barrett(m); }
    static mint raw(int v) { mint x; return x._v = v, x; }
    dynamic_modint() :_v(0) {}
    template<class T> dynamic_modint(T v) { ll x = v % (int)bt.m; _v = x < 0 ? x + bt.m : x; }
    uint val()const { return _v; }
    mint& operator++() { if(++_v == bt.m) _v = 0; return *this; }
    mint& operator--() { if(!_v--) _v = bt.m - 1; return *this; }
    mint operator++(int) { mint res = *this; ++*this; return res; }
    mint operator--(int) { mint res = *this; --*this; return res; }
    mint& operator+=(const mint& rhs) { _v += rhs._v; if(_v >= bt.m) _v -= bt.m; return *this; }
    mint& operator-=(const mint& rhs) { _v += bt.m - rhs._v; if(_v >= bt.m) _v -= bt.m; return *this; }
    mint& operator*=(const mint& rhs) { _v = bt.mul(_v, rhs._v); return *this; }
    mint& operator/=(const mint& rhs) { return *this = *this * rhs.inv(); }
    mint operator+()const { return *this; }
    mint operator-()const { return mint() - *this; }
    mint pow(ll n)const { assert(0 <= n); mint x = *this, r = 1; for(; n; n >>= 1, x *= x) if(n & 1) r *= x; return r; }
    mint inv()const { auto eg = inv_gcd(_v, bt.m); assert(eg.first == 1); return eg.second; }
    friend mint operator+(const mint& lhs, const mint& rhs) { return mint(lhs) += rhs; }
    friend mint operator-(const mint& lhs, const mint& rhs) { return mint(lhs) -= rhs; }
    friend mint operator*(const mint& lhs, const mint& rhs) { return mint(lhs) *= rhs; }
    friend mint operator/(const mint& lhs, const mint& rhs) { return mint(lhs) /= rhs; }
    friend bool operator==(const mint& lhs, const mint& rhs) { return lhs._v == rhs._v; }
    friend bool operator!=(const mint& lhs, const mint& rhs) { return lhs._v != rhs._v; }
  private:
    uint _v;
    static barrett bt;
};
barrett dynamic_modint::bt = 998244353;
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back
#define eb emplace_back
#define fi first 
#define se second
#define all(x) (x).begin(), (x).end()
#define SZ(x) int((x).size())
#define mid ((l + r) >> 1)
#define lc (o << 1)
#define rc (o << 1 | 1)
#define lch l, mid, lc
#define rch mid + 1, r, rc
#define lb lower_bound
#define ub upper_bound
#define bs binary_search

template<class T> T& cmin(T& a, const T& b) { if(b < a) a = b; return a; }
template<class T> T& cmax(T& a, const T& b) { if(a < b) a = b; return a; }

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

void solve() {
    
}
int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T;
    for(cin >> T; T--; solve());
}
```

## 随机数

### 函数

```cpp
using ull = unsigned ll;
ull gen(ull x) {
    const ull k = 0x9ddfea08eb382d69ULL;
    x *= k, x ^= x >> 47;
    x *= k, x ^= x >> 47;
    x *= k, x ^= x >> 47;
    return x * k;
}
int rnd() {
    static ull s = 2;
    return (s += gen(s)) & INT_MAX;
}
```

##  大模数取模

### 函数

```cpp
ll mul(ll a, ll b, ll p) {
    return (a * b - ll((long double)a / p * b + 0.5) * p + p) % p;
}
```

## `bash` 对拍

```bash
#!/bin/bash
while true; do
    ./gen > in
    ./a < in > 1
    ./b < in > 2
    diff 1 2
    if [ $? -ne 0 ] ; then break; fi
done
```

## 简易计算器

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)

using namespace std;
using ll = long long;

ll mul(ll a, ll b, ll p) {
    return (a * b - ll((long double)a / p * b + 0.5) * p + p) % p;
}
ll Pow(ll a, ll n, ll p, ll r = 1) {
    for(; n; n >>= 1, a = mul(a, a, p))
    if(n & 1) r = mul(r, a, p);
    return r;
}

namespace Pollard_Rho {
    int chk(ll n) {
        for(ll a : {2, 3, 7, 61, 24251}) {
            if(n == a) return 1;
            if(Pow(a, n - 1, n)  1) return 0;
            ll k = n - 1, t;
            while(~k & 1) k >>= 1;
            if((t = Pow(a, k, n)) == 1) continue;
            while(t != 1 && t != n - 1) t = mul(t, t, n);
            if(t != n - 1) return 0;
        }
        return 1;
    }
    ll f(ll x, ll c, ll p) { return (mul(x, x, p) + c) % p; }
    ll PR(ll n) {
        ll a = 0, b = 0, c = rand() % (n - 1) + 1, v = 1, g;
        for(int k = 1; ; k <<= 1, a = b, v = 1) {
            rep(i, 1, k) {
                b = f(b, c, n), v = mul(v, abs(a - b), n);
                if(!(i & 127) || i == k) {
                    g = __gcd(v, n);
                    if(g > 1) return g;
                }
            }
        }
    }
    ll ans[100]; int cnt;
    void solve(ll n) {
        if(chk(n)) return void(ans[++cnt] = n);
        ll d; do d = PR(n); while(d == n);
        solve(d), solve(n / d);
    }
    void work() {
        ll n;
        while(cin >> n) {
            cnt = 0;
            if(chk(n)) { puts("Prime"); continue; }
            solve(n), sort(ans + 1, ans + cnt + 1);
            int t = 0;
            rep(i, 1, cnt) {
                if(ans[i] != ans[i - 1]) cout << ans[i];
                t++; if(ans[i] != ans[i + 1]) {
                    if(t > 1) cout << '^' << t;
                    putchar(32), t = 0;
                }
            } putchar(10);
        }
    }
};
namespace Inv {
    void exgcd(ll a, ll b, ll& d, ll& x, ll& y) {
        if(b) exgcd(b, a % b, d, y, x), y -= a / b * x;
        else d = a, x = 1, y = 0;
    }
    void inv(ll a, ll p) {
        ll d, x, y; exgcd(a, p, d, x, y);
        if(d > 1) puts("Non-existent!");
        else cout << (x % p + p) % p;
    }
    void work() {
        ll a, p;
        while(cin >> a >> p) inv(a, p);
    }
};
namespace Prime {
    void work() {
        ll n;
        while(cin >> n) {
            while(!Pollard_Rho::chk(n)) n++;
            cout << n << endl;
        }
    }
};
namespace Cipolla {
    ll n, p, II;
    struct cmp {
        ll r, i;
        cmp operator *(const cmp& b) {
            return {(r * b.r + i * b.i % p * II) % p, (r * b.i + i * b.r) % p };
        }
    } U = { 1, 0 };
    int pow1(ll a, int n, ll r = 1) {
        for(; n; n >>= 1, a = a * a % p)
        if(n & 1) r = r * a % p;
        return r;
    }
    cmp pow2(cmp a, int n, cmp r = U) {
        for(; n; n >>= 1, a = a * a)
        if(n & 1) r = r * a;
        return r;
    }
    void work() {
        while(cin >> n >> p) {
            if(!n) { puts("0"); continue; }
            if(p == 2) { cout << n << endl; continue; }
            if(pow1(n, p / 2) != 1) { puts("Non-existent!"); continue; }
            ll a;
            do a = rand() % p, II = (a * a - n + p) % p; while(!a || pow1(II, p / 2) == 1);
            int x1 = pow2({a, 1}, p / 2 + 1).r, x2 = p - x1;
            if(x1 > x2) swap(x1, x2);
            printf("%d %d\n", x1, x2);
        }
    }
}; 
int main() {
    srand(time(0));
    puts("Press 1 for integer factorization.");
    puts("Press 2 to calculate the modular multiplicative inverse of a number.");
    puts("Press 3 to find the first prime number greater than or equal to a number.");
    puts("Press 4 to calculate the modular square root of a number.");
    int op;
    while(1) {
        cin >> op;
        if(op == 1) Pollard_Rho::work();
        else if(op == 2) Inv::work();
        else if(op == 3) Prime::work();
        else if(op == 4) Cipolla::work();
        else puts("Illegal input! Please re-enter your option.");
    }
}
```

# 数学

## NTT

### 普通版

#### 定义

```cpp
int lim = 1, rev[N];
```

#### 函数

```cpp
ll Pow(ll a, int n, ll r = 1) {
    for(; n; n >>= 1, a = a * a % P)
    if(n & 1) r = r * a % P;
    return r;
}
void NTT(ll a[], bool t) {
    For(i, 0, lim) if(rev[i] < i) swap(a[i], a[rev[i]]);
    for(int i = 1; i < lim; i <<= 1) {
        ll wn = Pow(3, P / 2 / i);
        for(int j = 0; j < lim; j += i << 1) {
            ll w = 1;
            For(k, j, j + i) {
                ll &x = a[k], y = a[k + i] * w % P;
                a[k + i] = (x - y) % P, (x += y) %= P, w = w * wn % P;
            }
        }
    }
    if(t) return;
    reverse(a + 1, a + lim);
    ll inv = Pow(lim, P - 2);
    For(i, 0, lim) (a[i] *= inv) %= P;
}
```

`modint` 版

```cpp
void NTT(mint a[], bool t) {
    For(i, 0, lim) if(rev[i] < i) swap(a[i], a[rev[i]]);
    for(int i = 1; i < lim; i <<= 1) {
        mint wn = mint(3).pow(P / 2 / i);
        for(int j = 0; j < lim; j += i << 1) {
            mint w = 1;
            For(k, j, j + i) {
                mint &x = a[k], y = a[k + i] * w;
                a[k + i] = x - y, x += y, w *= wn;
            }
        }
    }
    if(t) return;
    reverse(a + 1, a + lim);
    mint inv = mint(lim).inv();
    For(i, 0, lim) a[i] *= inv;
}
```

#### 预处理

```cpp
while(lim <= n + m) lim <<= 1;
For(i, 0, lim) rev[i] = rev[i >> 1] >> 1 | (i & 1 ? lim >> 1 : 1);
```

### 较快版

#### 定义

```cpp
ll w[N];
int lim = 1, rev[N]; 
```

`modint` 版

```cpp
mint w[N];
int lim = 1, rev[N]; 
```

#### 函数

```cpp
int add(int a, int b) { return a += b, a < P ? a : a - P; }
int sub(int a, int b) { return a -= b, a < 0 ? a + P : a; }
ll Pow(ll a, int n, ll r = 1) {
    for(; n; n >>= 1, a = a * a % P)
    if(n & 1) r = r * a % P;
    return r;
}
void NTT(ll a[], bool t) {
    For(i, 0, lim) if(rev[i] < i) swap(a[i], a[rev[i]]);
    for(int i = 1; i < lim; i <<= 1)
    for(int j = 0; j < lim; j += i << 1) For(k, j, j + i) {
        ll &x = a[k], y = a[k + i] * w[i + k - j] % P;
        a[k + i] = sub(x, y), x = add(x, y);
    }
    if(t) return;
    reverse(a + 1, a + lim);
    ll inv = Pow(lim, P - 2);
    For(i, 0, lim) (a[i] *= inv) %= P;
}
```

`modint` 版

```cpp
void NTT(mint a[], bool t) {
    For(i, 0, lim) if(rev[i] < i) swap(a[i], a[rev[i]]);
    for(int i = 1; i < lim; i <<= 1)
    for(int j = 0; j < lim; j += i << 1) For(k, j, j + i) {
        mint &x = a[k], y = a[k + i] * w[i + k - j];
        a[k + i] = x - y, x += y;
    }
    if(t) return;
    reverse(a + 1, a + lim);
    mint inv = mint(lim).inv();
    For(i, 0, lim) a[i] *= inv;
}
```

#### 预处理

```cpp
while(lim <= n + m) lim <<= 1;
For(i, 0, lim) rev[i] = rev[i >> 1] >> 1 | (i & 1 ? lim >> 1 : 0);
for(int i = 1; i < lim; i <<= 1) {
    ll wn = Pow(3, P / 2 / i);
    For(j, 0, i) w[i + j] = j ? w[i + j - 1] * wn % P : 1;
}
```

`modint` 版

```cpp
while(lim <= n + m) lim <<= 1;
For(i, 0, lim) rev[i] = rev[i >> 1] >> 1 | (i & 1 ? lim >> 1 : 0);
for(int i = 1; i < lim; i <<= 1) {
    mint wn = mint(3).pow(P / 2 / lim);
    For(j, 0, i) w[i + j] = j ? w[i + j - 1] * wn : 1;
}
```

## 任意模数 NTT / MTT

### 定义

```cpp
using cmp = complex<lf>;
const lf PI = acos(-1);
const cmp I(0, 1);

int n, m, P;
int M, lim = 1, rev[N];
cmp w[N], a0[N], a1[N], b0[N], b1[N];
```

`modint` 版

```cpp
using cmp = complex<lf>;
const lf PI = acos(-1);

int n, m, P;
int M, lim = 1, rev[N];
cmp w[N], a0[N], a1[N], b0[N], b1[N];
```

### 函数

```cpp
void FFT(cmp a[], bool t) {
    For(i, 0, lim) if(i < rev[i]) swap(a[i], a[rev[i]]);
    for(int i = 1; i < lim; i <<= 1)
    for(int j = 0; j < lim; j += i << 1) For(k, j, j + i) {
        cmp &x = a[k], y = a[k + i] * w[i + k - j];
        a[k + i] = x - y, x += y;
    }
    if(t) return;
    reverse(a + 1, a + lim);
    lf inv = 1. / lim;
    For(i, 0, lim) a[i] *= inv;
}
void FFT2(cmp a[], cmp b[]) {
    For(i, 0, lim) a[i] += b[i] * I;
    FFT(a, 0);
    For(i, 0, lim) b[i] = conj(a[i ? lim - i : 0]);
    For(i, 0, lim) {
        cmp x = a[i], y = b[i];
        a[i] = (y + x) * 0.5, b[i] = (y - x) * 0.5 * I;
    }
}
ll num(cmp x) { return M * ll(real(x) + 0.5) % P + ll(imag(x) + 0.5); }
```

`modint` 版

```cpp
void FFT(cmp a[], bool t) {
    For(i, 0, lim) if(i < rev[i]) swap(a[i], a[rev[i]]);
    for(int i = 1; i < lim; i <<= 1)
    for(int j = 0; j < lim; j += i << 1) For(k, j, j + i) {
        cmp &x = a[k], y = a[k + i] * w[i + k - j];
        a[k + i] = x - y, x += y;
    }
    if(t) return;
    reverse(a + 1, a + lim);
    lf inv = 1. / lim;
    For(i, 0, lim) a[i] *= inv;
}
void FFT2(cmp a[], cmp b[]) {
    For(i, 0, lim) a[i] += b[i] * 1i;
    FFT(a, 0);
    For(i, 0, lim) b[i] = conj(a[i ? lim - i : 0]);
    For(i, 0, lim) {
        cmp x = a[i], y = b[i];
        a[i] = (y + x) * 0.5, b[i] = (y - x) * 0.5i;
    }
}
mint num(cmp x) { return M * (mint)ll(real(x) + 0.5) % P + (mint)ll(imag(x) + 0.5); }
```

### 预处理

```cpp
M = sqrt(P);
rep(i, 0, n) a0[i] = A[i] / M, a1[i] = A[i] % M;
rep(i, 0, m) b0[i] = B[i] / M, b1[i] = B[i] % M;
while(lim <= n + m) lim <<= 1;
For(i, 0, lim) rev[i] = rev[i >> 1] >> 1 | (i & 1 ? lim >> 1 : 0);
for(int i = 1; i < lim; i <<= 1) For(j, 0, i)
    w[i + j] = cmp(cos(PI / i * j), sin(PI / i * j));
```

### 使用

```cpp
FFT2(a0, a1), FFT2(b0, b1);
For(i, 0, lim) {
    cmp t = a0[i] + I * a1[i];
    b0[i] *= t, b1[i] *= t;
}
FFT(b0, 1), FFT(b1, 1);
rep(i, 0, n + m) C[i] = (M * num(b0[i]) + num(b1[i])) % P;
```

`modint` 版

```cpp
FFT2(a0, a1), FFT2(b0, b1);
For(i, 0, lim) {
    cmp t = a0[i] + I * a1[i];
    b0[i] *= t, b1[i] *= t;
}
FFT(b0, 1), FFT(b1, 1);
rep(i, 0, n + m) C[i] = M * num(b0[i]) + num(b1[i]);
```

## FWT

### 函数

```cpp
void FWT(ll a[]) {
    For(i, 0, n) For(S, 0, 1 << n) if(S >> i & 1) {
        ll& x = a[S ^ 1 << i], y = a[S];
        a[S] = x - y, x += y;
    }
}
void IFWT(ll a[]) {
    For(i, 0, n) For(S, 0, 1 << n) if(S >> i & 1) {
        ll& x = a[S ^ 1 << i], y = a[S];
        a[S] = x - y >> 1, x = x + y >> 1;
    }
}
```

## 多项式求逆 / inv

**已弃用。**

### 定义

```cpp
int n, lim, rev[N];
```

### 函数

```cpp
ll Pow(ll a, int n, ll r = 1) {
    for(; n; n >>= 1, a = a * a % P)
    if(n & 1) r = r * a % P;
    return r;
}
void bld(int n) {
    lim = 1 << n--;
    For(i, 0, lim) rev[i] = rev[i >> 1] >> 1 | (i & 1) << n;
}
void NTT(ll a[], bool t) {
    For(i, 0, lim) if(rev[i] < i) swap(a[i], a[rev[i]]);
    for(int i = 1; i < lim; i <<= 1) {
        ll wn = Pow(3, P / 2 / i);
        for(int j = 0; j < lim; j += i << 1) {
            ll w = 1;
            For(k, j, j + i) {
                ll &x = a[k], y = a[k + i] * w % P;
                a[k + i] = (x - y) % P, (x += y) %= P, w = w * wn % P;
            }
        }
    }
    if(t) return;
    reverse(a + 1, a + lim);
    ll inv = Pow(lim, P - 2);
    For(i, 0, lim) (a[i] *= inv) %= P;
}
void Inv(ll a[], ll b[], int n) {
    static ll c[N];
    For(i, 0, 2 << n) b[i] = c[i] = 0;
    b[0] = Pow(a[0], P - 2);
    rep(i, 1, n) {
        For(j, 0, 1 << i) c[j] = a[j];
        bld(i + 1), NTT(c, 0), NTT(b, 0);
        For(j, 0, lim) b[j] = (b[j] * 2 - b[j] * b[j] % P * c[j]) % P;
        NTT(b, 1);
        For(j, 1 << i, lim) b[j] = 0;
    }
}
```

`modint` 版

```cpp
void bld(int n) {
    lim = 1 << n--;
    For(i, 0, lim) rev[i] = rev[i >> 1] >> 1 | (i & 1) << n;
}
void NTT(mint a[], bool t) {
    For(i, 0, lim) if(rev[i] < i) swap(a[i], a[rev[i]]);
    for(int i = 1; i < lim; i <<= 1) {
        mint wn = mint(3).pow(P / 2 / i);
        for(int j = 0; j < lim; j += i << 1) {
            mint w = 1;
            For(k, j, j + i) {
                mint &x = a[k], y = a[k + i] * w;
                a[k + i] = x - y, x += y, w *= wn;
            }
        }
    }
    if(t) return;
    reverse(a + 1, a + lim);
    mint inv = mint(lim).inv();
    For(i, 0, lim) a[i] *= inv;
}
void Inv(mint a[], mint b[], int n) {
    static mint c[N];
    For(i, 0, 2 << n) b[i] = c[i] = 0;
    b[0] = a[0].inv();
    rep(i, 1, n) {
        For(j, 0, 1 << i) c[j] = a[j];
        bld(i + 1), NTT(c, 0), NTT(b, 0);
        For(j, 0, lim) b[j] = b[j] * 2 - b[j] * b[j] * c[j];
        NTT(b, 1);
        For(j, 1 << i, lim) b[j] = 0;
    }
}
```

## 自然数等幂求和

## 中国剩余定理

## 扩展中国剩余定理 / exCRT

```cpp
ll mul(ll a, ll b, ll p) {
    return (a * b - (ll)((long double)a / p * b + 0.5) * p + p) % p;
}
void exgcd(ll a, ll b, ll& d, ll& x, ll& y) {
    if(!b) { d = a, x = 1, y = 0; return; }
    exgcd(b, a % b, d, y, x), y -= a / b * x;
}
void exCRT(ll& b1, ll& m1, ll b2, ll m2) {
    ll d, k1, k2; exgcd(m1, m2, d, k1, k2), m2 /= d;
    b1 = (b1 + mul(mul(k1 % m2, (b2 - b1) / d % m2, m2), m1, m1 * m2)) % (m1 *= m2);
}
```

## 杜教筛

## Min-25 筛 / min25

### 定义

```cpp
ll f1[N], f2[N];
```

### 函数

```cpp
ll min25(ll n) {
    int lim = sqrt(n);
    rep(i, 1, lim) f1[i] = i - 1, f2[i] = n / i - 1;
    rep(p, 2, lim) if (f1[p] != f1[p - 1]) {
        int w1 = lim / p;
        ll x = f1[p - 1], w3 = (ll)p * p, w2 = min((ll)lim, n / w3), d = n / p;
        rep(i, 1, w1) f2[i] -= f2[i * p] - x;
        rep(i, w1 + 1, w2) f2[i] -= f1[d / i] - x;
        per(i, lim, w3) f1[i] -= f1[i / p] - x;
    }
    return f2[1];
}
```

## exBSGS

### 定义

```cpp
map<int, int> mp;
```

### 函数

```cpp
int BSGS(ll pls, ll a, ll b, int p) {
    pls %= p, a %= p, b %= p; mp.clear();
    ll m = ceil(sqrt(p)), ls = 1, rs = 1;
    For(i, 0, m) mp[ls * b % p] = i, ls = ls * a % p;
    rep(i, 1, m) {
        rs = rs * ls % p;
        if(mp.count(rs * pls % p)) return i * m - mp[rs * pls % p];
    }
    return -1;
}
int exBSGS(int a, int b, int p) {
    a %= p, b %= p;
    int pls = 1, cnt = 0, g;
    while((g = __gcd(a, p)) > 1) {
        if(b == pls) return cnt;
        if(b % g) return -1;
        p /= g, b /= g, pls = pls * ll(a / g) % p, cnt++;
    }
    int ret = BSGS(pls, a, b, p);
    return ~ret ? ret + cnt : -1;
}
```

## cipolla

### 定义

```cpp
int n; ll II;
struct cmp {
    ll r, i;
    cmp operator *(const cmp& b) {
        return {(r * b.r + i * b.i % P * II) % P, (r * b.i + i * b.r) % P};
    }
} U = {1, 0};
```

`modint` 版

```cpp
mint n, II;
struct cmp {
    mint r, i;
    cmp operator *(const cmp& b) {
        return {r * b.r + i * b.i * II, r * b.i + i * b.r};
    }
} U = {1, 0};
```

### 函数

```cpp
int pow1(ll a, int n, ll r = 1) {
    for(; n; n >>= 1, a = a * a % P)
    if(n & 1) r = r * a % P;
    return r;
}
cmp pow2(cmp a, int n, cmp r = U) {
    for(; n; n >>= 1, a = a * a)
    if(n & 1) r = r * a;
    return r;
}
int cipolla(int n) {
    if(!n) return 0;
    if(P == 2) return n;
    if(pow1(n, P / 2) != 1) return -1;
    ll a;
    do a = rand() % P, II = (a * a - n + P) % P; while(!a || pow1(II, P / 2) == 1);
    return pow2({a, 1}, P / 2 + 1).r;
}
```

`modint` 版

```cpp
cmp Pow(cmp a, int n, cmp r = U) {
    for(; n; n >>= 1, a = a * a)
    if(n & 1) r = r * a;
    return r;
}
int cipolla(mint n) {
    if(n == 0) return 0;
    if(P == 2) return n.val();
    if(n.pow(P / 2) != 1) return -1;
    mint a;
    do a = rand(), II = a * a - n; while(a == 0 || II.pow(P / 2) == 1);
    return Pow({a, 1}, P / 2 + 1).r.val();
}
```

## Miller Rabin & Pollard Rho

### 定义

```cpp
vector<ll> as;
```

### 函数

```cpp
ll mul(ll a, ll b, ll p) {
    return (a * b - ll((long double)a / p * b + 0.5) * p + p) % p;
}
ll Pow(ll a, ll n, ll p, ll r = 1) {
    for(; n; n >>= 1, a = mul(a, a, p))
    if(n & 1) r = mul(r, a, p);
    return r;
}
int chk(ll n) {
    for(ll a : {2, 3, 7, 61, 24251}) {
        if(n == a) return 1;
        if(Pow(a, n - 1, n) != 1) return 0;
        ll k = n - 1, t;
        while(~k & 1) k >>= 1;
        if((t = Pow(a, k, n)) == 1) continue;
        while(t != 1 && t != n - 1) t = mul(t, t, n);
        if(t != n - 1) return 0;
    }
    return 1;
}
ll f(ll x, ll c, ll p) { return (mul(x, x, p) + c) % p; }
ll PR(ll n) {
    ll a = 0, b = 0, c = rand() % (n - 1) + 1, v = 1, g;
    for(int k = 1; ; k <<= 1, a = b, v = 1) {
        rep(i, 1, k) {
            b = f(b, c, n), v = mul(v, abs(a - b), n);
            if(!(i & 127) || i == k) {
                g = __gcd(v, n);
                if(g > 1) return g;
            }
        }
    }
}
void solve(ll n) {
    if(chk(n)) return as.pb(n);
    ll d; do d = PR(n); while(d == n);
    solve(d), solve(n / d);
}
```

# 数据结构

## 动态树 / LCT

### 普通版

#### 定义

```cpp
int c[N][2], f[N], r[N];
```

#### 函数

```cpp
bool id(int o) { return c[f[o]][1] == o; }
bool nrt(int o) { return c[f[o]][0] == o || c[f[o]][1] == o; }
void pu(int o) {
    
}
void rev(int o) {
    swap(c[o][0], c[o][1]), r[o] ^= 1;
}
void pd(int o) { if(r[o]) rev(c[o][0]), rev(c[o][1]), r[o] = 0; }
void rot(int o, int d) {
    int k = c[o][!d], &x = c[k][d];
    if(nrt(o)) c[f[o]][id(o)] = k;
    pu(x = f[c[o][!d] = x] = o), f[k] = f[o], pu(f[o] = k);
}
void dfs(int o) { if(nrt(o)) dfs(f[o]); pd(o); }
void splay(int o) {
    dfs(o);
    for(int fa; nrt(o); rot(f[o], !id(o)))
    if(nrt(fa = f[o])) rot(id(o) == id(fa) ? f[fa] : fa, !id(o));
}
void acc(int o) {
    for(int x = 0; o; o = f[x = o])
        splay(o), c[o][1] = x, pu(o);
}
void mkrt(int o) { acc(o), splay(o), rev(o); }
void link(int u, int v) {
    mkrt(u), acc(v), splay(v), pu(f[u] = v);
}
void cut(int u, int v) {
    mkrt(u), acc(v), splay(v), c[v][0] = f[u] = 0, pu(v);
}
```

### 维护子树 ```size``` 版

#### 定义

```cpp
int c[N][2], f[N], r[N], s[N], si[N];
```

#### 函数

```cpp
bool id(int o) { return c[f[o]][1] == o; }
bool nrt(int o) { return c[f[o]][0] == o || c[f[o]][1] == o; }
void pu(int o) {
    s[o] = s[c[o][0]] + s[c[o][1]] + si[o] + 1;
}
void rev(int o) {
    swap(c[o][0], c[o][1]), r[o] ^= 1;
}
void pd(int o) { if(r[o]) rev(c[o][0]), rev(c[o][1]), r[o] = 0; }
void rot(int o, int d) {
    int k = c[o][!d], &x = c[k][d];
    if(nrt(o)) c[f[o]][id(o)] = k;
    pu(x = f[c[o][!d] = x] = o), f[k] = f[o], pu(f[o] = k);
}
void dfs(int o) { if(nrt(o)) dfs(f[o]); pd(o); }
void splay(int o) {
    dfs(o);
    for(int fa; nrt(o); rot(f[o], !id(o)))
    if(nrt(fa = f[o])) rot(id(o) == id(fa) ? f[fa] : fa, !id(o));
}
void acc(int o) {
    for(int x = 0; o; o = f[x = o])
        splay(o), si[o] += s[c[o][1]] - s[x], c[o][1] = x, pu(o);
}
void mkrt(int o) { acc(o), splay(o), rev(o); }
void link(int u, int v) {
    mkrt(u), acc(v), splay(v), si[v] += s[u], pu(f[u] = v);
}
void cut(int u, int v) {
    mkrt(u), acc(v), splay(v), c[v][0] = f[u] = 0, pu(v);
}
```

## pbds::tree

#### 定义

```cpp
#include <bits/extc++.h>

using namespace __gnu_pbds;
template<class T> using Set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
template<class K, class V> using Map = tree<K, V, less<K>, rb_tree_tag, tree_order_statistics_node_update>;
#define ins(T, x) (T).insert(x)
#define era(T, x) (T).erase(x)
#define rk(T, x) ((T).order_of_key(x) + 1)
#define kth(T, x) (*(T).find_by_order(x - 1))
#define pre(T, x) (*--(T).lower_bound(x))
#define nxt(T, x) (*(T).upper_bound(x))
#define bld(T, l, r) (T).copy_from_range(l, r)
```

## RBS 树

### 普通版

#### 定义

```cpp
int rt, sz, lc[N], rc[N], c[N], s[N];
```

#### 函数

```cpp
using ull = unsigned ll;
ull fn(ull x) {
    const ull k = 0x9ddfea08eb382d69ULL;
    x *= k, x ^= x >> 47;
    x *= k, x ^= x >> 47;
    x *= k, x ^= x >> 47;
    return x * k;
}
int rand(int mod) {
    static ull s = 2;
    return (s += fn(s)) % mod;
}
int pu(int o) { s[o] = s[lc[o]] + s[rc[o]] + 1; return o; }
void bld(int l, int r, int& o) {
    if(l > r) return;
    int mid = l + r >> 1;
    c[o = ++sz] = a[mid];
    bld(l, mid - 1, lc[o]), bld(mid + 1, r, rc[o]), pu(o); 
}
void spt(int o, int x, int& u, int& v) {
    if(!o) u = v = 0;
    else if(x < c[o]) spt(lc[v = o], x, u, lc[o]), pu(o);
    else spt(rc[u = o], x, rc[o], v), pu(o);
}
int mrg(int u, int v) {
    if(!u || !v) return u + v;
    if(rand(s[u] + s[v]) < s[u])
        return rc[u] = mrg(rc[u], v), pu(u);
    return lc[v] = mrg(u, lc[v]), pu(v);
}
void ins(int& o, int x) {
    if(rand(s[o] + 1) == 0)
        c[++sz] = x, spt(o, x, lc[sz], rc[sz]), pu(o = sz);
    else ins(x < c[o] ? lc[o] : rc[o], x), pu(o);
}
void rmv(int& o, int x) {
    int t1, t2, t3;
    spt(o, x, t1, t3), spt(t1, x - 1, t1, t2);
    o = mrg(mrg(t1, mrg(lc[t2], rc[t2])), t3);
}
int rk(int o, int x, int re = 1) {
    while(o) o = x > c[o] ? re += s[lc[o]] + 1, rc[o] : lc[o]; 
    return re;
}
int kth(int o, int k) {
    while(k != s[lc[o]] + 1)
        o = k > s[lc[o]] ? k -= s[lc[o]] + 1, rc[o] : lc[o];
    return c[o];
}
int pre(int o, int x, int re = -inf) {
    while(o) o = x > c[o] ? re = max(re, c[o]), rc[o] : lc[o];
    return re;
}
int nxt(int o, int x, int re = inf) {
    while(o) o = x < c[o] ? re = min(re, c[o]), lc[o] : rc[o];
    return re;
}
```

### 可持久化版

```cpp
#include <cstdio>
#include <limits>
#include <vector>
using namespace std;

typedef long long int64;

inline int64 Fingerprint(int64 x) {
    const int64 kMul = 0x9ddfea08eb382d69ULL;
    x *= kMul, x ^= x >> 47;
    x *= kMul, x ^= x >> 47;
    x *= kMul, x ^= x >> 47;
    return x * kMul;
}

inline int64 Random() {
    static int64 Seed = 2;
    Seed += Fingerprint(Seed);
    return Seed & numeric_limits<int64>::max();
}

typedef int DataType;

struct RBST {
    RBST *ChildL, *ChildR;
    int Size;
    DataType Data;

    RBST() { ChildL = ChildR = 0; }
};

int GetSize(const RBST* root) { return root ? root->Size : 0; }

int LowerBoundIndex(const RBST* root, DataType x) {
    if (!root) return 0;
    if (x <= root->Data) return LowerBoundIndex(root->ChildL, x);
    int sizeL = GetSize(root->ChildL);
    return LowerBoundIndex(root->ChildR, x) + sizeL + 1;
}

DataType Select(const RBST* root, int index) {
    int sizeL = GetSize(root->ChildL);
    if (index == sizeL) return root->Data;
    if (index < sizeL) return Select(root->ChildL, index);
    return Select(root->ChildR, index - sizeL - 1);
}

RBST* SetSize(RBST* root) {
    root->Size = GetSize(root->ChildL) + GetSize(root->ChildR) + 1;
    return root;
}

struct RBSTree {
    vector<RBST*> Nodes;

    RBST* NewNode(RBST* node) {
        Nodes.push_back(new RBST(*node));
        return Nodes.back();
    }

    void Split(RBST* root, DataType x, RBST*& treeL, RBST*& treeR) {
        if (!root) {
            treeL = treeR = 0;
        } else if (x <= root->Data) {
            RBST *newRoot = NewNode(root);
            Split(root->ChildL, x, treeL, newRoot->ChildL);
            treeR = SetSize(newRoot);
        } else {
            RBST* newRoot = NewNode(root);
            Split(root->ChildR, x, newRoot->ChildR, treeR);
            treeL = SetSize(newRoot);
        }
    }

    RBST* Join(RBST* treeL, RBST* treeR) {
        int sizeL = GetSize(treeL);
        int sizeR = GetSize(treeR);
        int size = sizeL + sizeR;
        if (size == 0) return 0;
        if (Random() % size < sizeL) {
            RBST* newRoot = NewNode(treeL);
            newRoot->ChildR = Join(treeL->ChildR, treeR);
            return SetSize(newRoot);
        } else {
            RBST* newRoot = NewNode(treeR);
            newRoot->ChildL = Join(treeL, treeR->ChildL);
            return SetSize(newRoot);
        }
    }

    RBST* InsertAsRoot(RBST* root, DataType item) {
        Nodes.push_back(new RBST);
        RBST *newRoot = Nodes.back();
        newRoot->Data = item;
        Split(root, item + 1, newRoot->ChildL, newRoot->ChildR);
        return SetSize(newRoot);
    }

    RBST* Insert(RBST* root, DataType item) {
        if (Random() % (GetSize(root) + 1) == 0) {
            return InsertAsRoot(root, item);
        } else if (item < root->Data) {
            RBST *newRoot = NewNode(root);
            newRoot->ChildL = Insert(root->ChildL, item);
            return SetSize(newRoot);
        } else {
            RBST *newRoot = NewNode(root);
            newRoot->ChildR = Insert(root->ChildR, item);
            return SetSize(newRoot);
        }
    }

    RBST* Remove(RBST* root, DataType item) {
        RBST *tree1, *tree2, *tree3, *tree4 = 0;
        Split(root, item, tree1, tree2);
        Split(tree2, item + 1, tree2, tree3);
        if (tree2) tree4 = Join(tree2->ChildL, tree2->ChildR);
        return Join(Join(tree1, tree4), tree3);
    }

    void Destroy() {
        for (int i = 0; i < Nodes.size(); ++i) delete Nodes[i];
    }
};
```

## K-D 树

### 定义

```cpp
#define mid ((l + r) >> 1)
#define lc (o << 1)
#define rc (o << 1 | 1)
#define lch l, mid, lc
#define rch mid + 1, r, rc

int D, L[N * 4][K], R[N * 4][K];
struct node {
    int x[K];
    void read() { For(i, 0, K) scanf("%d", &x[i]); }
    bool operator <(const node& b)const {
        return x[D] < b.x[D];
    }
} a[N];
```

### 函数

```cpp
lf sq(lf x) { return x * x; }
void pu(int o) {
    For(i, 0, K) {
        L[o][i] = min(L[lc][i], L[rc][i]);
        R[o][i] = max(R[lc][i], R[rc][i]);
    }
}
void bld(int l, int r, int o) {
    if(l == r) {
        For(i, 0, K) L[o][i] = R[o][i] = a[l].x[i];
        return;
    }
    lf va[K] = {};
    For(i, 0, K) {
        lf av = 0;
        rep(j, l, r) av += a[j].x[i];
        av /= r - l + 1;
        rep(j, l, r) va[i] += sq(a[j].x[i] - av);
    }
    D = max_element(va, va + K) - va;
    nth_element(a + l, a + mid, a + r + 1);
    bld(lch), bld(rch), pu(o);
}
```

### 欧几里得距离平方

```cpp
ll sq(int x) { return 1ll * x * x; }
ll mine(int o, int x[], ll re = 0) {
    For(i, 0, K) re += sq(max(L[o][i] - x[i], 0)) + sq(max(x[i] - R[o][i], 0));
    return re;
}
ll maxe(int o, int x[], ll re = 0) {
    For(i, 0, K) re += max(sq(x[i] - L[o][i]), sq(x[i] - R[o][i]));
    return re;
}
```

### 曼哈顿距离

```cpp
int minm(int o, int x[], int re = 0) {
    For(i, 0, K) re += max(L[o][i] - x[i], 0) + max(x[i] - R[o][i], 0);
    return re;
}
int maxm(int o, int x[], int re = 0) {
    For(i, 0, K) re += max(abs(x[i] - L[o][i]), abs(x[i] - R[o][i]));
    return re;
}
```

### 切比雪夫距离

```cpp
int minc(int o, int x[], int re = inf) {
    For(i, 0, K) re = max(re, max(L[o][i] - x[i], 0) + max(x[i] - R[o][i], 0));
    return re;
}
int maxc(int o, int x[], int re = 0) {
    For(i, 0, K) re = max({re, abs(x[i] - L[o][i]), abs(x[i] - R[o][i])});
    return re;
}
```

### 最近点查询

#### 定义

```cpp
ll res;
```

#### 函数

```cpp
void qry(int x[], int l, int r, int o) {
    if(l == r) return void(res = mine(o, x));
    ll dl = mine(lc, x), dr = mine(rc, x);
    if(dl < dr) {
        if(dl < res) qry(x, lch);
        if(dr < res) qry(x, rch);
    } else {
        if(dr < res) qry(x, rch);
        if(dl < res) qry(x, lch);
    }
}
```

#### 初始化

```cpp
res = inf;
```
### 最远点查询

#### 定义

```cpp
ll res;
```

#### 函数

```cpp
void qry(int x[], int l, int r, int o) {
    if(l == r) return void(res = maxe(o, x));
    ll dl = maxe(lc, x), dr = maxe(rc, x);
    if(dl > dr) {
        if(dl > res) qry(x, lch);
        if(dr > res) qry(x, rch);
    } else {
        if(dr > res) qry(x, rch);
        if(dl > res) qry(x, lch);
    }
}
```

#### 初始化

```cpp
res = 0;
```

### 矩形判定

#### 矩形是否包含所有点

```cpp
int chk1(int l[], int r[], int o, int re = 1) {
    For(i, 0, K) re &= l[i] <= L[o][i] && R[o][i] <= r[i];
    return re;
}
```

#### 矩形是否可能包含点

```cpp
int chk2(int l[], int r[], int o, int re = 1) {
    For(i, 0, K) re &= max(l[i], L[o][i]) <= min(r[i], R[o][i]);
    return re;
}
```

### 圆判定

#### 圆是否包含所有点

```cpp
int chk1(int x[], int r, int o) {
    return maxe(x, o) <= 1ll * r * r;
}
```

#### 圆是否可能包含点

```cpp
int chk2(int x[], int r, int o) {
    return mine(x, o) <= 1ll * r * r;
}
```

# 图论

## 虚树 / vtree

### `vector` 版

#### 定义

```cpp
int n, a[N], tp, stk[N];
int idx, dfn[N], d[N], fa[20][N];
vector<int> G[N], T[N];
```

#### 函数

```cpp
void dfs(int u) {
    dfn[u] = ++idx;
    rep(i, 1, 19) fa[i][u] = fa[i - 1][fa[i - 1][u]];
    for(int v : G[u]) if(v != fa[0][u])
        d[v] = d[u] + 1, fa[0][v] = u, dfs(v);
}
int lca(int u, int v) {
    if(d[u] < d[v]) swap(u, v);
    rep(i, 0, 19) if(d[u] - d[v] >> i & 1) u = fa[i][u];
    if(u == v) return u;
    per(i, 19, 0) if(fa[i][u] != fa[i][v]) u = fa[i][u], v = fa[i][v];
    return fa[0][u];
}
void bld(int k) {
    sort(a + 1, a + k + 1, [](int a, int b) { return dfn[a] < dfn[b]; });
    stk[++tp] = 1;
    rep(i, 1, k) if(a[i] > 1) {
        int x = lca(a[i], stk[tp]);
        while(d[stk[tp]] > d[x]) tp--, T[d[stk[tp]] > d[x] ? stk[tp] : x].pb(stk[tp + 1]);
        if(stk[tp] != x) stk[++tp] = x;
        stk[++tp] = a[i];
    }
    while(--tp) T[stk[tp]].pb(stk[tp + 1]);
}
void clr(int u) {
    for(int v : G[u]) clr(v); T[u].clear();
}
```

### 快很多版

#### 定义

```cpp
int n, a[N], p, stk[N];
int d[N], fa[N], sz[N], son[N], tp[N], idx, dfn[N];
vector<int> G[N];
int eid, he[N];
struct edge { int v, nx; } e[N * 2];
```

#### 函数

```cpp
void dfs(int u) {
    sz[u] = 1;
    for(int v : G[u]) {
        d[v] = d[u] + 1, dfs(v), sz[u] += sz[v];
        if(sz[v] > sz[son[u]]) son[u] = v;
    }
}
void Dfs(int u, int top) {
    tp[u] = top, dfn[u] = ++idx;
    if(son[u]) Dfs(son[u], top);
    for(int v : G[u]) if(v != son[u]) Dfs(v, v);
}
int lca(int u, int v) {
    for(; tp[u] != tp[v]; d[tp[u]] > d[tp[v]] ? u = fa[tp[u]] : v = fa[tp[v]]);
    return d[u] < d[v] ? u : v;
}
void add(int u, int v, int t = 1) {
    e[++eid] = {v, h1[u]}, h1[u] = eid;
}
void bld(int k) {
    sort(a + 1, a + k + 1, [](int a, int b) { return dfn[a] < dfn[b]; });
    stk[++p] = 1;
    rep(i, 1, k) if(a[i] > 1) {
        int x = lca(a[i], stk[p]);
        while(d[stk[p]] > d[x])
            p--, add(d[stk[p]] > d[x] ? stk[p] : x, stk[p + 1]);
        if(stk[p] != x) stk[++p] = x;
        stk[++p] = a[i];
    }
    while(--p) add(stk[p], stk[p + 1]);
}
```

## 超快版倍增求 ```LCA```

### 定义

```cpp
int dl[N], dr[N], fa[20][N];
```

### 函数

```cpp
int lca(int u, int v) {
    if(dl[u] < dl[v]) swap(u, v);
    if(dl[v] <= dl[u] && dl[u] <= dr[v]) return v;
    per(i, 19, 0) {
        int x = fa[i][u];
        if(x && dl[x] > dl[v]) u = x;
    }
    return fa[0][u];
}
```

## 最大流 / mf

### ```int``` 版

#### 定义

```cpp
int n, m, s, t;
int eid = 1, he[N], nw[N], d[N], q[N];
struct edge { int v, nx, c; } e[M * 2];
```

#### 函数

```cpp
void add(int u, int v, int c) {
    e[++eid] = {v, he[u], c}, he[u] = eid;
    e[++eid] = {u, he[v], 0}, he[v] = eid;
}
bool bfs() {
    mem(d, 0), memcpy(nw, he, sizeof he);
    q[1] = s, d[s] = 1;
    for(int l = 1, r = 1; l <= r; l++) {
        int u = q[l];
        for(int i = he[u], v; v = e[i].v; i = e[i].nx)
        if(e[i].c && !d[v]) q[++r] = v, d[v] = d[u] + 1;
    }
    return d[t];
}
int dfs(int u, int lim, int re = 0) {
    if(u == t) return lim;
    for(int& i = nw[u], v; v = e[i].v; i = e[i].nx)
    if(e[i].c && d[v] == d[u] + 1) {
        int t = dfs(v, min(e[i].c, lim));
        e[i].c -= t, e[i ^ 1].c += t, re += t, lim -= t;
        if(!lim) break;
    }
    return re;
}
```

#### 使用

```cpp
int flow = 0;
while(bfs()) flow += dfs(s, inf);
```

### ```long long``` 版

```cpp
bool bfs() {
    mem(d, 0), memcpy(nw, he, sizeof he);
    q[1] = s, d[s] = 1;
    for(int l = 1, r = 1; l <= r; l++) {
        int u = q[l];
        for(int i = he[u], v; v = e[i].v; i = e[i].nx)
        if(e[i].c && !d[v]) q[++r] = v, d[v] = d[u] + 1;
    }
    return d[t];
}
ll dfs(int u, ll lim, ll re = 0) {
    if(u == t) return lim;
    for(int& i = nw[u], v; v = e[i].v; i = e[i].nx)
    if(e[i].c && d[v] == d[u] + 1) {
        ll t = dfs(v, min((ll)e[i].c, lim));
        e[i].c -= t, e[i ^ 1].c += t, re += t, lim -= t;
        if(!lim) break;
    }
    return re;
}
```

## 最小费用最大流 / mcf

### SSP 算法

#### 定义

```cpp
int n, m, s, t, flow, cost;
int eid = 1, he[N], nw[N], q[N], d[N], vis[N];
struct edge { int v, nx, c, w; } e[M * 2];
```

#### 函数

```cpp
void add(int u, int v, int c, int w) {
    e[++eid] = {v, he[u], c, w}, he[u] = eid;
    e[++eid] = {u, he[v], 0, -w}, he[v] = eid;
}
bool spfa() {
    mem(d, 63), memcpy(nw, he, sizeof he);
    q[0] = s, d[s] = 0, vis[s] = 1;
    for(int l = 0, r = 0; l <= r; l++) {
        int u = q[l % N]; vis[u] = 0;
        for(int i = he[u], v; v = e[i].v; i = e[i].nx)
        if(e[i].c && d[u] + e[i].w < d[v]) {
            d[v] = d[u] + e[i].w;
            if(!vis[v]) q[++r % N] = v, vis[v] = 1;
        }
    }
    return d[t] < d[0];
}
int dfs(int u, int lim, int re = 0) {
    if(u == t) return lim;
    vis[u] = 1;
    for(int& i = nw[u], v; v = e[i].v; i = e[i].nx)
    if(!vis[v] && e[i].c && d[v] == d[u] + e[i].w) {
        int t = dfs(v, min(lim, e[i].c));
        e[i].c -= t, e[i ^ 1].c += t, re += t, lim -= t, cost += t * e[i].w;
        if(!lim) break;
    }
    vis[u] = 0;
    return re;
}
```

#### 使用

```cpp
while(spfa()) flow += dfs(s, 1e9);
```

### Primal-Dual 原始对偶算法

#### 定义

```cpp
int n, m, s, t, flow, cost;
int eid = 1, he[N], d[N], pre[N], h[N];
struct edge { int v, nx, c, w; } e[M * 2];
struct node {
    int d, u;
    bool operator <(const node& b)const {
        return d > b.d;
    }
};
```

#### 函数

```cpp
void add(int u, int v, int c, int w) {
    e[++eid] = {v, he[u], c, w}, he[u] = eid;
    e[++eid] = {u, he[v], 0, -w}, he[v] = eid;
}
bool dij() {
    priority_queue<node> q;
    mem(d, 63), q.push({d[s] = 0, s});
    while(q.size()) {
        auto [dis, u] = q.top(); q.pop();
        if(dis > d[u]) continue;
        for(int i = he[u], v; v = e[i].v; i = e[i].nx) if(e[i].c) {
            int w = d[u] + e[i].w - h[v] + h[u];
            if(w < d[v]) pre[v] = i ^ 1, q.push({d[v] = w, v});
        }
    }
    rep(i, 1, /* Number of nodes */) h[i] += d[i];
    return d[t] < d[0];
}
```

#### 使用

```cpp
while(dij()) {
    int mi = inf;
    for(int u = t, i; i = pre[u]; u = e[i].v) mi = min(mi, e[i ^ 1].c);
    for(int u = t, i; i = pre[u]; u = e[i].v) e[i].c += mi, e[i ^ 1].c -= mi;
    flow += mi, cost += mi * h[t];
}
```

## 二分图最大匹配 / match

### 定义

```cpp
int n, m, vis[N], mat[N];
vector<int> G[N];
```

### 函数

```cpp
int dfs(int u, int s) {
    if(vis[u] == s) return 0;
    vis[u] = s;
    for(int v : G[u]) if(!mat[v] || dfs(mat[v], s)) return mat[v] = u;
    return 0;
}
```

### 使用

```cpp
int as = 0;
rep(i, 1, n) if(dfs(i, i)) as++;
```

## 2-SAT 问题

**已弃用。（复杂度不对）**

### 定义

```cpp
int n, m, co[N * 2], stk[N * 2], tp;
vector<int> G[N * 2];
```

### 函数

```cpp
void add(int u, int a, int v, int b) {
    G[u * 2 + !a].pb(v * 2 + b);
    G[v * 2 + !b].pb(u * 2 + a);
}
int dfs(int u) {
    if(co[u] | co[u ^ 1]) return co[u];
    co[u] = 1, stk[++tp] = u;
    for(int v : G[u]) if(!dfs(v)) return 0;
    return 1;
}
int twoSat() {
    rep(i, 1, n) {
        if(!co[i * 2] && !co[i * 2 + 1] && !dfs(i * 2)) {
            while(tp) co[stk[tp--]] = 0;
            if(!dfs(i * 2 + 1)) return 0;
        }
        tp = 0;
    }
    return 1;
}
```

# 字符串

## manacher 求偶回文串

```cpp
rep(i, 1, n) {
    int& j = ma > i ? min(R[p * 2 - i], ma - i) : 0;
    while(s[i - j] == s[i + j + 1]) j++;
    if(i + j > ma) ma = i + j, p = i;
}
```

## 回文自动机 PAM

### 普通版

#### 定义

```cpp
char s[N];
int n, sz = 1, nw, len[N], f[N], ch[N][26];
```

#### 函数
```cpp
void ins(int i) {
    auto jmp = [&](int o) {
        while(s[i - len[o] - 1] != s[i]) o = f[o];
        return o;
    };
    int o = jmp(nw), c = s[i] - 97;
    if(!ch[o][c]]) {
        f[++sz] = ch[jmp(f[o])][c];
        len[ch[o][c] = sz] = len[o] + 2;
    }
    nw = ch[o][c];
}
```

#### 预处理

```cpp
len[1] = -1, f[0] = 1;
```

### 偶回文版

#### 函数

```cpp
void ins(int i) {
    auto jmp = [&](int o) {
        while(o && s[i - len[o] - 1] != s[i]) o = f[o];
        return o;
    };
    int o = jmp(nw), c = s[i] - 97;
    if(!ch[o][c]]) {
        f[++sz] = ch[jmp(f[o])][c];
        len[ch[o][c] = sz] = len[o] + 2;
    }
    nw = ch[o][c];
}
```

#### 预处理

```cpp
rep(i, 0, 25) ch[0][i] = 1;
```

## 后缀数组 / SA

### 定义

```cpp
int n, sa[N], rk[N], tp[N], buc[N];
int h[20][N];
char s[N];
```

### 函数

```cpp
void SA() {
    int m = 128;
    rep(i, 1, n) buc[rk[i] = s[i]]++;
    rep(i, 1, m) buc[i] += buc[i - 1];
    per(i, n, 1) sa[buc[rk[i]]--] = i;
    for(int k = 1, p; memset(buc, p = 0, m * 4 + 4); k <<= 1) {
        rep(i, n - k + 1, n) tp[++p] = i;
        rep(i, 1, n) if(sa[i] > k) tp[++p] = sa[i] - k;
        rep(i, 1, n) buc[rk[i]]++;
        rep(i, 1, m) buc[i] += buc[i - 1];
        per(i, n, 1) sa[buc[rk[tp[i]]]--] = tp[i];
        memcpy(tp, rk, n * 4 + 4), p = 0;
        rep(i, 1, n) rk[sa[i]] = p += tp[sa[i]] != tp[sa[i - 1]] || tp[sa[i] + k] != tp[sa[i - 1] + k];
        if((m = p) >= n) break;
    }
}
void height() {
    int k = 0;
    rep(i, 1, n) {
        for(k ? k-- : 0; s[i + k] == s[sa[rk[i] - 1] + k]; k++);
        h[0][rk[i]] = k;
    }
    rep(i, 1, 19) rep(j, 1 << i, n)
        h[i][j] = min(h[i - 1][j], h[i - 1][j - (1 << i - 1)]);
}
```

## 后缀自动机 / SAM

### 定义

```cpp
char s[N];
int n, sz = 1, nw = 1, f[N * 2], len[N * 2], ch[N * 2][26];
```

### 函数

```cpp
void ins(int c) {
    int u = ++sz;
    len[u] = len[nw] + 1;
    while(nw && !ch[nw][c]) ch[nw][c] = u, nw = f[nw];
    if(!nw) f[u] = 1;
    else {
        int v = ch[nw][c];
        if(len[v] > len[nw] + 1) {
            f[++sz] = f[v], memcpy(ch[sz], ch[v], sizeof ch[v]);
            f[v] = f[u] = sz, len[sz] = len[nw] + 1;
            while(ch[nw][c] == v) ch[nw][c] = sz, nw = f[nw];
        } else f[u] = v;
    }
    nw = u;
}
```

## 最小表示法 / minnot

### 函数

```cpp
int calc(char s[]) {
    int i = 0, j = 1, k = 0;
    while(max({i, j, k}) < n)
    if(s[(i + k) % n] == s[(j + k) % n]) k++;
    else {
        if(s[(i + k) % n] > s[(j + k) % n]) swap(i, j);
        j += k + 1, k = 0;
        if(i == j) j++;
    }
    return min(i, j);
}
```

## 最小后缀 / minsuf

### 函数

```cpp
int calc(char s[]) {
    int i = 0, j = 1;
    while(s[j]) {
        int k = 0;
        while(s[i + k] == s[j + k]) k++;
        if(!s[j + k]) i = max(i + k, j + 1), swap(i, j);
        else if(s[i + k] > s[j + k]) i = max(i + k, j) + 1, swap(i, j);
        else j += k + 1;
    }
    return i;
}
```
