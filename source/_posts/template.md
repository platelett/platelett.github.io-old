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
```

# 有用的模板

## Fast IO 有符号整数版

```cpp
struct IO {
    static const int inSZ = 1 << 17;
    char inBuf[inSZ], *in1, *in2;
    template<class T> inline __attribute((always_inline))
    T read() {
        if (in1 > inBuf + inSZ - 32) [[unlikely]] {
            auto len = in2 - in1;
            memcpy(inBuf, in1, len);
            in1 = inBuf, in2 = inBuf + len;
            in2 += fread(in2, 1, inSZ - len, stdin);
            if (in2 != inBuf + inSZ) *in2 = 0;
        }
        T res = 0;
        unsigned char c;
        bool neg = 0;
        while ((c = *in1++) < 48) neg = c == 45;
        while (res = res * 10 + (c - 48), (c = *in1++) >= 48);
        return neg ? -res : res;
    }
    static const int outSZ = 1 << 21;
    char outBuf[outSZ], *out;
    template<class T> inline __attribute((always_inline))
    void write(T x) {
        if (out > outBuf + outSZ - 32) [[unlikely]]
            fwrite(outBuf, 1, out - outBuf, stdout), out = outBuf;
        if (!x) return *out++ = 48, void();
        if (x < 0) *out++ = 45, x = -x;
        alignas(2) const char* digits =
        "0001020304050607080910111213141516171819"
        "2021222324252627282930313233343536373839"
        "4041424344454647484950515253545556575859"
        "6061626364656667686970717273747576777879"
        "8081828384858687888990919293949596979899";
        alignas(64) static char buf[20];
        char* p = buf + 20;
        while (x >= 10) memcpy(p -= 2, digits + x % 100 * 2, 2), x /= 100;
        if (x) *--p = 48 + x;
        auto len = buf + 20 - p;
        memcpy(out, p, len), out += len;
    }
    IO() {
        in1 = in2 = inBuf + inSZ;
        out = outBuf;
    }
    ~IO() { fwrite(outBuf, 1, out - outBuf, stdout); }
} IO;
template<class T = int> inline T read() {
    return IO.read<T>();
}
template<class... Args> inline void read(Args&... args) {
    ((args = IO.read<Args>()), ...);
}
template<class T> inline void write(T x, char c = '\n') {
    IO.write(x), *IO.out++ = c;
}
```

## AtCoder-modint

```cpp
#include <bits/stdc++.h>

using namespace std;
using ll = long long;
using u32 = uint32_t;
using u64 = uint64_t;

constexpr ll safe_mod(ll x, ll m) {
    return x %= m, x < 0 ? x + m : x;
}
constexpr ll pow_mod_constexpr(ll x, ll n, int m) {
    if (m == 1) return 0;
    u32 _m = m;
    u64 r = 1, _x = safe_mod(x, m);
    for (; n; n >>= 1, _x = _x * _x % _m)
    if (n & 1) r = r * _x % m;
    return r;
}
constexpr bool is_prime_constexpr(int n) {
    if (n <= 1) return false;
    if (n == 2 || n == 7 || n == 61) return true;
    if (n % 2 == 0) return false;
    ll d = n - 1;
    while (~d & 1) d >>= 1;
    for (ll a : {2, 7, 61}) {
        ll t = d, y = pow_mod_constexpr(a, t, n);
        while (t != n - 1 && y != 1 && y != n - 1)
            y = y * y % n, t <<= 1;
        if (y != n - 1 && t % 2 == 0) return false;
    }
    return true;
}
constexpr pair<ll, ll> inv_gcd(ll a, ll b) {
    a = safe_mod(a, b);
    if (a == 0) return {b, 0};
    ll s = b, t = a, m0 = 0, m1 = 1;
    while (t) {
        ll u = s / t;
        s -= t * u, m0 -= m1 * u;
        ll tmp = s;
        s = t, t = tmp, tmp = m0, m0 = m1, m1 = tmp;
    }
    if (m0 < 0) m0 += b / s;
    return {s, m0};
}
struct barrett {
    u32 m; u64 im;
    barrett(u32 m) :m(m), im(~0ull / m + 1) {}
    u32 mul(u32 a, u32 b) const {
        u64 z = (u64)a * b;
        u64 x = (__uint128_t)z * im >> 64;
        u32 v = z - x * m;
        return (int)v < 0 ? v + m : v;
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
    u32 val() const { return _v; }
    mint& operator++() {
        if (++_v == m) _v = 0;
        return *this;
    }
    mint& operator--() {
        if (!_v--) _v = m - 1;
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
        if (_v >= m) _v -= m;
        return *this;
    }
    mint& operator-=(const mint& rhs) {
        _v -= rhs._v;
        if (_v >= m) _v += m;
        return *this;
    }
    mint& operator*=(const mint& rhs) {
        u64 z = _v;
        z *= rhs._v, _v = z % m;
        return *this;
    }
    mint& operator/=(const mint& rhs) { return *this = *this * rhs.inv(); }
    mint operator+() const { return *this; }
    mint operator-() const { return mint() - *this; }

    mint pow(ll n) const {
        assert(0 <= n);
        mint x = *this, r = 1;
        for (; n; n >>= 1, x *= x) if (n & 1) r *= x;
        return r;
    }
    mint inv() const {
        if (prime) {
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
    u32 _v;
    static constexpr bool prime = is_prime_constexpr(m);
};

template<int id> struct dynamic_modint {
    using mint = dynamic_modint;

  public:
    static int mod() { return bt.m; }
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

    u32 val() const { return _v; }

    mint& operator++() {
        if (++_v == bt.m) _v = 0;
        return *this;
    }
    mint& operator--() {
        if (!_v--) _v = bt.m - 1;
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
        if (_v >= bt.m) _v -= bt.m;
        return *this;
    }
    mint& operator-=(const mint& rhs) {
        _v += bt.m - rhs._v;
        if (_v >= bt.m) _v -= bt.m;
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
        for (; n; n >>= 1, x *= x) if (n & 1) r *= x;
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
    u32 _v;
    static barrett bt;
};
template<int id> barrett dynamic_modint<id>::bt = 998244353;
```

## 大模数取模

```cpp
using ll = long long;
ll mul(ll A, ll B, ll P) {
     C = A * B - (ll)((long double)A * B / P + 0.5) * P;
    return C < 0 ? C + P : C;
}
```

## `bash` 对拍

```bash
while true; do
    ./gen > in
    ./a < in > 1
    ./b < in > 2
    diff 1 2
    if (($? != 0)); then break; fi
done
```

# 数学

## NTT

```cpp
const int P = 998244353;

inline int add(int a, int b) { return (a += b) < P ? a : a - P; }
inline int sub(int a, int b) { return (a -= b) < 0 ? a + P : a; }
inline int mul(int a, int b) { return (uint64_t)(uint32_t)a * (uint32_t)b % P; }
inline int ceil2(int n) { return 2 << __builtin_ia32_bsrsi(n); }
int Pow(int a, int n) {
    int r = 1;
    for (; n; n >>= 1, a = mul(a, a))
        if (n & 1) r = mul(r, a);
    return r;
}
struct precalc {
    int w[23], iw[23];
    precalc() {
        int e[22], ie[22], now = 15311432, inow = 469870224;
        for (int i = 21; i >= 0; i--) {
            e[i] = now, ie[i] = inow;
            now = mul(now, now), inow = mul(inow, inow);
        }
        now = inow = 1;
        for (int i = 0; i <= 21; i++) {
            w[i] = mul(e[i], now), iw[i] = mul(ie[i], inow);
            now = mul(now, ie[i]), inow = mul(inow, e[i]);
        }
    }
} pre;
void DIF(int a[], int n) {
    for (int i = n >> 1, l = 1; i; i >>= 1, l <<= 1) {
        int now = 1;
        for (int j = 0; j < l; j++) {
            int p = j * i * 2;
            for (int k = p; k < p + i; k++) {
                int x = a[k], y = mul(a[k + i], now);
                a[k] = add(x, y), a[k + i] = sub(x, y);
            }
            now = mul(now, pre.w[__builtin_ctz(j + 1)]);
        }
    }
}
void IDIF(int a[], int n) {
    for (int i = 1, l = n >> 1; l; i <<= 1, l >>= 1) {
        int now = 1;
        for (int j = 0; j < l; j++) {
            int p = j * i * 2;
            for (int k = p; k < p + i; k++) {
                int x = a[k], y = a[k + i];
                a[k] = add(x, y), a[k + i] = mul(x - y + P, now);
            }
            now = mul(now, pre.iw[__builtin_ctz(j + 1)]);
        }
    }
    int inv = Pow(n, P - 2);
    for (int i = 0; i < n; i++) a[i] = mul(a[i], inv);
}
```

`modint` 版

```cpp
using mint = static_modint<998244353>;

inline int ceil2(int n) { return 2 << __builtin_ia32_bsrsi(n); }
struct precalc {
    mint w[23], iw[23];
    precalc() {
        mint e[22], ie[22], now = 15311432, inow = 469870224;
        for (int i = 21; i >= 0; i--) {
            e[i] = now, ie[i] = inow;
            now *= now, inow *= inow;
        }
        now = inow = 1;
        for (int i = 0; i <= 21; i++) {
            w[i] = e[i] * now, iw[i] = ie[i] * inow;
            now *= ie[i], inow *= e[i];
        }
    }
} pre;
void DIF(mint a[], int n) {
    for (int i = n >> 1, l = 1; i; i >>= 1, l <<= 1) {
        mint now = 1;
        for (int j = 0; j < l; j++) {
            int p = j * i * 2;
            for (int k = p; k < p + i; k++) {
                mint x = a[k], y = a[k + i] * now;
                a[k] = x + y, a[k + i] = x - y;
            }
            now *= pre.w[__builtin_ctz(j + 1)];
        }
    }
}
void IDIF(mint a[], int n) {
    for (int i = 1, l = n >> 1; l; i <<= 1, l >>= 1) {
        mint now = 1;
        for (int j = 0; j < l; j++) {
            int p = j * i * 2;
            for (int k = p; k < p + i; k++) {
                mint x = a[k], y = a[k + i];
                a[k] = x + y, a[k + i] = (x - y) * now;
            }
            now *= pre.iw[__builtin_ctz(j + 1)];
        }
    }
    mint inv = mint(n).inv();
    for (int i = 0; i < n; i++) a[i] *= inv;
}
```

## 任意模数 NTT / MTT

```cpp
using cplx = complex<double>;

const int N = /**/;
const double PI = acos(-1);

int rev[N];
cplx w[N];

void FFT(cplx a[], int n, bool t) {
    for (int i = 0; i < n; i++)
        if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int i = 1; i < n; i <<= 1)
        for (int j = 0; j < n; j += i << 1)
            for (int k = j; k < j + i; k++) {
                cplx x = a[k], y = a[k + i] * w[i + k - j];
                a[k] = x + y, a[k + i] = x - y;
            }
    if (t) return;
    reverse(a + 1, a + n);
    double inv = 1.0 / n;
    for (int i = 0; i < n; i++) a[i] *= inv;
}
void FFT2(cplx a[], cplx b[], int n) {
    for (int i = 0; i < n; i++) a[i] += b[i] * 1i;
    FFT(a, n, 1);
    b[0] = conj(a[0]);
    for (int i = 1; i < n; i++) b[i] = conj(a[n - i]);
    for (int i = 0; i < n; i++) {
        cplx x = a[i], y = b[i];
        a[i] = (y + x) * 0.5, b[i] = (y - x) * 0.5i;
    }
}
void mul(int n, int m, int P, int a[], int b[], int c[]) {
    static cplx a0[N], a1[N], b0[N], b1[N]; 
    int M = sqrt(P);
    for (int i = 0; i <= n; i++)
        a0[i] = a[i] / M, a1[i] = a[i] % M;
    for (int i = 0; i <= m; i++)
        b0[i] = b[i] / M, b1[i] = b[i] % M;
    int lim = 2 << __builtin_ia32_bsrsi(n + m);
    for (int i = 0; i < lim; i++)
        rev[i] = rev[i >> 1] >> 1 | (i & 1 ? lim >> 1 : 0);
    for (int i = 1; i < lim; i <<= 1)
        for (int j = 0; j < i; j++)
            w[i + j] = cplx(cos(PI / i * j), sin(PI / i * j));
    FFT2(a0, a1, lim), FFT2(b0, b1, lim);
    for (int i = 0; i < lim; i++) {
        cplx t = a0[i] + a1[i] * 1i;
        b0[i] *= t, b1[i] *= t;
    }
    FFT(b0, lim, 0), FFT(b1, lim, 0);
    for (int i = 0; i <= n + m; i++) {
        using u64 = uint64_t;
        u64 high = M * u64(real(b0[i]) + 0.5) + u64(imag(b0[i]) + 0.5);
        u64 low = M * u64(real(b1[i]) + 0.5) + u64(imag(b1[i]) + 0.5);
        c[i] = (high % P * M + low) % P;
    }
    for (int i = 0; i < lim; i++)
        a0[i] = a1[i] = b0[i] = b1[i] = 0;
}
```

`modint` 版

```cpp
using cplx = complex<double>;

const int N = /**/;
const double PI = acos(-1);

int rev[N];
cplx w[N];

void FFT(cplx a[], int n, bool t) {
    for (int i = 0; i < n; i++)
        if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int i = 1; i < n; i <<= 1)
        for (int j = 0; j < n; j += i << 1)
            for (int k = j; k < j + i; k++) {
                cplx x = a[k], y = a[k + i] * w[i + k - j];
                a[k] = x + y, a[k + i] = x - y;
            }
    if (t) return;
    reverse(a + 1, a + n);
    double inv = 1.0 / n;
    for (int i = 0; i < n; i++) a[i] *= inv;
}
void FFT2(cplx a[], cplx b[], int n) {
    for (int i = 0; i < n; i++) a[i] += b[i] * 1i;
    FFT(a, n, 1);
    b[0] = conj(a[0]);
    for (int i = 1; i < n; i++) b[i] = conj(a[n - i]);
    for (int i = 0; i < n; i++) {
        cplx x = a[i], y = b[i];
        a[i] = (y + x) * 0.5, b[i] = (y - x) * 0.5i;
    }
}
void mul(int n, int m, mint a[], mint b[], mint c[]) {
    static cplx a0[N], a1[N], b0[N], b1[N];
    int M = sqrt(mint::mod());
    for (int i = 0; i <= n; i++)
        a0[i] = a[i].val() / M, a1[i] = a[i].val() % M;
    for (int i = 0; i <= m; i++)
        b0[i] = b[i].val() / M, b1[i] = b[i].val() % M;
    int lim = 2 << __builtin_ia32_bsrsi(n + m);
    for (int i = 0; i < lim; i++)
        rev[i] = rev[i >> 1] >> 1 | (i & 1 ? lim >> 1 : 0);
    for (int i = 1; i < lim; i <<= 1)
        for (int j = 0; j < i; j++)
            w[i + j] = cplx(cos(PI / i * j), sin(PI / i * j));
    FFT2(a0, a1, lim), FFT2(b0, b1, lim);
    for (int i = 0; i < lim; i++) {
        cplx t = a0[i] + a1[i] * 1i;
        b0[i] *= t, b1[i] *= t;
    }
    FFT(b0, lim, 0), FFT(b1, lim, 0);
    for (int i = 0; i <= n + m; i++) {
        using u64 = uint64_t;
        u64 high = M * u64(real(b0[i]) + 0.5) + u64(imag(b0[i]) + 0.5);
        u64 low = M * u64(real(b1[i]) + 0.5) + u64(imag(b1[i]) + 0.5);
        c[i] = high % mint::mod() * M + low;
    }
    for (int i = 0; i < lim; i++)
        a0[i] = a1[i] = b0[i] = b1[i] = 0;
}
```

## 多项式求逆 / polyInv

```cpp
const int P = 998244353;
const int N = /**/;

inline int add(int a, int b) { return (a += b) < P ? a : a - P; }
inline int sub(int a, int b) { return (a -= b) < 0 ? a + P : a; }
inline int mul(int a, int b) { return (uint64_t)(uint32_t)a * (uint32_t)b % P; }
inline int ceil2(int n) { return 2 << __builtin_ia32_bsrsi(n); }
int Pow(int a, int n) {
    int r = 1;
    for (; n; n >>= 1, a = mul(a, a))
        if (n & 1) r = mul(r, a);
    return r;
}
struct precalc {
    int w[23], iw[23];
    precalc() {
        int e[22], ie[22], now = 15311432, inow = 469870224;
        for (int i = 21; i >= 0; i--) {
            e[i] = now, ie[i] = inow;
            now = mul(now, now), inow = mul(inow, inow);
        }
        now = inow = 1;
        for (int i = 0; i <= 21; i++) {
            w[i] = mul(e[i], now), iw[i] = mul(ie[i], inow);
            now = mul(now, ie[i]), inow = mul(inow, e[i]);
        }
    }
} pre;
void DIF(int a[], int n) {
    for (int i = n >> 1, l = 1; i; i >>= 1, l <<= 1) {
        int now = 1;
        for (int j = 0; j < l; j++) {
            int p = j * i * 2;
            for (int k = p; k < p + i; k++) {
                int x = a[k], y = mul(a[k + i], now);
                a[k] = add(x, y), a[k + i] = sub(x, y);
            }
            now = mul(now, pre.w[__builtin_ctz(j + 1)]);
        }
    }
}
void IDIF(int a[], int n) {
    for (int i = 1, l = n >> 1; l; i <<= 1, l >>= 1) {
        int now = 1;
        for (int j = 0; j < l; j++) {
            int p = j * i * 2;
            for (int k = p; k < p + i; k++) {
                int x = a[k], y = a[k + i];
                a[k] = add(x, y), a[k + i] = mul(x - y + P, now);
            }
            now = mul(now, pre.iw[__builtin_ctz(j + 1)]);
        }
    }
    int inv = Pow(n, P - 2);
    for (int i = 0; i < n; i++) a[i] = mul(a[i], inv);
}
void polyInv(int n, int a[], int b[]) {
    static int c[N];
    int lim = 2 << __builtin_ia32_bsrsi(n - 1);
    memset(b, 0, lim * 8);
    b[0] = 1;
    for (int k = 1; k < lim; k <<= 1) {
        memcpy(c, a, k * 8);
        DIF(b, k * 4), DIF(c, k * 4);
        for (int i = 0; i < k * 4; i++)
            b[i] = mul(b[i], 2 + P - mul(c[i], b[i]));
        IDIF(b, k * 4), memset(b + k * 2, 0, k * 8);
    }
    memset(c, 0, lim * 8);
}
```

`modint` 版

```cpp
using mint = static_modint<998244353>;
const int N = /**/;

inline int ceil2(int n) { return 2 << __builtin_ia32_bsrsi(n); }
struct precalc {
    mint w[23], iw[23];
    precalc() {
        mint e[22], ie[22], now = 15311432, inow = 469870224;
        for (int i = 21; i >= 0; i--) {
            e[i] = now, ie[i] = inow;
            now *= now, inow *= inow;
        }
        now = inow = 1;
        for (int i = 0; i <= 21; i++) {
            w[i] = e[i] * now, iw[i] = ie[i] * inow;
            now *= ie[i], inow *= e[i];
        }
    }
} pre;
void DIF(mint a[], int n) {
    for (int i = n >> 1, l = 1; i; i >>= 1, l <<= 1) {
        mint now = 1;
        for (int j = 0; j < l; j++) {
            int p = j * i * 2;
            for (int k = p; k < p + i; k++) {
                mint x = a[k], y = a[k + i] * now;
                a[k] = x + y, a[k + i] = x - y;
            }
            now *= pre.w[__builtin_ctz(j + 1)];
        }
    }
}
void IDIF(mint a[], int n) {
    for (int i = 1, l = n >> 1; l; i <<= 1, l >>= 1) {
        mint now = 1;
        for (int j = 0; j < l; j++) {
            int p = j * i * 2;
            for (int k = p; k < p + i; k++) {
                mint x = a[k], y = a[k + i];
                a[k] = x + y, a[k + i] = (x - y) * now;
            }
            now *= pre.iw[__builtin_ctz(j + 1)];
        }
    }
    mint inv = mint(n).inv();
    for (int i = 0; i < n; i++) a[i] *= inv;
}
void polyInv(int n, mint a[], mint b[]) {
    static mint c[N];
    int lim = 2 << __builtin_ia32_bsrsi(n - 1);
    memset(b, 0, lim * 8);
    b[0] = 1;
    for (int k = 1; k < lim; k <<= 1) {
        memcpy(c, a, k * 8);
        DIF(b, k * 4), DIF(c, k * 4);
        for (int i = 0; i < k * 4; i++)
            b[i] *= 2 - c[i] * b[i];
        IDIF(b, k * 4), memset(b + k * 2, 0, k * 8);
    }
    memset(c, 0, lim * 8);
}
```

## 自然数等幂求和

## 中国剩余定理

## 扩展中国剩余定理 / exCRT

```cpp
ll mul(ll A, ll B, ll P) {
    ll C = A * B - (ll)((long double)A * B / P + 0.5) * P;
    return C < 0 ? C + P : C;
}
void exgcd(ll a, ll b, ll& d, ll& x, ll& y) {
    if (!b) { d = a, x = 1, y = 0; return; }
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
        if (mp.count(rs * pls % p)) return i * m - mp[rs * pls % p];
    }
    return -1;
}
int exBSGS(int a, int b, int p) {
    a %= p, b %= p;
    int pls = 1, cnt = 0, g;
    while ((g = __gcd(a, p)) > 1) {
        if (b == pls) return cnt;
        if (b % g) return -1;
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
struct cplx {
    ll r, i;
    cplx operator *(const cplx& b) {
        return {(r * b.r + i * b.i % P * II) % P, (r * b.i + i * b.r) % P};
    }
} U = {1, 0};
```

`modint` 版

```cpp
mint n, II;
struct cplx {
    mint r, i;
    cplx operator *(const cplx& b) {
        return {r * b.r + i * b.i * II, r * b.i + i * b.r};
    }
} U = {1, 0};
```

### 函数

```cpp
int pow1(ll a, int n, ll r = 1) {
    for (; n; n >>= 1, a = a * a % P)
    if (n & 1) r = r * a % P;
    return r;
}
cplx pow2(cplx a, int n, cplx r = U) {
    for (; n; n >>= 1, a = a * a)
    if (n & 1) r = r * a;
    return r;
}
int cipolla(int n) {
    if (!n) return 0;
    if (P == 2) return n;
    if (pow1(n, P / 2) != 1) return -1;
    static mt19937 gen;
    ll a;
    do a = gen() % P, II = (a * a - n + P) % P; while (pow1(II, P / 2) == 1);
    return pow2({a, 1}, P / 2 + 1).r;
}
```

`modint` 版

```cpp
cplx Pow(cplx a, int n, cplx r = U) {
    for (; n; n >>= 1, a = a * a)
    if (n & 1) r = r * a;
    return r;
}
int cipolla(mint n) {
    if (n == 0) return 0;
    if (P == 2) return n.val();
    if (n.pow(P / 2) != 1) return -1;
    mint a;
    do a = rand(), II = a * a - n; while (a == 0 || II.pow(P / 2) == 1);
    return Pow({a, 1}, P / 2 + 1).r.val();
}
```

## Miller Rabin & Pollard Rho

### 定义

```cpp
vector<ll> prime;
```

### 函数

```cpp
inline ll mul(ll a, ll b, ll p) {
    ll c = a * b - ll((long double)a * b / p + 0.5) * p;
    return c < 0 ? c + p : c;
}
ll Pow(ll a, ll n, ll p) {
    ll r = 1;
    for (; n; n >>= 1, a = mul(a, a, p))
        if (n & 1) r = mul(r, a, p);
    return r;
}
bool check(ll n) {
    ll d = n - 1 >> __builtin_ctzll(n - 1);
    for (ll a : {2, 3, 5, 7, 11, 13, 17, 19, 23}) {
        if (a == n) return 1;
        ll t = d, x = Pow(a, d, n);
        if (x == 1) continue;
        while (t != n - 1 & x != 1 & x != n - 1)
            x = mul(x, x, n), t <<= 1;
        if (x != n - 1) return 0;
    }
    return 1;
}
ll find(ll n) {
    static mt19937_64 gen;
    ll x = gen() % n;
    for (int L = 128;; L <<= 1) {
        ll y = x;
        for (int i = 0; i < L; i += 128) {
            ll z = x, p = 1;
            rep(j, 0, 127) {
                x = mul(x, x, n) + 1;
                p = mul(p, abs(x - y), n);
            }
            if (gcd(p, n) == 1) continue;
            rep(j, 0, 127) {
                z = mul(z, z, n) + 1;
                ll g = gcd(abs(z - y), n);
                if (g > 1) return g;
            }
        }
    }
}
void factorize(ll n) {
    if (n == 1) return;
    if (check(n)) return prime.pb(n);
    ll d;
    do d = find(n); while (d == n);
    factorize(d), factorize(n / d);
}
```

# 数据结构

## 动态树 / LCT

### 普通版

#### 定义

```cpp
struct { int c[2], f; bool r; } c[N];
```

#### 函数

```cpp
inline bool id(int o) { return c[c[o].f].c[1] == o; }
inline bool nrt(int o) { return c[c[o].f].c[0] == o | c[c[o].f].c[1] == o; }
inline void pu(int o) {

}
inline void rev(int o) {
    swap(c[o].c[0], c[o].c[1]), c[o].r ^= 1;
}
inline void pd(int o) {
    if (c[o].r) rev(c[o].c[0]), rev(c[o].c[1]), c[o].r = 0;
}
void rot(int o, int d) {
    int k = c[o].c[!d], &x = c[k].c[d];
    if (nrt(o)) c[c[o].f].c[id(o)] = k;
    c[k].f = c[o].f;
    pu(x = c[c[o].c[!d] = x].f = o);
    pu(c[o].f = k);
}
void dfs(int o) { if (nrt(o)) dfs(c[o].f); pd(o); }
void splay(int o) {
    dfs(o);
    for (int f; nrt(o); rot(c[o].f, !id(o)))
    if (nrt(f = c[o].f)) rot(id(o) == id(f) ? c[f].f : f, !id(o));
}
int acc(int o) {
    int x = 0;
    for (; o; o = c[x = o].f) splay(o), c[o].c[1] = x, pu(o);
    return x;
}
void link(int u, int v) {
    rev(acc(u)), splay(u), c[u].f = v;
}
void cut(int u, int v) {
    rev(acc(u)), acc(v), splay(v), c[v].c[0] = c[u].f = 0, pu(v);
}
```

### 维护子树 ```size``` 版

#### 定义

```cpp
struct { int c[2], f, s, si; bool r; } c[N];
```

#### 函数

```cpp
inline bool id(int o) { return c[c[o].f].c[1] == o; }
inline bool nrt(int o) { return c[c[o].f].c[0] == o | c[c[o].f].c[1] == o; }
inline void pu(int o) {
    c[o].s = c[c[o].c[0]].s + c[c[o].c[1]].s + c[o].si + 1;
}
inline void rev(int o) {
    swap(c[o].c[0], c[o].c[1]), c[o].r ^= 1;
}
inline void pd(int o) {
    if (c[o].r) rev(c[o].c[0]), rev(c[o].c[1]), c[o].r = 0;
}
void rot(int o, int d) {
    int k = c[o].c[!d], &x = c[k].c[d];
    if (nrt(o)) c[c[o].f].c[id(o)] = k;
    c[k].f = c[o].f;
    pu(x = c[c[o].c[!d] = x].f = o);
    pu(c[o].f = k);
}
void dfs(int o) { if (nrt(o)) dfs(c[o].f); pd(o); }
void splay(int o) {
    dfs(o);
    for (int f; nrt(o); rot(c[o].f, !id(o)))
    if (nrt(f = c[o].f)) rot(id(o) == id(f) ? c[f].f : f, !id(o));
}
int acc(int o) {
    int x = 0;
    for (; o; o = c[x = o].f) {
        splay(o);
        c[o].si += c[c[o].c[1]].s - c[x].s;
        c[o].c[1] = x, pu(o);
    }
    return x;
}
void link(int u, int v) {
    rev(acc(u)), c[u].f = v, c[v].s += c[u].s, c[v].si += c[u].s;
}
void cut(int u, int v) {
    rev(acc(u)), acc(v), splay(v), c[v][0] = f[u] = 0, pu(v);
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
#define pre(T, x) (*prev((T).lower_bound(x)))
#define nxt(T, x) (*(T).upper_bound(x))
#define bld(T, l, r) (T).copy_from_range(l, r)
```

## RBST

### 普通版

#### 定义

```cpp
struct { int l, r, s; } c[N];
```

#### 函数

```cpp
inline void pu(int o) {
    c[o].s = c[c[o].l].s + c[c[o].r].s + 1;
}
void split(int o, int k, int& u, int& v) {
    if (!o) u = v = 0;
    else if (k <= c[c[o].l].s) split(c[v = o].l, k, u, c[o].l), pu(o);
    else split(c[u = o].r, k - c[c[o].l].s - 1, c[o].r, v), pu(o);
}
int merge(int u, int v) {
    static mt19937 gen;
    if (!u || !v) return u + v;
    if (gen() % (c[u].s + c[v].s) < c[u].s)
        return c[u].r = merge(c[u].r, v), pu(u), u;
    return c[v].l = merge(u, c[v].l), pu(v), v;
}
int build(int l, int r) {
    if (l > r) return 0;
    int m = l + r >> 1;
    c[m].s = r - l + 1;
    c[m].l = build(l, m - 1);
    c[m].r = build(m + 1, r);
    return m;
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
    bool operator <(const node& rhs)const {
        return x[D] < rhs.x[D];
    }
} a[N];
```

### 函数

```cpp
double sq(double x) { return x * x; }
void pu(int o) {
    For(i, 0, K) {
        L[o][i] = min(L[lc][i], L[rc][i]);
        R[o][i] = max(R[lc][i], R[rc][i]);
    }
}
void build(int l, int r, int o) {
    if (l == r) {
        For(i, 0, K) L[o][i] = R[o][i] = a[l].x[i];
        return;
    }
    double va[K] = {};
    For(i, 0, K) {
        double av = 0;
        rep(j, l, r) av += a[j].x[i];
        av /= r - l + 1;
        rep(j, l, r) va[i] += sq(a[j].x[i] - av);
    }
    D = max_element(va, va + K) - va;
    nth_element(a + l, a + mid, a + r + 1);
    build(lch), build(rch), pu(o);
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
    if (l == r) return void(res = mine(o, x));
    ll dl = mine(lc, x), dr = mine(rc, x);
    if (dl < dr) {
        if (dl < res) qry(x, lch);
        if (dr < res) qry(x, rch);
    } else {
        if (dr < res) qry(x, rch);
        if (dl < res) qry(x, lch);
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
    if (l == r) return void(res = maxe(o, x));
    ll dl = maxe(lc, x), dr = maxe(rc, x);
    if (dl > dr) {
        if (dl > res) qry(x, lch);
        if (dr > res) qry(x, rch);
    } else {
        if (dr > res) qry(x, rch);
        if (dl > res) qry(x, lch);
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

# 图论和树

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
    for (int v : G[u]) if (v != fa[0][u])
        d[v] = d[u] + 1, fa[0][v] = u, dfs(v);
}
int lca(int u, int v) {
    if (d[u] < d[v]) swap(u, v);
    rep(i, 0, 19) if (d[u] - d[v] >> i & 1) u = fa[i][u];
    if (u == v) return u;
    per(i, 19, 0) if (fa[i][u] != fa[i][v]) u = fa[i][u], v = fa[i][v];
    return fa[0][u];
}
void bld(int k) {
    sort(a + 1, a + k + 1, [](int a, int b) { return dfn[a] < dfn[b]; });
    stk[++tp] = 1;
    rep(i, 1, k) if (a[i] > 1) {
        int x = lca(a[i], stk[tp]);
        while (d[stk[tp]] > d[x]) tp--, T[d[stk[tp]] > d[x] ? stk[tp] : x].pb(stk[tp + 1]);
        if (stk[tp] != x) stk[++tp] = x;
        stk[++tp] = a[i];
    }
    while (--tp) T[stk[tp]].pb(stk[tp + 1]);
}
void clr(int u) {
    for (int v : G[u]) clr(v); T[u].clear();
}
```

### 快很多版

#### 定义

```cpp
int n, a[N], p, stk[N];
int d[N], fa[N], sz[N], son[N], tp[N], idx, dfn[N];
vector<int> G[N];
int eid, he[N];
struct edge { int v, n; } e[N << 1];
```

#### 函数

```cpp
void dfs1(int u) {
    sz[u] = 1;
    for (int v : G[u]) {
        d[v] = d[u] + 1, dfs1(v), sz[u] += sz[v];
        if (sz[v] > sz[son[u]]) son[u] = v;
    }
}
void dfs2(int u, int top) {
    tp[u] = top, dfn[u] = ++idx;
    if (son[u]) dfs2(son[u], top);
    for (int v : G[u]) if (v != son[u]) dfs2(v, v);
}
int lca(int u, int v) {
    for (; tp[u] != tp[v]; d[tp[u]] > d[tp[v]] ? u = fa[tp[u]] : v = fa[tp[v]]);
    return d[u] < d[v] ? u : v;
}
inline void add(int u, int v) {
    e[++eid] = {v, he[u]}, he[u] = eid;
}
void bld(int k) {
    sort(a + 1, a + k + 1, [](int a, int b) { return dfn[a] < dfn[b]; });
    stk[++p] = 1;
    rep(i, 1, k) if (a[i] > 1) {
        int x = lca(a[i], stk[p]);
        while (d[stk[p]] > d[x])
            p--, add(d[stk[p]] > d[x] ? stk[p] : x, stk[p + 1]);
        if (stk[p] != x) stk[++p] = x;
        stk[++p] = a[i];
    }
    while (--p) add(stk[p], stk[p + 1]);
}
```

## 最大流 / mf

### 定义

```cpp
const int N = /**/, M = /**/;

int eid, deg[N], G[M << 1], cur[N], d[N];
struct { int v, c; } e[M << 1];
```

### 函数

```cpp
inline void add(int u, int v, int c) {
    deg[u]++, e[eid++] = {v, c};
    deg[v]++, e[eid++] = {u, 0};
}
bool bfs(int s, int t) {
    memcpy(cur, deg, sizeof deg);
    memset(d, 0x3f, sizeof d);
    static int q[N];
    d[t] = 0, q[0] = t;
    for (int L = 0, R = 0; L <= R; L++) {
        int u = q[L];
        for (int i = deg[u]; i != deg[u + 1]; i++) {
            int k = G[i], v = e[k].v;
            if (d[v] != 0x3f3f3f3f | !e[k ^ 1].c) continue;
            d[v] = d[u] + 1;
            if (v == s) return true;
            q[++R] = v;
        }
    }
    return false;
}
int dfs(int u, int t, int lim) {
    if (u == t) return lim;
    int dis = d[u], res = 0;
    for (int& i = cur[u]; i != deg[u + 1]; i++) {
        int k = G[i], v = e[k].v, c = e[k].c;
        if (!c | d[v] >= dis) continue;
        int f = dfs(v, t, min(lim, c));
        e[k].c -= f, e[k ^ 1].c += f;
        res += f, lim -= f;
        if (!lim) return res;
    }
    d[u] = 0x3f3f3f3f;
    return res;
}
int maxFlow(int n, int s, int t) {
    for (int i = 1; i <= n; i++) deg[i + 1] += deg[i];
    for (int i = 0; i < eid; i += 2) {
        G[--deg[e[i + 1].v]] = i;
        G[--deg[e[i].v]] = i + 1;
    }
    int res = 0;
    while (bfs(s, t))
        for (int i = deg[s]; i != deg[s + 1]; i++) {
            int k = G[i], f = dfs(e[k].v, t, e[k].c);
            res += f, e[k].c -= f, e[k ^ 1].c += f;
        }
    return res;
}
```

## 最小费用最大流 / mcf

### 原始对偶算法（单路增广）

#### 定义

```cpp
const int N = /**/, M = /**/, SZ = /**/;

int eid, deg[N], G[M << 1], d[N], h[N], pre[N];
struct { int v, c, w; } e[M << 1];
uint64_t c[SZ << 1];
```

#### 函数

```cpp

inline void add(int u, int v, int c, int w) {
    deg[u]++, e[eid++] = {v, c, w};
    deg[v]++, e[eid++] = {u, 0, -w};
}
inline void push(int i, uint32_t v0) {
    auto v = (uint64_t)v0 << 32 | i;
    for (i += SZ; i; i >>= 1) c[i] = min(c[i], v);
}
inline void pop(int i) {
    uint64_t v = -1;
    for (i += SZ; i; i >>= 1) c[i] = v, v = min(v, c[i ^ 1]);
}
bool dijkstra(int s, int t) {
    memset(c, -1, sizeof c);
    memset(d, 0x3f, sizeof d);
    d[s] = 0, push(s, 0);
    while (~c[1]) {
        int u = c[1];
        auto dis = d[u] + h[u];
        pop(u);
        for (int i = deg[u]; i != deg[u + 1]; i++) {
            int k = G[i], v = e[k].v;
            if (!e[k].c) continue;
            auto w = dis + e[k].w - h[v];
            if (w < d[v]) pre[v] = k ^ 1, d[v] = w, push(v, w);
        }
    }
    return d[t] < d[0];
}
pair<int, int> minCostFlow(int n, int s, int t) {
    for (int i = 1; i <= n; i++) deg[i + 1] += deg[i];
    for (int i = 0; i < eid; i += 2) {
        G[--deg[e[i + 1].v]] = i;
        G[--deg[e[i].v]] = i + 1;
    }
    int flow = 0, cost = 0;
    while (dijkstra(s, t)) {
        int f = 0x3f3f3f3f;
        for (int u = t, k; u != s; u = e[k].v)
            k = pre[u], f = min(f, e[k ^ 1].c);
        for (int u = t, k; u != s; u = e[k].v)
            k = pre[u], e[k ^ 1].c -= f, e[k].c += f;
        flow += f, cost += f * h[t];
    }
    return {flow, cost};
}
```

### 原始对偶算法（多路增广）

#### 定义

```cpp
const int N = /**/, M = /**/, SZ = /**/;

int eid, deg[N], G[M << 1], d[N], h[N], pre[N];
struct { int v, c, w; } e[M << 1];
uint64_t c[SZ << 1];
int deg0[N], G0[M << 1];
```

#### 函数

```cpp
inline void add(int u, int v, int c, int w) {
    deg[u]++, e[eid++] = {v, c, w};
    deg[v]++, e[eid++] = {u, 0, -w};
}
inline void push(int i, uint32_t v0) {
    auto v = (uint64_t)v0 << 32 | i;
    for (i += SZ; i; i >>= 1) c[i] = min(c[i], v);
}
inline void pop(int i) {
    uint64_t v = -1;
    for (i += SZ; i; i >>= 1) c[i] = v, v = min(v, c[i ^ 1]);
}
bool dijkstra(int s, int t) {
    memset(c, -1, sizeof c);
    memset(d, 0x3f, sizeof d);
    d[s] = 0, push(s, 0);
    while (~c[1]) {
        int u = c[1];
        auto dis = d[u] + h[u];
        pop(u);
        for (int i = deg[u]; i != deg[u + 1]; i++) {
            int k = G[i], v = e[k].v;
            if (!e[k].c) continue;
            auto w = dis + e[k].w - h[v];
            if (w < d[v]) pre[v] = k ^ 1, d[v] = w, push(v, w);
        }
    }
    return d[t] < d[0];
}
bool bfs(int s, int t) {
    static int q[N];
    memset(pre, -1, sizeof pre);
    pre[s] = 0, q[0] = s;
    for (int l = 0, r = 0; l <= r; l++) {
        int u = q[l];
        for (int i = deg0[u]; i != deg0[u + 1]; i++) {
            int k = G0[i], v = e[k].v;
            if ((pre[v] < 0) & (e[k].c != 0)) {
                pre[v] = k ^ 1, q[++r] = v;
                if (v == t) return 1;
            }
        }
    }
    return 0;
}
pair<int, int> minCostFlow(int n, int s, int t) {
    for (int i = 1; i <= n; i++) deg[i + 1] += deg[i];
    for (int i = 0; i < eid; i += 2) {
        G[--deg[e[i + 1].v]] = i;
        G[--deg[e[i].v]] = i + 1;
    }
    int flow = 0, cost = 0;
    while (dijkstra(s, t)) {
        for (int i = 1; i <= n; i++) h[i] += d[i];
        for (int i = 0; i < eid; i += 2) {
            int u = e[i + 1].v, v = e[i].v;
            if (e[i].w == h[v] - h[u])
                deg0[u]++, deg0[v]++;
        }
        for (int i = 1; i <= n; i++) deg0[i + 1] += deg0[i];
        for (int i = 0; i < eid; i += 2) {
            int u = e[i + 1].v, v = e[i].v;
            if (e[i].w == h[v] - h[u])
                G0[--deg0[u]] = i, G0[--deg0[v]] = i + 1;
        }
        do {
            int f = 0x3f3f3f3f;
            for (int u = t, k; u != s; u = e[k].v)
                k = pre[u], f = min(f, e[k ^ 1].c);
            for (int u = t, k; u != s; u = e[k].v)
                k = pre[u], e[k ^ 1].c -= f, e[k].c += f;
            flow += f, cost += f * h[t];
        } while (bfs(s, t));
        memset(deg0, 0, sizeof deg0);
    }
    return {flow, cost};
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
bool dfs(int u, int s) {
    if (vis[u] == s) return 0;
    vis[u] = s;
    for (int v : G[u]) if (!mat[v] || dfs(mat[v], s)) return mat[v] = u, 1;
    return 0;
}
```

### 使用

```cpp
int as = 0;
rep(i, 1, n) ans += dfs(i, i);
```

## 2-SAT 问题

求字典序最小的解，随机数据是 $O(N+M)$，最坏复杂度 $O(NM)$。

### 定义

```cpp
int n, m, co[N << 1], stk[N << 1], tp;
vector<int> G[N << 1];
```

### 函数

```cpp
inline void add(int u, int a, int v, int b) {
    G[u * 2 + !a].pb(v * 2 + b);
    G[v * 2 + !b].pb(u * 2 + a);
}
int dfs(int u) {
    if (co[u] | co[u ^ 1]) return co[u];
    co[u] = 1, stk[++tp] = u;
    for (int v : G[u]) if (!dfs(v)) return 0;
    return 1;
}
int twoSat() {
    rep(i, 1, n) {
        if (!co[i * 2] && !co[i * 2 + 1] && !dfs(i * 2)) {
            while (tp) co[stk[tp--]] = 0;
            if (!dfs(i * 2 + 1)) return 0;
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
    int& j = R[i] = ma > i ? min(R[p * 2 - i], ma - i) : 0;
    while (s[i - j] == s[i + j + 1]) j++;
    if (i + j > ma) ma = i + j, p = i;
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
        while (s[i - len[o] - 1] != s[i]) o = f[o];
        return o;
    };
    int o = jmp(nw), c = s[i] - 97;
    if (!ch[o][c]) {
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
        while (o && s[i - len[o] - 1] != s[i]) o = f[o];
        return o;
    };
    int o = jmp(nw), c = s[i] - 97;
    if (!ch[o][c]) {
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
int n, sa[N], rk[N], tp[N], px[N], buc[N];
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
    for (int k = 1, p; memset(buc, p = 0, m + 1 << 2); k <<= 1) {
        rep(i, n - k + 1, n) p++, px[p] = rk[tp[p] = i];
        rep(i, 1, n) if (sa[i] > k) p++, px[p] = rk[tp[p] = sa[i] - k];
        rep(i, 1, n) buc[rk[i]]++;
        rep(i, 1, m) buc[i] += buc[i - 1];
        per(i, n, 1) sa[buc[px[i]]--] = tp[i];
        memcpy(tp, rk, n + 1 << 2), p = 0;
        rep(i, 1, n) {
            int a = sa[i], b = sa[i - 1];
            rk[a] = p += tp[a] != tp[b] || tp[a + k] != tp[b + k];
        }
        if ((m = p) >= n) break;
    }
}
void height() {
    int p = 0;
    rep(i, 1, n) {
        for (p && p--; s[i + p] == s[sa[rk[i] - 1] + p]; p++);
        h[0][rk[i]] = p;
    }
    rep(i, 1, 19) rep(j, 1 << i, n)
        h[i][j] = min(h[i - 1][j], h[i - 1][j - (1 << i - 1)]);
}
```

## 后缀自动机 / SAM

### 定义

```cpp
char s[N];
int n, sz = 1, nw = 1, f[N << 1], len[N << 1], ch[N << 1][26];
```

### 函数

```cpp
void ins(int c) {
    int u = ++sz;
    len[u] = len[nw] + 1;
    while (nw && !ch[nw][c]) ch[nw][c] = u, nw = f[nw];
    if (!nw) f[u] = 1;
    else {
        int v = ch[nw][c];
        if (len[v] > len[nw] + 1) {
            f[++sz] = f[v], memcpy(ch[sz], ch[v], sizeof ch[v]);
            f[v] = f[u] = sz, len[sz] = len[nw] + 1;
            while (ch[nw][c] == v) ch[nw][c] = sz, nw = f[nw];
        } else f[u] = v;
    }
    nw = u;
}
```

## 最小表示法 / smallest cyclic shift

### 函数

```cpp
int calc(char s[]) {
    int i = 0, j = 1, k = 0;
    while (max({i, j, k}) < n)
    if (s[(i + k) % n] == s[(j + k) % n]) k++;
    else {
        if (s[(i + k) % n] > s[(j + k) % n]) swap(i, j);
        j += k + 1, k = 0;
        if (i == j) j++;
    }
    return min(i, j);
}
```

## 最小后缀 / smallest suffix

### 函数

```cpp
int calc(char s[]) {
    int i = 0, j = 1;
    while (s[j]) {
        int k = 0;
        while (s[i + k] == s[j + k]) k++;
        if (!s[j + k]) i = max(i + k, j + 1), swap(i, j);
        else if (s[i + k] > s[j + k]) i = max(i + k, j) + 1, swap(i, j);
        else j += k + 1;
    }
    return i;
}
```
