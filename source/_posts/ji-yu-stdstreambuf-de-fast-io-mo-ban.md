---
title: 基于 std::streambuf 的 Fast IO 模板
date: 2021-03-13 22:22:39
updated: 2021-03-13 22:22:39
tags: [模板]
---
### 读入输出非负整数版

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

**注：```main``` 函数开头请加入这 $3$ 句话**

```cpp
ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
```

其实后两句可以选择性添加，见文末。

**不可与  ```cin, cout, scanf, printf, gets, puts, getchar, putchar``` 等其他读入输出方式同时使用**

其实也不是全部不能同时使用，见文末。

**```read``` 函数只能通过文件输入或者标准输入完后按 ```Ctrl + Z``` （仅 ```WIN 10```）。**

### 读入输出整数版

```cpp
int read() {
    const int M = 1e6;
    static streambuf* in = cin.rdbuf();
    #define gc (p1 == p2 && (p2 = (p1 = buf) + in -> sgetn(buf, M), p1 == p2) ? -1 : *p1++)
    static char buf[M], *p1, *p2;
    int c = gc, r = 0, f = 1;
    while(c < 48) { if(c == 45) f = -1; c = gc; }
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

### 文末补充

- 在使用 ```read``` 函数时要加 ```cin.tie(0)```，在使用 ```wrt``` 函数时要加 ```cout.tie(0)```，但 ```ios::sync_with_stdio(0)``` **必须加。**

- 两个函数不可与 ```scanf, printf, gets, puts, getchar, putchar``` 等 ```stdio``` 的读入输出方式同时使用。

  在 ```read``` 函数第一次调用前可以随便使用 ```cin``` 等 ```streambuf```  的读入方式。

   ```wrt``` 函数可以与 ```cout``` 等 ```streambuf```  的输出方式随便交叉使用。