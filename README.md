# device_ptr
device_ptr is a simple wrapper around a pointer, used in CUDA code to mark the pointer as a device pointer.
Given that `T*` is an object pointer, `device_ptr<T>` is identical to a `T*` except
1. `operator*`, `operator->`, `operator[]` can only be used on device.
2. Converting from `T*` is required to be explicit.
3. Casting to `T*` and `void*` is required to be explicit.
4. Casting to `bool` is required to be explicit.
5. Can't be initialized with `NULL` (use `nullptr` instead).

There is also a free function `get` to fetch `T*`.

Reinterpreting `device_ptr<T>` as `device_ptr<U>` is impossible, it has to round-trip through actual pointers
```c++
device_ptr<T> d1 = // ...
device_ptr<U> d2{(U*)get(d1)};
device_ptr    d3{(U*)get(d1)};  // with CTAD
 ```

## Installation
Just `#include"device_ptr.h"`, requires C++14. Forward compatible up to C++20.
