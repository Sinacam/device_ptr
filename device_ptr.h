#ifndef DEVICE_PTR_H_INCLUDED
#define DEVICE_PTR_H_INCLUDED

/*
device_ptr is a simple wrapper around a pointer, used in CUDA code to mark the pointer as a device pointer.
Given that T* is an object pointer, device_ptr<T> is identical to a T* except
    1. operator*, operator->, operator[] can only be used on device.
    2. Converting from T* is required to be explicit.
    3. Casting to T* and void* is required to be explicit.
    4. Casting to bool is required to be explicit.
    5. Can't be initialized with NULL (use nullptr instead).
There is also a free function get() to fetch T*.

Reinterpreting device_ptr<T> as device_ptr<U> is impossible, it has to round-trip through actual pointers
    device_ptr<T> d1 = // ...
    device_ptr<U> d2{(U*)get(d1)};
    device_ptr    d3{(U*)get(d1)};  // with CTAD
*/

#include<utility>
#include<cstdint>
#include<iterator>

#ifndef __host__
#define __host__
#define UNDEF_HOST
#endif

#ifndef __device__
#define __device__
#define UNDEF_DEVICE
#endif

namespace detail
{
    using sfinae = int;
#if __cplusplus >= 201703L
    using iterator_category = std::contiguous_iterator_tag;
#else
    using iterator_category = std::random_access_iterator_tag;
#endif
}

template<typename T, typename = detail::sfinae>
class device_ptr
{
    template<typename, typename>
    friend class device_ptr;
public:
    using iterator_category = detail::iterator_category;
    using iterator_concept = iterator_category;
    using value_type = T;
    using reference = T&;
    using pointer = T*;
    using difference_type = std::ptrdiff_t;

    __host__ __device__
    device_ptr() = default;
    __host__ __device__
    device_ptr(decltype(nullptr)) : ptr{nullptr} {}
    __host__ __device__
    explicit device_ptr(T* ptr) : ptr{ptr} {}

    template<typename U = detail::sfinae, std::enable_if_t<std::is_const<T>::value, U> = 0>
    __host__ __device__
    device_ptr(device_ptr<std::remove_const_t<T>> dp) : ptr{dp.ptr} {}

    template<typename Void, std::enable_if_t<std::is_void<Void>::value, int> = 0>
    __host__ __device__
    explicit device_ptr(device_ptr<Void> dp) : ptr{dp.ptr} {}
    template<typename Void, std::enable_if_t<std::is_void<Void>::value, int> = 0>
    __host__ __device__
    explicit device_ptr(device_ptr<const Void> dp) : ptr{dp.ptr} {}

    __device__
    T& operator*() const
    {
        return *ptr;
    }
    __device__
    T& operator[](std::ptrdiff_t n) const
    {
        return ptr[n];
    }
    __device__
    T* operator->() const
    {
        return ptr;
    }

    __host__ __device__
    device_ptr operator++(int)
    {
        auto tmp = *this;
        ptr++;
        return tmp;
    }
    __host__ __device__
    device_ptr& operator++()
    {
        ptr++;
        return *this;
    }
    __host__ __device__
    device_ptr& operator+=(std::ptrdiff_t n)
    {
        ptr += n;
        return *this;
    }
    __host__ __device__
    device_ptr operator+(std::ptrdiff_t n) const
    {
        auto tmp = *this;
        return tmp += n;
    }
    __host__ __device__
    device_ptr operator--(int)
    {
        auto tmp = *this;
        ptr--;
        return tmp;
    }
    __host__ __device__
    device_ptr& operator--()
    {
        ptr--;
        return *this;
    }
    __host__ __device__
    device_ptr& operator-=(std::ptrdiff_t n)
    {
        ptr -= n;
        return *this;
    }
    __host__ __device__
    device_ptr operator-(std::ptrdiff_t n) const
    {
        auto tmp = *this;
        return tmp -= n;
    }
    __host__ __device__
    std::ptrdiff_t operator-(device_ptr rhs) const
    {
        return ptr - rhs.ptr;
    }

    __host__ __device__
    friend bool operator==(device_ptr x, device_ptr y)
    {
        return x.ptr == y.ptr;
    }
    __host__ __device__
    friend bool operator!=(device_ptr x, device_ptr y)
    {
        return x.ptr != y.ptr;
    }
    __host__ __device__
    friend bool operator<(device_ptr x, device_ptr y)
    {
        return x.ptr < y.ptr;
    }
    __host__ __device__
    friend bool operator<=(device_ptr x, device_ptr y)
    {
        return x.ptr <= y.ptr;
    }
    __host__ __device__
    friend bool operator>(device_ptr x, device_ptr y)
    {
        return x.ptr > y.ptr;
    }
    __host__ __device__
    friend bool operator>=(device_ptr x, device_ptr y)
    {
        return x.ptr >= y.ptr;
    }

    template<typename Void, std::enable_if_t<std::is_void<Void>::value, int> = 0>
    __host__ __device__
    operator device_ptr<Void>() const
    {
        return device_ptr<Void>{ptr};
    }
    template<typename Integral, std::enable_if_t<std::is_integral<Integral>::value, int> = 0>
    __host__ __device__
    explicit operator Integral() const
    {
        return (Integral)ptr;
    }
    __host__ __device__
    explicit operator bool() const
    {
        return ptr;
    }
    __host__ __device__
    explicit operator T*() const
    {
        return ptr;
    }
    __host__ __device__
    explicit operator void*() const
    {
        return ptr;
    }
    __host__ __device__
    explicit operator const void*() const
    {
        return ptr;
    }

    __host__ __device__
    friend void swap(device_ptr& x, device_ptr& y)
    {
        std::swap(x.ptr, y.ptr);
    }

    __host__ __device__
    friend T* get(device_ptr x)
    {
        return x.ptr;
    }

private:
    T* ptr;
};

template<typename Void>
class device_ptr<Void, std::enable_if_t<std::is_void<std::remove_cv_t<Void>>::value, detail::sfinae>>
{
    template<typename, typename>
    friend class device_ptr;
public:

    __host__ __device__
    device_ptr() = default;
    __host__ __device__
    device_ptr(decltype(nullptr)) : ptr{nullptr} {}
    __host__ __device__
    explicit device_ptr(void* ptr) : ptr{ptr} {}

    __host__ __device__
    friend bool operator==(device_ptr x, device_ptr y)
    {
        return x.ptr == y.ptr;
    }
    __host__ __device__
    friend bool operator!=(device_ptr x, device_ptr y)
    {
        return x.ptr != y.ptr;
    }
    __host__ __device__
    friend bool operator<(device_ptr x, device_ptr y)
    {
        return x.ptr < y.ptr;
    }
    __host__ __device__
    friend bool operator<=(device_ptr x, device_ptr y)
    {
        return x.ptr <= y.ptr;
    }
    __host__ __device__
    friend bool operator>(device_ptr x, device_ptr y)
    {
        return x.ptr > y.ptr;
    }
    __host__ __device__
    friend bool operator>=(device_ptr x, device_ptr y)
    {
        return x.ptr >= y.ptr;
    }

    template<typename Integral, std::enable_if_t<std::is_integral<Integral>::value, int> = 0>
    __host__ __device__
    explicit operator Integral() const
    {
        return (Integral)ptr;
    }
    __host__ __device__
    explicit operator bool() const
    {
        return ptr;
    }
    template<typename T>
    __host__ __device__
    explicit operator T*() const
    {
        return static_cast<T*>(ptr);
    }
    template<typename T>
    __host__ __device__
    explicit operator const T*() const
    {
        return static_cast<T*>(ptr);
    }
    __host__ __device__
    explicit operator void*() const
    {
        return ptr;
    }
    __host__ __device__
    explicit operator const void*() const
    {
        return ptr;
    }

    __host__ __device__
    friend void swap(device_ptr& x, device_ptr& y)
    {
        std::swap(x.ptr, y.ptr);
    }

    __host__ __device__
    friend void* get(device_ptr x)
    {
        return x.ptr;
    }

private:
    void* ptr;
};

#ifdef UNDEF_HOST
#undef UNDEF_HOST
#undef __host__
#endif

#ifdef UNDEF_DEVICE
#undef UNDEF_DEVICE
#undef __device__
#endif



#endif
