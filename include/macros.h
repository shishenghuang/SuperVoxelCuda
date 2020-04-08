//
// Copyright 2011 Yangbin Lin. All Rights Reserved.
//
// Author: yblin@jmu.edu.cn (Yangbin Lin)
//
// This file is part of the Code Library.
//
// Description: macros of Code Library.
//

#ifndef BASE_MACROS_H_
#define BASE_MACROS_H_

/**
 * A macro to disallow the copy constructor and operator= functions.
 * This should be used in the private: declarations for a class.
 */
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
        TypeName(const TypeName&);         \
        void operator=(const TypeName&)

/**
 * Use this annotation at the end of a struct/class definition to prevent the
 * compiler from optimizing away instances that are never used.
 *
 * This is useful when all interesting logic happens inside the constructor and
 * destructor.
 *
 * Usage:
 *
 * struct Foo {
 *     Foo() { ... }
 * } ATTRIBUTE_UNUSED;
 *
 * Also use it after a variable or parameter declaration to tell the compiler
 * the variable/parameter does not have to be used.
 */
#if defined(__GNUC__) && !defined(COMPILER_ICC)
# define ATTRIBUTE_UNUSED __attribute__ ((unused))
#else
# define ATTRIBUTE_UNUSED
#endif

/**
 * @return the integer i such as 2^i <= n < 2^(i+1).
 *
 * Equal to n_bits - __builtin_clz(n)
 */
template <typename IntType>
int Log2Floor(IntType n) {
    assert(n > 0);

    int start = 0;
    switch (std::numeric_limits<IntType>::digits) {
    case 7:
    case 8:
        start = 2;
        break;
    case 15:
    case 16:
        start = 3;
        break;
    case 31:
    case 32:
        start = 4;
        break;
    case 63:
    case 64:
        start = 5;
        break;
    default:
        assert(false && "The input type is not integer.");
    }

    int log = 0;
    for (int i = start; i >= 0; --i) {
        int shift = (1 << i);
        IntType x = n >> shift;
        if (x != 0) {
            n = x;
            log += shift;
        }
    }

    return log;
}

#endif // BASE_MACROS_H_
