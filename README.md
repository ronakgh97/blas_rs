Rewriting **BLAS FORTRAN 77 Prototype** Kernels in modern rust. `ONLY x86_64`

> This will not cover all kernels for every single BLAS routine, but the most commonly used ones. (excluding complex
> type)

ref:

- https://www.netlib.org/blas/
- https://icl.utk.edu/~mgates3/docs/lapack.html
- https://www.netlib.org/lapack/explore-html/
- https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.htm
- https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/blas-routines.html
- https://doc.rust-lang.org/core/arch/x86_64/index.html#functions