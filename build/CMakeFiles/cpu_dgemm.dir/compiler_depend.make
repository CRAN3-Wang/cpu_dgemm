# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

CMakeFiles/cpu_dgemm.dir/src/kernel_int_8x4.c.o: /home/crane/dev/cpu_dgemm/src/kernel_int_8x4.c

CMakeFiles/cpu_dgemm.dir/src/main.c.o: /home/crane/dev/cpu_dgemm/src/main.c \
  /home/crane/dev/cpu_dgemm/include/config.h \
  /home/crane/dev/cpu_dgemm/include/my_dgemm.h \
  /home/crane/dev/cpu_dgemm/include/utils.h \
  /usr/include/alloca.h \
  /usr/include/endian.h \
  /usr/include/features-time64.h \
  /usr/include/features.h \
  /usr/include/inttypes.h \
  /usr/include/math.h \
  /usr/include/stdc-predef.h \
  /usr/include/stdint.h \
  /usr/include/stdio.h \
  /usr/include/stdlib.h \
  /usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h \
  /usr/include/x86_64-linux-gnu/bits/byteswap.h \
  /usr/include/x86_64-linux-gnu/bits/endian.h \
  /usr/include/x86_64-linux-gnu/bits/endianness.h \
  /usr/include/x86_64-linux-gnu/bits/floatn-common.h \
  /usr/include/x86_64-linux-gnu/bits/floatn.h \
  /usr/include/x86_64-linux-gnu/bits/flt-eval-method.h \
  /usr/include/x86_64-linux-gnu/bits/fp-fast.h \
  /usr/include/x86_64-linux-gnu/bits/fp-logb.h \
  /usr/include/x86_64-linux-gnu/bits/libc-header-start.h \
  /usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h \
  /usr/include/x86_64-linux-gnu/bits/long-double.h \
  /usr/include/x86_64-linux-gnu/bits/math-vector.h \
  /usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h \
  /usr/include/x86_64-linux-gnu/bits/mathcalls.h \
  /usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h \
  /usr/include/x86_64-linux-gnu/bits/pthreadtypes.h \
  /usr/include/x86_64-linux-gnu/bits/select.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-intn.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-least.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-uintn.h \
  /usr/include/x86_64-linux-gnu/bits/stdio_lim.h \
  /usr/include/x86_64-linux-gnu/bits/stdlib-float.h \
  /usr/include/x86_64-linux-gnu/bits/struct_mutex.h \
  /usr/include/x86_64-linux-gnu/bits/struct_rwlock.h \
  /usr/include/x86_64-linux-gnu/bits/thread-shared-types.h \
  /usr/include/x86_64-linux-gnu/bits/time64.h \
  /usr/include/x86_64-linux-gnu/bits/timesize.h \
  /usr/include/x86_64-linux-gnu/bits/types.h \
  /usr/include/x86_64-linux-gnu/bits/types/FILE.h \
  /usr/include/x86_64-linux-gnu/bits/types/__FILE.h \
  /usr/include/x86_64-linux-gnu/bits/types/__fpos64_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__mbstate_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/clock_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/clockid_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/cookie_io_functions_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/sigset_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h \
  /usr/include/x86_64-linux-gnu/bits/types/time_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/timer_t.h \
  /usr/include/x86_64-linux-gnu/bits/typesizes.h \
  /usr/include/x86_64-linux-gnu/bits/uintn-identity.h \
  /usr/include/x86_64-linux-gnu/bits/waitflags.h \
  /usr/include/x86_64-linux-gnu/bits/waitstatus.h \
  /usr/include/x86_64-linux-gnu/bits/wchar.h \
  /usr/include/x86_64-linux-gnu/bits/wordsize.h \
  /usr/include/x86_64-linux-gnu/cblas.h \
  /usr/include/x86_64-linux-gnu/cblas_mangling.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs-64.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs.h \
  /usr/include/x86_64-linux-gnu/sys/cdefs.h \
  /usr/include/x86_64-linux-gnu/sys/select.h \
  /usr/include/x86_64-linux-gnu/sys/types.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stdarg___gnuc_va_list.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_max_align_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_null.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_offsetof.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_ptrdiff_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_size_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_wchar_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/inttypes.h \
  /usr/lib/llvm-18/lib/clang/18/include/stdarg.h \
  /usr/lib/llvm-18/lib/clang/18/include/stddef.h \
  /usr/lib/llvm-18/lib/clang/18/include/stdint.h

CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o: /home/crane/dev/cpu_dgemm/src/my_dgemm.c \
  /home/crane/dev/cpu_dgemm/include/config.h \
  /home/crane/dev/cpu_dgemm/include/my_dgemm.h \
  /home/crane/dev/cpu_dgemm/include/utils.h \
  /usr/include/alloca.h \
  /usr/include/endian.h \
  /usr/include/features-time64.h \
  /usr/include/features.h \
  /usr/include/inttypes.h \
  /usr/include/stdc-predef.h \
  /usr/include/stdint.h \
  /usr/include/stdlib.h \
  /usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h \
  /usr/include/x86_64-linux-gnu/bits/byteswap.h \
  /usr/include/x86_64-linux-gnu/bits/endian.h \
  /usr/include/x86_64-linux-gnu/bits/endianness.h \
  /usr/include/x86_64-linux-gnu/bits/floatn-common.h \
  /usr/include/x86_64-linux-gnu/bits/floatn.h \
  /usr/include/x86_64-linux-gnu/bits/libc-header-start.h \
  /usr/include/x86_64-linux-gnu/bits/long-double.h \
  /usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h \
  /usr/include/x86_64-linux-gnu/bits/pthreadtypes.h \
  /usr/include/x86_64-linux-gnu/bits/select.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-intn.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-least.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-uintn.h \
  /usr/include/x86_64-linux-gnu/bits/stdlib-float.h \
  /usr/include/x86_64-linux-gnu/bits/struct_mutex.h \
  /usr/include/x86_64-linux-gnu/bits/struct_rwlock.h \
  /usr/include/x86_64-linux-gnu/bits/thread-shared-types.h \
  /usr/include/x86_64-linux-gnu/bits/time64.h \
  /usr/include/x86_64-linux-gnu/bits/timesize.h \
  /usr/include/x86_64-linux-gnu/bits/types.h \
  /usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/clock_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/clockid_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/sigset_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h \
  /usr/include/x86_64-linux-gnu/bits/types/time_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/timer_t.h \
  /usr/include/x86_64-linux-gnu/bits/typesizes.h \
  /usr/include/x86_64-linux-gnu/bits/uintn-identity.h \
  /usr/include/x86_64-linux-gnu/bits/waitflags.h \
  /usr/include/x86_64-linux-gnu/bits/waitstatus.h \
  /usr/include/x86_64-linux-gnu/bits/wchar.h \
  /usr/include/x86_64-linux-gnu/bits/wordsize.h \
  /usr/include/x86_64-linux-gnu/cblas.h \
  /usr/include/x86_64-linux-gnu/cblas_mangling.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs-64.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs.h \
  /usr/include/x86_64-linux-gnu/sys/cdefs.h \
  /usr/include/x86_64-linux-gnu/sys/select.h \
  /usr/include/x86_64-linux-gnu/sys/types.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_max_align_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_null.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_offsetof.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_ptrdiff_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_size_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_wchar_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/inttypes.h \
  /usr/lib/llvm-18/lib/clang/18/include/stddef.h \
  /usr/lib/llvm-18/lib/clang/18/include/stdint.h

CMakeFiles/cpu_dgemm.dir/src/utils.c.o: /home/crane/dev/cpu_dgemm/src/utils.c \
  /home/crane/dev/cpu_dgemm/include/config.h \
  /home/crane/dev/cpu_dgemm/include/utils.h \
  /usr/include/alloca.h \
  /usr/include/endian.h \
  /usr/include/features-time64.h \
  /usr/include/features.h \
  /usr/include/inttypes.h \
  /usr/include/stdc-predef.h \
  /usr/include/stdint.h \
  /usr/include/stdio.h \
  /usr/include/stdlib.h \
  /usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h \
  /usr/include/x86_64-linux-gnu/bits/byteswap.h \
  /usr/include/x86_64-linux-gnu/bits/endian.h \
  /usr/include/x86_64-linux-gnu/bits/endianness.h \
  /usr/include/x86_64-linux-gnu/bits/floatn-common.h \
  /usr/include/x86_64-linux-gnu/bits/floatn.h \
  /usr/include/x86_64-linux-gnu/bits/libc-header-start.h \
  /usr/include/x86_64-linux-gnu/bits/long-double.h \
  /usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h \
  /usr/include/x86_64-linux-gnu/bits/pthreadtypes.h \
  /usr/include/x86_64-linux-gnu/bits/select.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-intn.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-least.h \
  /usr/include/x86_64-linux-gnu/bits/stdint-uintn.h \
  /usr/include/x86_64-linux-gnu/bits/stdio_lim.h \
  /usr/include/x86_64-linux-gnu/bits/stdlib-float.h \
  /usr/include/x86_64-linux-gnu/bits/struct_mutex.h \
  /usr/include/x86_64-linux-gnu/bits/struct_rwlock.h \
  /usr/include/x86_64-linux-gnu/bits/thread-shared-types.h \
  /usr/include/x86_64-linux-gnu/bits/time64.h \
  /usr/include/x86_64-linux-gnu/bits/timesize.h \
  /usr/include/x86_64-linux-gnu/bits/types.h \
  /usr/include/x86_64-linux-gnu/bits/types/FILE.h \
  /usr/include/x86_64-linux-gnu/bits/types/__FILE.h \
  /usr/include/x86_64-linux-gnu/bits/types/__fpos64_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__mbstate_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/clock_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/clockid_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/cookie_io_functions_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/sigset_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h \
  /usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h \
  /usr/include/x86_64-linux-gnu/bits/types/time_t.h \
  /usr/include/x86_64-linux-gnu/bits/types/timer_t.h \
  /usr/include/x86_64-linux-gnu/bits/typesizes.h \
  /usr/include/x86_64-linux-gnu/bits/uintn-identity.h \
  /usr/include/x86_64-linux-gnu/bits/waitflags.h \
  /usr/include/x86_64-linux-gnu/bits/waitstatus.h \
  /usr/include/x86_64-linux-gnu/bits/wchar.h \
  /usr/include/x86_64-linux-gnu/bits/wordsize.h \
  /usr/include/x86_64-linux-gnu/cblas.h \
  /usr/include/x86_64-linux-gnu/cblas_mangling.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs-64.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs.h \
  /usr/include/x86_64-linux-gnu/sys/cdefs.h \
  /usr/include/x86_64-linux-gnu/sys/select.h \
  /usr/include/x86_64-linux-gnu/sys/types.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stdarg___gnuc_va_list.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_max_align_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_null.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_offsetof.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_ptrdiff_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_size_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_wchar_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/inttypes.h \
  /usr/lib/llvm-18/lib/clang/18/include/stdarg.h \
  /usr/lib/llvm-18/lib/clang/18/include/stddef.h \
  /usr/lib/llvm-18/lib/clang/18/include/stdint.h


/usr/lib/llvm-18/lib/clang/18/include/stddef.h:

/usr/lib/llvm-18/lib/clang/18/include/stdarg.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_wchar_t.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_size_t.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_ptrdiff_t.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_offsetof.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_null.h:

/home/crane/dev/cpu_dgemm/src/my_dgemm.c:

/usr/include/x86_64-linux-gnu/sys/types.h:

/usr/include/x86_64-linux-gnu/sys/select.h:

/usr/include/x86_64-linux-gnu/gnu/stubs-64.h:

/home/crane/dev/cpu_dgemm/src/utils.c:

/usr/include/x86_64-linux-gnu/cblas_mangling.h:

/usr/include/x86_64-linux-gnu/cblas.h:

/usr/include/x86_64-linux-gnu/bits/wchar.h:

/usr/include/x86_64-linux-gnu/bits/waitstatus.h:

/usr/include/x86_64-linux-gnu/bits/waitflags.h:

/usr/include/x86_64-linux-gnu/bits/typesizes.h:

/usr/include/x86_64-linux-gnu/bits/math-vector.h:

/usr/include/x86_64-linux-gnu/bits/endian.h:

/usr/include/x86_64-linux-gnu/bits/floatn.h:

/home/crane/dev/cpu_dgemm/src/main.c:

/usr/include/stdlib.h:

/usr/include/x86_64-linux-gnu/bits/floatn-common.h:

/usr/include/stdio.h:

/usr/include/x86_64-linux-gnu/bits/endianness.h:

/usr/include/math.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h:

/usr/include/x86_64-linux-gnu/bits/wordsize.h:

/home/crane/dev/cpu_dgemm/src/kernel_int_8x4.c:

/usr/include/x86_64-linux-gnu/bits/struct_rwlock.h:

/usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h:

/home/crane/dev/cpu_dgemm/include/my_dgemm.h:

/usr/include/x86_64-linux-gnu/bits/fp-fast.h:

/usr/include/stdint.h:

/usr/lib/llvm-18/lib/clang/18/include/stdint.h:

/usr/include/x86_64-linux-gnu/bits/uintn-identity.h:

/usr/include/x86_64-linux-gnu/bits/fp-logb.h:

/usr/lib/llvm-18/lib/clang/18/include/inttypes.h:

/usr/include/x86_64-linux-gnu/bits/timesize.h:

/usr/include/features-time64.h:

/usr/include/x86_64-linux-gnu/bits/types/cookie_io_functions_t.h:

/usr/include/x86_64-linux-gnu/bits/flt-eval-method.h:

/usr/include/x86_64-linux-gnu/bits/libc-header-start.h:

/usr/include/x86_64-linux-gnu/gnu/stubs.h:

/usr/include/inttypes.h:

/home/crane/dev/cpu_dgemm/include/config.h:

/usr/include/alloca.h:

/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h:

/usr/include/features.h:

/usr/include/x86_64-linux-gnu/bits/types/FILE.h:

/usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h:

/usr/include/x86_64-linux-gnu/bits/types/clockid_t.h:

/usr/include/x86_64-linux-gnu/bits/time64.h:

/usr/include/x86_64-linux-gnu/bits/types/sigset_t.h:

/usr/lib/llvm-18/lib/clang/18/include/__stdarg___gnuc_va_list.h:

/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h:

/usr/include/x86_64-linux-gnu/bits/select.h:

/usr/include/x86_64-linux-gnu/bits/stdint-intn.h:

/usr/include/x86_64-linux-gnu/bits/stdint-least.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_max_align_t.h:

/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h:

/usr/include/x86_64-linux-gnu/bits/types/__mbstate_t.h:

/home/crane/dev/cpu_dgemm/include/utils.h:

/usr/include/x86_64-linux-gnu/bits/stdio_lim.h:

/usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h:

/usr/include/x86_64-linux-gnu/bits/types.h:

/usr/include/stdc-predef.h:

/usr/include/x86_64-linux-gnu/bits/stdlib-float.h:

/usr/include/x86_64-linux-gnu/bits/mathcalls.h:

/usr/include/x86_64-linux-gnu/bits/struct_mutex.h:

/usr/include/endian.h:

/usr/include/x86_64-linux-gnu/bits/types/timer_t.h:

/usr/include/x86_64-linux-gnu/bits/thread-shared-types.h:

/usr/include/x86_64-linux-gnu/sys/cdefs.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h:

/usr/include/x86_64-linux-gnu/bits/long-double.h:

/usr/include/x86_64-linux-gnu/bits/byteswap.h:

/usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__FILE.h:

/usr/include/x86_64-linux-gnu/bits/types/clock_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__fpos64_t.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h:

/usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h:

/usr/include/x86_64-linux-gnu/bits/types/time_t.h:
