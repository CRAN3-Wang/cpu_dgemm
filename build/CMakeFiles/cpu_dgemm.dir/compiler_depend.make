# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

CMakeFiles/cpu_dgemm.dir/src/kernels.c.o: /home/crane/dev/cpu_dgemm/src/kernels.c \
  /usr/include/alloca.h \
  /usr/include/endian.h \
  /usr/include/features-time64.h \
  /usr/include/features.h \
  /usr/include/stdc-predef.h \
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
  /usr/include/x86_64-linux-gnu/bits/wordsize.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs-64.h \
  /usr/include/x86_64-linux-gnu/gnu/stubs.h \
  /usr/include/x86_64-linux-gnu/sys/cdefs.h \
  /usr/include/x86_64-linux-gnu/sys/select.h \
  /usr/include/x86_64-linux-gnu/sys/types.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_null.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_size_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__stddef_wchar_t.h \
  /usr/lib/llvm-18/lib/clang/18/include/__wmmintrin_aes.h \
  /usr/lib/llvm-18/lib/clang/18/include/__wmmintrin_pclmul.h \
  /usr/lib/llvm-18/lib/clang/18/include/adcintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/adxintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/amxcomplexintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/amxfp16intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/amxintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx2intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512bf16intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512bitalgintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512bwintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512cdintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512dqintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512erintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512fintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512fp16intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512ifmaintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512ifmavlintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512pfintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vbmi2intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vbmiintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vbmivlintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vlbf16intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vlbitalgintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vlbwintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vlcdintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vldqintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vlfp16intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vlintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vlvbmi2intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vlvnniintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vlvp2intersectintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vnniintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vp2intersectintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vpopcntdqintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avx512vpopcntdqvlintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avxifmaintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avxintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avxneconvertintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avxvnniint16intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avxvnniint8intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/avxvnniintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/bmi2intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/bmiintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/cetintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/cldemoteintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/clflushoptintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/clwbintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/cmpccxaddintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/crc32intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/emmintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/enqcmdintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/f16cintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/fmaintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/fxsrintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/gfniintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/hresetintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/immintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/invpcidintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/keylockerintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/lzcntintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/mm_malloc.h \
  /usr/lib/llvm-18/lib/clang/18/include/mmintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/movdirintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/pconfigintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/pkuintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/pmmintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/popcntintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/prfchiintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/ptwriteintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/raointintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/rdseedintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/rtmintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/serializeintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/sgxintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/sha512intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/shaintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/sm3intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/sm4intrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/smmintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/stddef.h \
  /usr/lib/llvm-18/lib/clang/18/include/tmmintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/tsxldtrkintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/uintrintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/usermsrintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/vaesintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/vpclmulqdqintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/waitpkgintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/wbnoinvdintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/wmmintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/x86gprintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/xmmintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/xsavecintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/xsaveintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/xsaveoptintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/xsavesintrin.h \
  /usr/lib/llvm-18/lib/clang/18/include/xtestintrin.h

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
  /home/crane/dev/cpu_dgemm/include/kernels.h \
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


/home/crane/dev/cpu_dgemm/src/utils.c:

/usr/lib/llvm-18/lib/clang/18/include/stdint.h:

/usr/lib/llvm-18/lib/clang/18/include/stdarg.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_ptrdiff_t.h:

/usr/lib/llvm-18/lib/clang/18/include/__stdarg___gnuc_va_list.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h:

/usr/include/x86_64-linux-gnu/bits/wchar.h:

/usr/include/x86_64-linux-gnu/bits/types/cookie_io_functions_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__FILE.h:

/usr/include/x86_64-linux-gnu/bits/types/FILE.h:

/usr/include/x86_64-linux-gnu/bits/stdio_lim.h:

/usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h:

/home/crane/dev/cpu_dgemm/include/kernels.h:

/usr/include/x86_64-linux-gnu/bits/math-vector.h:

/usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h:

/usr/include/x86_64-linux-gnu/bits/fp-logb.h:

/usr/include/x86_64-linux-gnu/bits/fp-fast.h:

/usr/include/stdint.h:

/usr/include/math.h:

/usr/include/inttypes.h:

/home/crane/dev/cpu_dgemm/include/my_dgemm.h:

/home/crane/dev/cpu_dgemm/include/config.h:

/home/crane/dev/cpu_dgemm/src/main.c:

/usr/lib/llvm-18/lib/clang/18/include/xtestintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/xsavesintrin.h:

/usr/include/x86_64-linux-gnu/bits/stdint-least.h:

/usr/lib/llvm-18/lib/clang/18/include/x86gprintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/wmmintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/tsxldtrkintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/tmmintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx2intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/amxcomplexintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/adcintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/invpcidintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/amxfp16intrin.h:

/usr/include/x86_64-linux-gnu/bits/flt-eval-method.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_size_t.h:

/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h:

/usr/include/x86_64-linux-gnu/sys/select.h:

/usr/include/x86_64-linux-gnu/bits/waitstatus.h:

/usr/include/x86_64-linux-gnu/bits/waitflags.h:

/usr/include/x86_64-linux-gnu/cblas.h:

/usr/include/x86_64-linux-gnu/gnu/stubs-64.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vlbitalgintrin.h:

/usr/include/x86_64-linux-gnu/sys/types.h:

/usr/lib/llvm-18/lib/clang/18/include/pkuintrin.h:

/usr/include/x86_64-linux-gnu/bits/types/time_t.h:

/usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h:

/usr/include/x86_64-linux-gnu/sys/cdefs.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h:

/usr/include/x86_64-linux-gnu/bits/floatn.h:

/usr/include/x86_64-linux-gnu/bits/select.h:

/usr/include/x86_64-linux-gnu/bits/byteswap.h:

/usr/include/x86_64-linux-gnu/bits/long-double.h:

/usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512bwintrin.h:

/usr/include/x86_64-linux-gnu/bits/floatn-common.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512ifmavlintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/xmmintrin.h:

/home/crane/dev/cpu_dgemm/src/kernels.c:

/usr/include/x86_64-linux-gnu/bits/libc-header-start.h:

/usr/lib/llvm-18/lib/clang/18/include/cldemoteintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/inttypes.h:

/usr/include/features.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vlfp16intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/sha512intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/amxintrin.h:

/usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h:

/usr/include/x86_64-linux-gnu/bits/time64.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512cdintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512fp16intrin.h:

/usr/include/stdc-predef.h:

/usr/include/x86_64-linux-gnu/bits/uintn-identity.h:

/usr/include/x86_64-linux-gnu/bits/stdlib-float.h:

/usr/lib/llvm-18/lib/clang/18/include/mm_malloc.h:

/usr/include/x86_64-linux-gnu/bits/wordsize.h:

/usr/include/stdlib.h:

/usr/lib/llvm-18/lib/clang/18/include/xsaveoptintrin.h:

/usr/include/x86_64-linux-gnu/bits/timesize.h:

/usr/lib/llvm-18/lib/clang/18/include/xsaveintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avxifmaintrin.h:

/usr/include/x86_64-linux-gnu/bits/typesizes.h:

/home/crane/dev/cpu_dgemm/src/my_dgemm.c:

/usr/lib/llvm-18/lib/clang/18/include/fmaintrin.h:

/usr/include/x86_64-linux-gnu/bits/endianness.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vbmiintrin.h:

/usr/include/endian.h:

/usr/include/x86_64-linux-gnu/bits/types/timer_t.h:

/usr/lib/llvm-18/lib/clang/18/include/uintrintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/crc32intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/vpclmulqdqintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/usermsrintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_wchar_t.h:

/usr/lib/llvm-18/lib/clang/18/include/smmintrin.h:

/usr/include/x86_64-linux-gnu/cblas_mangling.h:

/usr/include/x86_64-linux-gnu/bits/types.h:

/usr/include/x86_64-linux-gnu/bits/stdint-intn.h:

/usr/lib/llvm-18/lib/clang/18/include/pmmintrin.h:

/usr/include/x86_64-linux-gnu/bits/struct_mutex.h:

/usr/lib/llvm-18/lib/clang/18/include/__wmmintrin_aes.h:

/usr/lib/llvm-18/lib/clang/18/include/adxintrin.h:

/usr/include/x86_64-linux-gnu/bits/struct_rwlock.h:

/usr/include/stdio.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512fintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/fxsrintrin.h:

/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h:

/usr/lib/llvm-18/lib/clang/18/include/hresetintrin.h:

/usr/include/alloca.h:

/usr/include/x86_64-linux-gnu/bits/types/clock_t.h:

/usr/include/x86_64-linux-gnu/bits/types/sigset_t.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512dqintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vnniintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512erintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_offsetof.h:

/usr/include/features-time64.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vbmi2intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vbmivlintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vlbf16intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vlbwintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_max_align_t.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vlcdintrin.h:

/home/crane/dev/cpu_dgemm/include/utils.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vldqintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vlintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/__stddef_null.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vlvbmi2intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vlvnniintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/bmiintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/stddef.h:

/usr/include/x86_64-linux-gnu/bits/endian.h:

/usr/include/x86_64-linux-gnu/gnu/stubs.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vlvp2intersectintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vp2intersectintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vpopcntdqintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/serializeintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/__wmmintrin_pclmul.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512vpopcntdqvlintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512bitalgintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avxneconvertintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avxintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/sm3intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/keylockerintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avxvnniint16intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avxvnniintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/rdseedintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/bmi2intrin.h:

/usr/include/x86_64-linux-gnu/bits/thread-shared-types.h:

/usr/lib/llvm-18/lib/clang/18/include/cetintrin.h:

/usr/include/x86_64-linux-gnu/bits/types/clockid_t.h:

/usr/lib/llvm-18/lib/clang/18/include/clflushoptintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/f16cintrin.h:

/usr/include/x86_64-linux-gnu/bits/mathcalls.h:

/usr/lib/llvm-18/lib/clang/18/include/enqcmdintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/gfniintrin.h:

/usr/include/x86_64-linux-gnu/bits/types/__mbstate_t.h:

/usr/lib/llvm-18/lib/clang/18/include/immintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/lzcntintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/popcntintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/movdirintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/emmintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512pfintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/pconfigintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/wbnoinvdintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/cmpccxaddintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/prfchiintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/xsavecintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/ptwriteintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/waitpkgintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512bf16intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/clwbintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avx512ifmaintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/mmintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/raointintrin.h:

/usr/include/x86_64-linux-gnu/bits/types/__fpos64_t.h:

/usr/lib/llvm-18/lib/clang/18/include/rtmintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/sgxintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/avxvnniint8intrin.h:

/usr/lib/llvm-18/lib/clang/18/include/shaintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/vaesintrin.h:

/usr/lib/llvm-18/lib/clang/18/include/sm4intrin.h:
