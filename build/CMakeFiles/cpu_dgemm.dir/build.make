# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/crane/dev/cpu_dgemm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/crane/dev/cpu_dgemm/build

# Include any dependencies generated for this target.
include CMakeFiles/cpu_dgemm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cpu_dgemm.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cpu_dgemm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cpu_dgemm.dir/flags.make

CMakeFiles/cpu_dgemm.dir/src/main.c.o: CMakeFiles/cpu_dgemm.dir/flags.make
CMakeFiles/cpu_dgemm.dir/src/main.c.o: /home/crane/dev/cpu_dgemm/src/main.c
CMakeFiles/cpu_dgemm.dir/src/main.c.o: CMakeFiles/cpu_dgemm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/crane/dev/cpu_dgemm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/cpu_dgemm.dir/src/main.c.o"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpu_dgemm.dir/src/main.c.o -MF CMakeFiles/cpu_dgemm.dir/src/main.c.o.d -o CMakeFiles/cpu_dgemm.dir/src/main.c.o -c /home/crane/dev/cpu_dgemm/src/main.c

CMakeFiles/cpu_dgemm.dir/src/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpu_dgemm.dir/src/main.c.i"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/crane/dev/cpu_dgemm/src/main.c > CMakeFiles/cpu_dgemm.dir/src/main.c.i

CMakeFiles/cpu_dgemm.dir/src/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpu_dgemm.dir/src/main.c.s"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/crane/dev/cpu_dgemm/src/main.c -o CMakeFiles/cpu_dgemm.dir/src/main.c.s

CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o: CMakeFiles/cpu_dgemm.dir/flags.make
CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o: /home/crane/dev/cpu_dgemm/src/my_dgemm.c
CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o: CMakeFiles/cpu_dgemm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/crane/dev/cpu_dgemm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o -MF CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o.d -o CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o -c /home/crane/dev/cpu_dgemm/src/my_dgemm.c

CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.i"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/crane/dev/cpu_dgemm/src/my_dgemm.c > CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.i

CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.s"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/crane/dev/cpu_dgemm/src/my_dgemm.c -o CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.s

CMakeFiles/cpu_dgemm.dir/src/utils.c.o: CMakeFiles/cpu_dgemm.dir/flags.make
CMakeFiles/cpu_dgemm.dir/src/utils.c.o: /home/crane/dev/cpu_dgemm/src/utils.c
CMakeFiles/cpu_dgemm.dir/src/utils.c.o: CMakeFiles/cpu_dgemm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/crane/dev/cpu_dgemm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/cpu_dgemm.dir/src/utils.c.o"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpu_dgemm.dir/src/utils.c.o -MF CMakeFiles/cpu_dgemm.dir/src/utils.c.o.d -o CMakeFiles/cpu_dgemm.dir/src/utils.c.o -c /home/crane/dev/cpu_dgemm/src/utils.c

CMakeFiles/cpu_dgemm.dir/src/utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpu_dgemm.dir/src/utils.c.i"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/crane/dev/cpu_dgemm/src/utils.c > CMakeFiles/cpu_dgemm.dir/src/utils.c.i

CMakeFiles/cpu_dgemm.dir/src/utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpu_dgemm.dir/src/utils.c.s"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/crane/dev/cpu_dgemm/src/utils.c -o CMakeFiles/cpu_dgemm.dir/src/utils.c.s

CMakeFiles/cpu_dgemm.dir/src/kernels.c.o: CMakeFiles/cpu_dgemm.dir/flags.make
CMakeFiles/cpu_dgemm.dir/src/kernels.c.o: /home/crane/dev/cpu_dgemm/src/kernels.c
CMakeFiles/cpu_dgemm.dir/src/kernels.c.o: CMakeFiles/cpu_dgemm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/crane/dev/cpu_dgemm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/cpu_dgemm.dir/src/kernels.c.o"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cpu_dgemm.dir/src/kernels.c.o -MF CMakeFiles/cpu_dgemm.dir/src/kernels.c.o.d -o CMakeFiles/cpu_dgemm.dir/src/kernels.c.o -c /home/crane/dev/cpu_dgemm/src/kernels.c

CMakeFiles/cpu_dgemm.dir/src/kernels.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cpu_dgemm.dir/src/kernels.c.i"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/crane/dev/cpu_dgemm/src/kernels.c > CMakeFiles/cpu_dgemm.dir/src/kernels.c.i

CMakeFiles/cpu_dgemm.dir/src/kernels.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cpu_dgemm.dir/src/kernels.c.s"
	clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/crane/dev/cpu_dgemm/src/kernels.c -o CMakeFiles/cpu_dgemm.dir/src/kernels.c.s

# Object files for target cpu_dgemm
cpu_dgemm_OBJECTS = \
"CMakeFiles/cpu_dgemm.dir/src/main.c.o" \
"CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o" \
"CMakeFiles/cpu_dgemm.dir/src/utils.c.o" \
"CMakeFiles/cpu_dgemm.dir/src/kernels.c.o"

# External object files for target cpu_dgemm
cpu_dgemm_EXTERNAL_OBJECTS =

cpu_dgemm: CMakeFiles/cpu_dgemm.dir/src/main.c.o
cpu_dgemm: CMakeFiles/cpu_dgemm.dir/src/my_dgemm.c.o
cpu_dgemm: CMakeFiles/cpu_dgemm.dir/src/utils.c.o
cpu_dgemm: CMakeFiles/cpu_dgemm.dir/src/kernels.c.o
cpu_dgemm: CMakeFiles/cpu_dgemm.dir/build.make
cpu_dgemm: /usr/lib/x86_64-linux-gnu/libblas.so
cpu_dgemm: CMakeFiles/cpu_dgemm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/crane/dev/cpu_dgemm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking C executable cpu_dgemm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpu_dgemm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cpu_dgemm.dir/build: cpu_dgemm
.PHONY : CMakeFiles/cpu_dgemm.dir/build

CMakeFiles/cpu_dgemm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cpu_dgemm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cpu_dgemm.dir/clean

CMakeFiles/cpu_dgemm.dir/depend:
	cd /home/crane/dev/cpu_dgemm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/crane/dev/cpu_dgemm /home/crane/dev/cpu_dgemm /home/crane/dev/cpu_dgemm/build /home/crane/dev/cpu_dgemm/build /home/crane/dev/cpu_dgemm/build/CMakeFiles/cpu_dgemm.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cpu_dgemm.dir/depend

