# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

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
CMAKE_COMMAND = /soft/centos7/cmake/3.21.3/bin/cmake

# The command to remove a file.
RM = /soft/centos7/cmake/3.21.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/castro/NHHySEA_202410/src_lb

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/castro/NHHySEA_202410/bin_lb

# Include any dependencies generated for this target.
include CMakeFiles/CargaX.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CargaX.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CargaX.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CargaX.dir/flags.make

CMakeFiles/CargaX.dir/CargaX.cxx.o: CMakeFiles/CargaX.dir/flags.make
CMakeFiles/CargaX.dir/CargaX.cxx.o: /home/castro/NHHySEA_202410/src_lb/CargaX.cxx
CMakeFiles/CargaX.dir/CargaX.cxx.o: CMakeFiles/CargaX.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/castro/NHHySEA_202410/bin_lb/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CargaX.dir/CargaX.cxx.o"
	/soft/centos7/gcc/11.2.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CargaX.dir/CargaX.cxx.o -MF CMakeFiles/CargaX.dir/CargaX.cxx.o.d -o CMakeFiles/CargaX.dir/CargaX.cxx.o -c /home/castro/NHHySEA_202410/src_lb/CargaX.cxx

CMakeFiles/CargaX.dir/CargaX.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CargaX.dir/CargaX.cxx.i"
	/soft/centos7/gcc/11.2.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/castro/NHHySEA_202410/src_lb/CargaX.cxx > CMakeFiles/CargaX.dir/CargaX.cxx.i

CMakeFiles/CargaX.dir/CargaX.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CargaX.dir/CargaX.cxx.s"
	/soft/centos7/gcc/11.2.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/castro/NHHySEA_202410/src_lb/CargaX.cxx -o CMakeFiles/CargaX.dir/CargaX.cxx.s

# Object files for target CargaX
CargaX_OBJECTS = \
"CMakeFiles/CargaX.dir/CargaX.cxx.o"

# External object files for target CargaX
CargaX_EXTERNAL_OBJECTS =

libCargaX.a: CMakeFiles/CargaX.dir/CargaX.cxx.o
libCargaX.a: CMakeFiles/CargaX.dir/build.make
libCargaX.a: CMakeFiles/CargaX.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/castro/NHHySEA_202410/bin_lb/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libCargaX.a"
	$(CMAKE_COMMAND) -P CMakeFiles/CargaX.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CargaX.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CargaX.dir/build: libCargaX.a
.PHONY : CMakeFiles/CargaX.dir/build

CMakeFiles/CargaX.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CargaX.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CargaX.dir/clean

CMakeFiles/CargaX.dir/depend:
	cd /home/castro/NHHySEA_202410/bin_lb && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/castro/NHHySEA_202410/src_lb /home/castro/NHHySEA_202410/src_lb /home/castro/NHHySEA_202410/bin_lb /home/castro/NHHySEA_202410/bin_lb /home/castro/NHHySEA_202410/bin_lb/CMakeFiles/CargaX.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CargaX.dir/depend

