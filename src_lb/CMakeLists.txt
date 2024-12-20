CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(get_load_balancing)

# Find NetCDF
FIND_PACKAGE(NetCDF)
IF (NOT NETCDF_FOUND)
        MESSAGE(FATAL_ERROR "NetCDF not found")
ENDIF (NOT NETCDF_FOUND)
INCLUDE_DIRECTORIES(${NETCDF_INCLUDES})

# Set the build type to Release
IF (NOT CMAKE_BUILD_TYPE)
        SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type (Debug, Release)" FORCE)
ENDIF (NOT CMAKE_BUILD_TYPE)

ADD_LIBRARY(netcdf netcdf.cxx)
TARGET_LINK_LIBRARIES(netcdf ${NETCDF_LIBRARIES})

ADD_LIBRARY(CargaX CargaX.cxx)
ADD_LIBRARY(CargaY CargaY.cxx)

ADD_EXECUTABLE(get_load_balancing ShallowWater.cxx)
TARGET_LINK_LIBRARIES(get_load_balancing netcdf CargaX CargaY)

