cmake_minimum_required(VERSION 3.7.2)

project (METAOPT)

set(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 11)

FIND_PACKAGE( Boost 1.55 COMPONENTS program_options REQUIRED )

include_directories( 
	${Boost_INCLUDE_DIR}
	${METAOPT_SOURCE_DIR}/include
)

link_directories(
	${Boost_LIBRARY_DIR}
	${METAOPT_SOURCE_DIR}/lib
)

set(
	CMAKE_CXX_LINK_FLAGS 
)

#Generate the shared library from the sources
#add_library(exchange SHARED ${SOURCES})

add_executable(
    metaoptd
	${METAOPT_SOURCE_DIR}/src/metaoptd.cpp
)

add_executable(
    metaopt
	${METAOPT_SOURCE_DIR}/src/metaopt.cpp
)

TARGET_LINK_LIBRARIES(
    metaoptd
    ${Boost_LIBRARY_DIR}
)

TARGET_LINK_LIBRARIES(
    metaopt
    ${Boost_LIBRARY_DIR}
	-lboost_program_options
)

install(TARGETS metaoptd DESTINATION /usr/local/bin)
install(TARGETS metaopt DESTINATION /usr/local/bin)
