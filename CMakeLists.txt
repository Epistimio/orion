cmake_minimum_required(VERSION 3.8.0)

project (evolve)

set(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 11)

set(BOOST_INCLUDE_DIR /usr/local/Cellar/boost/1.65.1/include)
set(BOOST_LIBRARY_DIR /usr/local/Cellar/boost/1.65.1/lib)

find_package( Boost 1.65 COMPONENTS system REQUIRED )

include_directories( 
	${BOOST_INCLUDE_DIR}
	${FOOD_SOURCE_DIR}/include
)

link_directories(
	${BOOST_LIBRARY_DIR}
	${FOOD_SOURCE_DIR}/lib
)

set(
	CMAKE_CXX_LINK_FLAGS 
)

#Generate the shared library from the sources
#add_library(exchange SHARED ${SOURCES})

add_executable(
    evolve
)

TARGET_LINK_LIBRARIES(
    evolve
    ${Boost_LIBRARY_DIR}
)

install(TARGETS evolve DESTINATION /usr/local/bin)
