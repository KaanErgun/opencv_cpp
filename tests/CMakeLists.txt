include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2)
FetchContent_MakeAvailable(Catch2)

add_executable(vision_tests
  test_source_spec.cpp
  test_tracker.cpp)

target_link_libraries(vision_tests PRIVATE
  vision_core
  Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(vision_tests)
