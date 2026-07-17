# INTERFACE target carrying the project's warning flags. Link it into every
# first-party target so warnings stay consistent. WARNINGS_AS_ERRORS is OFF by
# default (legacy code isn't warning-clean yet); flip it on after a cleanup pass.
add_library(project_warnings INTERFACE)

option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(_warnings
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wnon-virtual-dtor
  -Wcast-align
  -Woverloaded-virtual
  -Wsign-compare)

target_compile_options(project_warnings INTERFACE ${_warnings})

if(WARNINGS_AS_ERRORS)
  target_compile_options(project_warnings INTERFACE -Werror)
endif()
