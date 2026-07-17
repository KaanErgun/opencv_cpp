function(add_vision_app name)
  add_executable(app_${name} ${name}/main.cpp)
  target_link_libraries(app_${name} PRIVATE vision_core project_warnings)
  set_target_properties(app_${name} PROPERTIES OUTPUT_NAME app_${name})
endfunction()

add_vision_app(detect)
add_vision_app(rtsp_record)
add_vision_app(multicam)
add_vision_app(alpr)

# multicam parses raw JSON directly for its per-camera config.
target_link_libraries(app_multicam PRIVATE nlohmann_json::nlohmann_json)
