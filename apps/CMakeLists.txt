function(add_vision_app name)
  add_executable(app_${name} ${name}/main.cpp)
  target_link_libraries(app_${name} PRIVATE vision_core project_warnings)
  set_target_properties(app_${name} PROPERTIES OUTPUT_NAME app_${name})
endfunction()

# Links an OpenCV module target only when this install provides it. OpenCV 5.x
# split imgproc: contourArea/moments -> geometry, goodFeaturesToTrack -> features,
# CascadeClassifier -> xobjdetect. On 4.x those all live in the base components
# already linked through vision_core, so the extra target simply doesn't exist.
function(link_cv_module_if_present app module)
  if(TARGET ${module})
    target_link_libraries(${app} PRIVATE ${module})
  endif()
endfunction()

# --- DNN / pipeline apps ------------------------------------------------------
add_vision_app(detect)
add_vision_app(rtsp_record)
add_vision_app(multicam)
add_vision_app(alpr)

# multicam parses raw JSON directly for its per-camera config.
target_link_libraries(app_multicam PRIVATE nlohmann_json::nlohmann_json)

# --- Educational / classic-CV apps --------------------------------------------
add_vision_app(image_ops)
add_vision_app(filters)
add_vision_app(edges)

add_vision_app(contours)
link_cv_module_if_present(app_contours opencv_geometry)

add_vision_app(color_track)
link_cv_module_if_present(app_color_track opencv_geometry)

add_vision_app(face_detect)
link_cv_module_if_present(app_face_detect opencv_xobjdetect)

add_vision_app(motion_detect)
link_cv_module_if_present(app_motion_detect opencv_geometry)

add_vision_app(optical_flow)
link_cv_module_if_present(app_optical_flow opencv_features)

add_vision_app(qr_scanner)

if(VISION_HAVE_TRACKING)
  add_vision_app(object_track)
  target_link_libraries(app_object_track PRIVATE opencv_tracking)
endif()

# --- Client/server ALPR system (needs Tesseract; see VISION_HAVE_OCR) ---------
if(VISION_HAVE_OCR)
  add_executable(app_alpr_server alpr_server/main.cpp)
  target_link_libraries(app_alpr_server PRIVATE
    vision_alpr vision_core project_warnings
    httplib::httplib nlohmann_json::nlohmann_json)
  target_include_directories(app_alpr_server PRIVATE alpr_server)

  # The client does no vision work — just capture + HTTP + draw.
  add_executable(app_alpr_client alpr_client/main.cpp)
  target_link_libraries(app_alpr_client PRIVATE
    vision_core project_warnings
    httplib::httplib nlohmann_json::nlohmann_json)
endif()
