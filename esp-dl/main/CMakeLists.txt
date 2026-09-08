set(ACTIVE_APP "har" CACHE STRING "App variant to build: 'har' (default) or a subdirectory of main/apps") 

if(ACTIVE_APP STREQUAL "har")
    set(src_dirs        ./)

    set(include_dirs    ./)
elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/apps/${ACTIVE_APP}")
    set(src_dirs        apps/${ACTIVE_APP})

    set(include_dirs    ./ apps/${ACTIVE_APP})
else()
    message(FATAL_ERROR "Unknown ACTIVE_APP '${ACTIVE_APP}': expected 'har' or an existing directory under main/apps")
endif()

list(APPEND requires esp32_s3_eye_noglib
                      esp_lcd
                      espressif__esp-dl)

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)
set(embed_files "model/model.espdl")

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

target_add_aligned_binary_data(${COMPONENT_LIB} ${embed_files} BINARY)
