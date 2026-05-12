set(src_dirs        ./)

set(include_dirs    ./)

list(APPEND requires esp32_s3_eye_noglib
                      esp_lcd
                      espressif__esp-dl)

set(embed_files
    "model/har-mamba-1.espdl"
)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})
