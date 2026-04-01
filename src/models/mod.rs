pub mod mymodel {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/",
        env!("DATASET"),
        "-",
        env!("MODEL"),
        ".rs"
    ));
}
