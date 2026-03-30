pub mod mymodel {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/mnist-",
        env!("MODEL"),
        ".rs"
    ));
}
