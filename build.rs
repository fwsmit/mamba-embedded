use burn_onnx::ModelGen;

fn main() {
    generate_model();

    linker_be_nice();
    // make sure linkall.x is the last linker script (otherwise might cause problems with flip-link)
    println!("cargo:rustc-link-arg=-Tlinkall.x");
}

fn linker_be_nice() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let kind = &args[1];
        let what = &args[2];

        match kind.as_str() {
            "undefined-symbol" => match what.as_str() {
                what if what.starts_with("_defmt_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `defmt` not found - make sure `defmt.x` is added as a linker script and you have included `use defmt_rtt as _;`"
                    );
                    eprintln!();
                }
                "_stack_start" => {
                    eprintln!();
                    eprintln!("💡 Is the linker script `linkall.x` missing?");
                    eprintln!();
                }
                what if what.starts_with("esp_rtos_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `esp-radio` has no scheduler enabled. Make sure you have initialized `esp-rtos` or provided an external scheduler."
                    );
                    eprintln!();
                }
                "embedded_test_linker_file_not_added_to_rustflags" => {
                    eprintln!();
                    eprintln!(
                        "💡 `embedded-test` not found - make sure `embedded-test.x` is added as a linker script for tests"
                    );
                    eprintln!();
                }
                "free"
                | "malloc"
                | "calloc"
                | "get_free_internal_heap_size"
                | "malloc_internal"
                | "realloc_internal"
                | "calloc_internal"
                | "free_internal" => {
                    eprintln!();
                    eprintln!(
                        "💡 Did you forget the `esp-alloc` dependency or didn't enable the `compat` feature on it?"
                    );
                    eprintln!();
                }
                _ => (),
            },
            // we don't have anything helpful for "missing-lib" yet
            _ => {
                std::process::exit(1);
            }
        }

        std::process::exit(0);
    }

    println!(
        "cargo:rustc-link-arg=-Wl,--error-handling-script={}",
        std::env::current_exe().unwrap().display()
    );
}

fn generate_model() {
    println!("cargo::rerun-if-env-changed=MODEL");
    println!("cargo::rerun-if-env-changed=DATASET");
    let model_type = env!("MODEL");
    let dataset = env!("DATASET");

    // Set feature flags based on dataset for conditional compilation
    match dataset {
        "mnist" => println!("cargo:rustc-cfg=dataset_mnist"),
        "har" => println!("cargo:rustc-cfg=dataset_har"),
        _ => println!("cargo:rustc-cfg=dataset_mnist"),
    }

    // Generate the model code from the ONNX file.
    // Format: src/models/{dataset}-{model_type}.onnx (e.g., src/models/mnist-mamba-1.onnx)
    let model_path = ["src/models/", dataset, "-", model_type, ".onnx"].join("");
    println!("cargo::rerun-if-changed={}", model_path);

    ModelGen::new()
        .input(&model_path)
        .out_dir("model/")
        .embed_states(true)
        .run_from_script();
}
