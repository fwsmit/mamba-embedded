#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use burn::backend::NdArray;

use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use log::info;

#[cfg(dataset_har)]
use mamba_embedded::data::har_tensor::input_tensor;

#[cfg(dataset_mnist)]
use mamba_embedded::data::mnist_tensor::input_tensor;

use mamba_embedded::mymodel::Model;
// Set the backend to NdArray with f32
type Backend = NdArray<f32>;
// type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;

// type InputType = Tensor<Backend, 4>;

// This creates a default app-descriptor required by the esp-idf bootloader.
// For more information see: <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/app_image_format.html#application-description>
esp_bootloader_esp_idf::esp_app_desc!();

#[allow(
    clippy::large_stack_frames,
    reason = "it's not unusual to allocate larger buffers etc. in main"
)]
#[main]
fn main() -> ! {
    // Get the backend device we use (cpu)

    // Initialize allocator with a size
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    // generator version: 1.2.0
    esp_println::logger::init_logger_from_env();
    info!("Started");

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    // Get a default device for the backend
    // let device = BackendDevice::default();

    // Create a new model and load the state
    let model: Model<Backend> = Model::default();

    info!("Running inference");

    // Select dataset based on compile-time DATASET environment variable
    let dataset = env!("DATASET");

    loop {
        // Use conditional compilation to select the correct input tensor type
        // let input = mnist_tensor();

        let input = input_tensor();
        info!("Input shape: {:?}", input.shape());

        // #[cfg(not(any(feature = "dataset_mnist", feature = "dataset_har")))]
        // let input = mnist_tensor();

        let inference_timer = Instant::now();
        let output = model.forward(input);
        let inference_time = inference_timer.elapsed();

        // Print the output
        info!("{:?}", &output);
        let guess = output.argmax(1).into_scalar();
        info!("Guess: {:?}", guess);
        info!("Dataset: {}", dataset);
        info!("Inference done in: {:?} ms", inference_time.as_millis());
        info!("Finished");
        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(500) {}
    }
}
