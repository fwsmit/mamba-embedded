#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

// Set the backend to NdArray with f32
type Backend = NdArray<f32>;
// Get the backend device we use (cpu)
type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;

use burn::{backend::NdArray, tensor::Tensor};
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use log::info;
use mamba_embedded::sine::Model;

// This creates a default app-descriptor required by the esp-idf bootloader.
// For more information see: <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/app_image_format.html#application-description>
esp_bootloader_esp_idf::esp_app_desc!();

#[allow(
    clippy::large_stack_frames,
    reason = "it's not unusual to allocate larger buffers etc. in main"
)]
#[main]
fn main() -> ! {
    // Initialize allocator with a size
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    // generator version: 1.2.0

    esp_println::logger::init_logger_from_env();

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    // Get a default device for the backend
    let device = BackendDevice::default();

    // Create a new model and load the state
    let model: Model<Backend> = Model::default();

    // Define input, this is the `x` in the function `y = sin(x)` that we are
    // approximating with our model
    let mut input = 0.0;

    loop {
        info!("Hello world!");
        if input > 2.0 {
            input = 0.0
        }
        input += 0.05;

        // Run the model
        let output = run_model(&model, &device, input);

        // Output the values
        match output.into_data().as_slice::<f32>() {
            Ok(slice) => log::info!("input: {:.3} - output: {:?}", input, slice),
            Err(err) => core::panic!("err: {:?}", err),
        };

        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(500) {}
    }

    // for inspiration have a look at the examples at https://github.com/esp-rs/esp-hal/tree/esp-hal-v1.0.0/examples
}

fn run_model(model: &Model<NdArray>, device: &BackendDevice, input: f32) -> Tensor<Backend, 2> {
    // Define the tensor
    let input = Tensor::<Backend, 2>::from_floats([[input]], device);

    // Run the model on the input
    model.forward(input)
}
