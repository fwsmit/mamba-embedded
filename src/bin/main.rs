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
use mamba_embedded::mymodel::Model;

type InputType = Tensor<NdArray<f32>, 2>;

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
    // info!("Started");

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    // Get a default device for the backend
    let device = BackendDevice::default();

    // Create a new model and load the state
    let model: Model<Backend> = Model::default();

    info!("Running inference");
    let i = 0.12;
    let input = InputType::from_floats([[i]], &device);
    let output = model.forward(input);
    // Create a new input tensor (all zeros for demonstration purposes)
    // let input = InputType::zeros([1, 1, 28, 28], &device);

    // Run the model
    // let output = run_model(&model, &device, input);

    // Print the output
    info!("{:?}", output);
    info!("Finished");
    loop {
        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(500) {
            // info!("Waiting")
        }
    }
}

// fn run_model(
//     model: &Model<NdArray>,
//     device: &BackendDevice,
//     input: InputType,
// ) -> Tensor<Backend, 2> {
//     // Define the tensor
//     // let input = Tensor::<Backend, 4>::from_floats([[input]], device);
//     info!("Running inference");
//
//     // Run the model on the input
//     let output = model.forward(input);
//     info!("Got output");
//     info!("output is {:?}", output);
//
//     output
// }
