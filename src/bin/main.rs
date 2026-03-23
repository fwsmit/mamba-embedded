#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use burn::{backend::NdArray, tensor::Tensor};

use esp_alloc::HeapStats;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use log::info;
use mamba_embedded::mymodel::Model;
// Set the backend to NdArray with f32
type Backend1 = NdArray<f32>;
type BackendDevice = <Backend1 as burn::tensor::backend::Backend>::Device;

type InputType = Tensor<NdArray<f32>, 4>;

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
    // info!("Started");

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    // Get a default device for the backend
    let device = BackendDevice::default();

    // Create a new model and load the state
    let model: Model<Backend1> = Model::default();

    info!("Running inference");
    let i = 0.12;
    // let input = InputType::from_floats([[i]], &device);
    // let input = InputType::zeros([1, 1], &device);
    let input = InputType::zeros([1, 1, 28, 28], &device);
    let output = model.forward(input);
    // Create a new input tensor (all zeros for demonstration purposes)

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
//

// Generated from ONNX "src/model/mnist.onnx" by burn-onnx
// use burn::nn::Linear;
// use burn::nn::LinearConfig;
// use burn::nn::LinearLayout;
// use burn::prelude::*;
// use burn::tensor::activation::log_softmax;
// use burn_store::BurnpackStore;
// use burn_store::ModuleSnapshot;
//
// #[derive(Module, Debug)]
// pub struct Model<B: Backend> {
//     linear1: Linear<B>,
//     phantom: core::marker::PhantomData<B>,
//     device: burn::module::Ignored<B::Device>,
// }
//
// #[repr(C, align(256))]
// struct Aligned256([u8; 31784usize]);
// static ALIGNED_DATA: Aligned256 = Aligned256(*include_bytes!(
//     "/home/friso/Documents/Nextcloud/TU/Thesis/Software/mamba_embedded/target/xtensa-esp32s3-none-elf/release/build/mamba-embedded-8949ea7b983b41f2/out/model/mnist.bpk"
// ));
// static EMBEDDED_STATES: &[u8] = &ALIGNED_DATA.0;
//
// impl<B: Backend> Default for Model<B> {
//     fn default() -> Self {
//         Self::from_embedded(&Default::default())
//     }
// }
//
// impl<B: Backend> Model<B> {
//     /// Load model weights from embedded burnpack data (zero-copy at store level).
//     ///
//     /// The embedded data stays in the binary's .rodata section without heap allocation.
//     /// Tensor data is sliced directly from the static bytes.
//     ///
//     /// Note: Some backends (e.g., NdArray) may still copy data internally.
//     /// See <https://github.com/tracel-ai/burn/issues/4153> for true backend zero-copy.
//     ///
//     /// See <https://github.com/tracel-ai/burn/issues/4123>
//     pub fn from_embedded(device: &B::Device) -> Self {
//         let mut model = Self::new(device);
//         let mut store = BurnpackStore::from_static(EMBEDDED_STATES);
//         model
//             .load_from(&mut store)
//             .expect("Failed to load embedded burnpack");
//         model
//     }
// }
//
// impl<B: Backend> Model<B> {
//     #[allow(unused_variables)]
//     pub fn new(device: &B::Device) -> Self {
//         let linear1 = LinearConfig::new(784, 10)
//             .with_bias(true)
//             .with_layout(LinearLayout::Col)
//             .init(device);
//         Self {
//             linear1,
//             phantom: core::marker::PhantomData,
//             device: burn::module::Ignored(device.clone()),
//         }
//     }
//
//     #[allow(clippy::let_and_return, clippy::approx_constant)]
//     pub fn forward(&self, onnx_flatten_0: Tensor<B, 4>) -> Tensor<B, 2> {
//         let device = BackendDevice::default();
//         let input_ex = InputType::zeros([1, 1, 28, 28], &device);
//         // let input_ex = Tensor::<B, 4>::zeros([1, 1, 28, 28], device);
//         info!("print 1");
//         let flatten1_out1 = {
//             let leading_dim = input_ex.shape().dims::<4>()[..1].iter().product::<usize>() as i32;
//             onnx_flatten_0.reshape::<2, _>([leading_dim, -1])
//         };
//         let stats: HeapStats = esp_alloc::HEAP.stats();
//         info!("{}", stats);
//         info!("print 1.5");
//         let test_alloc = InputType::zeros([1, 20, 28, 28], &device);
//         let stats: HeapStats = esp_alloc::HEAP.stats();
//         info!("{}", stats);
//         // HeapStats implements the Display and defmt::Format traits, so you can
//         // pretty-print the heap stats.
//         info!("print 2");
//         let linear1_out1 = self.linear1.forward(flatten1_out1);
//         info!("print 3");
//         let relu1_out1 = burn::tensor::activation::relu(linear1_out1);
//         let logsoftmax1_out1 = log_softmax(relu1_out1, 1);
//         logsoftmax1_out1
//     }
// }
