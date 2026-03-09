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

use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use log::info;
use burn::{backend::NdArray, tensor::Tensor};
// use embassy_executor::Spawner;
// use embassy_rp::{
//     bind_interrupts,
//     gpio::{Level, Output},
//     peripherals::USB,
//     usb::{Driver, InterruptHandler},
// };
// use embassy_time::Timer;
// use embedded_alloc::LlffHeap as Heap;
use mamba_embedded::sine::Model;
// use {defmt_rtt as _, panic_probe as _};


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

// #[global_allocator]
// static HEAP: Heap = Heap::empty();
//
// bind_interrupts!(struct Irqs {
//     USBCTRL_IRQ => InterruptHandler<USB>;
// });

// This is the main function for the program. Where execution starts.
// #[embassy_executor::main]
// async fn main(spawner: Spawner) {
    // Initializes the allocator, must be done before use.
    // {
        // use core::mem::MaybeUninit;
        // // Watch out for this, if it is too big or small, program may crash
        // // this is in u8 bytes, as such this is a total of 100kb
        // const HEAP_SIZE: usize = 100 * 1024;
        // static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        // unsafe { HEAP.init(&raw mut HEAP_MEM as usize, HEAP_SIZE) } // Initialize the heap
    // }

    // This is just setup to make the microcontroller output to serial.
    // let p = embassy_rp::init(Default::default());
    // let driver = Driver::new(p.USB, Irqs);
    // spawner.spawn(logger_task(driver)).unwrap();
    //
    // // Set the onboard LED to high to help indicate that the program is working properly.
    // let mut led = Output::new(p.PIN_25, Level::Low);
    // led.set_high();

// }

fn run_model(model: &Model<NdArray>, device: &BackendDevice, input: f32) -> Tensor<Backend, 2> {
    // Define the tensor
    let input = Tensor::<Backend, 2>::from_floats([[input]], device);

    // Run the model on the input
    model.forward(input)
}

// This runs as a separate task whenever there is an await
// #[embassy_executor::task]
// async fn logger_task(driver: Driver<'static, USB>) {
//     // This just makes a logger that outputs to serial.
//     embassy_usb_logger::run!(1024, log::LevelFilter::Info, driver);
// }
