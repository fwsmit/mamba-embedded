#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use burn::backend::NdArray;

use esp_alloc::HeapStats;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use log::debug;
use log::info;

use mamba_embedded::data::test_tensor::input_tensor;
use mamba_embedded::mymodel::Model;
use mamba_embedded::utils::stack_paint::*;

// Set the backend to NdArray with f32
type Backend = NdArray<f32>;

esp_bootloader_esp_idf::esp_app_desc!();

#[allow(
    clippy::large_stack_frames,
    reason = "it's not unusual to allocate larger buffers etc. in main"
)]
#[main]
fn main() -> ! {
    // -----------------------------------------------------------------------
    // Paint the stack FIRST, before any significant work is done.
    // The earlier this runs, the more accurate the high-water mark will be.
    // -----------------------------------------------------------------------
    // SAFETY: called at the very top of main, before the heap allocator or any
    // large local variables are initialised.
    unsafe { paint() };

    // Initialize allocator with a size
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    esp_println::logger::init_logger_from_env();
    info!("Started");

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    // Log baseline stack usage right after init — before inference overhead.
    // SAFETY: paint() was called above.
    let baseline = unsafe { peak_bytes() };
    info!("Stack after init (baseline): {} bytes", baseline);

    let model: Model<Backend> = Model::default();
    info!("Running inference");

    let dataset = env!("DATASET");
    let n_loops = 5;
    let mut peak_usage_paint = 0;
    let mut peak_usage_alloc = 0;
    let mut latency_total = 0;

    for _i in 0..n_loops {
        let input = input_tensor();

        let inference_timer = Instant::now();
        let output = model.forward(input);
        let inference_time = inference_timer.elapsed();

        debug!("{:?}", &output);
        let guess = output.argmax(1).into_scalar();
        debug!("Guess: {:?}", guess);
        debug!("Inference done in: {:?} ms", inference_time.as_millis());
        latency_total += inference_time.as_millis();

        // -----------------------------------------------------------------------
        // Report peak stack usage since paint() was called.
        // This accumulates over the loop — the number monotonically increases
        // (or stays flat) as deeper call paths are exercised.
        // -----------------------------------------------------------------------
        // SAFETY: paint() was called at entry to main.
        let peak_paint = unsafe { peak_bytes() };
        let stats: HeapStats = esp_alloc::HEAP.stats();
        debug!("Peak stack usage: {} bytes", peak_paint);
        debug!("Peak stack usage (heapstats){}", stats.max_usage);
        peak_usage_alloc += stats.max_usage;
        peak_usage_paint += peak_paint;
    }

    info!("Average stats:");
    info!(
        "Peak memory: {} (paint) {} (allocator)",
        peak_usage_paint / n_loops,
        peak_usage_alloc / n_loops,
    );
    info!("Latency {} (avg)", latency_total / n_loops as u64);
    info!("Finished");
    loop {
        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(500) {}
    }
}
