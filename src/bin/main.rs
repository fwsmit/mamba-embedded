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

use mamba_embedded::data::test_tensor::input_tensor;

use mamba_embedded::mymodel::Model;

// Set the backend to NdArray with f32
type Backend = NdArray<f32>;

// ---------------------------------------------------------------------------
// Stack watermark — paint-and-scan high-water mark
//
// _stack_end   = lowest address of the stack region  (grows down toward here)
// _stack_start = highest address of the stack region (initial SP)
//
// These symbols come from esp-hal's link.x / esp-idf linker scripts.
// If your build fails to link, check the exact names with:
//   xtensa-esp32s3-elf-nm target/.../firmware | grep -i stack
// ---------------------------------------------------------------------------
mod stack_watermark {
    /// Byte written across the unpainted stack at startup.
    const PATTERN: u8 = 0xAA;

    /// Minimum guard margin left below the active frame when painting, so we
    /// don't accidentally stomp our own return address or saved registers.
    const PAINT_GUARD_BYTES: usize = 256;

    unsafe extern "C" {
        /// Bottom of the stack region (lowest valid stack address).
        static _stack_end: u8;
    }

    /// Fill the region from `_stack_end` up to just below the current frame
    /// with `PATTERN`. Call this as early as possible in `main`.
    ///
    /// # Safety
    /// Writes to stack memory below the current frame. The `PAINT_GUARD_BYTES`
    /// margin keeps us from touching live data, but call this before any
    /// non-trivial work is pushed onto the stack.
    pub unsafe fn paint() {
        // Use a local variable's address as a conservative approximation of SP.
        let local: u8 = 0;
        let current_sp = core::ptr::addr_of!(local) as usize;
        let bottom = core::ptr::addr_of!(_stack_end) as usize;

        let paint_top = current_sp.saturating_sub(PAINT_GUARD_BYTES);
        if paint_top <= bottom {
            return; // nothing safe to paint
        }

        let len = paint_top - bottom;
        // SAFETY: this range is below our current frame and within the stack.
        let slice = unsafe { core::slice::from_raw_parts_mut(bottom as *mut u8, len) };
        slice.fill(PATTERN);
    }

    /// Scan from `_stack_end` upward and return the number of bytes that have
    /// been touched since `paint()` was called, i.e. the peak stack depth.
    ///
    /// # Safety
    /// Reads stack memory. Call after `paint()` has been called.
    pub unsafe fn peak_bytes() -> usize {
        let local: u8 = 0;
        let current_sp = core::ptr::addr_of!(local) as usize;
        let bottom = core::ptr::addr_of!(_stack_end) as usize;

        if current_sp <= bottom {
            return 0;
        }

        let len = current_sp - bottom;
        // SAFETY: this range is within the stack region we previously painted.
        let slice = unsafe { core::slice::from_raw_parts(bottom as *const u8, len) };

        // Count intact pattern bytes from the very bottom of the stack.
        let intact = slice.iter().take_while(|&&b| b == PATTERN).count();

        // Peak usage = total scanned range minus the still-pristine prefix.
        len - intact
    }
}

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
    unsafe { stack_watermark::paint() };

    // Initialize allocator with a size
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    esp_println::logger::init_logger_from_env();
    info!("Started");

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    // Log baseline stack usage right after init — before inference overhead.
    // SAFETY: paint() was called above.
    let baseline = unsafe { stack_watermark::peak_bytes() };
    info!("Stack after init (baseline): {} bytes", baseline);

    let model: Model<Backend> = Model::default();
    info!("Running inference");

    let dataset = env!("DATASET");

    loop {
        let input = input_tensor();
        info!("Input shape: {:?}", input.shape());

        let inference_timer = Instant::now();
        let output = model.forward(input);
        let inference_time = inference_timer.elapsed();

        info!("{:?}", &output);
        let guess = output.argmax(1).into_scalar();
        info!("Guess: {:?}", guess);
        info!("Dataset: {}", dataset);
        info!("Inference done in: {:?} ms", inference_time.as_millis());

        // -----------------------------------------------------------------------
        // Report peak stack usage since paint() was called.
        // This accumulates over the loop — the number monotonically increases
        // (or stays flat) as deeper call paths are exercised.
        // -----------------------------------------------------------------------
        // SAFETY: paint() was called at entry to main.
        let peak = unsafe { stack_watermark::peak_bytes() };
        info!("Peak stack usage: {} bytes", peak);

        info!("Finished");
        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(500) {}
    }
}
