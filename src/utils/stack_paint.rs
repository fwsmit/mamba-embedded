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
