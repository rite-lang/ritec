# MIR - Middle intermediary representation

## Types
The following are types that mir can represent.

 * Void 
 * Booleans
 * Integers, eg. `i32`, `u32`, `isize`, `usize`
 * Floating point numbers, eg. `f32`, `f64`
 * Pointers, eg. `*i32`, `*mut i32`
 * Arrays, eg. `[i32; 6]`
 * Structs, eg. `struct { i32, bool, *mut f64 }`
 * Function pointers:, eg. `fn(i32, u32) -> bool`
