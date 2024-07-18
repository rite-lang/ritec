# Rite Module system overview

## `mod` keyword

The `mod` keyword exposes a different file or folder module in the current context
as a namespace with the file or folder name.

When using `mod` on

```
main.ri
mul.ri
ops/
    - mod.ri
    - add.ri
    - sub.fi
ext/
    - div.ri
```

> Example module structure for this example.

```rite

// ext/div.ri
fn div(a, b) => a / b

// ops/sub.ri
fn sub(a, b) => a - b

// ops/add.ri
fn add(a, b) => a + b

// ops/mod.ri
mod sub
mod add

sub::sub(1, 2) // 1 - 2 = -1
add::add(1, 2) // 1 + 2 = 3

// mul.ri
fn mul(a, b) => a * b

// main.ri
mod mul
mod ops
mod ext

mul::mul(2, 3)      // 2 * 3 = 6
ops::add::add(2, 3) // 2 + 3 = 5
ops::sub::sub(2, 3) // 2 - 3 = -1
ext::div::div(2, 3) // 2 / 3 = 0.66
```

## `use` keyword

```rite
use ops::add::add
use ops::{add::add, sub::sub as subtract}
```

```rite
use Option::{Some, None}
```
