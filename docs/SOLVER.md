# Type solver

## Type variables
 * Unknown type, eg. `'a`, is_parameter
 * Known type, eg. `*mut [i32]`, `struct vec['a]`, `fn[void, i32]`
    - Item, eg. `i32`, `pointer`, `struct`, `fn`
    - Parameters, `[type variables]`
 * Projections 
    - Field `'b`, `"foo"`
    - Associated type `'c`, optional trait, `"Output"`

## Constraints
 * Unify
     - (Unknown a, any b) => a = b
     - (Known a, Known b) => 
        valid if a.item == b.item,
        union parameters

```
where
| 'a: Add<u32, Output>
| 'a::Output: Mul<u32>
```

<'a as Add<'b>>::Output = 