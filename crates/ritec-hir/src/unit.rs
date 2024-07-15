use crate::{Bodies, Modules, Types};

/// A a single compilation unit.
#[derive(Clone, Debug, Default)]
pub struct Unit {
    pub modules: Modules,
    pub types: Types,
    pub bodies: Bodies,
}
