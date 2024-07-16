use crate::{Bodies, BodyId};

#[derive(Clone, Debug, Default)]
pub struct Unit {
    pub bodies: Bodies,
    pub entry: Option<BodyId>,
}
