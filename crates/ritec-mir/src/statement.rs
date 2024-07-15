use crate::{Place, Value};

#[derive(Clone, Debug)]
pub enum Statement {
    Assign(Place, Value),
}
