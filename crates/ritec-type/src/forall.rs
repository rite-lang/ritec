use std::fmt::Display;

use crate::WhereId;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Forall {
    pub where_: WhereId,
}

impl Display for Forall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "forall")
    }
}
