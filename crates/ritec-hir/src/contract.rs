use ritec_diagnostic::Span;

use crate::{KnownTy, TraitId};

ritec_arena::arena!(Contracts[ContractId]: Contract);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Contract {
    pub bounds: Vec<Bound>,
}

impl Contract {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bound {
    pub implementor: KnownTy,
    pub trait_id: TraitId,
    pub generics: Vec<KnownTy>,
    pub assocs: Vec<Option<KnownTy>>,
    pub span: Option<Span>,
}
