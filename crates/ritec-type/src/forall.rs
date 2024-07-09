use std::{
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{Partial, Projected, Projection, Variable, WhereId};

/// A unique identifier for a generic type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Forall {
    where_id: WhereId,
    id: usize,
}

impl Forall {
    /// Create a new unique identifier.
    pub fn new(where_id: WhereId) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            where_id,
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
        }
    }

    pub fn where_id(&self) -> WhereId {
        self.where_id
    }
}

impl Display for Forall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "'{}", self.id)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Specialization {
    items: Vec<(Forall, Variable)>,
}

impl Specialization {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, generic: Forall, variable: Variable) {
        self.items.push((generic, variable));
    }

    pub fn get(&self, generic: Forall) -> Option<&Variable> {
        (self.items.iter()).find_map(|(g, v)| (*g == generic).then_some(v))
    }

    pub fn specialize(&self, variable: &Variable) -> Variable {
        match variable {
            Variable::Unknown(_) => variable.clone(),
            Variable::Partial(partial) => {
                let mut params = Vec::with_capacity(partial.params.len());

                for param in &partial.params {
                    params.push(self.specialize(param));
                }

                Variable::Partial(Partial {
                    item: partial.item.clone(),
                    params,
                })
            }
            Variable::Projected(projected) => {
                let base = self.specialize(&projected.base);

                let projection = match projected.projection {
                    Projection::Associated {
                        trait_,
                        ref generics,
                        index,
                    } => {
                        let mut generics = generics.clone();

                        for generic in &mut generics {
                            *generic = self.specialize(generic);
                        }

                        Projection::Associated {
                            trait_,
                            generics,
                            index,
                        }
                    }
                };

                Variable::Projected(Projected {
                    where_: projected.where_,
                    base: Box::new(base),
                    projection,
                })
            }
            Variable::Forall(forall) => match self.get(*forall) {
                Some(variable) => variable.clone(),
                None => Variable::Forall(*forall),
            },
        }
    }
}

impl Display for Specialization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let items: Vec<_> = self
            .items
            .iter()
            .map(|(generic, variable)| format!("{} = {}", generic, variable))
            .collect();

        write!(f, "<{}>", items.join(", "))
    }
}
