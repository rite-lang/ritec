use std::{
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{Partial, Projected, Projection, Type};

/// A unique identifier for a generic type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Generic {
    id: usize,
}

impl Generic {
    /// Create a new unique identifier.
    pub fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
        }
    }
}

impl Display for Generic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "'{}", self.id)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Specialization {
    items: Vec<(Generic, Type)>,
}

impl Specialization {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, generic: Generic, variable: Type) {
        self.items.push((generic, variable));
    }

    pub fn get(&self, generic: Generic) -> Option<&Type> {
        (self.items.iter()).find_map(|(g, v)| (*g == generic).then_some(v))
    }

    pub fn specialize(&self, variable: &Type) -> Type {
        match variable {
            Type::Unknown(_) => variable.clone(),
            Type::Partial(partial) => {
                let mut params = Vec::with_capacity(partial.params.len());

                for param in &partial.params {
                    params.push(self.specialize(param));
                }

                Type::Partial(Partial {
                    item: partial.item.clone(),
                    params,
                })
            }
            Type::Projected(projected) => {
                let base = self.specialize(&projected.base);

                let projection = match projected.projection {
                    Projection::Associated {
                        trait_id: trait_,
                        ref generics,
                        index,
                    } => {
                        let mut generics = generics.clone();

                        for generic in &mut generics {
                            *generic = self.specialize(generic);
                        }

                        Projection::Associated {
                            trait_id: trait_,
                            generics,
                            index,
                        }
                    }
                    _ => projected.projection.clone(),
                };

                Type::Projected(Projected {
                    contract: projected.contract,
                    base: Box::new(base),
                    projection,
                })
            }
            Type::Generic(forall) => match self.get(*forall) {
                Some(variable) => self.specialize(variable),
                None => Type::Generic(*forall),
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
