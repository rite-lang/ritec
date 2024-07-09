use std::{
    collections::HashMap,
    ops::{Index, IndexMut},
};

use ritec_diagnostic::Diagnostic;

use crate::{Known, Trait, TraitId, TraitImpl, Traits, Uid, Variable, Where, WhereId, Wheres};

#[derive(Clone, Debug, Default)]
pub struct World {
    pub traits: Traits,
    pub trait_impls: Vec<TraitImpl>,
    pub wheres: Wheres,
    pub substitutions: HashMap<Uid, Variable>,
}

impl World {
    pub fn new() -> World {
        World {
            traits: Traits::new(),
            trait_impls: Vec::new(),
            wheres: Wheres::new(),
            substitutions: HashMap::new(),
        }
    }

    pub fn add_substitution(&mut self, uid: Uid, substitute: Variable) {
        self.substitutions.insert(uid, substitute);
    }

    pub fn try_substitute(&self, variable: &Variable) -> Option<&Variable> {
        match variable {
            Variable::Unknown(unknown) => self.substitutions.get(&unknown.uid),
            _ => None,
        }
    }

    pub fn substitute(&self, variable: &Variable) -> Variable {
        match self.try_substitute(variable) {
            Some(substitute) => self.substitute(substitute),
            None => variable.clone(),
        }
    }

    pub fn query(&self, variable: &Variable) -> Result<Known, Diagnostic> {
        if let Some(substitute) = self.try_substitute(variable) {
            return self.query(substitute);
        }

        match variable {
            Variable::Unknown(_) => {
                let diagnostic = Diagnostic::new("unknown type");
                Err(diagnostic)
            }
            Variable::Partial(partial) => {
                let mut params = Vec::with_capacity(partial.params.len());

                for param in &partial.params {
                    params.push(self.query(param)?);
                }

                Ok(Known {
                    item: partial.item.clone(),
                    params,
                })
            }
            Variable::Projected(projected) => self.query(&self.project(projected)?),
            Variable::Forall(_) => todo!(),
        }
    }
}

impl Index<TraitId> for World {
    type Output = Trait;

    fn index(&self, index: TraitId) -> &Self::Output {
        &self.traits[index]
    }
}

impl IndexMut<TraitId> for World {
    fn index_mut(&mut self, index: TraitId) -> &mut Self::Output {
        &mut self.traits[index]
    }
}

impl Index<WhereId> for World {
    type Output = Where;

    fn index(&self, index: WhereId) -> &Self::Output {
        &self.wheres[index]
    }
}

impl IndexMut<WhereId> for World {
    fn index_mut(&mut self, index: WhereId) -> &mut Self::Output {
        &mut self.wheres[index]
    }
}
