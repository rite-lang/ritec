use std::collections::HashMap;

use crate::{Traits, Variable, Wheres};

#[derive(Clone, Debug, Default)]
pub struct World {
    pub traits: Traits,
    pub wheres: Wheres,
    substitions: HashMap<Variable, Variable>,
}

impl World {
    pub fn new() -> World {
        World {
            traits: Traits::new(),
            wheres: Wheres::new(),
            substitions: HashMap::new(),
        }
    }

    pub fn add_substitution(&mut self, variable: Variable, substitute: Variable) {
        self.substitions.insert(variable, substitute);
    }

    pub fn substitute(&self, variable: &Variable) -> Variable {
        match self.substitions.get(variable) {
            Some(substitute) => self.substitute(substitute),
            None => variable.clone(),
        }
    }
}
