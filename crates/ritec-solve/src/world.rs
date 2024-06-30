use std::collections::HashMap;

use crate::Variable;

#[derive(Clone, Debug, Default)]
pub struct World {
    substitions: HashMap<Variable, Variable>,
}

impl World {
    pub fn new() -> World {
        World {
            substitions: HashMap::new(),
        }
    }
}
