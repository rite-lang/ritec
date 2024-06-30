use std::collections::HashMap;

use crate::Variable;

#[derive(Clone, Debug)]
pub struct World {
    substitions: HashMap<Variable, Variable>,
}
