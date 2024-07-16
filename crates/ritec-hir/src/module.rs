use std::collections::HashMap;

use crate::{BodyId, Builtins, EnumId, StructId, TraitId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vis {
    Public,
    Private,
}

#[derive(Clone, Debug, Default)]
pub struct Module {
    pub modules: HashMap<String, ModuleId>,
    pub structs: HashMap<String, StructId>,
    pub traits: HashMap<String, TraitId>,
    pub enums: HashMap<String, EnumId>,
    pub funcs: HashMap<String, BodyId>,
}

impl Module {
    pub fn use_builtins(&mut self, builtins: &Builtins) {
        self.traits.insert(String::from("Add"), builtins.add_trait);
        self.traits.insert(String::from("Sub"), builtins.sub_trait);
        self.traits.insert(String::from("Mul"), builtins.mul_trait);
        self.traits.insert(String::from("Div"), builtins.div_trait);
        self.traits.insert(String::from("Eq"), builtins.eq_trait);
    }
}

ritec_arena::arena!(Modules[ModuleId]: Module);
