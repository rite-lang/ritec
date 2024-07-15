use std::collections::HashMap;

use crate::{BodyId, EnumId, StructId, TraitId};

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

ritec_arena::arena!(Modules[ModuleId]: Module);
