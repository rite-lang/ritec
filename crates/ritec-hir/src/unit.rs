use crate::{
    Bodies, Body, BodyId, Contract, ContractId, Contracts, EnumDef, EnumId, Enums, Impl, LangTrait,
    Module, ModuleId, Modules, StructDef, StructId, Structs, TraitDef, TraitId, TraitImpl, Traits,
    TyEnv,
};

/// A  single compilation unit.
#[derive(Clone, Debug, Default)]
pub struct Unit {
    pub modules: Modules,
    pub bodies: Bodies,
    pub structs: Structs,
    pub enums: Enums,
    pub traits: Traits,
    pub contracts: Contracts,
    pub env: TyEnv,
    pub impls: Vec<Impl>,
    pub trait_impls: Vec<TraitImpl>,
}

impl Unit {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_lang_trait(&self, lang_trait: LangTrait) -> Option<TraitId> {
        self.traits.iter().find_map(|(id, trait_def)| {
            if trait_def.lang_trait == Some(lang_trait) {
                Some(id)
            } else {
                None
            }
        })
    }
}

macro_rules! impl_unit_index {
    ($unit:ident, $items:ident, $item:ty, $index:ty) => {
        impl ::std::ops::Index<$index> for $unit {
            type Output = $item;

            fn index(&self, index: $index) -> &Self::Output {
                &self.$items[index]
            }
        }

        impl ::std::ops::IndexMut<$index> for $unit {
            fn index_mut(&mut self, index: $index) -> &mut Self::Output {
                &mut self.$items[index]
            }
        }
    };
}

impl_unit_index!(Unit, modules, Module, ModuleId);
impl_unit_index!(Unit, bodies, Body, BodyId);
impl_unit_index!(Unit, structs, StructDef, StructId);
impl_unit_index!(Unit, enums, EnumDef, EnumId);
impl_unit_index!(Unit, traits, TraitDef, TraitId);
impl_unit_index!(Unit, contracts, Contract, ContractId);
