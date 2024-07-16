use std::{
    collections::{HashMap, VecDeque},
    ops::{Index, IndexMut},
};

use ritec_diagnostic::Diagnostic;

use crate::{
    Contract, ContractId, Contracts, Enums, Item, Known, Specialization, Struct, StructId, Structs,
    Trait, TraitId, TraitImpl, Traits, Type, Uid, UnknownKind,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Goal {
    Unify(Type, Type),
    Satisfy(ContractId, Specialization),
}

#[derive(Clone, Debug, Default)]
pub struct Types {
    pub structs: Structs,
    pub traits: Traits,
    pub enums: Enums,
    pub trait_impls: Vec<TraitImpl>,
    pub contracts: Contracts,
    pub substitutions: HashMap<Uid, Type>,
    pub goals: VecDeque<Goal>,
}

impl Types {
    pub fn new() -> Types {
        Types {
            structs: Structs::new(),
            traits: Traits::new(),
            enums: Enums::new(),
            trait_impls: Vec::new(),
            contracts: Contracts::new(),
            substitutions: HashMap::new(),
            goals: VecDeque::new(),
        }
    }

    pub fn add_substitution(&mut self, uid: Uid, substitute: Type) {
        self.substitutions.insert(uid, substitute);
    }

    pub fn try_substitute(&self, variable: &Type) -> Option<&Type> {
        match variable {
            Type::Unknown(unknown) => self.substitutions.get(&unknown.uid),
            _ => None,
        }
    }

    pub fn substitute(&self, variable: &Type) -> Type {
        match self.try_substitute(variable) {
            Some(substitute) => self.substitute(substitute),
            None => variable.clone(),
        }
    }

    pub fn query(
        &self,
        variable: &Type,
        specialization: &Specialization,
    ) -> Result<Known, Diagnostic> {
        if let Some(substitute) = self.try_substitute(variable) {
            return self.query(substitute, specialization);
        }

        match variable {
            Type::Unknown(unknown) => match unknown.kind {
                UnknownKind::Number { float } if float => {
                    let known = Known {
                        item: Item::Float { width: 32 },
                        params: Vec::new(),
                    };

                    Ok(known)
                }
                UnknownKind::Number { .. } => {
                    let known = Known {
                        item: Item::Int {
                            signed: true,
                            width: Some(32),
                        },
                        params: Vec::new(),
                    };

                    Ok(known)
                }
                _ => {
                    let diagnostic = Diagnostic::new("unknown type");
                    Err(diagnostic)
                }
            },
            Type::Partial(partial) => {
                let mut params = Vec::with_capacity(partial.params.len());

                for param in &partial.params {
                    params.push(self.query(param, specialization)?);
                }

                Ok(Known {
                    item: partial.item.clone(),
                    params,
                })
            }
            Type::Projected(projected) => match self.project(projected, specialization)? {
                Some(projected) => self.query(&projected, specialization),
                None => {
                    let diagnostic = Diagnostic::new("projection failed");
                    Err(diagnostic)
                }
            },
            Type::Generic(generic) => match specialization.get(*generic) {
                Some(variable) => self.query(variable, specialization),
                None => {
                    let diagnostic = Diagnostic::new("generic not specialized");
                    Err(diagnostic)
                }
            },
        }
    }

    pub fn push_goal(&mut self, goal: Goal) {
        self.goals.push_back(goal);
    }

    pub fn unify(&mut self, a: Type, b: Type) {
        self.push_goal(Goal::Unify(a, b));
    }

    pub fn satisfy(&mut self, contract: ContractId, specialization: &Specialization) {
        self.push_goal(Goal::Satisfy(contract, specialization.clone()));
    }

    fn solve_goal(&mut self, goal: &Goal) -> Result<bool, Diagnostic> {
        match goal {
            Goal::Unify(a, b) => self.unify_var_var(a, b),
            Goal::Satisfy(contract, specialization) => {
                self.satisfy_contract(*contract, specialization)?;
                Ok(true)
            }
        }
    }

    pub fn solve(&mut self) -> Result<(), Diagnostic> {
        let mut fail_count = 0;

        while let Some(goal) = self.goals.pop_front() {
            if self.solve_goal(&goal)? {
                fail_count = 0;
                continue;
            }

            fail_count += 1;
            self.goals.push_back(goal.clone());

            if fail_count > self.goals.len() + 10 {
                let diagnostic = Diagnostic::new("failed to solve goal");
                return Err(diagnostic);
            }
        }

        Ok(())
    }
}

impl Index<StructId> for Types {
    type Output = Struct;

    fn index(&self, index: StructId) -> &Self::Output {
        &self.structs[index]
    }
}

impl IndexMut<StructId> for Types {
    fn index_mut(&mut self, index: StructId) -> &mut Self::Output {
        &mut self.structs[index]
    }
}

impl Index<TraitId> for Types {
    type Output = Trait;

    fn index(&self, index: TraitId) -> &Self::Output {
        &self.traits[index]
    }
}

impl IndexMut<TraitId> for Types {
    fn index_mut(&mut self, index: TraitId) -> &mut Self::Output {
        &mut self.traits[index]
    }
}

impl Index<ContractId> for Types {
    type Output = Contract;

    fn index(&self, index: ContractId) -> &Self::Output {
        &self.contracts[index]
    }
}

impl IndexMut<ContractId> for Types {
    fn index_mut(&mut self, index: ContractId) -> &mut Self::Output {
        &mut self.contracts[index]
    }
}
