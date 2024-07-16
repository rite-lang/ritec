use ritec_diagnostic::Diagnostic;

use crate::{
    ContractId, Item, Projected, Projection, Specialization, StructId, TraitId, TraitImpl, Type,
    Types,
};

impl Types {
    fn applies(
        &self,
        candidate: &Type,
        implementor: &Type,
        specialization: &mut Specialization,
    ) -> bool {
        match (candidate, implementor) {
            (Type::Unknown(candidate), implementor) => {
                match self.substitutions.get(&candidate.uid) {
                    Some(candidate) => self.applies(candidate, implementor, specialization),
                    None => false,
                }
            }

            (candidate, Type::Unknown(implementor)) => {
                match self.substitutions.get(&implementor.uid) {
                    Some(implementor) => self.applies(candidate, implementor, specialization),
                    None => false,
                }
            }

            (Type::Projected(candidate), _) => match self.project(candidate, specialization) {
                Ok(Some(candidate)) => self.applies(&candidate, implementor, specialization),
                Ok(None) => false,
                Err(_) => false,
            },

            (Type::Generic(candidate), implementor) => {
                match specialization.get(*candidate).cloned() {
                    Some(candidate) => self.applies(&candidate, implementor, specialization),
                    None => true,
                }
            }

            (candidate, Type::Generic(implementor)) => {
                match specialization.get(*implementor).cloned() {
                    Some(implementor) => self.applies(candidate, &implementor, specialization),
                    None => {
                        specialization.insert(*implementor, candidate.clone());
                        true
                    }
                }
            }

            (Type::Partial(candidate), Type::Partial(implementor)) => {
                let mut applies = candidate.item == implementor.item;

                for (candidate, implementor) in candidate.params.iter().zip(&implementor.params) {
                    applies &= self.applies(candidate, implementor, specialization);
                }

                applies
            }

            _ => false,
        }
    }

    fn equal(&self, candidate: &Type, implementor: &Type, specialization: &Specialization) -> bool {
        match (candidate, implementor) {
            (Type::Unknown(candidate), implementor) => {
                match self.substitutions.get(&candidate.uid) {
                    Some(candidate) => self.equal(candidate, implementor, specialization),
                    None => false,
                }
            }

            (candidate, Type::Unknown(implementor)) => {
                match self.substitutions.get(&implementor.uid) {
                    Some(implementor) => self.equal(candidate, implementor, specialization),
                    None => false,
                }
            }

            (Type::Projected(candidate), _) => match self.project(candidate, specialization) {
                Ok(Some(candidate)) => self.equal(&candidate, implementor, specialization),
                Ok(None) => false,
                Err(_) => false,
            },

            (_, Type::Projected(implementor)) => match self.project(implementor, specialization) {
                Ok(Some(implementor)) => self.equal(candidate, &implementor, specialization),
                Ok(None) => false,
                Err(_) => false,
            },

            (Type::Generic(candidate), implementor) => {
                match specialization.get(*candidate).cloned() {
                    Some(candidate) => self.equal(&candidate, implementor, specialization),
                    None => true,
                }
            }

            (Type::Partial(candidate), Type::Partial(implementor)) => {
                if candidate.item != implementor.item {
                    return false;
                }

                if candidate.params.len() != implementor.params.len() {
                    return false;
                }

                for (candidate, implementor) in candidate.params.iter().zip(&implementor.params) {
                    if !self.equal(candidate, implementor, specialization) {
                        return false;
                    }
                }

                true
            }

            _ => false,
        }
    }

    pub fn fetch_trait_impl(
        &self,
        trait_id: TraitId,
        generics: &[Type],
        implementor: &Type,
        specialization: &Specialization,
    ) -> Result<(&TraitImpl, Specialization), Diagnostic> {
        for trait_impl in &self.trait_impls {
            if trait_impl.trait_id != trait_id {
                continue;
            }

            let mut specialization = specialization.clone();

            if !self.applies(implementor, &trait_impl.implementor, &mut specialization) {
                continue;
            }

            let mut applies = true;

            for (generic, implementor) in generics.iter().zip(&trait_impl.generics) {
                let applied = self.applies(generic, implementor, &mut specialization);
                applies &= applied;
            }

            if !applies {
                continue;
            }

            if self
                .satisfy_contract(trait_impl.contract, &specialization)
                .is_err()
            {
                continue;
            }

            return Ok((trait_impl, specialization));
        }

        let trait_ = &self.traits[trait_id];
        let generics: Vec<_> = generics
            .iter()
            .map(|ty| match self.query(ty, specialization) {
                Ok(k) => format!("{}", k),
                Err(_) => String::from("_"),
            })
            .collect();

        let message = format!(
            "trait {}<{}> not implemented for {}",
            trait_.name.as_deref().unwrap_or("unnamed"),
            generics.join(", "),
            match self.query(implementor, specialization) {
                Ok(k) => format!("{}", k),
                Err(_) => String::from("_"),
            }
        );

        let diagnostic = Diagnostic::new(message);
        Err(diagnostic)
    }

    pub fn satisfy_contract(
        &self,
        contract: ContractId,
        specialization: &Specialization,
    ) -> Result<(), Diagnostic> {
        let contract = &self[contract];

        for bound in &contract.bounds {
            let base = specialization.specialize(&bound.base);

            let mut generics = Vec::with_capacity(bound.generics.len());

            for generic in &bound.generics {
                generics.push(specialization.specialize(generic));
            }

            self.fetch_trait_impl(bound.trait_id, &generics, &base, specialization)?;
        }

        Ok(())
    }

    fn associated(
        &self,
        implementor: &Type,
        trait_id: TraitId,
        generics: &[Type],
        contract_id: ContractId,
        index: usize,
        specialization: &Specialization,
    ) -> Result<Option<Type>, Diagnostic> {
        let contract = &self[contract_id];

        for bound in &contract.bounds {
            if bound.trait_id != trait_id {
                continue;
            }

            if !self.equal(&bound.base, implementor, specialization) {
                continue;
            }

            for (generic, implementor) in generics.iter().zip(&bound.generics) {
                if !self.equal(generic, implementor, specialization) {
                    continue;
                }
            }

            match bound.types[index] {
                Some(ref ty) => return Ok(Some(ty.clone())),
                None => return Ok(None),
            }
        }

        let (trait_impl, specialization) =
            self.fetch_trait_impl(trait_id, generics, implementor, specialization)?;
        let assoc = trait_impl.types[index].clone();
        Ok(Some(specialization.specialize(&assoc)))
    }

    pub fn fetch_field_index(
        &self,
        base: &Type,
        name: &str,
        specialization: &Specialization,
    ) -> Result<(StructId, Vec<Type>, usize), Diagnostic> {
        let partial = match base {
            Type::Unknown(unknown) => match self.substitutions.get(&unknown.uid) {
                Some(substitute) => {
                    return self.fetch_field_index(substitute, name, specialization)
                }
                None => {
                    let message = "unknown type";
                    return Err(Diagnostic::new(message));
                }
            },

            Type::Projected(projected) => match self.project(projected, &Default::default())? {
                Some(projected) => return self.fetch_field_index(&projected, name, specialization),
                None => {
                    let message = "projection failed";
                    return Err(Diagnostic::new(message));
                }
            },

            Type::Generic(generics) => match specialization.get(*generics) {
                Some(variable) => return self.fetch_field_index(variable, name, specialization),
                None => {
                    let message = "generic not specialized";
                    return Err(Diagnostic::new(message));
                }
            },

            Type::Partial(partial) => partial,
        };

        let Item::Struct(struct_id) = partial.item else {
            let message = format!("expected struct, found {}", base);
            return Err(Diagnostic::new(message));
        };

        match self[struct_id].field_index(name) {
            Some(index) => Ok((struct_id, partial.params.clone(), index)),
            None => {
                let message = format!("field {} not found in struct {}", name, base);
                Err(Diagnostic::new(message))
            }
        }
    }

    pub(crate) fn project(
        &self,
        projected: &Projected,
        specialization: &Specialization,
    ) -> Result<Option<Type>, Diagnostic> {
        match projected.projection {
            Projection::Associated {
                trait_id,
                ref generics,
                index,
            } => self.associated(
                &projected.base,
                trait_id,
                generics,
                projected.contract,
                index,
                specialization,
            ),
            Projection::Field { ref name } => {
                let (struct_id, params, index) =
                    self.fetch_field_index(&projected.base, name, specialization)?;

                let mut specialization = specialization.clone();

                for (&generic, param) in self[struct_id].generics.iter().zip(&params) {
                    specialization.insert(generic, param.clone());
                }

                let ty = self[struct_id].fields[index].ty.clone();

                Ok(Some(specialization.specialize(&ty)))
            }
        }
    }
}
