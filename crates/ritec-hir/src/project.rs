use ritec_diagnostic::{Diagnostic, Span};

use crate::{
    ContractId, Item, Method, Partial, Projected, Projection, Spec, StructId, TraitId, TraitImpl,
    Type, Types,
};

impl Types {
    fn applies(
        &self,
        candidate: &Type,
        implementor: &Type,
        specialization: &mut Spec,
        infer: bool,
    ) -> bool {
        match (candidate, implementor) {
            (Type::Unknown(candidate), implementor) => {
                match self.substitutions.get(&candidate.uid) {
                    None => infer,
                    Some(candidate) => self.applies(candidate, implementor, specialization, infer),
                }
            }

            (candidate, Type::Unknown(implementor)) => {
                match self.substitutions.get(&implementor.uid) {
                    Some(implementor) => {
                        self.applies(candidate, implementor, specialization, infer)
                    }
                    None => false,
                }
            }

            (candidate, Type::Generic(generic)) => {
                if candidate == implementor {
                    return true;
                }

                match specialization.get(*generic).cloned() {
                    Some(implementor) => {
                        self.applies(candidate, &implementor, specialization, infer)
                    }
                    None => {
                        specialization.insert(*generic, candidate.clone());
                        true
                    }
                }
            }

            (Type::Generic(generic), implementor) => {
                if candidate == implementor {
                    return true;
                }

                match specialization.get(*generic).cloned() {
                    Some(candidate) => self.applies(&candidate, implementor, specialization, infer),
                    None => false,
                }
            }

            (Type::Projected(candidate), _) => match self.project(candidate, specialization) {
                Ok(Some(candidate)) => self.applies(&candidate, implementor, specialization, infer),
                Ok(None) => false,
                Err(_) => false,
            },

            (Type::Partial(candidate), Type::Partial(implementor)) => {
                let mut applies = candidate.item == implementor.item;

                for (candidate, implementor) in candidate.params.iter().zip(&implementor.params) {
                    applies &= self.applies(candidate, implementor, specialization, infer);
                }

                applies
            }

            _ => false,
        }
    }

    fn equal(&self, candidate: &Type, implementor: &Type, specialization: &Spec) -> bool {
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
        specialization: &Spec,
    ) -> Result<(&TraitImpl, Spec), Diagnostic> {
        for trait_impl in &self.trait_impls {
            if trait_impl.trait_id != trait_id {
                continue;
            }

            let mut specialization = specialization.clone();

            if !self.applies(
                implementor,
                &trait_impl.implementor,
                &mut specialization,
                false,
            ) {
                continue;
            }

            let mut applies = true;

            for (generic, implementor) in generics.iter().zip(&trait_impl.generics) {
                applies &= self.applies(generic, implementor, &mut specialization, false);
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
            .map(|ty| match self.know(ty, specialization) {
                Ok(k) => format!("{}", k),
                Err(_) => String::from("_"),
            })
            .collect();

        let message = format!(
            "trait {}<{}> not implemented for {}",
            trait_.name.as_deref().unwrap_or("unnamed"),
            generics.join(", "),
            match self.know(implementor, specialization) {
                Ok(k) => format!("{}", k),
                Err(_) => String::from("_"),
            }
        );

        Err(Diagnostic::new(message))
    }

    pub fn satisfy_contract(
        &self,
        contract: ContractId,
        specialization: &Spec,
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

    fn project_associated(
        &self,
        implementor: &Type,
        trait_id: TraitId,
        generics: &[Type],
        contract_id: ContractId,
        index: usize,
        specialization: &Spec,
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
        specialization: &Spec,
    ) -> Result<(StructId, Vec<Type>, usize, usize), Diagnostic> {
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

        if let Item::Pointer { .. } = partial.item {
            let (a, b, c, d) = self.fetch_field_index(&partial.params[0], name, specialization)?;

            return Ok((a, b, c, d + 1));
        }

        let Item::Struct(struct_id) = partial.item else {
            let message = format!("expected struct, found {}", base);
            return Err(Diagnostic::new(message));
        };

        match self[struct_id].field_index(name) {
            Some(index) => Ok((struct_id, partial.params.clone(), index, 0)),
            None => {
                let message = format!("field {} not found in struct {}", name, base);
                Err(Diagnostic::new(message))
            }
        }
    }

    pub fn fetch_assoc_method(
        &self,
        implementor: &Type,
        name: &str,
        generics: &[Type],
        arguments: Option<&[Type]>,
        specialization: &Spec,
    ) -> Result<(&Method, Spec), Diagnostic> {
        for impl_ in self.impls.iter() {
            for method in impl_.methods.iter() {
                if method.name != name {
                    continue;
                }

                let mut specialization = specialization.clone();

                if let Some(arguments) = arguments {
                    let mut arguments = arguments.iter();

                    let mut applies = self.applies(
                        arguments.next().unwrap(),
                        &method.output,
                        &mut specialization,
                        true,
                    );

                    for (ty, argument) in arguments.zip(&method.arguments) {
                        applies &= self.applies(ty, argument, &mut specialization, true);
                    }

                    if !applies {
                        continue;
                    }
                }

                if !self.applies(implementor, &impl_.implementor, &mut specialization, true) {
                    continue;
                }

                if (self.satisfy_contract(impl_.contract, &specialization)).is_err() {
                    continue;
                }

                for generic in &impl_.generics {
                    specialization.insert(*generic, Type::unknown(Span { lo: 0, hi: 0 }));
                }

                for (generic, ty) in method.generics.iter().zip(generics) {
                    specialization.insert(*generic, ty.clone());
                }

                return Ok((method, specialization));
            }
        }

        // TODO: Add a span
        let message = format!("method {} not found in {}", name, implementor);
        Err(Diagnostic::new(message))
    }

    pub(crate) fn project(
        &self,
        projected: &Projected,
        specialization: &Spec,
    ) -> Result<Option<Type>, Diagnostic> {
        match projected.projection {
            Projection::TraitType {
                trait_id,
                ref generics,
                index,
            } => self.project_associated(
                &projected.base,
                trait_id,
                generics,
                projected.contract,
                index,
                specialization,
            ),
            Projection::AssocMethod {
                ref name,
                ref generics,
                ref arguments,
            } => {
                let (method, specialization) = self.fetch_assoc_method(
                    &projected.base,
                    name,
                    generics,
                    arguments.as_deref(),
                    specialization,
                )?;

                let mut params = Vec::new();

                params.push(method.output.clone());
                params.extend(method.arguments.clone());

                let ty = Type::Partial(Partial {
                    item: Item::Function,
                    params,
                });

                Ok(Some(specialization.specialize(&ty)))
            }
            Projection::Field { ref name } => {
                let (struct_id, params, index, _) =
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
