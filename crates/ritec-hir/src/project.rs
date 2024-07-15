use ritec_diagnostic::Diagnostic;

use crate::{ContractId, Projected, Projection, Specialization, TraitId, TraitImpl, Type, Types};

impl Types {
    fn applies(
        &self,
        implementor: &Type,
        candidate: &Type,
        specialization: &mut Specialization,
    ) -> bool {
        match (candidate, implementor) {
            (Type::Unknown(implementor), candidate) => {
                match self.substitutions.get(&implementor.uid) {
                    Some(implementor) => self.applies(candidate, implementor, specialization),
                    None => false,
                }
            }
            (_, Type::Unknown(candidate)) => match self.substitutions.get(&candidate.uid) {
                Some(candidate) => self.applies(candidate, implementor, specialization),
                None => false,
            },
            (_, Type::Generic(forall)) => match specialization.get(*forall).cloned() {
                Some(implementor) => self.applies(candidate, &implementor, specialization),
                None => {
                    specialization.insert(*forall, candidate.clone());
                    true
                }
            },
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

    fn fetch_trait_impl(
        &self,
        trait_: TraitId,
        generics: &[Type],
        for_: &Type,
    ) -> Result<&TraitImpl, Diagnostic> {
        for trait_impl in &self.trait_impls {
            if trait_impl.trait_ != trait_ {
                continue;
            }

            let mut specialization = Specialization::new();

            if !self.applies(&trait_impl.for_, for_, &mut specialization) {
                continue;
            }

            let mut applies = true;

            for (generic, implementor) in generics.iter().zip(&trait_impl.generics) {
                applies &= self.applies(generic, implementor, &mut specialization);
            }

            if !applies {
                continue;
            }

            if self
                .satisfy_specialized(trait_impl.contract, &specialization)
                .is_err()
            {
                continue;
            }

            return Ok(trait_impl);
        }

        let message = format!("trait {} not implemented for {}", trait_, for_);
        let diagnostic = Diagnostic::new(message);
        Err(diagnostic)
    }

    pub(crate) fn satisfy_specialized(
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

            self.fetch_trait_impl(bound.trait_id, &generics, &base)?;
        }

        Ok(())
    }

    pub fn satisfy(&self, contract: ContractId) -> Result<(), Diagnostic> {
        self.satisfy_specialized(contract, &Specialization::new())
    }

    pub(crate) fn project(&self, projected: &Projected) -> Result<Type, Diagnostic> {
        match projected.projection {
            Projection::Associated {
                trait_id: trait_,
                ref generics,
                index,
            } => {
                let trait_impl = self.fetch_trait_impl(trait_, generics, &projected.base)?;
                Ok(trait_impl.types[index].clone())
            }
        }
    }
}
