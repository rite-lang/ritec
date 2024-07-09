use ritec_diagnostic::Diagnostic;

use crate::{Projected, Projection, Specialization, TraitId, TraitImpl, Variable, WhereId, World};

impl World {
    fn applies(
        &self,
        implementor: &Variable,
        candidate: &Variable,
        specialization: &mut Specialization,
    ) -> bool {
        match (candidate, implementor) {
            (Variable::Unknown(implementor), candidate) => {
                match self.substitutions.get(&implementor.uid) {
                    Some(implementor) => self.applies(candidate, implementor, specialization),
                    None => false,
                }
            }
            (_, Variable::Unknown(candidate)) => match self.substitutions.get(&candidate.uid) {
                Some(candidate) => self.applies(candidate, implementor, specialization),
                None => false,
            },
            (_, Variable::Forall(forall)) => match specialization.get(*forall).cloned() {
                Some(implementor) => self.applies(candidate, &implementor, specialization),
                None => {
                    specialization.insert(*forall, candidate.clone());
                    true
                }
            },
            (Variable::Partial(candidate), Variable::Partial(implementor)) => {
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
        generics: &[Variable],
        for_: &Variable,
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
                .satisfy_specialized(trait_impl.where_, &specialization)
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
        where_id: WhereId,
        specialization: &Specialization,
    ) -> Result<(), Diagnostic> {
        let where_ = &self[where_id];

        for bound in &where_.bounds {
            let base = specialization.specialize(&bound.base);

            let mut generics = Vec::with_capacity(bound.generics.len());

            for generic in &bound.generics {
                generics.push(specialization.specialize(generic));
            }

            self.fetch_trait_impl(bound.trait_, &generics, &base)?;
        }

        Ok(())
    }

    pub(crate) fn satisfy(&self, where_id: WhereId) -> Result<(), Diagnostic> {
        self.satisfy_specialized(where_id, &Specialization::new())
    }

    pub(crate) fn project(&self, projected: &Projected) -> Result<Variable, Diagnostic> {
        match projected.projection {
            Projection::Associated {
                trait_,
                ref generics,
                index,
            } => {
                let trait_impl = self.fetch_trait_impl(trait_, generics, &projected.base)?;
                Ok(trait_impl.types[index].clone())
            }
        }
    }
}
