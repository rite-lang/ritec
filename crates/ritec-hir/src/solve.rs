use crate::{Bound, ContractId, KnownTy, Projection, TraitId, Ty, Unit};

impl Unit {
    pub fn is_sub_ty(&self, ty: &Ty, implementor: &KnownTy) -> bool {
        match (ty, implementor) {
            (Ty::Unknown(unknown), implementor) => {
                let ty = self.env.get_substitution(unknown)?;
                self.is_sub_ty(ty, implementor)
            }
            _ => false,
        }
    }

    pub fn try_resolve_ty(&self, ty: &Ty) -> Option<KnownTy> {
        match ty {
            Ty::Unknown(unknown) => {
                let ty = self.env.get_substitution(unknown)?;
                self.try_resolve_ty(ty)
            }
            Ty::Partial(part, params) => {
                let params = params
                    .iter()
                    .map(|param| self.try_resolve_ty(param))
                    .collect::<Option<_>>()?;

                Some(KnownTy::Partial(part.clone(), params))
            }
            Ty::Generic(generic) => Some(KnownTy::Generic(*generic)),
            Ty::Proj(proj) => match proj.proj {
                Projection::Assoc {
                    trait_id,
                    ref trait_generics,
                    assoc_index,
                } => {
                    let base = self.try_resolve_ty(&proj.base)?;
                    let trait_generics = trait_generics
                        .iter()
                        .map(|ty| self.try_resolve_ty(ty))
                        .collect::<Option<_>>()?;

                    Some(KnownTy::Assoc(
                        Box::new(base),
                        trait_id,
                        trait_generics,
                        assoc_index,
                    ))
                }
                Projection::Deref => todo!(),
                Projection::Method {
                    ref name,
                    ref generics,
                } => todo!(),
                Projection::Field { ref name } => todo!(),
            },
        }
    }

    pub fn applicable_bounds(&self, contract: ContractId, ty: &Ty) -> &[Bound] {
        let contract = &self[contract];

        todo!()
    }

    pub fn contract_requires(
        &self,
        contract: ContractId,
        ty: &Ty,
        trait_id: TraitId,
        generics: &[Ty],
        assocs: &[Option<Ty>],
    ) -> bool {
        let contract = &self[contract];

        for bound in &contract.bounds {
            if bound.trait_id != trait_id {
                continue;
            }

            if bound.generics.len() != generics.len() {
                continue;
            }
        }

        false
    }
}
