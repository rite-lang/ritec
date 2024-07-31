use ritec_diagnostic::{Diagnostic, Span};

use crate::{BodyId, ContractId, Generic, KnownTy, Spec, Ty};

ritec_arena::arena!(Structs[StructId]: StructDef);
ritec_arena::arena!(Enums[EnumId]: EnumDef);

#[derive(Clone, Debug, PartialEq)]
pub struct StructDef {
    pub name: Option<String>,
    pub generics: Vec<Generic>,
    pub contract: ContractId,
    pub fields: Vec<FieldDef>,
    pub span: Option<Span>,
}

impl StructDef {
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    pub fn field_ty_known(&self, generics: &[KnownTy], index: usize) -> Result<Ty, Diagnostic> {
        let generics: Vec<_> = generics.iter().map(KnownTy::to_ty).collect();
        let spec = Spec::specified(&self.generics, &generics)?;
        Ok(spec.specialize(&self.fields[index].ty))
    }

    pub fn field_ty(&self, generics: &[Ty], index: usize) -> Result<Ty, Diagnostic> {
        let spec = Spec::specified(&self.generics, generics)?;
        Ok(spec.specialize(&self.fields[index].ty))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldDef {
    pub name: String,
    pub ty: KnownTy,
    pub span: Option<Span>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnumDef {
    pub name: Option<String>,
    pub generics: Vec<Generic>,
    pub contract: ContractId,
    pub variants: Vec<VariantDef>,
    pub span: Option<Span>,
}

impl EnumDef {
    pub fn variant_index(&self, name: &str) -> Option<usize> {
        self.variants.iter().position(|v| v.name == name)
    }

    pub fn field_ty(
        &self,
        generics: &[KnownTy],
        variant: usize,
        field: usize,
    ) -> Result<Ty, Diagnostic> {
        let generics: Vec<_> = generics.iter().map(KnownTy::to_ty).collect();
        let spec = Spec::specified(&self.generics, &generics)?;
        Ok(spec.specialize(&self.variants[variant].fields[field]))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VariantDef {
    pub name: String,
    pub fields: Vec<KnownTy>,
    pub builder: Option<BodyId>,
    pub span: Option<Span>,
}
