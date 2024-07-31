use std::{
    collections::{HashMap, VecDeque},
    sync::atomic::{AtomicUsize, Ordering},
};

use ritec_diagnostic::Diagnostic;

use crate::{EnumId, StructId, TraitId};

/// A representation of a type in the HIR.
#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    /// A type that is not yet known.
    Unknown(UnknownTy),

    /// A type that is partially or fully known.
    Partial(TyPart, Vec<Ty>),

    /// A generic type.
    Generic(Generic),

    /// A type that is a projection of another type.
    Proj(ProjTy),
}

impl Ty {
    pub const VOID: Self = Self::Partial(TyPart::Void, Vec::new());
    pub const BOOL: Self = Self::Partial(TyPart::Bool, Vec::new());

    pub fn new_unknown(kind: Option<UnknownKind>) -> Self {
        Self::Unknown(UnknownTy::new(kind))
    }
}

impl From<Generic> for Ty {
    fn from(generic: Generic) -> Self {
        Self::Generic(generic)
    }
}

/// A known type.
#[derive(Clone, Debug, PartialEq)]
pub enum KnownTy {
    /// A type that is partially or fully known.
    Partial(TyPart, Vec<KnownTy>),

    /// A generic type.
    Generic(Generic),

    /// A type that is associated with a trait implementation.
    Assoc(Box<KnownTy>, TraitId, Vec<KnownTy>, usize),
}

impl KnownTy {
    pub const VOID: Self = Self::Partial(TyPart::Void, Vec::new());
    pub const BOOL: Self = Self::Partial(TyPart::Bool, Vec::new());

    fn slice_to_tyes(slice: &[KnownTy]) -> Vec<Ty> {
        slice.iter().map(|ty| ty.to_ty()).collect()
    }

    pub fn to_ty(&self) -> Ty {
        match self {
            Self::Partial(part, tys) => Ty::Partial(part.clone(), Self::slice_to_tyes(tys)),

            Self::Generic(generic) => Ty::Generic(*generic),

            Self::Assoc(base, trait_id, trait_generics, assoc_index) => {
                let proj = Projection::Assoc {
                    trait_id: *trait_id,
                    trait_generics: Self::slice_to_tyes(trait_generics),
                    assoc_index: *assoc_index,
                };

                Ty::Proj(ProjTy {
                    base: Box::new(base.to_ty()),
                    proj,
                })
            }
        }
    }
}

impl From<Generic> for KnownTy {
    fn from(generic: Generic) -> Self {
        Self::Generic(generic)
    }
}

pub trait FromPartial: Sized {
    fn new_int(signed: bool, width: Option<u16>) -> Self {
        Self::from_partial(TyPart::Int { signed, width }, Vec::new())
    }

    fn new_float(width: u16) -> Self {
        Self::from_partial(TyPart::Float { width }, Vec::new())
    }

    fn new_pointer(mutable: bool, pointee: Self) -> Self {
        Self::from_partial(TyPart::Pointer { mutable }, vec![pointee])
    }

    fn new_func(args: Vec<Self>, output: Self) -> Self {
        let mut arguments = Vec::with_capacity(args.len() + 1);
        arguments.push(output);
        arguments.extend(args);
        Self::from_partial(TyPart::Func, arguments)
    }

    fn new_struct(id: StructId, generics: Vec<Self>) -> Self {
        Self::from_partial(TyPart::Struct(id), generics)
    }

    fn new_enum(id: EnumId, generics: Vec<Self>) -> Self {
        Self::from_partial(TyPart::Enum(id), generics)
    }

    fn from_partial(part: TyPart, tys: Vec<Self>) -> Self;
}

impl FromPartial for KnownTy {
    fn from_partial(part: TyPart, tys: Vec<Self>) -> Self {
        Self::Partial(part, tys)
    }
}

impl FromPartial for Ty {
    fn from_partial(part: TyPart, tys: Vec<Self>) -> Self {
        Self::Partial(part, tys)
    }
}

/// A part of a known type.
#[derive(Clone, Debug, PartialEq)]
pub enum TyPart {
    Void,
    Bool,
    Int { signed: bool, width: Option<u16> },
    Float { width: u16 },
    Pointer { mutable: bool },
    Tuple,
    Slice,
    Func,
    Struct(StructId),
    Enum(EnumId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnknownKind {
    Number(bool),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnknownTy {
    kind: Option<UnknownKind>,
    index: usize,
}

impl UnknownTy {
    pub fn new(kind: Option<UnknownKind>) -> Self {
        static INDEX: AtomicUsize = AtomicUsize::new(0);
        let index = INDEX.fetch_add(1, Ordering::SeqCst);
        Self { kind, index }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Generic {
    id: usize,
}

impl Default for Generic {
    fn default() -> Self {
        Self::new()
    }
}

impl Generic {
    pub fn new() -> Self {
        static ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: ID.fetch_add(1, Ordering::SeqCst),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjTy {
    pub base: Box<Ty>,
    pub proj: Projection,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Projection {
    Assoc {
        trait_id: TraitId,
        trait_generics: Vec<Ty>,
        assoc_index: usize,
    },

    Deref,

    Method {
        name: String,
        generics: Vec<Ty>,
    },

    Field {
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Spec<T = Ty> {
    generics: Vec<(Generic, T)>,
}

impl<T> Default for Spec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Spec<T> {
    pub fn new() -> Self {
        Self {
            generics: Vec::new(),
        }
    }

    pub fn specify<'a>(
        &mut self,
        generics: impl IntoIterator<Item = &'a Generic>,
        tys: impl IntoIterator<Item = &'a T>,
    ) -> Result<(), Diagnostic>
    where
        T: Clone + From<Generic> + 'a,
    {
        let mut generics = generics.into_iter();

        for ty in tys {
            match generics.next() {
                Some(generic) => self.generics.push((*generic, ty.clone())),
                None => {
                    let message = "too many types specified";
                    return Err(Diagnostic::new(message));
                }
            }
        }

        for &generic in generics {
            self.insert(generic, T::from(generic));
        }

        Ok(())
    }

    pub fn specified<'a>(
        generics: impl IntoIterator<Item = &'a Generic>,
        tys: impl IntoIterator<Item = &'a T>,
    ) -> Result<Self, Diagnostic>
    where
        T: Clone + From<Generic> + 'a,
    {
        let mut spec = Self::new();
        spec.specify(generics, tys)?;
        Ok(spec)
    }

    pub fn insert(&mut self, generic: Generic, ty: T) {
        self.generics.push((generic, ty));
    }

    pub fn extend(&mut self, other: Self) {
        self.generics.extend(other.generics);
    }

    pub fn get(&self, generic: Generic) -> Option<&T> {
        self.generics
            .iter()
            .find(|(g, _)| *g == generic)
            .map(|(_, ty)| ty)
    }

    pub fn specialize(&self, ty: &KnownTy) -> T
    where
        T: Specialize + Clone,
    {
        match ty {
            KnownTy::Partial(part, tys) => {
                let tys = tys.iter().map(|ty| self.specialize(ty)).collect();
                T::partial(part.clone(), tys)
            }
            KnownTy::Generic(generic) => match self.get(*generic) {
                Some(ty) => ty.clone(),
                None => T::generic(*generic),
            },
            KnownTy::Assoc(base, id, tys, idx) => {
                let base = Box::new(self.specialize(base));
                let tys = tys.iter().map(|ty| self.specialize(ty)).collect();

                T::assoc(base, *id, tys, *idx)
            }
        }
    }
}

impl Spec<KnownTy> {
    pub fn to_ty(&self) -> Spec<Ty> {
        let generics = self
            .generics
            .iter()
            .map(|(generic, ty)| (*generic, ty.to_ty()))
            .collect();

        Spec { generics }
    }
}

pub trait Specialize: Sized {
    fn partial(part: TyPart, tys: Vec<Self>) -> Self;
    fn generic(generic: Generic) -> Self;
    fn assoc(base: Box<Self>, id: TraitId, tys: Vec<Self>, idx: usize) -> Self;
}

impl Specialize for Ty {
    fn partial(part: TyPart, tys: Vec<Self>) -> Self {
        Self::Partial(part, tys)
    }

    fn generic(generic: Generic) -> Self {
        Self::Generic(generic)
    }

    fn assoc(base: Box<Self>, id: TraitId, tys: Vec<Self>, idx: usize) -> Self {
        Self::Proj(ProjTy {
            base,
            proj: Projection::Assoc {
                trait_id: id,
                trait_generics: tys,
                assoc_index: idx,
            },
        })
    }
}

impl Specialize for KnownTy {
    fn partial(part: TyPart, tys: Vec<Self>) -> Self {
        Self::Partial(part, tys)
    }

    fn generic(generic: Generic) -> Self {
        Self::Generic(generic)
    }

    fn assoc(base: Box<Self>, id: TraitId, tys: Vec<Self>, idx: usize) -> Self {
        Self::Assoc(base, id, tys, idx)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TyConstraint {
    /// A constraint that signifies that data from the source
    /// type flows into the destination type.
    Assign { src: Ty, dst: Ty },
}

#[derive(Clone, Debug, Default)]
pub struct TyEnv {
    constraints: VecDeque<TyConstraint>,
    substitutions: HashMap<UnknownTy, Ty>,
}

impl TyEnv {
    pub fn new() -> Self {
        Self {
            constraints: VecDeque::new(),
            substitutions: HashMap::new(),
        }
    }

    pub fn add_constraint(&mut self, constraint: TyConstraint) {
        self.constraints.push_back(constraint);
    }

    pub fn add_substitution(&mut self, unknown: UnknownTy, ty: Ty) {
        self.substitutions.insert(unknown, ty);
    }

    pub fn assign(&mut self, src: Ty, dst: Ty) {
        self.add_constraint(TyConstraint::Assign { src, dst });
    }

    pub fn get_substitution(&self, unknown: &UnknownTy) -> Option<&Ty> {
        self.substitutions.get(unknown)
    }
}
