use std::ops::Index;

use crate::Type;

#[derive(Clone, Debug)]
pub struct LocalDecl {
    pub mutable: bool,
    pub ty: Type,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Local {
    pub index: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Locals {
    decls: Vec<Option<LocalDecl>>,
}

impl Locals {
    pub fn new() -> Self {
        Locals { decls: Vec::new() }
    }

    pub fn insert(&mut self, local: Local, decl: LocalDecl) {
        if local.index >= self.decls.len() {
            self.decls.resize_with(local.index + 1, || None);
        }

        self.decls[local.index] = Some(decl);
    }

    pub fn push(&mut self, decl: LocalDecl) -> Local {
        let local = Local {
            index: self.decls.len(),
        };

        self.decls.push(Some(decl));

        local
    }

    pub fn iter(&self) -> impl Iterator<Item = (Local, &LocalDecl)> {
        self.decls.iter().enumerate().filter_map(|(index, decl)| {
            // don't one-line please
            decl.as_ref().map(|decl| (Local { index }, decl))
        })
    }
}

impl Index<Local> for Locals {
    type Output = LocalDecl;

    fn index(&self, local: Local) -> &Self::Output {
        self.decls[local.index].as_ref().unwrap()
    }
}

impl Index<usize> for Locals {
    type Output = LocalDecl;

    fn index(&self, index: usize) -> &Self::Output {
        self.decls[index].as_ref().unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct Place {
    pub ty: Type,
    pub base: Local,
    pub projections: Vec<Projection>,
}

impl Place {
    pub fn ty(&self) -> &Type {
        match self.projections.last() {
            Some(projection) => &projection.ty,
            None => &self.ty,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Projection {
    pub kind: ProjectionKind,
    pub ty: Type,
}

#[derive(Clone, Debug)]
pub enum ProjectionKind {
    Deref,
    Field(usize),
}
