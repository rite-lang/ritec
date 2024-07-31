use hir::FromPartial;
use ritec_ast as ast;
use ritec_ast::PathSegment;
use ritec_diagnostic::Diagnostic;
use ritec_hir as hir;
use ritec_hir::ModuleId;

pub struct TyCx<'a> {
    pub module: ModuleId,
    pub generics: &'a mut Vec<(String, hir::Generic)>,
    pub allow_new_generics: bool,
    pub trait_id: Option<hir::TraitId>,
    pub self_ty: Option<hir::KnownTy>,
    pub spec: Option<&'a hir::Spec<hir::KnownTy>>,
}

#[derive(Debug)]
pub enum Resolved {
    Struct(hir::StructId, Vec<hir::KnownTy>),
    Trait(hir::TraitId, Vec<hir::KnownTy>),
    Enum(hir::EnumId, Vec<hir::KnownTy>),
    Func(hir::BodyId, Vec<hir::KnownTy>),
    AssocTy(hir::KnownTy, hir::TraitId, Vec<hir::KnownTy>, usize),
    Assoc(hir::KnownTy, String, Vec<hir::KnownTy>),
    TraitMethod(hir::TraitId, Vec<hir::KnownTy>, usize, Vec<hir::KnownTy>),
    EnumVariant(hir::EnumId, Vec<hir::KnownTy>, usize),
    Generic(hir::Generic),
    Module(hir::ModuleId),
    Ty(hir::KnownTy),
    SelfArgument,
    SelfTy,
}

impl<'a> TyCx<'a> {
    fn get_generic(&mut self, ast: &ast::Generic) -> Result<hir::Generic, Diagnostic> {
        for (name, generic) in self.generics.iter() {
            if name == &ast.name {
                return Ok(*generic);
            }
        }

        if !self.allow_new_generics {
            let message = format!("generic `{}` not found", ast.name);
            return Err(Diagnostic::new(message).with_span(ast.span));
        }

        let generic = hir::Generic::new();
        self.generics.push((ast.name.clone(), generic));
        Ok(generic)
    }

    pub fn resolve_path(
        &mut self,
        unit: &hir::Unit,
        ast: &ast::Path,
    ) -> Result<Resolved, Diagnostic> {
        // If the ast entry is just an identifier
        // And the type context has a trait_id
        // We are looking for an associated type from the trait
        if let (Some(ident), Some(trait_id), Some(self_ty)) =
            (ast.ident(), self.trait_id, self.self_ty.clone())
        {
            let trait_def = &unit[trait_id];

            if let Some(index) = trait_def.assoc_index(ident) {
                let generics: Vec<_> = trait_def
                    .generics
                    .iter()
                    .copied()
                    .map(hir::KnownTy::Generic)
                    .collect();

                return Ok(Resolved::AssocTy(
                    self_ty.clone(),
                    trait_id,
                    generics.clone(),
                    index,
                ));
            }
        }

        let mut resolved = Resolved::Module(self.module);

        for (i, segment) in ast.segments.iter().enumerate() {
            match resolved {
                Resolved::Module(module) => match segment {
                    PathSegment::Named(named) => {
                        let name = &named.name;
                        let module = &unit[module];
                        let generics: Vec<_> = named
                            .generics
                            .iter()
                            .map(|g| self.lower_type(unit, g))
                            .collect::<Result<_, _>>()?;

                        if let Some(&struct_id) = module.structs.get(name) {
                            resolved = Resolved::Struct(struct_id, generics);
                            continue;
                        }

                        if let Some(&enum_id) = module.enums.get(name) {
                            resolved = Resolved::Enum(enum_id, generics);
                            continue;
                        }

                        if let Some(&trait_id) = module.traits.get(name) {
                            resolved = Resolved::Trait(trait_id, generics);
                            continue;
                        }

                        if let Some(&func_id) = module.funcs.get(name) {
                            resolved = Resolved::Func(func_id, generics);
                            continue;
                        }

                        if let Some(module_id) = module.modules.get(name) {
                            if !generics.is_empty() {
                                return Err(Diagnostic::new("modules do not have generics")
                                    .with_span(named.span));
                            }

                            resolved = Resolved::Module(*module_id);
                            continue;
                        }

                        let message = format!("no item found for {:?}", segment);
                        return Err(Diagnostic::new(message).with_span(named.span));
                    }

                    PathSegment::Assoc(assoc) if i == 0 => {
                        let implementor = self.lower_type(unit, &assoc.implementor)?;
                        let trait_ = self.resolve_path(unit, &assoc.trait_path)?;

                        let Resolved::Trait(trait_id, generics) = trait_ else {
                            let message = "expected trait path";
                            return Err(Diagnostic::new(message).with_span(assoc.trait_path.span));
                        };

                        let Some(index) = unit[trait_id].assoc_index(&assoc.name) else {
                            let message = format!("no associated type found for `{}`", assoc.name);
                            return Err(Diagnostic::new(message).with_span(assoc.span));
                        };

                        resolved = Resolved::AssocTy(implementor, trait_id, generics, index);
                    }

                    PathSegment::Generic(generic) if i == 0 => {
                        let generic = self.get_generic(generic)?;

                        resolved = match self.spec {
                            Some(ref spec) => match spec.get(generic) {
                                Some(ty) => Resolved::Ty(ty.clone()),
                                None => Resolved::Generic(generic),
                            },
                            None => Resolved::Generic(generic),
                        };
                    }

                    PathSegment::SelfLower(_) if i == 0 => {
                        resolved = Resolved::SelfArgument;
                    }

                    PathSegment::SelfUpper(_) if i == 0 => {
                        resolved = Resolved::SelfTy;
                    }

                    _ => {
                        return Err(Diagnostic::new("unexpected path segment (1)")
                            .with_span(segment.span()))
                    }
                },

                Resolved::Struct(struct_id, ref generics) => match segment {
                    PathSegment::Named(named) => {
                        let ty = hir::KnownTy::new_struct(struct_id, generics.to_vec());

                        let generics: Vec<_> = named
                            .generics
                            .iter()
                            .map(|g| self.lower_type(unit, g))
                            .collect::<Result<_, _>>()?;

                        resolved = Resolved::Assoc(ty, named.name.clone(), generics);
                    }
                    _ => {
                        let message = "unexpected path segment (2)";
                        return Err(Diagnostic::new(message).with_span(segment.span()));
                    }
                },
                Resolved::Enum(enum_id, ref generics) => match segment {
                    PathSegment::Named(named) => {
                        let ty = hir::KnownTy::new_enum(enum_id, generics.to_vec());

                        if let Some(index) = unit[enum_id].variant_index(&named.name) {
                            if !named.generics.is_empty() {
                                let message = "enums variants do not have generics";
                                return Err(Diagnostic::new(message).with_span(named.span));
                            }

                            resolved = Resolved::EnumVariant(enum_id, generics.clone(), index);
                        } else {
                            let generics = named
                                .generics
                                .iter()
                                .map(|g| self.lower_type(unit, g))
                                .collect::<Result<_, _>>()?;

                            resolved = Resolved::Assoc(ty, named.name.clone(), generics);
                        }
                    }
                    _ => {
                        let message = "unexpected path segment (3)";
                        return Err(Diagnostic::new(message).with_span(segment.span()));
                    }
                },
                Resolved::Generic(generic) => match segment {
                    PathSegment::Named(named) => {
                        let ty = hir::KnownTy::Generic(generic);

                        let generics = named
                            .generics
                            .iter()
                            .map(|g| self.lower_type(unit, g))
                            .collect::<Result<_, _>>()?;

                        resolved = Resolved::Assoc(ty, named.name.clone(), generics);
                    }
                    _ => {
                        let message = "unexpected path segment (4)";
                        return Err(Diagnostic::new(message).with_span(segment.span()));
                    }
                },
                Resolved::SelfTy => match segment {
                    PathSegment::Named(named) => {
                        let Some(ty) = self.self_ty.clone() else {
                            let message = "no self type found";
                            return Err(Diagnostic::new(message).with_span(named.span));
                        };

                        let generics = named
                            .generics
                            .iter()
                            .map(|g| self.lower_type(unit, g))
                            .collect::<Result<_, _>>()?;

                        resolved = Resolved::Assoc(ty, named.name.clone(), generics);
                    }
                    _ => {
                        let message = "unexpected path segment (5)";
                        return Err(Diagnostic::new(message).with_span(segment.span()));
                    }
                },
                Resolved::Trait(trait_id, trait_generics) => match segment {
                    PathSegment::Named(named) => {
                        let hir_trait = &unit[trait_id];

                        let Some(method_index) = hir_trait.method_index(&named.name) else {
                            let message = format!("no method found for `{}`", named.name);
                            return Err(Diagnostic::new(message).with_span(named.span));
                        };

                        let method_generics = named
                            .generics
                            .iter()
                            .map(|g| self.lower_type(unit, g))
                            .collect::<Result<_, _>>()?;

                        resolved = Resolved::TraitMethod(
                            trait_id,
                            trait_generics,
                            method_index,
                            method_generics,
                        );
                    }
                    _ => {
                        let message = "unexpected path segment (6)";
                        return Err(Diagnostic::new(message).with_span(segment.span()));
                    }
                },
                _ => {
                    let message = "unexpected path segment (7)";
                    return Err(Diagnostic::new(message).with_span(segment.span()));
                }
            }
        }

        Ok(resolved)
    }

    pub fn lower_type(
        &mut self,
        unit: &hir::Unit,
        ast: &ast::Type,
    ) -> Result<hir::KnownTy, Diagnostic> {
        Ok(match ast {
            ast::Type::Void(_) => hir::KnownTy::VOID,

            ast::Type::Bool(_) => hir::KnownTy::BOOL,

            ast::Type::Int(ty) => hir::KnownTy::new_int(ty.signed, ty.width),

            ast::Type::Float(ty) => hir::KnownTy::new_float(ty.width),

            ast::Type::Pointer(ty) => {
                let pointee = self.lower_type(unit, &ty.pointee)?;
                hir::KnownTy::new_pointer(ty.mutable, pointee)
            }

            ast::Type::Function(ty) => {
                let mut arguments = Vec::with_capacity(ty.arguments.len());

                for arg in &ty.arguments {
                    arguments.push(self.lower_type(unit, arg)?);
                }

                let output = self.lower_type(unit, &ty.output)?;
                hir::KnownTy::new_func(arguments, output)
            }

            ast::Type::Array(_) => todo!(),
            ast::Type::Slice(_) => todo!(),
            ast::Type::Tuple(_) => todo!(),

            ast::Type::Path(item) => match self.resolve_path(unit, item)? {
                Resolved::Struct(id, generics) => hir::KnownTy::new_struct(id, generics),
                Resolved::Enum(id, generics) => hir::KnownTy::new_enum(id, generics),
                Resolved::Generic(generic) => hir::KnownTy::Generic(generic),
                Resolved::AssocTy(implementor, trait_id, trait_generics, assoc_index) => {
                    hir::KnownTy::Assoc(
                        Box::new(implementor),
                        trait_id,
                        trait_generics,
                        assoc_index,
                    )
                }
                Resolved::SelfTy => match self.self_ty.clone() {
                    Some(ty) => ty,
                    None => {
                        let message = "no self type found";
                        return Err(Diagnostic::new(message).with_span(item.span));
                    }
                },
                _ => {
                    let message = "unexpected path query";
                    return Err(Diagnostic::new(message).with_span(item.span));
                }
            },
        })
    }
}
