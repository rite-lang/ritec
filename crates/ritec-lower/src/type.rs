use ritec_ast as ast;
use ritec_ast::PathSegment;
use ritec_diagnostic::Diagnostic;
use ritec_hir as hir;
use ritec_hir::ModuleId;

pub struct TypeContext<'a> {
    pub module: ModuleId,
    pub generics: &'a mut Vec<(String, hir::Generic)>,
    pub allow_new_generics: bool,
    pub trait_id: Option<hir::TraitId>,
    pub self_type: Option<hir::Type>,
    pub specialization: Option<&'a hir::Spec>,
}

#[derive(Debug)]
pub enum Resolved {
    Struct(hir::StructId, Vec<hir::Type>),
    Trait(hir::TraitId, Vec<hir::Type>),
    Enum(hir::EnumId, Vec<hir::Type>),
    Func(hir::BodyId, Vec<hir::Type>),
    AssocType(hir::Type, hir::TraitId, Vec<hir::Type>, usize),
    Assoc(hir::Type, String, Vec<hir::Type>),
    TraitMethod(hir::TraitId, Vec<hir::Type>, usize, Vec<hir::Type>),
    EnumVariant(hir::EnumId, Vec<hir::Type>, usize),
    Generic(hir::Generic),
    Module(ModuleId),
    SelfArgument,
    SelfType,
}

impl<'a> TypeContext<'a> {
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
        if let (Some(ident), Some(trait_id)) = (ast.ident(), self.trait_id) {
            let trait_ = &unit.types[trait_id];

            for (index, assoc) in trait_.assocs.iter().enumerate() {
                if assoc.name != ident {
                    continue;
                }

                let generics: Vec<_> = trait_
                    .generics
                    .iter()
                    .copied()
                    .map(hir::Type::Generic)
                    .collect();

                return Ok(Resolved::AssocType(
                    trait_.self_type(),
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
                        let module = &unit.modules[module];
                        let mut generics: Vec<_> = named
                            .generics
                            .iter()
                            .map(|g| self.lower_type(unit, g))
                            .collect::<Result<_, _>>()?;

                        if let Some(&struct_id) = module.structs.get(name) {
                            while generics.len() < unit.types[struct_id].generics.len() {
                                generics.push(hir::Type::unknown(ast.span));
                            }

                            resolved = Resolved::Struct(struct_id, generics);
                            continue;
                        }

                        if let Some(&enum_id) = module.enums.get(name) {
                            while generics.len() < unit.types[enum_id].generics.len() {
                                generics.push(hir::Type::unknown(ast.span));
                            }

                            resolved = Resolved::Enum(enum_id, generics);
                            continue;
                        }

                        if let Some(&trait_id) = module.traits.get(name) {
                            resolved = Resolved::Trait(trait_id, generics);
                            continue;
                        }

                        if let Some(&func_id) = module.funcs.get(name) {
                            let mut generics = generics.clone();

                            while generics.len() < unit.bodies[func_id].generics.len() {
                                generics.push(hir::Type::unknown(ast.span));
                            }

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

                        let Some(index) = unit.types[trait_id].assoc_index(&assoc.name) else {
                            let message = format!("no associated type found for `{}`", assoc.name);
                            return Err(Diagnostic::new(message).with_span(assoc.span));
                        };

                        resolved = Resolved::AssocType(implementor, trait_id, generics, index);
                    }

                    PathSegment::Generic(generic) if i == 0 => {
                        resolved = Resolved::Generic(self.get_generic(generic)?);
                    }

                    PathSegment::SelfLower(_) if i == 0 => {
                        resolved = Resolved::SelfArgument;
                    }

                    PathSegment::SelfUpper(_) if i == 0 => {
                        resolved = Resolved::SelfType;
                    }

                    _ => {
                        return Err(Diagnostic::new("unexpected path segment (1)")
                            .with_span(segment.span()))
                    }
                },

                Resolved::Struct(struct_id, ref generics) => match segment {
                    PathSegment::Named(named) => {
                        let ty = hir::Type::Partial(hir::Partial {
                            item: hir::Item::Struct(struct_id),
                            params: generics.clone(),
                        });

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
                        let ty = hir::Type::Partial(hir::Partial {
                            item: hir::Item::Enum(enum_id),
                            params: generics.clone(),
                        });

                        if let Some(index) = unit.types[enum_id].variant_index(&named.name) {
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
                        let ty = hir::Type::Generic(generic);

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
                Resolved::SelfType => match segment {
                    PathSegment::Named(named) => {
                        let Some(ty) = self.self_type.clone() else {
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
                        let hir_trait = &unit.types[trait_id];

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
    ) -> Result<hir::Type, Diagnostic> {
        let ty = match ast {
            ast::Type::Void(_) => hir::Type::Partial(hir::Partial {
                item: hir::Item::Void,
                params: Vec::new(),
            }),

            ast::Type::Bool(_) => hir::Type::Partial(hir::Partial {
                item: hir::Item::Bool,
                params: Vec::new(),
            }),

            ast::Type::Int(ty) => hir::Type::Partial(hir::Partial {
                item: hir::Item::Int {
                    signed: ty.signed,
                    width: ty.width,
                },
                params: Vec::new(),
            }),

            ast::Type::Float(ty) => hir::Type::Partial(hir::Partial {
                item: hir::Item::Float { width: ty.width },
                params: Vec::new(),
            }),

            ast::Type::Pointer(ty) => {
                let pointee = self.lower_type(unit, &ty.pointee)?;

                hir::Type::Partial(hir::Partial {
                    item: hir::Item::Pointer {
                        mutable: ty.mutable,
                    },
                    params: vec![pointee],
                })
            }

            ast::Type::Array(_) => todo!(),
            ast::Type::Slice(_) => todo!(),
            ast::Type::Tuple(_) => todo!(),
            ast::Type::Function(_) => todo!(),
            ast::Type::Path(item) => match self.resolve_path(unit, item)? {
                Resolved::Struct(id, generics) => hir::Type::Partial(hir::Partial {
                    item: hir::Item::Struct(id),
                    params: generics,
                }),
                Resolved::Enum(id, generics) => hir::Type::Partial(hir::Partial {
                    item: hir::Item::Enum(id),
                    params: generics,
                }),
                Resolved::Generic(generic) => hir::Type::Generic(generic),
                Resolved::AssocType(implementor, trait_id, generics, index) => {
                    hir::Type::Projected(hir::Projected {
                        contract: unit.types[trait_id].contract,
                        base: Box::new(implementor),
                        projection: hir::Projection::TraitType {
                            trait_id,
                            generics,
                            index,
                        },
                    })
                }
                Resolved::SelfType => match self.self_type.clone() {
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
        };

        match self.specialization {
            Some(specialization) => Ok(specialization.specialize(&ty)),
            None => Ok(ty),
        }
    }
}
