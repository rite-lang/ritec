use ritec_ast as ast;
use ritec_diagnostic::Diagnostic;
use ritec_hir as hir;
use ritec_hir::ModuleId;

pub struct TypeContext<'a> {
    pub module: ModuleId,
    pub generics: &'a mut Vec<(String, hir::Generic)>,
    pub allow_new_generics: bool,
    pub trait_id: Option<hir::TraitId>,
    pub self_type: Option<hir::Type>,
    pub specialization: Option<&'a hir::Specialization>,
}

pub enum ItemQuery {
    Struct(hir::StructId, Vec<hir::Type>),
    Trait(hir::TraitId, Vec<hir::Type>),
    Enum(hir::EnumId, Vec<hir::Type>),
    Func(hir::BodyId, Vec<hir::Type>),
    Assoc(hir::Type, hir::TraitId, Vec<hir::Type>, usize),
    Generic(hir::Generic),
    SelfArgument,
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

    pub fn query_item(
        &mut self,
        unit: &hir::Unit,
        ast: &ast::Item,
    ) -> Result<ItemQuery, Diagnostic> {
        if let Some(ident) = ast.ident() {
            if let Some(trait_id) = self.trait_id {
                let trait_ = &unit.types[trait_id];

                let generics: Vec<_> = trait_
                    .generics
                    .iter()
                    .copied()
                    .map(hir::Type::Generic)
                    .collect();

                for (index, assoc) in trait_.assocs.iter().enumerate() {
                    if assoc.name == ident {
                        return Ok(ItemQuery::Assoc(
                            trait_.self_type(),
                            trait_id,
                            generics.clone(),
                            index,
                        ));
                    }
                }
            }
        }

        let mut module = self.module;

        let mut segments = ast.segments.iter();

        loop {
            let Some(segment) = segments.next() else {
                break;
            };

            match segment {
                ast::ItemSegment::Named(ast) => {
                    let mut generics = Vec::new();

                    for generic in &ast.generics {
                        let generic = self.lower_type(unit, generic)?;
                        generics.push(generic);
                    }

                    if let Some(id) = unit.modules[module].enums.get(&ast.name) {
                        return Ok(ItemQuery::Enum(*id, generics));
                    }

                    if let Some(id) = unit.modules[module].structs.get(&ast.name) {
                        return Ok(ItemQuery::Struct(*id, generics));
                    }

                    if let Some(id) = unit.modules[module].traits.get(&ast.name) {
                        return Ok(ItemQuery::Trait(*id, generics));
                    }

                    if let Some(id) = unit.modules[module].funcs.get(&ast.name) {
                        return Ok(ItemQuery::Func(*id, generics));
                    }

                    if !generics.is_empty() {
                        let message = format!("item `{}` not found", ast.name);
                        return Err(Diagnostic::new(message).with_span(ast.span));
                    }

                    if let Some(id) = unit.modules[module].modules.get(&ast.name) {
                        module = *id;
                        continue;
                    }

                    let message = format!("item `{}` not found", ast.name);
                    return Err(Diagnostic::new(message).with_span(ast.span));
                }

                ast::ItemSegment::Assoc(ast) => {
                    let implementor = self.lower_type(unit, &ast.implementor)?;

                    let query = self.query_item(unit, &ast.trait_item)?;
                    let ItemQuery::Trait(trait_id, generics) = query else {
                        let message = "expected trait";
                        return Err(Diagnostic::new(message).with_span(ast.trait_item.span));
                    };

                    let Some(index) = unit.types[trait_id].assoc_index(&ast.name) else {
                        let message = format!("assoc `{}` not found", ast.name);
                        return Err(Diagnostic::new(message).with_span(ast.span));
                    };

                    return Ok(ItemQuery::Assoc(implementor, trait_id, generics, index));
                }

                ast::ItemSegment::Generic(ast) => {
                    if segments.next().is_some() {
                        let message = "unexpected generic segment";
                        return Err(Diagnostic::new(message).with_span(ast.span));
                    }

                    let generic = self.get_generic(ast)?;
                    return Ok(ItemQuery::Generic(generic));
                }

                ast::ItemSegment::SelfLower(_) => {
                    return Ok(ItemQuery::SelfArgument);
                }

                ast::ItemSegment::SelfUpper(_) => todo!(),
            }
        }

        todo!()
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
            ast::Type::Item(item) => match self.query_item(unit, item)? {
                ItemQuery::Struct(id, generics) => hir::Type::Partial(hir::Partial {
                    item: hir::Item::Struct(id),
                    params: generics,
                }),
                ItemQuery::Enum(id, generics) => hir::Type::Partial(hir::Partial {
                    item: hir::Item::Enum(id),
                    params: generics,
                }),
                ItemQuery::Generic(generic) => hir::Type::Generic(generic),
                ItemQuery::Assoc(implementor, trait_id, generics, index) => {
                    hir::Type::Projected(hir::Projected {
                        contract: unit.types[trait_id].contract,
                        base: Box::new(implementor),
                        projection: hir::Projection::Associated {
                            trait_id,
                            generics,
                            index,
                        },
                    })
                }
                _ => {
                    let message = "unexpected item query";
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
