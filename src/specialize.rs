use std::collections::HashMap;

use crate::{
    number::IntKind,
    rir::{
        self, Adt, Argument, Block, Capture, Constant, Func, Local, Operand, Place, Projection,
        Specific, Statement, Ty, Unit, Value,
    },
};

pub fn specialize(generic: Unit, func: usize) -> (usize, Unit<Specific>) {
    let mut specific = Unit::default();
    let mut adts = HashMap::new();
    let mut funcs = HashMap::new();

    let generics = vec![Specific::Int(IntKind::Int); generic.funcs[func].generics.len()];
    let index = specialize_func(
        &generic,
        &mut specific,
        &mut adts,
        &mut funcs,
        func,
        &generics,
    );

    (index, specific)
}

struct Specializer<'a> {
    generic: &'a Unit,
    specific: &'a mut Unit<Specific>,
    adts: &'a mut HashMap<(usize, Vec<Specific>), usize>,
    funcs: &'a mut HashMap<(usize, Vec<Specific>), usize>,
    generics: &'a [Specific],
    func: Option<usize>,
}

fn specialize_func(
    generic: &Unit,
    specific: &mut Unit<Specific>,
    adts: &mut HashMap<(usize, Vec<Specific>), usize>,
    funcs: &mut HashMap<(usize, Vec<Specific>), usize>,
    func_index: usize,
    generics: &[Specific],
) -> usize {
    let key = (func_index, generics.to_vec());
    if let Some(&id) = funcs.get(&key) {
        return id;
    }

    let func = &generic.funcs[func_index];
    assert_eq!(func.generics.len(), generics.len());

    let mut spec = Specializer {
        generic,
        specific,
        adts,
        funcs,
        generics,
        func: Some(func_index),
    };

    let input = func
        .input
        .iter()
        .map(|arg| Argument {
            ty: specialize_ty(&mut spec, &arg.ty),
            span: arg.span,
        })
        .collect();

    let output = specialize_ty(&mut spec, &func.output);

    let locals = func
        .locals
        .iter()
        .map(|local| Local {
            ty: specialize_ty(&mut spec, &local.ty),
        })
        .collect();

    let captures = func
        .captures
        .iter()
        .map(|capture| Capture {
            ty: specialize_ty(&mut spec, &capture.ty),
        })
        .collect();

    let index = spec.specific.funcs.len();
    spec.specific.funcs.push(Func {
        decorators: func.decorators.clone(),
        name: func.name.clone(),
        generics: func.generics.clone(),
        input,
        output,
        locals,
        captures,
        body: rir::Block::new(),
    });

    spec.funcs.insert(key, index);

    let body = specialize_block(&mut spec, func.body.clone());
    spec.specific.funcs[index].body = body;

    index
}

fn specialize_adt(spec: &mut Specializer, adt: usize, generics: &[Specific]) -> usize {
    let key = (adt, generics.to_vec());
    if let Some(&id) = spec.adts.get(&key) {
        return id;
    }

    let id = spec.specific.adts.len();
    spec.specific.adts.push(Adt::default());
    spec.adts.insert(key, id);

    let adt = &spec.generic.adts[adt];
    assert_eq!(adt.generics.len(), generics.len());

    let mut spec = Specializer {
        generic: spec.generic,
        specific: spec.specific,
        adts: spec.adts,
        funcs: spec.funcs,
        generics,
        func: None,
    };

    let mut variants = Vec::new();

    for variant in &adt.variants {
        let mut fields = Vec::new();

        for field in &variant.fields {
            let ty = specialize_ty(&mut spec, &field.ty);
            fields.push(Argument { ty, span: None });
        }

        variants.push(rir::Variant { fields });
    }

    spec.specific.adts[id] = Adt {
        decorators: adt.decorators.clone(),
        name: adt.name.clone(),
        generics: adt.generics.clone(),
        variants,
    };

    id
}

fn specialize_ty(spec: &mut Specializer, ty: &Ty) -> Specific {
    match ty {
        Ty::Void => Specific::Void,
        Ty::Bool => Specific::Bool,
        Ty::Str => Specific::Str,
        Ty::Ref(ty) => Specific::Ref(Box::new(specialize_ty(spec, ty))),
        Ty::Int(kind) => Specific::Int(*kind),
        Ty::Float(kind) => Specific::Float(*kind),
        Ty::List(ty) => Specific::List(Box::new(specialize_ty(spec, ty))),
        Ty::Tuple(items) => {
            let items = items.iter().map(|ty| specialize_ty(spec, ty)).collect();
            Specific::Tuple(items)
        }
        Ty::Func(input, output) => {
            let input = input.iter().map(|ty| specialize_ty(spec, ty)).collect();
            let output = Box::new(specialize_ty(spec, output));
            Specific::Func(input, output)
        }
        Ty::Adt(adt, args) => {
            let args: Vec<_> = args.iter().map(|ty| specialize_ty(spec, ty)).collect();
            let adt = specialize_adt(spec, *adt, &args);
            Specific::Adt(adt, args)
        }
        Ty::Generic(index) => {
            if spec.generics.len() <= *index {
                if let Some(func) = spec.func {
                    let name = &spec.generic.funcs[func].name;
                    panic!("{}: generic type out of bounds", name);
                }
            }

            spec.generics[*index].clone()
        }
    }
}

fn specialize_block(spec: &mut Specializer, block: Block) -> Block<Specific> {
    let mut specific = Block::new();

    for statement in block.statements {
        let statement = match statement {
            Statement::Use { value } => Statement::Use {
                value: specialize_value(spec, value),
            },
            Statement::Call { place, func, args } => {
                let place = specialize_place(spec, place);

                let func = specialize_operand(spec, func);

                let args = args
                    .into_iter()
                    .map(|arg| specialize_operand(spec, arg))
                    .collect();

                Statement::Call { place, func, args }
            }
            Statement::Return { value } => Statement::Return {
                value: value.map(|value| specialize_value(spec, value)),
            },
            Statement::Panic { message } => Statement::Panic { message },
            Statement::Assign { place, value } => Statement::Assign {
                place: specialize_place(spec, place),
                value: specialize_value(spec, value),
            },
            Statement::MatchBool {
                input,
                r#true,
                r#false,
            } => Statement::MatchBool {
                input: specialize_operand(spec, input),
                r#true: specialize_block(spec, r#true),
                r#false: specialize_block(spec, r#false),
            },
            Statement::MatchAdt {
                input,
                variants,
                default,
            } => {
                let input = specialize_operand(spec, input);

                let variants = variants
                    .into_iter()
                    .map(|variant| variant.map(|block| specialize_block(spec, block)))
                    .collect();

                let default = default.map(|block| specialize_block(spec, block));

                Statement::MatchAdt {
                    input,
                    variants,
                    default,
                }
            }
        };

        specific.statements.push(statement);
    }

    specific
}

fn specialize_value(spec: &mut Specializer, value: Value) -> Value<Specific> {
    match value {
        Value::Use(operand) => Value::Use(specialize_operand(spec, operand)),
        Value::Cast(cast, operand) => Value::Cast(cast, specialize_operand(spec, operand)),
        Value::Func(index, args, generics) => {
            let args = args
                .into_iter()
                .map(|arg| specialize_operand(spec, arg))
                .collect();

            let generics = generics
                .into_iter()
                .map(|ty| specialize_ty(spec, &ty))
                .collect::<Vec<_>>();

            let index = specialize_func(
                spec.generic,
                spec.specific,
                spec.adts,
                spec.funcs,
                index,
                &generics,
            );

            Value::Func(index, args, generics)
        }
        Value::List(items, tail) => {
            let items = items
                .into_iter()
                .map(|item| specialize_operand(spec, item))
                .collect();

            let tail = tail.map(|tail| specialize_operand(spec, tail));

            Value::List(items, tail)
        }
        Value::ListHead(list) => Value::ListHead(specialize_operand(spec, list)),
        Value::ListTail(list) => Value::ListTail(specialize_operand(spec, list)),
        Value::ListEmpty(list) => Value::ListEmpty(specialize_operand(spec, list)),
        Value::Binary(op, lhs, rhs) => Value::Binary(
            op,
            specialize_operand(spec, lhs),
            specialize_operand(spec, rhs),
        ),
        Value::Unary(op, operand) => Value::Unary(op, specialize_operand(spec, operand)),
        Value::IsVariant(operand, variant) => {
            Value::IsVariant(specialize_operand(spec, operand), variant)
        }
        Value::Ref(place) => Value::Ref(specialize_place(spec, place)),
        Value::Tuple(items) => {
            let items = items
                .into_iter()
                .map(|item| specialize_operand(spec, item))
                .collect();

            Value::Tuple(items)
        }
        Value::Adt(variant, fields) => {
            let fields = fields
                .into_iter()
                .map(|field| specialize_operand(spec, field))
                .collect();

            Value::Adt(variant, fields)
        }
    }
}

fn specialize_operand(spec: &mut Specializer, operand: Operand) -> Operand<Specific> {
    match operand {
        Operand::Copy(place) => Operand::Copy(specialize_place(spec, place)),
        Operand::Move(place) => Operand::Move(specialize_place(spec, place)),
        Operand::Constant(constant) => Operand::Constant(specialize_constant(spec, constant)),
    }
}

fn specialize_constant(spec: &mut Specializer, constant: Constant) -> Constant<Specific> {
    match constant {
        Constant::Void => Constant::Void,
        Constant::Bool(b) => Constant::Bool(b),
        Constant::Int(negative, base, digits, ty) => {
            Constant::Int(negative, base, digits, specialize_ty(spec, &ty))
        }
        Constant::String(s) => Constant::String(s),
    }
}

fn specialize_place(spec: &mut Specializer, place: Place) -> Place<Specific> {
    let projection = place
        .projection
        .into_iter()
        .map(|projection| specialize_projection(spec, projection))
        .collect();

    let ty = specialize_ty(spec, &place.ty);

    Place {
        location: place.location,
        projection,
        ty,
    }
}

fn specialize_projection(spec: &mut Specializer, projection: Projection) -> Projection<Specific> {
    Projection {
        kind: projection.kind,
        ty: specialize_ty(spec, &projection.ty),
        span: projection.span,
    }
}
