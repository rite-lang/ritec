use crate::{
    Assoc, BinaryOp, Bodies, Body, Contract, Expr, ExprKind, Generic, Local, Locals, Method,
    Modules, Projected, Projection, Trait, TraitId, TraitImpl, TraitMethod, Type, Types,
};

#[derive(Clone, Debug)]
pub struct Builtins {
    pub add_trait: TraitId,
    pub sub_trait: TraitId,
    pub mul_trait: TraitId,
    pub div_trait: TraitId,
    pub eq_trait: TraitId,
}

/// A  single compilation unit.
#[derive(Clone, Debug)]
pub struct Unit {
    pub modules: Modules,
    pub types: Types,
    pub bodies: Bodies,
    pub builtins: Builtins,
}

impl Default for Unit {
    fn default() -> Self {
        Self::new()
    }
}

impl Unit {
    pub fn new() -> Self {
        let modules = Modules::new();
        let mut types = Types::new();
        let mut bodies = Bodies::new();

        let builtins = Self::generate_builtins(&mut types, &mut bodies);

        Self {
            modules,
            types,
            bodies,
            builtins,
        }
    }

    fn generate_builtins(types: &mut Types, bodies: &mut Bodies) -> Builtins {
        const MATH_TYPES: &[Type] = &[
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::USIZE,
            Type::ISIZE,
            Type::F32,
            Type::F64,
        ];

        const EQ_TYPES: &[Type] = &[
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::USIZE,
            Type::ISIZE,
            Type::F32,
            Type::F64,
            Type::BOOL,
        ];

        let add_trait = Self::generate_binary_trait(types, "Add");
        let sub_trait = Self::generate_binary_trait(types, "Sub");
        let mul_trait = Self::generate_binary_trait(types, "Mul");
        let div_trait = Self::generate_binary_trait(types, "Div");
        let eq_trait = Self::generate_eq_trait(types);

        for ty in MATH_TYPES {
            Self::generate_binary_trait_impl(
                types,
                bodies,
                add_trait,
                "add",
                BinaryOp::Add,
                ty.clone(),
            );

            Self::generate_binary_trait_impl(
                types,
                bodies,
                sub_trait,
                "sub",
                BinaryOp::Sub,
                ty.clone(),
            );

            Self::generate_binary_trait_impl(
                types,
                bodies,
                mul_trait,
                "mul",
                BinaryOp::Mul,
                ty.clone(),
            );

            Self::generate_binary_trait_impl(
                types,
                bodies,
                div_trait,
                "div",
                BinaryOp::Div,
                ty.clone(),
            );
        }

        for ty in EQ_TYPES {
            Self::generate_eq_trait_impl(types, bodies, eq_trait, ty.clone());
        }

        Builtins {
            add_trait,
            sub_trait,
            mul_trait,
            div_trait,
            eq_trait,
        }
    }

    fn generate_binary_trait(types: &mut Types, name: &str) -> TraitId {
        let self_generic = Generic::new();
        let rhs = Generic::new();

        let contract = types.contracts.push(Contract::new());

        let trait_id = types.traits.alloc();

        let trait_ = Trait {
            self_generic,
            name: Some(String::from(name)),
            generics: vec![rhs],
            contract,
            assocs: vec![Assoc {
                name: String::from("Output"),
            }],
            methods: vec![TraitMethod {
                name: name.to_lowercase(),
                generics: Vec::new(),
                arguments: vec![Type::Generic(self_generic), Type::Generic(rhs)],
                output: Type::Projected(Projected {
                    contract,
                    base: Box::new(Type::Generic(self_generic)),
                    projection: Projection::Associated {
                        trait_id,
                        generics: vec![Type::Generic(rhs)],
                        index: 0,
                    },
                }),
                contract,
            }],
        };

        types.traits.insert(trait_id, trait_);

        trait_id
    }

    fn generate_eq_trait(types: &mut Types) -> TraitId {
        let self_generic = Generic::new();
        let rhs = Generic::new();

        let contract = types.contracts.push(Contract::new());

        let trait_id = types.traits.alloc();

        let trait_ = Trait {
            self_generic,
            name: Some(String::from("Eq")),
            generics: vec![rhs],
            contract,
            assocs: Vec::new(),
            methods: vec![TraitMethod {
                name: String::from("eq"),
                generics: Vec::new(),
                arguments: vec![Type::Generic(self_generic), Type::Generic(rhs)],
                output: Type::BOOL,
                contract,
            }],
        };

        types.traits.insert(trait_id, trait_);

        trait_id
    }

    fn generate_binary_trait_impl(
        types: &mut Types,
        bodies: &mut Bodies,
        trait_id: TraitId,
        name: &str,
        op: BinaryOp,
        ty: Type,
    ) {
        let contract = types.contracts.push(Contract::new());

        let mut locals = Locals::new();
        let lhs_local = locals.push(Local {
            mutable: false,
            name: None,
            ty: ty.clone(),
        });

        let rhs_local = locals.push(Local {
            mutable: false,
            name: None,
            ty: ty.clone(),
        });

        let lhs = Expr {
            kind: ExprKind::Local(lhs_local),
            span: None,
            ty: ty.clone(),
        };

        let rhs = Expr {
            kind: ExprKind::Local(rhs_local),
            span: None,
            ty: ty.clone(),
        };

        let body = Body {
            name: Some(name.to_lowercase()),
            arguments: vec![lhs_local, rhs_local],
            output: ty.clone(),
            generics: vec![Generic::new()],
            contract,
            locals,
            expr: Expr {
                kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
                span: None,
                ty: ty.clone(),
            },
        };

        let body_id = bodies.push(body);

        let contract = types.contracts.push(Contract::new());

        let trait_impl = TraitImpl {
            trait_id,
            generics: vec![ty.clone()],
            implementor: ty.clone(),
            contract,
            types: vec![ty.clone()],
            methods: vec![Method {
                name: name.to_lowercase(),
                body: body_id,
            }],
        };

        types.trait_impls.push(trait_impl);
    }

    fn generate_eq_trait_impl(types: &mut Types, bodies: &mut Bodies, trait_id: TraitId, ty: Type) {
        let contract = types.contracts.push(Contract::new());

        let mut locals = Locals::new();
        let lhs_local = locals.push(Local {
            mutable: false,
            name: None,
            ty: ty.clone(),
        });

        let rhs_local = locals.push(Local {
            mutable: false,
            name: None,
            ty: ty.clone(),
        });

        let lhs = Expr {
            kind: ExprKind::Local(lhs_local),
            span: None,
            ty: ty.clone(),
        };

        let rhs = Expr {
            kind: ExprKind::Local(rhs_local),
            span: None,
            ty: ty.clone(),
        };

        let body = Body {
            name: Some(String::from("eq")),
            arguments: vec![lhs_local, rhs_local],
            output: Type::BOOL,
            generics: vec![Generic::new()],
            contract,
            locals,
            expr: Expr {
                kind: ExprKind::Binary(BinaryOp::Eq, Box::new(lhs), Box::new(rhs)),
                span: None,
                ty: Type::BOOL,
            },
        };

        let body_id = bodies.push(body);

        let contract = types.contracts.push(Contract::new());

        let trait_impl = TraitImpl {
            trait_id,
            generics: vec![ty.clone()],
            implementor: ty.clone(),
            contract,
            types: vec![ty.clone()],
            methods: vec![Method {
                name: String::from("eq"),
                body: body_id,
            }],
        };

        types.trait_impls.push(trait_impl);
    }
}
