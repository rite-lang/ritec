use crate::{
    ast::BinOp,
    hir,
    number::{Base, IntKind},
    span::Span,
};

#[derive(Debug, Default)]
pub struct Unit {
    pub funcs: Vec<Func>,
    pub adts: Vec<Adt>,
}

#[derive(Debug)]
pub struct Func {
    pub name: String,
    pub generics: Vec<Generic>,
    pub input: Vec<Argument>,
    pub output: Ty,
    pub locals: Vec<Ty>,
    pub captures: Vec<Ty>,
    pub body: Expr,
}

impl Default for Func {
    fn default() -> Self {
        Self {
            name: String::new(),
            generics: Vec::new(),
            input: Vec::new(),
            output: Ty::Void,
            locals: Vec::new(),
            captures: Vec::new(),
            body: Expr {
                kind: ExprKind::Void,
                ty: Ty::Void,
            },
        }
    }
}

#[derive(Debug)]
pub struct Adt {
    pub name: String,
    pub generics: Vec<Generic>,
    pub variants: Vec<Variant>,
}

#[derive(Debug)]
pub struct Variant {
    pub fields: Vec<Argument>,
}

#[derive(Debug)]
pub struct Generic {}

#[derive(Debug)]
pub struct Argument {
    pub ty: Ty,
    pub span: Option<Span>,
}

#[derive(Clone, Debug)]
pub enum Ty {
    Void,
    Bool,
    Str,
    Int(IntKind),
    List(Box<Ty>),
    Tuple(Vec<Ty>),
    Func(Vec<Ty>, Box<Ty>),
    Adt(usize, Vec<Ty>),
    Generic(usize),
}

#[derive(Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Ty,
}

#[derive(Debug)]
pub enum ExprKind {
    Void,
    Int(bool, Base, Vec<u8>),
    String(&'static str),
    Bool(bool),
    Func(usize, Vec<Expr>),
    Variant(usize, usize),
    Local(usize),
    Argument(usize),
    Captured(usize),
    Tuple(Vec<Expr>),
    List(Vec<Expr>, Option<Box<Expr>>),
    ListHead(Box<Expr>),
    ListTail(Box<Expr>),
    Block(Vec<Expr>),
    Field(Box<Expr>, usize),
    VariantField(Box<Expr>, usize, usize),
    Call(Box<Expr>, Vec<Expr>),
    Pipe(Box<Expr>, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Let(usize, Box<Expr>),
    Match(Box<Expr>, Match),
}

#[derive(Debug)]
pub enum Match {
    Bool(Box<Expr>, Box<Expr>),
    Adt(Vec<Option<Expr>>, Option<Box<Expr>>),
    List(Box<Expr>, Box<Expr>),
}

impl Unit {
    pub fn from_hir(unit: hir::Unit) -> miette::Result<Self> {
        let mut funcs = Vec::new();
        funcs.resize_with(unit.funcs.len(), Func::default);

        let mut rir = Self {
            adts: Adt::from_hir_vec(&unit, &unit.adts)?,
            funcs,
        };

        for (index, func) in unit.funcs.iter().enumerate() {
            rir.funcs[index] = Func::from_hir(&mut rir, &unit, func)?;
        }

        Ok(rir)
    }
}

type Generics = Vec<(Option<hir::Tid>, Generic)>;

impl Func {
    pub fn from_hir(rir: &mut Unit, unit: &hir::Unit, func: &hir::Func) -> miette::Result<Self> {
        let mut generics = func.generics.iter().map(|_| (None, Generic {})).collect();
        let input = Argument::from_hir_vec(unit, &mut generics, &func.input)?;
        let output = Ty::from_hir(unit, &mut generics, &func.output)?;

        let locals = func
            .locals
            .iter()
            .map(|local| Ty::from_hir(unit, &mut generics, &local.ty))
            .collect::<miette::Result<_>>()?;

        let captures = func
            .captures
            .iter()
            .map(|ty| Ty::from_hir(unit, &mut generics, ty))
            .collect::<miette::Result<_>>()?;

        let body = Expr::from_hir(rir, unit, &mut generics, &func.body)?;

        let generics = generics.into_iter().map(|(_, g)| g).collect();

        Ok(Self {
            name: func.name.to_string(),
            generics,
            input,
            output,
            locals,
            captures,
            body,
        })
    }

    pub fn ty(&self) -> Ty {
        let input = self.input.iter().map(|arg| arg.ty.clone()).collect();
        Ty::Func(input, Box::new(self.output.clone()))
    }

    fn from_hir_vec(
        rir: &mut Unit,
        unit: &hir::Unit,
        funcs: &[hir::Func],
    ) -> miette::Result<Vec<Self>> {
        funcs
            .iter()
            .map(|func| Func::from_hir(rir, unit, func))
            .collect::<miette::Result<_>>()
    }
}

impl Adt {
    fn from_hir(unit: &hir::Unit, adt: &hir::Adt) -> miette::Result<Self> {
        let mut generics = adt.generics.iter().map(|_| (None, Generic {})).collect();
        let variants = Variant::from_hir_vec(unit, &mut generics, &adt.variants)?;

        let generics = generics.into_iter().map(|(_, g)| g).collect();

        Ok(Self {
            name: adt.name.to_string(),
            generics,
            variants,
        })
    }

    fn from_hir_vec(unit: &hir::Unit, adts: &[hir::Adt]) -> miette::Result<Vec<Self>> {
        adts.iter()
            .map(|adt| Adt::from_hir(unit, adt))
            .collect::<miette::Result<_>>()
    }
}

impl Variant {
    fn from_hir(
        unit: &hir::Unit,
        generics: &mut Generics,
        variant: &hir::Variant,
    ) -> miette::Result<Self> {
        let fields = Argument::from_hir_vec(unit, generics, &variant.fields)?;
        Ok(Self { fields })
    }

    fn from_hir_vec(
        unit: &hir::Unit,
        generics: &mut Generics,
        variants: &[hir::Variant],
    ) -> miette::Result<Vec<Self>> {
        variants
            .iter()
            .map(|variant| Variant::from_hir(unit, generics, variant))
            .collect::<miette::Result<_>>()
    }
}

impl Argument {
    fn from_hir(
        unit: &hir::Unit,
        generics: &mut Generics,
        arg: &hir::Argument,
    ) -> miette::Result<Self> {
        Ok(Self {
            ty: Ty::from_hir(unit, generics, &arg.ty)?,
            span: Some(arg.span),
        })
    }

    fn from_hir_vec(
        unit: &hir::Unit,
        generics: &mut Generics,
        args: &[hir::Argument],
    ) -> miette::Result<Vec<Self>> {
        args.iter()
            .map(|arg| Argument::from_hir(unit, generics, arg))
            .collect::<miette::Result<_>>()
    }
}

impl Ty {
    fn from_hir(unit: &hir::Unit, generics: &mut Generics, ty: &hir::Ty) -> miette::Result<Self> {
        match unit.env.get(ty) {
            hir::Ty::Inferred(tid, kind, func) => {
                if let Some(index) = generics.iter().position(|(t, _)| *t == Some(tid)) {
                    return Ok(Ty::Generic(index));
                }

                if func.is_some() {
                    let index = generics.len();
                    generics.push((Some(tid), Generic {}));
                    return Ok(Ty::Generic(index));
                }

                match kind {
                    hir::Inferred::Any => todo!("{:?}", ty),
                    hir::Inferred::Int(kind) => Ok(Ty::Int(kind)),
                    hir::Inferred::Float(_) => todo!(),
                }
            }
            hir::Ty::Partial(part, arguments) => match part {
                hir::Part::Void => {
                    assert!(arguments.is_empty());
                    Ok(Ty::Void)
                }
                hir::Part::Bool => {
                    assert!(arguments.is_empty());
                    Ok(Ty::Bool)
                }
                hir::Part::Str => {
                    assert!(arguments.is_empty());
                    Ok(Ty::Str)
                }
                hir::Part::List => {
                    assert_eq!(arguments.len(), 1);
                    Ok(Ty::List(Box::new(Ty::from_hir(
                        unit,
                        generics,
                        &arguments[0],
                    )?)))
                }
                hir::Part::Tuple => {
                    let tuple = Ty::vec_from_hir(unit, generics, &arguments)?;
                    Ok(Ty::Tuple(tuple))
                }
                hir::Part::Func => {
                    let input = arguments
                        .iter()
                        .take(arguments.len() - 1)
                        .map(|ty| Ty::from_hir(unit, generics, ty))
                        .collect::<miette::Result<_>>()?;

                    let output = arguments.last().unwrap();
                    let output = Ty::from_hir(unit, generics, output)?;

                    Ok(Ty::Func(input, Box::new(output)))
                }
                hir::Part::Int(kind) => {
                    assert!(arguments.is_empty());
                    Ok(Ty::Int(kind))
                }
                hir::Part::Adt(index) => {
                    let generics = Ty::vec_from_hir(unit, generics, &arguments)?;
                    Ok(Ty::Adt(index, generics))
                }
                hir::Part::Generic(index) => {
                    assert!(arguments.is_empty());
                    Ok(Ty::Generic(index))
                }
            },
            hir::Ty::Field(_, _)
            | hir::Ty::Tuple(_, _)
            | hir::Ty::Call(_, _)
            | hir::Ty::Pipe(_, _) => {
                unreachable!("unexpected field, call or pipe: {}", ty)
            }
        }
    }

    fn vec_from_hir(
        unit: &hir::Unit,
        generics: &mut Generics,
        tys: &[hir::Ty],
    ) -> miette::Result<Vec<Ty>> {
        tys.iter()
            .map(|ty| Ty::from_hir(unit, generics, ty))
            .collect::<miette::Result<_>>()
    }
}

impl Expr {
    fn from_hir(
        rir: &mut Unit,
        unit: &hir::Unit,
        generics: &mut Generics,
        expr: &hir::Expr,
    ) -> miette::Result<Self> {
        Ok(Self {
            kind: ExprKind::from_hir(rir, unit, generics, &expr.kind)?,
            ty: Ty::from_hir(unit, generics, &expr.ty)?,
        })
    }

    fn vec_from_hir(
        rir: &mut Unit,
        unit: &hir::Unit,
        generics: &mut Generics,
        exprs: &[hir::Expr],
    ) -> miette::Result<Vec<Expr>> {
        exprs
            .iter()
            .map(|expr| Expr::from_hir(rir, unit, generics, expr))
            .collect::<miette::Result<_>>()
    }
}

impl ExprKind {
    fn from_hir(
        rir: &mut Unit,
        unit: &hir::Unit,
        generics: &mut Generics,
        kind: &hir::ExprKind,
    ) -> miette::Result<Self> {
        match kind {
            hir::ExprKind::Void => Ok(ExprKind::Void),
            hir::ExprKind::String(s) => Ok(ExprKind::String(s)),
            hir::ExprKind::Int(n, base, bytes) => Ok(ExprKind::Int(*n, *base, bytes.clone())),
            hir::ExprKind::Bool(b) => Ok(ExprKind::Bool(*b)),
            hir::ExprKind::Func(i) => Ok(ExprKind::Func(*i, Vec::new())),
            hir::ExprKind::Variant(i, j) => Ok(ExprKind::Variant(*i, *j)),
            hir::ExprKind::Local(i) => Ok(ExprKind::Local(*i)),
            hir::ExprKind::Argument(i) => Ok(ExprKind::Argument(*i)),
            hir::ExprKind::Tuple(exprs) => {
                let tuple = Expr::vec_from_hir(rir, unit, generics, exprs)?;

                Ok(ExprKind::Tuple(tuple))
            }
            hir::ExprKind::List(exprs, rest) => {
                let list = Expr::vec_from_hir(rir, unit, generics, exprs)?;
                let rest = match rest {
                    Some(rest) => Some(Box::new(Expr::from_hir(rir, unit, generics, rest)?)),
                    None => None,
                };

                Ok(ExprKind::List(list, rest))
            }
            hir::ExprKind::ListHead(expr) => {
                let expr = Box::new(Expr::from_hir(rir, unit, generics, expr)?);

                Ok(ExprKind::ListHead(expr))
            }
            hir::ExprKind::ListTail(expr) => {
                let expr = Box::new(Expr::from_hir(rir, unit, generics, expr)?);

                Ok(ExprKind::ListTail(expr))
            }
            hir::ExprKind::Block(exprs) => {
                let block = Expr::vec_from_hir(rir, unit, generics, exprs)?;

                Ok(ExprKind::Block(block))
            }
            hir::ExprKind::Call(callee, args) => {
                let callee = Expr::from_hir(rir, unit, generics, callee)?;

                let Ty::Func(ref input, ref output) = callee.ty else {
                    unreachable!("unexpected callee: {:?}", callee.ty)
                };

                if !args.iter().any(Option::is_none) && args.len() == input.len() {
                    let args = args
                        .iter()
                        .map(Option::as_ref)
                        .map(Option::unwrap)
                        .map(|arg| Expr::from_hir(rir, unit, generics, arg))
                        .collect::<miette::Result<_>>()?;

                    return Ok(ExprKind::Call(Box::new(callee), args));
                }

                let output = output.as_ref().clone();

                let mut arguments = Vec::new();
                let mut captured = Vec::new();

                let mut exprs = Vec::new();

                for (provided, arg) in args.iter().zip(input.iter()) {
                    match provided {
                        Some(arg) => {
                            let expr = Expr::from_hir(rir, unit, generics, arg)?;

                            let index = captured.len();
                            exprs.push(Expr {
                                kind: ExprKind::Captured(index),
                                ty: expr.ty.clone(),
                            });

                            captured.push(expr);
                        }
                        None => {
                            let index = arguments.len();

                            exprs.push(Expr {
                                kind: ExprKind::Argument(index),
                                ty: arg.clone(),
                            });

                            arguments.push(Argument {
                                ty: arg.clone(),
                                span: None,
                            });
                        }
                    }
                }

                for arg in input.iter().skip(args.len()) {
                    let index = arguments.len();

                    exprs.push(Expr {
                        kind: ExprKind::Argument(index),
                        ty: arg.clone(),
                    });

                    arguments.push(Argument {
                        ty: arg.clone(),
                        span: None,
                    });
                }

                let captures = captured.iter().map(|expr| expr.ty.clone()).collect();

                let body = Expr {
                    kind: ExprKind::Call(Box::new(callee), exprs),
                    ty: output.clone(),
                };

                let generics = generics.iter().map(|_| Generic {}).collect();

                let func = Func {
                    name: String::new(),
                    generics,
                    input: arguments,
                    output,
                    locals: Vec::new(),
                    captures,
                    body,
                };

                let index = unit.funcs.len();
                rir.funcs.push(func);

                Ok(ExprKind::Func(index, captured))
            }
            hir::ExprKind::Field(expr, field) => {
                let expr = Box::new(Expr::from_hir(rir, unit, generics, expr)?);

                let Ty::Adt(index, _) = expr.ty else {
                    unreachable!("unexpected field: {:?}", expr.ty)
                };

                let (index, _) = unit.adts[index].find_field(field)?;

                Ok(ExprKind::Field(expr, index))
            }
            hir::ExprKind::VariantField(expr, i, j) => {
                let expr = Box::new(Expr::from_hir(rir, unit, generics, expr)?);

                Ok(ExprKind::VariantField(expr, *i, *j))
            }
            hir::ExprKind::TupleField(expr, i) => {
                let expr = Box::new(Expr::from_hir(rir, unit, generics, expr)?);

                Ok(ExprKind::Field(expr, *i))
            }
            hir::ExprKind::Pipe(lhs, rhs) => {
                let lhs = Box::new(Expr::from_hir(rir, unit, generics, lhs)?);
                let rhs = Box::new(Expr::from_hir(rir, unit, generics, rhs)?);

                Ok(ExprKind::Pipe(lhs, rhs))
            }
            hir::ExprKind::Binary(op, lhs, rhs) => {
                let lhs = Box::new(Expr::from_hir(rir, unit, generics, lhs)?);
                let rhs = Box::new(Expr::from_hir(rir, unit, generics, rhs)?);

                Ok(ExprKind::Binary(*op, lhs, rhs))
            }
            hir::ExprKind::Let(local, expr) => {
                let expr = Box::new(Expr::from_hir(rir, unit, generics, expr)?);

                Ok(ExprKind::Let(*local, expr))
            }
            hir::ExprKind::Match(input, r#match) => {
                let input = Box::new(Expr::from_hir(rir, unit, generics, input)?);

                let r#match = match r#match {
                    hir::Match::Bool(r#true, r#false) => {
                        let r#true = Expr::from_hir(rir, unit, generics, r#true)?;
                        let r#false = Expr::from_hir(rir, unit, generics, r#false)?;

                        Match::Bool(Box::new(r#true), Box::new(r#false))
                    }
                    hir::Match::Adt(_, variants, default) => {
                        let variants = variants
                            .iter()
                            .map(|variant| {
                                variant
                                    .as_ref()
                                    .map(|expr| Expr::from_hir(rir, unit, generics, expr))
                                    .transpose()
                            })
                            .collect::<miette::Result<_>>()?;

                        let default = default
                            .as_ref()
                            .map(|expr| Expr::from_hir(rir, unit, generics, expr))
                            .transpose()?
                            .map(Box::new);

                        Match::Adt(variants, default)
                    }
                    hir::Match::List(some, none) => {
                        let some = Expr::from_hir(rir, unit, generics, some)?;
                        let none = Expr::from_hir(rir, unit, generics, none)?;

                        Match::List(Box::new(some), Box::new(none))
                    }
                };

                Ok(ExprKind::Match(input, r#match))
            }
        }
    }
}
