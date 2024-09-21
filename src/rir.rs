use crate::{
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
    pub body: Expr,
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
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Ty {
    Void,
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
    Func(usize),
    Variant(usize, usize),
    Local(usize),
    Argument(usize),
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    Block(Vec<Expr>),
    Field(Box<Expr>, usize),
    Call(Box<Expr>, Vec<Expr>),
    Pipe(Box<Expr>, Box<Expr>),
    Let(usize, Box<Expr>),
}

impl Unit {
    pub fn from_hir(unit: hir::Unit) -> miette::Result<Self> {
        let funcs = Func::from_hir_vec(&unit, &unit.funcs)?;
        let adts = Adt::from_hir_vec(&unit, &unit.adts)?;

        Ok(Self { funcs, adts })
    }
}

type Generics = Vec<(Option<hir::Tid>, Generic)>;

impl Func {
    pub fn from_hir(unit: &hir::Unit, func: &hir::Func) -> miette::Result<Self> {
        let mut generics = func.generics.iter().map(|_| (None, Generic {})).collect();
        let input = Argument::from_hir_vec(unit, &mut generics, &func.input)?;
        let output = Ty::from_hir(unit, &mut generics, &func.output)?;
        let locals = func
            .locals
            .iter()
            .map(|local| Ty::from_hir(unit, &mut generics, &local.ty))
            .collect::<miette::Result<_>>()?;
        let body = Expr::from_hir(unit, &mut generics, &func.body)?;

        let generics = generics.into_iter().map(|(_, g)| g).collect();

        Ok(Self {
            name: func.name.to_string(),
            generics,
            input,
            output,
            locals,
            body,
        })
    }

    pub fn ty(&self) -> Ty {
        let input = self.input.iter().map(|arg| arg.ty.clone()).collect();
        Ty::Func(input, Box::new(self.output.clone()))
    }

    fn from_hir_vec(unit: &hir::Unit, funcs: &[hir::Func]) -> miette::Result<Vec<Self>> {
        funcs
            .iter()
            .map(|func| Func::from_hir(unit, func))
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
            span: arg.span,
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
            hir::Ty::Field(_, _) | hir::Ty::Call(_, _) | hir::Ty::Pipe(_, _) => {
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
        unit: &hir::Unit,
        generics: &mut Generics,
        expr: &hir::Expr,
    ) -> miette::Result<Self> {
        Ok(Self {
            kind: ExprKind::from_hir(unit, generics, &expr.kind)?,
            ty: Ty::from_hir(unit, generics, &expr.ty)?,
        })
    }

    fn vec_from_hir(
        unit: &hir::Unit,
        generics: &mut Generics,
        exprs: &[hir::Expr],
    ) -> miette::Result<Vec<Expr>> {
        exprs
            .iter()
            .map(|expr| Expr::from_hir(unit, generics, expr))
            .collect::<miette::Result<_>>()
    }
}

impl ExprKind {
    fn from_hir(
        unit: &hir::Unit,
        generics: &mut Generics,
        kind: &hir::ExprKind,
    ) -> miette::Result<Self> {
        match kind {
            hir::ExprKind::Void => Ok(ExprKind::Void),
            hir::ExprKind::Int(n, base, bytes) => Ok(ExprKind::Int(*n, *base, bytes.clone())),
            hir::ExprKind::Func(i) => Ok(ExprKind::Func(*i)),
            hir::ExprKind::Variant(i, j) => Ok(ExprKind::Variant(*i, *j)),
            hir::ExprKind::Local(i) => Ok(ExprKind::Local(*i)),
            hir::ExprKind::Argument(i) => Ok(ExprKind::Argument(*i)),
            hir::ExprKind::Tuple(exprs) => {
                let tuple = Expr::vec_from_hir(unit, generics, exprs)?;

                Ok(ExprKind::Tuple(tuple))
            }
            hir::ExprKind::List(exprs) => {
                let list = Expr::vec_from_hir(unit, generics, exprs)?;

                Ok(ExprKind::List(list))
            }
            hir::ExprKind::Block(exprs) => {
                let block = Expr::vec_from_hir(unit, generics, exprs)?;

                Ok(ExprKind::Block(block))
            }
            hir::ExprKind::Call(callee, args) => {
                let callee = Box::new(Expr::from_hir(unit, generics, callee)?);
                let args = Expr::vec_from_hir(unit, generics, args)?;

                Ok(ExprKind::Call(callee, args))
            }
            hir::ExprKind::Field(expr, field) => {
                let expr = Box::new(Expr::from_hir(unit, generics, expr)?);

                let Ty::Adt(index, _) = expr.ty else {
                    unreachable!("unexpected field: {:?}", expr.ty)
                };

                let (index, _) = unit.adts[index].find_field(field)?;

                Ok(ExprKind::Field(expr, index))
            }
            hir::ExprKind::Pipe(lhs, rhs) => {
                let lhs = Box::new(Expr::from_hir(unit, generics, lhs)?);
                let rhs = Box::new(Expr::from_hir(unit, generics, rhs)?);

                Ok(ExprKind::Pipe(lhs, rhs))
            }
            hir::ExprKind::Let(local, expr) => {
                let expr = Box::new(Expr::from_hir(unit, generics, expr)?);

                Ok(ExprKind::Let(*local, expr))
            }
        }
    }
}
