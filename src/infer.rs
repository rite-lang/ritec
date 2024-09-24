use std::collections::{hash_map::Entry, HashMap, VecDeque};

use crate::hir::{self, format_partial, Inferred, Part, Tid, Ty, Unit};

#[derive(Debug, Default)]
pub struct TyEnv {
    constraints: VecDeque<(Ty, Ty)>,
    substitutions: HashMap<Ty, Ty>,
    the_matrix: HashMap<usize, (Vec<Tid>, Vec<Vec<Ty>>)>,
}

impl TyEnv {
    pub fn unify(&mut self, a: Ty, b: Ty) {
        self.constraints.push_back((a, b));
    }

    pub fn normalize(&mut self, ty: Ty) {
        self.unify(ty.clone(), ty);
    }

    pub fn substitute(&self, ty: &Ty) -> Option<&Ty> {
        self.substitutions.get(ty)
    }

    pub fn get(&self, ty: &Ty) -> Ty {
        match self.substitute(ty) {
            Some(sub) => self.get(sub),
            None => ty.clone(),
        }
    }

    pub fn use_ty(&mut self, generics: &mut HashMap<usize, Ty>, ty: &Ty) -> Ty {
        if let Some(ty) = self.substitute(ty) {
            return self.use_ty(generics, &ty.clone());
        }

        match ty {
            Ty::Inferred(tid, kind, func) => {
                let ty = Ty::inferred(*kind);

                if let Some(func) = func {
                    let (index, table) = self.the_matrix.entry(*func).or_default();

                    match index.iter().position(|i| i == tid) {
                        Some(index) => table[index].push(ty.clone()),
                        None => {
                            index.push(*tid);
                            table.push(vec![ty.clone()]);
                        }
                    }
                }

                ty
            }
            Ty::Partial(Part::Generic(index), arguments) => {
                assert!(arguments.is_empty());

                match generics.entry(*index) {
                    Entry::Occupied(entry) => entry.get().clone(),
                    Entry::Vacant(entry) => {
                        let ty = Ty::inferred(Inferred::Any);
                        entry.insert(ty).clone()
                    }
                }
            }
            Ty::Partial(part, arguments) => {
                let arguments = arguments
                    .iter()
                    .map(|arg| self.use_ty(generics, arg))
                    .collect();
                Ty::Partial(*part, arguments)
            }
            Ty::Field(ty, field) => {
                let ty = self.use_ty(generics, ty);
                Ty::Field(Box::new(ty), field)
            }
            Ty::Tuple(ty, index) => {
                let ty = self.use_ty(generics, ty);
                Ty::Tuple(Box::new(ty), *index)
            }
            Ty::Call(func, arguments) => {
                let func = self.use_ty(generics, func);
                let arguments = arguments
                    .iter()
                    .map(|arg| arg.as_ref().map(|arg| self.use_ty(generics, arg)))
                    .collect();
                Ty::Call(Box::new(func), arguments)
            }
            Ty::Pipe(lhs, rhs) => {
                let lhs = self.use_ty(generics, lhs);
                let rhs = self.use_ty(generics, rhs);
                Ty::Pipe(Box::new(lhs), Box::new(rhs))
            }
        }
    }
}

#[derive(Debug)]
enum RetryOrError {
    Retry,
    Error(miette::Report),
}

impl From<miette::Report> for RetryOrError {
    fn from(report: miette::Report) -> Self {
        Self::Error(report)
    }
}

pub fn infer(unit: &mut Unit) -> miette::Result<()> {
    let mut retried = 0;

    while let Some((a, b)) = unit.env.constraints.pop_front() {
        match unify_ty_ty(unit, &a, &b) {
            Ok(()) => {
                retried = 0;
            }
            Err(RetryOrError::Retry) => {
                unit.env.constraints.push_back((a, b));
                retried += 1;
            }
            Err(RetryOrError::Error(report)) => return Err(report),
        }

        if retried > unit.env.constraints.len() {
            return Err(miette::miette!("not enough information to infer types"));
        }
    }

    Ok(())
}

fn unify_ty_ty(unit: &mut Unit, a: &Ty, b: &Ty) -> Result<(), RetryOrError> {
    let a = normalize(unit, a)?;
    let b = normalize(unit, b)?;

    match (a, b) {
        (a @ Ty::Inferred(a_tid, a_kind, a_func), b @ Ty::Inferred(b_tid, b_kind, b_func)) => {
            if a_tid == b_tid {
                return Ok(());
            }

            if is_sub_kind(&a_kind, &b_kind) {
                unify_inferred_inferred(unit, a_tid, a_kind, a_func, b_tid, b_kind, b_func)?;
                Ok(())
            } else if is_sub_kind(&b_kind, &a_kind) {
                unify_inferred_inferred(unit, b_tid, b_kind, b_func, a_tid, a_kind, a_func)?;
                Ok(())
            } else {
                Err(miette::miette!("expected `{}` but found `{}`", a, b).into())
            }
        }
        (inferred @ Ty::Inferred(tid, kind, func), mut ty)
        | (mut ty, inferred @ Ty::Inferred(tid, kind, func)) => {
            if let Some(func) = func {
                ty = with_func(unit, &ty, func);

                if let Some((ref index, ref table)) = unit.env.the_matrix.get(&func) {
                    let index = index.iter().position(|&i| i == tid).unwrap();
                    for used in table[index].clone().iter() {
                        let ty = unit.env.use_ty(&mut HashMap::new(), &ty);
                        unify_ty_ty(unit, used, &ty)?;
                    }
                }
            }

            match kind {
                Inferred::Int(kind) if ty != Ty::int(kind) => {
                    return Err(miette::miette!(
                        "expected `{}` but found `{}`",
                        hir::Ty::int(kind),
                        ty
                    )
                    .into());
                }
                Inferred::Float(_) => todo!(),
                _ => {}
            }

            unit.env.substitutions.insert(inferred.clone(), ty.clone());
            Ok(())
        }
        (Ty::Partial(a_part, a_args), Ty::Partial(b_part, b_args)) => {
            unify_partial_partial(unit, &a_part, &a_args, &b_part, &b_args)
        }
        _ => unreachable!(),
    }
}

fn is_sub_kind(a: &Inferred, b: &Inferred) -> bool {
    match (a, b) {
        (Inferred::Any, _) => true,
        (Inferred::Int(a), Inferred::Int(b)) => a == b,
        (Inferred::Float(a), Inferred::Float(b)) => a == b,
        _ => false,
    }
}

fn unify_inferred_inferred(
    unit: &mut Unit,
    a_tid: Tid,
    a_kind: Inferred,
    a_func: Option<usize>,
    b_tid: Tid,
    b_kind: Inferred,
    b_func: Option<usize>,
) -> Result<(), RetryOrError> {
    let a = Ty::Inferred(a_tid, a_kind, a_func);
    let b = Ty::Inferred(b_tid, b_kind, b_func);

    match (a_func, b_func) {
        (None, None) | (None, Some(_)) => {
            unit.env.substitutions.insert(a, b);

            Ok(())
        }
        (Some(a_func), None) => {
            let b_with_func = Ty::Inferred(b_tid, b_kind, Some(a_func));

            if let Some((ref index, ref table)) = unit.env.the_matrix.get(&a_func) {
                let a_index = index.iter().position(|&i| i == a_tid).unwrap();

                for a in table[a_index].clone() {
                    let b = unit.env.use_ty(&mut HashMap::new(), &b_with_func);
                    unify_ty_ty(unit, &a, &b)?;
                }
            }

            unit.env.substitutions.insert(b, b_with_func.clone());
            unit.env.substitutions.insert(a, b_with_func);

            Ok(())
        }
        (Some(a_func), Some(b_func)) => {
            assert_eq!(a_func, b_func);

            if let Some((ref index, ref table)) = unit.env.the_matrix.get(&a_func) {
                let a_index = index.iter().position(|&i| i == a_tid).unwrap();
                let b_index = index.iter().position(|&i| i == b_tid).unwrap();

                let a_row = table[a_index].clone();
                let b_row = table[b_index].clone();

                assert_eq!(a_row.len(), b_row.len());

                for (a, b) in a_row.into_iter().zip(b_row) {
                    unify_ty_ty(unit, &a, &b)?;
                }
            }

            unit.env.substitutions.insert(a, b);

            Ok(())
        }
    }
}

fn normalize(unit: &mut Unit, ty: &Ty) -> Result<Ty, RetryOrError> {
    if let Some(ty) = unit.env.substitute(ty) {
        return normalize(unit, &ty.clone());
    }

    match ty {
        Ty::Inferred(_, _, _) => Ok(ty.clone()),
        Ty::Partial(_, _) => Ok(ty.clone()),
        Ty::Field(adt, field) => {
            let normalized = normalize_field(unit, adt, field)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
        Ty::Tuple(base, index) => {
            let normalized = normalize_tuple(unit, base, *index)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
        Ty::Call(callee, arguments) => {
            let normalized = normalize_call(unit, callee, arguments)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
        Ty::Pipe(lhs, rhs) => {
            let normalized = normalize_pipe(unit, lhs, rhs)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
    }
}

fn with_func(unit: &mut Unit, ty: &Ty, new_func: usize) -> Ty {
    match ty {
        Ty::Inferred(_, kind, func) => {
            assert!(func.is_none() || *func == Some(new_func));

            let new_ty = Ty::Inferred(Tid::new(), *kind, Some(new_func));
            unit.unify(ty.clone(), new_ty.clone());
            new_ty
        }
        Ty::Partial(part, arguments) => {
            let arguments = arguments
                .iter()
                .map(|arg| with_func(unit, arg, new_func))
                .collect();
            Ty::Partial(*part, arguments)
        }
        Ty::Field(base, field) => {
            let base = with_func(unit, base, new_func);

            let ty = Ty::Field(Box::new(base), field);
            unit.normalize(ty.clone());
            ty
        }
        Ty::Tuple(base, index) => {
            let base = with_func(unit, base, new_func);

            let ty = Ty::Tuple(Box::new(base), *index);
            unit.normalize(ty.clone());
            ty
        }
        Ty::Call(callee, arguments) => {
            let callee = with_func(unit, callee, new_func);
            let arguments = arguments
                .iter()
                .map(|arg| arg.as_ref().map(|arg| with_func(unit, arg, new_func)))
                .collect();

            let ty = Ty::Call(Box::new(callee), arguments);
            unit.normalize(ty.clone());
            ty
        }
        Ty::Pipe(lhs, rhs) => {
            let lhs = with_func(unit, lhs, new_func);
            let rhs = with_func(unit, rhs, new_func);

            let ty = Ty::Pipe(Box::new(lhs), Box::new(rhs));
            unit.normalize(ty.clone());
            ty
        }
    }
}

fn normalize_field(unit: &mut Unit, adt: &Ty, field: &str) -> Result<Ty, RetryOrError> {
    let adt = normalize(unit, adt)?;

    let (index, generics) = match adt {
        Ty::Inferred(_, _, _) => return Err(RetryOrError::Retry),
        Ty::Partial(Part::Adt(index), generics) => (index, generics),
        Ty::Partial(_, _) => return Err(miette::miette!("expected an ADT").into()),
        Ty::Field(_, _) | Ty::Tuple(_, _) | Ty::Call(_, _) | Ty::Pipe(_, _) => {
            unreachable!()
        }
    };

    let (_, ty) = unit.adts[index].find_field(field)?;
    Ok(ty.specialize(&generics))
}

fn normalize_tuple(unit: &mut Unit, base: &Ty, index: usize) -> Result<Ty, RetryOrError> {
    let base = normalize(unit, base)?;

    let (index, items) = match base {
        Ty::Inferred(_, _, _) => return Err(RetryOrError::Retry),
        Ty::Partial(Part::Tuple, items) => (index, items),
        Ty::Partial(_, _) => return Err(miette::miette!("expected a tuple").into()),
        Ty::Field(_, _) | Ty::Tuple(_, _) | Ty::Call(_, _) | Ty::Pipe(_, _) => {
            unreachable!()
        }
    };

    Ok(items[index].clone())
}

fn normalize_call(
    unit: &mut Unit,
    callee: &Ty,
    arguments: &[Option<Ty>],
) -> Result<Ty, RetryOrError> {
    let callee = normalize(unit, callee)?;

    let mut callee_arguments = match callee {
        Ty::Inferred(_, _, _) => return Err(RetryOrError::Retry),
        Ty::Partial(Part::Func, arguments) => arguments,
        Ty::Partial(_, _) => return Err(miette::miette!("expected a function").into()),
        Ty::Field(_, _) | Ty::Tuple(_, _) | Ty::Call(_, _) | Ty::Pipe(_, _) => {
            unreachable!()
        }
    };

    let output = callee_arguments
        .pop()
        .expect("funcs have least one argument");

    if arguments.len() > callee_arguments.len() {
        return Err(miette::miette!(
            "wrong number of arguments: expected {} but found {}",
            callee_arguments.len(),
            arguments.len()
        )
        .into());
    }

    let mut remaining = Vec::new();

    for (a, b) in callee_arguments.iter().zip(arguments) {
        match b {
            Some(b) => unify_ty_ty(unit, a, b)?,
            None => remaining.push(a.clone()),
        }
    }

    remaining.extend(callee_arguments.iter().skip(arguments.len()).cloned());

    if remaining.is_empty() {
        return normalize(unit, &output);
    }

    remaining.push(output);

    let func = Ty::Partial(Part::Func, remaining);
    normalize(unit, &func)
}

fn normalize_pipe(unit: &mut Unit, lhs: &Ty, rhs: &Ty) -> Result<Ty, RetryOrError> {
    let rhs = normalize(unit, rhs)?;

    let mut rhs_arguments = match rhs {
        Ty::Inferred(_, _, _) => return Err(RetryOrError::Retry),
        Ty::Partial(Part::Func, arguments) => arguments,
        Ty::Partial(_, _) => return Err(miette::miette!("expected a function").into()),
        Ty::Field(_, _) | Ty::Tuple(_, _) | Ty::Call(_, _) | Ty::Pipe(_, _) => {
            unreachable!()
        }
    };

    let output = rhs_arguments.pop().expect("funcs have least one argument");

    match rhs_arguments.len() {
        0 => Err(miette::miette!("expected a function").into()),
        1 => {
            unify_ty_ty(unit, lhs, &rhs_arguments[0])?;
            normalize(unit, &output)
        }
        _ => {
            let tuple = Ty::Partial(Part::Tuple, rhs_arguments);
            unify_ty_ty(unit, lhs, &tuple)?;
            normalize(unit, &output)
        }
    }
}

fn unify_partial_partial(
    unit: &mut Unit,
    a_part: &Part,
    a_args: &[Ty],
    b_part: &Part,
    b_args: &[Ty],
) -> Result<(), RetryOrError> {
    if a_part != b_part || a_args.len() != b_args.len() {
        return Err(miette::miette!(
            "expected `{}` but found `{}`",
            format_partial(a_part, a_args),
            format_partial(b_part, b_args)
        )
        .into());
    }

    for (a, b) in a_args.iter().zip(b_args) {
        unify_ty_ty(unit, a, b)?;
    }

    Ok(())
}
