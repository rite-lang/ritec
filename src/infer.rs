use std::collections::{hash_map::Entry, HashMap, VecDeque};

use crate::hir::{self, format_partial};

#[derive(Debug, Default)]
pub struct TyEnv {
    constraints: VecDeque<(hir::Ty, hir::Ty)>,
    substitutions: HashMap<hir::Ty, hir::Ty>,
    the_matrix: HashMap<usize, (Vec<hir::Tid>, Vec<Vec<hir::Ty>>)>,
}

impl TyEnv {
    pub fn unify(&mut self, a: hir::Ty, b: hir::Ty) {
        self.constraints.push_back((a, b));
    }

    pub fn normalize(&mut self, ty: hir::Ty) {
        self.unify(ty.clone(), ty);
    }

    pub fn substitute(&self, ty: &hir::Ty) -> Option<&hir::Ty> {
        self.substitutions.get(ty)
    }

    pub fn get(&self, ty: &hir::Ty) -> hir::Ty {
        match self.substitute(ty) {
            Some(sub) => self.get(sub),
            None => ty.clone(),
        }
    }

    pub fn use_ty(&mut self, generics: &mut HashMap<usize, hir::Ty>, ty: &hir::Ty) -> hir::Ty {
        match ty {
            hir::Ty::Inferred(tid, kind, func) => {
                let ty = hir::Ty::inferred(*kind);

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
            hir::Ty::Partial(hir::Part::Generic(index), arguments) => {
                assert!(arguments.is_empty());

                match generics.entry(*index) {
                    Entry::Occupied(entry) => entry.get().clone(),
                    Entry::Vacant(entry) => {
                        let ty = hir::Ty::inferred(hir::Inferred::Any);
                        entry.insert(ty).clone()
                    }
                }
            }
            hir::Ty::Partial(part, arguments) => {
                let arguments = arguments
                    .iter()
                    .map(|arg| self.use_ty(generics, arg))
                    .collect();
                hir::Ty::Partial(*part, arguments)
            }
            hir::Ty::Field(ty, field) => {
                let ty = self.use_ty(generics, ty);
                hir::Ty::Field(Box::new(ty), field)
            }
            hir::Ty::Tuple(ty, index) => {
                let ty = self.use_ty(generics, ty);
                hir::Ty::Tuple(Box::new(ty), *index)
            }
            hir::Ty::Call(func, arguments) => {
                let func = self.use_ty(generics, func);
                let arguments = arguments
                    .iter()
                    .map(|arg| self.use_ty(generics, arg))
                    .collect();
                hir::Ty::Call(Box::new(func), arguments)
            }
            hir::Ty::Pipe(lhs, rhs) => {
                let lhs = self.use_ty(generics, lhs);
                let rhs = self.use_ty(generics, rhs);
                hir::Ty::Pipe(Box::new(lhs), Box::new(rhs))
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

pub fn infer(unit: &mut hir::Unit) -> miette::Result<()> {
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

fn unify_ty_ty(unit: &mut hir::Unit, a: &hir::Ty, b: &hir::Ty) -> Result<(), RetryOrError> {
    let a = normalize(unit, a)?;
    let b = normalize(unit, b)?;

    match (a, b) {
        (
            a @ hir::Ty::Inferred(a_tid, a_kind, a_func),
            b @ hir::Ty::Inferred(b_tid, b_kind, b_func),
        ) => {
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
        (inferred @ hir::Ty::Inferred(tid, kind, func), mut ty)
        | (mut ty, inferred @ hir::Ty::Inferred(tid, kind, func)) => {
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
                hir::Inferred::Int(kind) if ty != hir::Ty::int(kind) => {
                    return Err(miette::miette!(
                        "expected `{}` but found `{}`",
                        hir::Ty::int(kind),
                        ty
                    )
                    .into());
                }
                hir::Inferred::Float(_) => todo!(),
                _ => {}
            }

            unit.env.substitutions.insert(inferred.clone(), ty.clone());
            Ok(())
        }
        (hir::Ty::Partial(a_part, a_args), hir::Ty::Partial(b_part, b_args)) => {
            unify_partial_partial(unit, &a_part, &a_args, &b_part, &b_args)
        }
        _ => unreachable!(),
    }
}

fn is_sub_kind(a: &hir::Inferred, b: &hir::Inferred) -> bool {
    match (a, b) {
        (hir::Inferred::Any, _) => true,
        (hir::Inferred::Int(a), hir::Inferred::Int(b)) => a == b,
        (hir::Inferred::Float(a), hir::Inferred::Float(b)) => a == b,
        _ => false,
    }
}

fn unify_inferred_inferred(
    unit: &mut hir::Unit,
    a_tid: hir::Tid,
    a_kind: hir::Inferred,
    a_func: Option<usize>,
    b_tid: hir::Tid,
    b_kind: hir::Inferred,
    b_func: Option<usize>,
) -> Result<(), RetryOrError> {
    let a = hir::Ty::Inferred(a_tid, a_kind, a_func);
    let b = hir::Ty::Inferred(b_tid, b_kind, b_func);

    match (a_func, b_func) {
        (None, None) | (None, Some(_)) => {
            unit.env.substitutions.insert(a, b);

            Ok(())
        }
        (Some(a_func), None) => {
            let b = hir::Ty::Inferred(b_tid, b_kind, Some(a_func));

            if let Some((ref index, ref table)) = unit.env.the_matrix.get(&a_func) {
                let a_index = index.iter().position(|&i| i == a_tid).unwrap();

                for a in table[a_index].clone() {
                    let b = unit.env.use_ty(&mut HashMap::new(), &b);
                    unify_ty_ty(unit, &a, &b)?;
                }
            }

            unit.env.substitutions.insert(a, b);

            Ok(())
        }
        (Some(a_func), Some(b_func)) => {
            assert_eq!(a_func, b_func);

            if let Some((ref index, ref table)) = unit.env.the_matrix.get(&a_func) {
                let a_index = index.iter().position(|&i| i == a_tid).unwrap();
                let b_index = index.iter().position(|&i| i == b_tid).unwrap();

                let a_row = table[a_index].clone();
                let b_row = table[b_index].clone();

                for (a, b) in a_row.into_iter().zip(b_row) {
                    unify_ty_ty(unit, &a, &b)?;
                }
            }

            unit.env.substitutions.insert(a, b);

            Ok(())
        }
    }
}

fn normalize(unit: &mut hir::Unit, ty: &hir::Ty) -> Result<hir::Ty, RetryOrError> {
    if let Some(ty) = unit.env.substitute(ty) {
        return normalize(unit, &ty.clone());
    }

    match ty {
        hir::Ty::Inferred(_, _, _) => Ok(ty.clone()),
        hir::Ty::Partial(_, _) => Ok(ty.clone()),
        hir::Ty::Field(adt, field) => {
            let normalized = normalize_field(unit, adt, field)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
        hir::Ty::Tuple(base, index) => {
            let normalized = normalize_tuple(unit, base, *index)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
        hir::Ty::Call(callee, arguments) => {
            let normalized = normalize_call(unit, callee, arguments)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
        hir::Ty::Pipe(lhs, rhs) => {
            let normalized = normalize_pipe(unit, lhs, rhs)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
    }
}

fn with_func(unit: &mut hir::Unit, ty: &hir::Ty, new_func: usize) -> hir::Ty {
    match ty {
        hir::Ty::Inferred(_, kind, func) => {
            assert!(func.is_none() || *func == Some(new_func));

            let new_ty = hir::Ty::Inferred(hir::Tid::new(), *kind, Some(new_func));
            unit.unify(ty.clone(), new_ty.clone());
            new_ty
        }
        hir::Ty::Partial(part, arguments) => {
            let arguments = arguments
                .iter()
                .map(|arg| with_func(unit, arg, new_func))
                .collect();
            hir::Ty::Partial(*part, arguments)
        }
        hir::Ty::Field(base, field) => {
            let base = with_func(unit, base, new_func);

            let ty = hir::Ty::Field(Box::new(base), field);
            unit.normalize(ty.clone());
            ty
        }
        hir::Ty::Tuple(base, index) => {
            let base = with_func(unit, base, new_func);

            let ty = hir::Ty::Tuple(Box::new(base), *index);
            unit.normalize(ty.clone());
            ty
        }
        hir::Ty::Call(callee, arguments) => {
            let callee = with_func(unit, callee, new_func);
            let arguments = arguments
                .iter()
                .map(|arg| with_func(unit, arg, new_func))
                .collect();

            let ty = hir::Ty::Call(Box::new(callee), arguments);
            unit.normalize(ty.clone());
            ty
        }
        hir::Ty::Pipe(lhs, rhs) => {
            let lhs = with_func(unit, lhs, new_func);
            let rhs = with_func(unit, rhs, new_func);

            let ty = hir::Ty::Pipe(Box::new(lhs), Box::new(rhs));
            unit.normalize(ty.clone());
            ty
        }
    }
}

fn normalize_field(
    unit: &mut hir::Unit,
    adt: &hir::Ty,
    field: &str,
) -> Result<hir::Ty, RetryOrError> {
    let adt = normalize(unit, adt)?;

    let (index, generics) = match adt {
        hir::Ty::Inferred(_, _, _) => return Err(RetryOrError::Retry),
        hir::Ty::Partial(hir::Part::Adt(index), generics) => (index, generics),
        hir::Ty::Partial(_, _) => return Err(miette::miette!("expected an ADT").into()),
        hir::Ty::Field(_, _) | hir::Ty::Tuple(_, _) | hir::Ty::Call(_, _) | hir::Ty::Pipe(_, _) => {
            unreachable!()
        }
    };

    let (_, ty) = unit.adts[index].find_field(field)?;
    Ok(ty.specialize(&generics))
}

fn normalize_tuple(
    unit: &mut hir::Unit,
    base: &hir::Ty,
    index: usize,
) -> Result<hir::Ty, RetryOrError> {
    let base = normalize(unit, base)?;

    let (index, items) = match base {
        hir::Ty::Inferred(_, _, _) => return Err(RetryOrError::Retry),
        hir::Ty::Partial(hir::Part::Tuple, items) => (index, items),
        hir::Ty::Partial(_, _) => return Err(miette::miette!("expected a tuple").into()),
        hir::Ty::Field(_, _) | hir::Ty::Tuple(_, _) | hir::Ty::Call(_, _) | hir::Ty::Pipe(_, _) => {
            unreachable!()
        }
    };

    Ok(items[index].clone())
}

fn normalize_call(
    unit: &mut hir::Unit,
    callee: &hir::Ty,
    arguments: &[hir::Ty],
) -> Result<hir::Ty, RetryOrError> {
    let callee = normalize(unit, callee)?;

    let mut callee_arguments = match callee {
        hir::Ty::Inferred(_, _, _) => return Err(RetryOrError::Retry),
        hir::Ty::Partial(hir::Part::Func, arguments) => arguments,
        hir::Ty::Partial(_, _) => return Err(miette::miette!("expected a function").into()),
        hir::Ty::Field(_, _) | hir::Ty::Tuple(_, _) | hir::Ty::Call(_, _) | hir::Ty::Pipe(_, _) => {
            unreachable!()
        }
    };

    let output = callee_arguments
        .pop()
        .expect("funcs have least one argument");

    if arguments.len() != callee_arguments.len() {
        return Err(miette::miette!(
            "wrong number of arguments: expected {} but found {}",
            callee_arguments.len(),
            arguments.len()
        )
        .into());
    }

    for (a, b) in callee_arguments.iter().zip(arguments) {
        unify_ty_ty(unit, a, b)?;
    }

    normalize(unit, &output)
}

fn normalize_pipe(
    unit: &mut hir::Unit,
    lhs: &hir::Ty,
    rhs: &hir::Ty,
) -> Result<hir::Ty, RetryOrError> {
    let rhs = normalize(unit, rhs)?;

    let mut rhs_arguments = match rhs {
        hir::Ty::Inferred(_, _, _) => return Err(RetryOrError::Retry),
        hir::Ty::Partial(hir::Part::Func, arguments) => arguments,
        hir::Ty::Partial(_, _) => return Err(miette::miette!("expected a function").into()),
        hir::Ty::Field(_, _) | hir::Ty::Tuple(_, _) | hir::Ty::Call(_, _) | hir::Ty::Pipe(_, _) => {
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
            let tuple = hir::Ty::Partial(hir::Part::Tuple, rhs_arguments);
            unify_ty_ty(unit, lhs, &tuple)?;
            normalize(unit, &output)
        }
    }
}

fn unify_partial_partial(
    unit: &mut hir::Unit,
    a_part: &hir::Part,
    a_args: &[hir::Ty],
    b_part: &hir::Part,
    b_args: &[hir::Ty],
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
