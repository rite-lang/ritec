use std::collections::{HashMap, VecDeque};

use crate::{
    hir::{self, Inferred, Part, Tid, Ty, Unit},
    span::Span,
};

#[derive(Debug, Default)]
pub struct TyEnv {
    constraints: VecDeque<(Ty, Ty, Span)>,
    substitutions: HashMap<Ty, Ty>,
    the_matrix: HashMap<usize, (Vec<Tid>, Vec<Vec<Ty>>)>,
    the_gatrix: HashMap<usize, Vec<Vec<Ty>>>,
}

impl TyEnv {
    pub fn unify(&mut self, a: Ty, b: Ty, span: Span) {
        self.constraints.push_back((a, b, span));
    }

    pub fn normalize(&mut self, ty: Ty, span: Span) {
        self.unify(ty.clone(), ty, span);
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

    pub fn next_column(&mut self, func: usize) -> usize {
        let table = self.the_gatrix.entry(func).or_default();
        table.first().map_or(0, Vec::len)
    }

    pub fn use_ty(&mut self, column: usize, ty: &Ty, span: Span) -> Ty {
        if let Some(ty) = self.substitute(ty) {
            return self.use_ty(column, &ty.clone(), span);
        }

        match ty {
            Ty::Inferred(tid, kind, func, span) => {
                let ty = Ty::inferred(*kind, *span);

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
            Ty::Partial(Part::Generic(index, func), arguments, _) => {
                assert!(arguments.is_empty());

                let func = func.expect("generics must have a function");
                let table = self.the_gatrix.entry(func).or_default();

                if table.len() <= *index {
                    table.resize(index + 1, Vec::new());
                }

                if table[*index].len() <= column {
                    table[*index].push(Ty::inferred(Inferred::Any, span));
                }

                table[*index][column].clone()
            }
            Ty::Partial(part, arguments, ty_span) => {
                let arguments = arguments
                    .iter()
                    .map(|arg| self.use_ty(column, arg, span))
                    .collect();
                Ty::Partial(*part, arguments, *ty_span)
            }
            Ty::Field(ty, field, ty_span) => {
                let ty = self.use_ty(column, ty, span);
                Ty::Field(Box::new(ty), field, *ty_span)
            }
            Ty::Tuple(ty, index, ty_span) => {
                let ty = self.use_ty(column, ty, span);
                Ty::Tuple(Box::new(ty), *index, *ty_span)
            }
            Ty::Call(func, arguments, ty_span) => {
                let func = self.use_ty(column, func, span);
                let arguments = arguments
                    .iter()
                    .map(|arg| arg.as_ref().map(|arg| self.use_ty(column, arg, span)))
                    .collect();
                Ty::Call(Box::new(func), arguments, *ty_span)
            }
            Ty::Pipe(lhs, rhs, arguments, ty_span) => {
                let lhs = self.use_ty(column, lhs, span);
                let rhs = self.use_ty(column, rhs, span);

                let arguments = arguments
                    .iter()
                    .map(|arg| arg.as_ref().map(|arg| self.use_ty(column, arg, span)))
                    .collect();

                Ty::Pipe(Box::new(lhs), Box::new(rhs), arguments, *ty_span)
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

    while let Some((a, b, span)) = unit.env.constraints.pop_front() {
        match unify_ty_ty(unit, &a, &b, span) {
            Ok(()) => {
                retried = 0;
            }
            Err(RetryOrError::Retry) => {
                unit.env.constraints.push_back((a, b, span));
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

fn unify_ty_ty(unit: &mut Unit, a: &Ty, b: &Ty, constraint_span: Span) -> Result<(), RetryOrError> {
    let a = normalize(unit, a, constraint_span)?;
    let b = normalize(unit, b, constraint_span)?;

    match (a, b) {
        (
            a @ Ty::Inferred(a_tid, a_kind, a_func, a_span),
            b @ Ty::Inferred(b_tid, b_kind, b_func, b_span),
        ) => {
            if a_tid == b_tid {
                return Ok(());
            }

            if is_sub_kind(&a_kind, &b_kind) {
                unify_inferred_inferred(
                    unit,
                    a_tid,
                    a_kind,
                    a_func,
                    a_span,
                    b_tid,
                    b_kind,
                    b_func,
                    b_span,
                    constraint_span,
                )?;
                Ok(())
            } else if is_sub_kind(&b_kind, &a_kind) {
                unify_inferred_inferred(
                    unit,
                    b_tid,
                    b_kind,
                    b_func,
                    b_span,
                    a_tid,
                    a_kind,
                    a_func,
                    a_span,
                    constraint_span,
                )?;
                Ok(())
            } else {
                Err(miette::miette!(
                    "expected `{}` but found `{}`",
                    a.format(unit),
                    b.format(unit),
                )
                .into())
            }
        }
        (inferred @ Ty::Inferred(tid, kind, func, span), mut ty)
        | (mut ty, inferred @ Ty::Inferred(tid, kind, func, span)) => {
            if let Some(func) = func {
                ty = with_func(unit, &ty, func);

                if let Some((ref index, ref table)) = unit.env.the_matrix.get(&func) {
                    let index = index.iter().position(|&i| i == tid).unwrap();
                    for (i, used) in table[index].clone().iter().enumerate() {
                        let ty = unit.env.use_ty(i, &ty, span);
                        unify_ty_ty(unit, used, &ty, constraint_span)?;
                    }
                }
            }

            match kind {
                Inferred::Any => {}
                Inferred::Unsigned => {
                    if !matches!(ty, Ty::Partial(Part::Int(_), _, _)) {
                        return Err(miette::miette!(
                            labels = [span.label("here"), constraint_span.label("here")],
                            "expected `{}` but found `{}`",
                            inferred.format(unit),
                            ty.format(unit),
                        )
                        .with_source_code(span)
                        .into());
                    }
                }
                Inferred::Signed => {
                    let hir::Ty::Partial(hir::Part::Int(kind), _, _) = ty.clone() else {
                        return Err(miette::miette!(
                            labels = [span.label("here"), constraint_span.label("here")],
                            "expected `{}` but found `{}`",
                            inferred.format(unit),
                            ty.format(unit),
                        )
                        .with_source_code(span)
                        .into());
                    };

                    if !kind.is_signed() {
                        return Err(miette::miette!(
                            labels = [span.label("here"), constraint_span.label("here")],
                            "expected `{}` but found `{}`",
                            inferred.format(unit),
                            ty.format(unit),
                        )
                        .with_source_code(span)
                        .into());
                    }
                }
                Inferred::Float => todo!(),
            }

            unit.env.substitutions.insert(inferred.clone(), ty.clone());
            Ok(())
        }
        (Ty::Partial(a_part, a_args, a_span), Ty::Partial(b_part, b_args, b_span)) => {
            unify_partial_partial(
                unit,
                &a_part,
                &a_args,
                a_span,
                &b_part,
                &b_args,
                b_span,
                constraint_span,
            )
        }
        _ => {
            unreachable!()
        }
    }
}

fn is_sub_kind(a: &Inferred, b: &Inferred) -> bool {
    #[allow(clippy::match_like_matches_macro)]
    match (a, b) {
        (Inferred::Any, _) => true,
        (Inferred::Unsigned, Inferred::Unsigned | Inferred::Signed) => true,
        (Inferred::Signed, Inferred::Signed) => true,
        (Inferred::Float, Inferred::Float) => true,
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn unify_inferred_inferred(
    unit: &mut Unit,
    a_tid: Tid,
    a_kind: Inferred,
    a_func: Option<usize>,
    a_span: Span,
    b_tid: Tid,
    b_kind: Inferred,
    b_func: Option<usize>,
    b_span: Span,
    constraint_span: Span,
) -> Result<(), RetryOrError> {
    let a = Ty::Inferred(a_tid, a_kind, a_func, a_span);
    let b = Ty::Inferred(b_tid, b_kind, b_func, b_span);

    match (a_func, b_func) {
        (None, None) | (None, Some(_)) => {
            unit.env.substitutions.insert(a, b);

            Ok(())
        }
        (Some(a_func), None) => {
            let b_with_func = Ty::Inferred(b_tid, b_kind, Some(a_func), b_span);

            if let Some((ref index, ref table)) = unit.env.the_matrix.get(&a_func) {
                let a_index = index.iter().position(|&i| i == a_tid).unwrap();

                let mut row = Vec::new();

                for table_a in &table[a_index] {
                    // this check prevents a stack overflow
                    // when a function calls itself
                    if unit.env.get(table_a) == a {
                        row.push(table_a.clone());
                    } else {
                        row.push(Ty::inferred(b_kind, b_span));
                    }
                }

                let (ref mut index, ref mut table) = unit.env.the_matrix.entry(a_func).or_default();

                index.push(b_tid);
                table.push(row);
            }

            unit.env.substitutions.insert(b, b_with_func.clone());
            unify_ty_ty(unit, &a, &b_with_func, constraint_span)?;

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

                for (table_a, table_b) in a_row.into_iter().zip(b_row) {
                    if unit.env.get(&table_a) == a {
                        continue;
                    }

                    if unit.env.get(&table_b) == b {
                        continue;
                    }

                    unify_ty_ty(unit, &table_a, &table_b, constraint_span)?;
                }
            }

            unit.env.substitutions.insert(a, b);

            Ok(())
        }
    }
}

fn normalize(unit: &mut Unit, ty: &Ty, constraint_span: Span) -> Result<Ty, RetryOrError> {
    if let Some(ty) = unit.env.substitute(ty) {
        return normalize(unit, &ty.clone(), constraint_span);
    }

    match ty {
        Ty::Inferred(_, _, _, _) => Ok(ty.clone()),
        Ty::Partial(_, _, _) => Ok(ty.clone()),
        Ty::Field(adt, field, _) => {
            let normalized = normalize_field(unit, adt, field, constraint_span)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
        Ty::Tuple(base, index, _) => {
            let normalized = normalize_tuple(unit, base, *index, constraint_span)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
        Ty::Call(callee, arguments, _) => {
            let normalized = normalize_call(unit, callee, arguments, constraint_span)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
        Ty::Pipe(lhs, rhs, arguments, _) => {
            let normalized = normalize_pipe(unit, lhs, rhs, arguments, constraint_span)?;
            (unit.env.substitutions).insert(ty.clone(), normalized.clone());
            Ok(normalized)
        }
    }
}

fn with_func(unit: &mut Unit, ty: &Ty, new_func: usize) -> Ty {
    match ty {
        Ty::Inferred(_, kind, func, span) => {
            assert!(func.is_none() || *func == Some(new_func));

            let new_ty = Ty::Inferred(Tid::new(), *kind, Some(new_func), *span);
            unit.unify(ty.clone(), new_ty.clone(), *span);
            new_ty
        }
        Ty::Partial(part, arguments, span) => {
            let arguments = arguments
                .iter()
                .map(|arg| with_func(unit, arg, new_func))
                .collect();
            Ty::Partial(*part, arguments, *span)
        }
        Ty::Field(base, field, span) => {
            let base = with_func(unit, base, new_func);

            let ty = Ty::Field(Box::new(base), field, *span);
            unit.normalize(ty.clone(), *span);
            ty
        }
        Ty::Tuple(base, index, span) => {
            let base = with_func(unit, base, new_func);

            let ty = Ty::Tuple(Box::new(base), *index, *span);
            unit.normalize(ty.clone(), *span);
            ty
        }
        Ty::Call(callee, arguments, span) => {
            let callee = with_func(unit, callee, new_func);
            let arguments = arguments
                .iter()
                .map(|arg| arg.as_ref().map(|arg| with_func(unit, arg, new_func)))
                .collect();

            let ty = Ty::Call(Box::new(callee), arguments, *span);
            unit.normalize(ty.clone(), *span);
            ty
        }
        Ty::Pipe(lhs, rhs, arguments, span) => {
            let lhs = with_func(unit, lhs, new_func);
            let rhs = with_func(unit, rhs, new_func);
            let arguments = arguments
                .iter()
                .map(|arg| arg.as_ref().map(|arg| with_func(unit, arg, new_func)))
                .collect();

            let ty = Ty::Pipe(Box::new(lhs), Box::new(rhs), arguments, *span);
            unit.normalize(ty.clone(), *span);
            ty
        }
    }
}

fn normalize_field(
    unit: &mut Unit,
    adt: &Ty,
    field: &str,
    constraint_span: Span,
) -> Result<Ty, RetryOrError> {
    let adt = normalize(unit, adt, constraint_span)?;

    if let Ty::Partial(Part::Ref, args, _) = adt {
        return normalize_field(unit, &args[0], field, constraint_span);
    }

    let (index, generics) = match adt {
        Ty::Inferred(_, _, _, _) => return Err(RetryOrError::Retry),
        Ty::Partial(Part::Adt(index), generics, _) => (index, generics),
        Ty::Partial(_, _, _) => {
            return Err(miette::miette!(
                labels = [constraint_span.label("here")],
                "tried to get field of non ADT type"
            )
            .with_source_code(constraint_span)
            .into())
        }
        Ty::Field(_, _, _) | Ty::Tuple(_, _, _) | Ty::Call(_, _, _) | Ty::Pipe(_, _, _, _) => {
            unreachable!()
        }
    };

    let Some((_, ty)) = unit.adts[index].find_field(field) else {
        return Err(miette::miette!(
            labels = [constraint_span.label("here")],
            "field `{}` not found",
            field
        )
        .with_source_code(constraint_span)
        .into());
    };

    normalize(unit, &ty.specialize(&generics), constraint_span)
}

fn normalize_tuple(
    unit: &mut Unit,
    base: &Ty,
    index: usize,
    constraint_span: Span,
) -> Result<Ty, RetryOrError> {
    let base = normalize(unit, base, constraint_span)?;

    let (index, items) = match base {
        Ty::Inferred(_, _, _, _) => return Err(RetryOrError::Retry),
        Ty::Partial(Part::Tuple, items, _) => (index, items),
        Ty::Partial(_, _, _) => {
            return Err(miette::miette!(
                labels = [constraint_span.label("here")],
                "expected a tuple"
            )
            .with_source_code(constraint_span)
            .into())
        }
        Ty::Field(_, _, _) | Ty::Tuple(_, _, _) | Ty::Call(_, _, _) | Ty::Pipe(_, _, _, _) => {
            unreachable!()
        }
    };

    normalize(unit, &items[index], constraint_span)
}

fn normalize_call(
    unit: &mut Unit,
    callee: &Ty,
    arguments: &[Option<Ty>],
    constraint_span: Span,
) -> Result<Ty, RetryOrError> {
    let callee = normalize(unit, callee, constraint_span)?;

    let (mut callee_arguments, span) = match callee {
        Ty::Inferred(_, _, _, _) => return Err(RetryOrError::Retry),
        Ty::Partial(Part::Func, arguments, span) => (arguments, span),
        Ty::Partial(_, _, span) => {
            return Err(miette::miette!(
                labels = [span.label("here"), constraint_span.label("here")],
                "expected a function"
            )
            .with_source_code(span.source)
            .into())
        }
        Ty::Field(_, _, _) | Ty::Tuple(_, _, _) | Ty::Call(_, _, _) | Ty::Pipe(_, _, _, _) => {
            unreachable!()
        }
    };

    let output = callee_arguments
        .pop()
        .expect("funcs have least one argument");

    if arguments.len() > callee_arguments.len() {
        return Err(miette::miette!(
            labels = [span.label("here"), constraint_span.label("here")],
            "wrong number of arguments: expected {} but found {}",
            callee_arguments.len(),
            arguments.len()
        )
        .with_source_code(constraint_span)
        .into());
    }

    let mut remaining = Vec::new();

    for (a, b) in callee_arguments.iter().zip(arguments) {
        match b {
            Some(b) => unify_ty_ty(unit, a, b, constraint_span)?,
            None => remaining.push(a.clone()),
        }
    }

    remaining.extend(callee_arguments.iter().skip(arguments.len()).cloned());

    if remaining.is_empty() {
        return normalize(unit, &output, constraint_span);
    }

    remaining.push(output);

    let func = Ty::Partial(Part::Func, remaining, span);
    normalize(unit, &func, constraint_span)
}

fn normalize_pipe(
    unit: &mut Unit,
    lhs: &Ty,
    rhs: &Ty,
    arguments: &[Option<Ty>],
    constraint_span: Span,
) -> Result<Ty, RetryOrError> {
    let rhs = normalize(unit, rhs, constraint_span)?;

    let (mut rhs_arguments, span) = match rhs {
        Ty::Inferred(_, _, _, _) => return Err(RetryOrError::Retry),
        Ty::Partial(Part::Func, arguments, span) => (arguments, span),
        Ty::Partial(_, _, span) => {
            return Err(miette::miette!(
                labels = [span.label("here"), constraint_span.label("here")],
                "expected a function"
            )
            .with_source_code(span.source)
            .into())
        }
        Ty::Field(_, _, _) | Ty::Tuple(_, _, _) | Ty::Call(_, _, _) | Ty::Pipe(_, _, _, _) => {
            unreachable!()
        }
    };

    let output = rhs_arguments.pop().expect("funcs have least one argument");

    let has_slot = arguments.iter().any(Option::is_none);

    if arguments.len() + !has_slot as usize > rhs_arguments.len() {
        return Err(miette::miette!(
            labels = [span.label("here"), constraint_span.label("here")],
            "wrong number of arguments: expected at most {} but found {}",
            rhs_arguments.len() - 1,
            arguments.len()
        )
        .with_source_code(constraint_span)
        .into());
    }

    let missing = rhs_arguments.len() - arguments.len();

    let mut tys = Vec::new();

    tys.extend(rhs_arguments.iter().take(missing).cloned());

    for arg in rhs_arguments.iter().skip(missing).zip(arguments) {
        match arg {
            (a, Some(b)) => unify_ty_ty(unit, a, b, constraint_span)?,
            (a, None) => tys.push(a.clone()),
        }
    }

    match tys.len() {
        0 => panic!(),
        1 => {
            unify_ty_ty(unit, lhs, &tys[0], constraint_span)?;
            normalize(unit, &output, constraint_span)
        }
        _ => {
            let tuple = Ty::Partial(Part::Tuple, tys, span);
            unify_ty_ty(unit, lhs, &tuple, constraint_span)?;
            normalize(unit, &output, constraint_span)
        }
    }
}

fn part_eq(a_part: &Part, b_part: &Part) -> bool {
    match (a_part, b_part) {
        (Part::Generic(a, _), Part::Generic(b, _)) => a == b,
        _ => a_part == b_part,
    }
}

#[allow(clippy::too_many_arguments)]
fn unify_partial_partial(
    unit: &mut Unit,
    a_part: &Part,
    a_args: &[Ty],
    a_span: Span,
    b_part: &Part,
    b_args: &[Ty],
    b_span: Span,
    constraint_span: Span,
) -> Result<(), RetryOrError> {
    if !part_eq(a_part, b_part) || a_args.len() != b_args.len() {
        return Err(miette::miette!(
            labels = [
                a_span.label("here"),
                b_span.label("here"),
                constraint_span.label("here")
            ],
            "expected `{}` but found `{}`",
            Ty::format_partial(unit, a_part, a_args),
            Ty::format_partial(unit, b_part, b_args),
        )
        .with_source_code(a_span)
        .into());
    }

    for (a, b) in a_args.iter().zip(b_args) {
        unify_ty_ty(unit, a, b, constraint_span)?;
    }

    Ok(())
}
