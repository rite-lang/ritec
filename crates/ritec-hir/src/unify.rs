use ritec_diagnostic::Diagnostic;

use crate::{Generic, Item, Partial, Projected, Type, Types, Unknown, UnknownKind};

impl Types {
    pub(crate) fn unify_var_var(&mut self, a: &Type, b: &Type) -> Result<bool, Diagnostic> {
        let a = self.substitute(a);
        let b = self.substitute(b);

        match (a, b) {
            (Type::Unknown(a), b) | (b, Type::Unknown(a)) => self.unify_unknown_var(&a, &b),
            (Type::Partial(a), Type::Partial(b)) => self.unify_partial_partial(&a, &b),
            (Type::Projected(a), b) | (b, Type::Projected(a)) => self.unify_projected_var(&a, &b),
            (Type::Generic(a), b) | (b, Type::Generic(a)) => self.unify_generic_var(&a, &b),
        }
    }

    fn unify_unknown_var(&mut self, unknown: &Unknown, other: &Type) -> Result<bool, Diagnostic> {
        match unknown.kind {
            UnknownKind::Any => {
                self.add_substitution(unknown.uid, other.clone());
            }
            UnknownKind::Number { float } => match other {
                Type::Unknown(other) => {
                    self.add_substitution(other.uid, Type::Unknown(unknown.clone()));
                }
                Type::Partial(partial) => match partial.item {
                    Item::Int { .. } if !float => {
                        self.add_substitution(unknown.uid, other.clone());
                    }
                    Item::Float { .. } => {
                        self.add_substitution(unknown.uid, other.clone());
                    }
                    _ => {
                        let diagnostic = Diagnostic::new("expected number");
                        return Err(diagnostic);
                    }
                },
                Type::Projected(projected) => {
                    if let Some(var) = self.project(projected, &Default::default())? {
                        self.unify_unknown_var(unknown, &var)?;
                    } else {
                        return Ok(false);
                    }
                }
                _ => {
                    let diagnostic = Diagnostic::new("expected number");
                    return Err(diagnostic);
                }
            },
        }

        Ok(true)
    }

    fn unify_partial_partial(&mut self, a: &Partial, b: &Partial) -> Result<bool, Diagnostic> {
        if a.item != b.item {
            let diagnostic = Diagnostic::new("type mismatch");
            return Err(diagnostic);
        }

        if a.params.len() != b.params.len() {
            let diagnostic = Diagnostic::new("generic count mismatch");
            return Err(diagnostic);
        }

        for (a, b) in a.params.iter().zip(b.params.iter()) {
            self.unify_var_var(a, b)?;
        }

        Ok(true)
    }

    fn unify_projected_var(
        &mut self,
        projected: &Projected,
        other: &Type,
    ) -> Result<bool, Diagnostic> {
        match self.project(projected, &Default::default()) {
            Ok(Some(var)) => self.unify_var_var(&var, other),
            Ok(None) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn unify_generic_var(&mut self, forall: &Generic, other: &Type) -> Result<bool, Diagnostic> {
        Ok(true)
    }
}
