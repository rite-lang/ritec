use ritec_diagnostic::Diagnostic;

use crate::{Forall, Partial, Projected, Solver, Unknown, Variable};

impl Solver {
    pub(crate) fn unify_var_var(&mut self, a: &Variable, b: &Variable) -> Result<bool, Diagnostic> {
        let a = self.world.substitute(a);
        let b = self.world.substitute(b);

        match (a, b) {
            (Variable::Unknown(a), b) | (b, Variable::Unknown(a)) => self.unify_unknown_var(&a, &b),
            (Variable::Partial(a), Variable::Partial(b)) => self.unify_partial_partial(&a, &b),
            (Variable::Projected(a), b) | (b, Variable::Projected(a)) => {
                self.unify_projected_var(&a, &b)
            }
            (Variable::Forall(a), b) | (b, Variable::Forall(a)) => self.unify_forall_var(&a, &b),
        }
    }

    fn unify_unknown_var(
        &mut self,
        unknown: &Unknown,
        other: &Variable,
    ) -> Result<bool, Diagnostic> {
        self.world.add_substitution(unknown.uid, other.clone());

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
        other: &Variable,
    ) -> Result<bool, Diagnostic> {
        match self.world.project(projected) {
            Ok(var) => self.unify_var_var(&var, other),
            Err(_) => Ok(false),
        }
    }

    fn unify_forall_var(&mut self, forall: &Forall, other: &Variable) -> Result<bool, Diagnostic> {
        Ok(true)
    }
}
