use ritec_diagnostic::Diagnostic;

use crate::{Partial, Solver, TypeError, Unknown, Variable};

impl Solver {
    pub(crate) fn unify_var_var(&mut self, a: &Variable, b: &Variable) -> Result<(), TypeError> {
        let a = self.world.substitute(a);
        let b = self.world.substitute(b);

        match (a, b) {
            (Variable::Unknown(a), b) | (b, Variable::Unknown(a)) => {
                self.unify_unknown_var(&a, &b)?;
            }
            (Variable::Partial(a), Variable::Partial(b)) => {
                self.unify_partial_partial(&a, &b)?;
            }
            _ => todo!(),
        }

        Ok(())
    }

    fn unify_unknown_var(&mut self, unknown: &Unknown, other: &Variable) -> Result<(), TypeError> {
        let lhs = Variable::Unknown(unknown.clone());

        // ensure that we don't substitute a variable with itself
        if lhs == *other {
            return Ok(());
        }

        self.world.add_substitution(lhs, other.clone());

        Ok(())
    }

    fn unify_partial_partial(&mut self, a: &Partial, b: &Partial) -> Result<(), TypeError> {
        if a.item != b.item {
            let diagnostic = Diagnostic::new("type mismatch");
            return Err(TypeError::from(diagnostic));
        }

        if a.params.len() != b.params.len() {
            let diagnostic = Diagnostic::new("generic count mismatch");
            return Err(TypeError::from(diagnostic));
        }

        for (a, b) in a.params.iter().zip(b.params.iter()) {
            self.unify_var_var(a, b)?;
        }

        Ok(())
    }
}
