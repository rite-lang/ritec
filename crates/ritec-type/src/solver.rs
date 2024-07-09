use std::collections::VecDeque;

use ritec_diagnostic::Diagnostic;

use crate::{Variable, WhereId, World};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Goal {
    Unify(WhereId, Variable, Variable),
}

#[derive(Debug, Default)]
pub struct Solver {
    pub world: World,
    pub(crate) goals: VecDeque<Goal>,
}

impl Solver {
    pub fn new() -> Solver {
        Solver {
            world: World::new(),
            goals: VecDeque::new(),
        }
    }

    pub fn push_goal(&mut self, goal: Goal) {
        self.goals.push_back(goal);
    }

    pub fn unify(&mut self, where_id: WhereId, a: Variable, b: Variable) {
        self.push_goal(Goal::Unify(where_id, a, b));
    }

    fn solve_goal(&mut self, goal: &Goal) -> Result<bool, Diagnostic> {
        match goal {
            Goal::Unify(_, a, b) => self.unify_var_var(a, b),
        }
    }

    pub fn solve(&mut self) -> Result<(), Diagnostic> {
        let mut fail_count = 0;

        while let Some(goal) = self.goals.pop_front() {
            if self.solve_goal(&goal)? {
                fail_count = 0;
                continue;
            }

            fail_count += 1;
            self.goals.push_back(goal.clone());

            if fail_count > self.goals.len() + 10 {
                let diagnostic = Diagnostic::new("failed to solve goal");
                return Err(diagnostic);
            }
        }

        Ok(())
    }
}
