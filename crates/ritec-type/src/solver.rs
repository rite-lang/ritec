use std::collections::VecDeque;

use crate::{TypeError, Variable, World};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Goal {
    Unify(Variable, Variable),
}

#[derive(Debug, Default)]
pub struct Solver {
    pub(crate) world: World,
    pub(crate) goals: VecDeque<Goal>,
}

impl Solver {
    pub fn new() -> Solver {
        Solver {
            world: World::new(),
            goals: VecDeque::new(),
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn push_goal(&mut self, goal: Goal) {
        self.goals.push_back(goal);
    }

    pub fn unify(&mut self, a: Variable, b: Variable) {
        self.push_goal(Goal::Unify(a, b));
    }

    fn solve_goal(&mut self, goal: Goal) -> Result<(), TypeError> {
        match goal {
            Goal::Unify(a, b) => self.unify_var_var(&a, &b),
        }
    }

    pub fn solve(&mut self) -> Result<(), TypeError> {
        while let Some(goal) = self.goals.pop_front() {
            self.solve_goal(goal)?;
        }

        Ok(())
    }
}
