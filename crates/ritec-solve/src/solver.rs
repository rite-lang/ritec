use crate::World;

#[derive(Debug, Default)]
pub struct Solver {
    world: World,
}

impl Solver {
    pub fn new() -> Solver {
        Solver {
            world: World::new(),
        }
    }
}
