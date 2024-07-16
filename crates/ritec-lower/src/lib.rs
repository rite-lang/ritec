mod body;
mod lower;
mod populate;
mod r#type;

use ritec_hir as hir;

#[derive(Clone, Debug, Default)]
pub struct Lowerer {
    pub unit: hir::Unit,
}
