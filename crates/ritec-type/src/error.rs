use ritec_diagnostic::Diagnostic;

#[derive(Debug)]
pub struct TypeError {
    diagnostic: Diagnostic,
}

impl From<Diagnostic> for TypeError {
    fn from(diagnostic: Diagnostic) -> Self {
        Self { diagnostic }
    }
}
