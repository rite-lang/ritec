#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub lo: usize,
    pub hi: usize,
}
