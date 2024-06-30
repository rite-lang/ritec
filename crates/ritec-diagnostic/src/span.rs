#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub lo: usize,
    pub hi: usize,
}

impl Span {
    pub fn new(lo: usize, hi: usize) -> Span {
        Span { lo, hi }
    }

    pub fn empty() -> Span {
        Span { lo: 0, hi: 0 }
    }

    pub fn merge(self: &Self, other: &Self) -> Span {
        Span {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    pub fn len(self: &Self) -> usize {
        self.hi - self.lo
    }
}
