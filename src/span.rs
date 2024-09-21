use std::ops::Range;

use miette::{LabeledSpan, MietteError, MietteSpanContents, SourceCode, SourceSpan, SpanContents};

#[derive(Clone, Copy)]
pub struct Span {
    pub lo: usize,
    pub hi: usize,
    pub file: &'static str,
    pub source: &'static str,
}

impl Span {
    pub fn range(self) -> Range<usize> {
        self.lo..self.hi
    }

    pub fn as_str(self) -> &'static str {
        &self.source[self.range()]
    }

    pub fn join(self, other: Span) -> Span {
        debug_assert_eq!(self.file, other.file);
        debug_assert_eq!(self.source, other.source);

        Span {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
            file: self.file,
            source: self.source,
        }
    }

    pub fn label(self, label: impl ToString) -> LabeledSpan {
        LabeledSpan::at(self, label.to_string())
    }
}

impl From<Span> for SourceSpan {
    fn from(span: Span) -> Self {
        SourceSpan::new(span.lo.into(), span.hi - span.lo)
    }
}

impl SourceCode for Span {
    fn read_span<'a>(
        &'a self,
        span: &SourceSpan,
        before: usize,
        after: usize,
    ) -> Result<Box<dyn SpanContents<'a> + 'a>, MietteError> {
        let inner_contents = self.source.read_span(span, before, after)?;

        Ok(Box::new(MietteSpanContents::new_named(
            self.file.to_string(),
            inner_contents.data(),
            *inner_contents.span(),
            inner_contents.line(),
            inner_contents.column(),
            inner_contents.line_count(),
        )))
    }
}

impl std::fmt::Debug for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}..{}", self.file, self.lo, self.hi)
    }
}
