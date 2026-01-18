
//---------------------------------------Paragraph splitter------------
pub struct Paragraph<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> Paragraph<'a> {
    pub fn new(text: &'a str) -> Self {
        Paragraph { text, position: 0 }
    }
}

impl<'a> Iterator for Paragraph<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.text.len() {
            return None;
        }

        let remaining_text = &self.text[self.position..];
        let next_paragraph_end = remaining_text.find("\n\n").unwrap_or(remaining_text.len());

        let paragraph = &remaining_text[..next_paragraph_end].trim();
        self.position += next_paragraph_end + 2;

        if paragraph.is_empty() {
            self.next()
        } else {
            Some(paragraph)
        }
    }
}

