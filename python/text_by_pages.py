from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import open_filename


def iter_text_per_page(pdf_file,  page_numbers=None,
                      codec='utf-8', laparams=None):
    if laparams is None:
        laparams = LAParams()

    with open_filename(pdf_file, "rb") as fp:
        rsrcmgr = PDFResourceManager(caching=caching)

        for page in PDFPage.get_pages(
                fp,
                page_numbers,
                maxpages=0,
                password='',
                caching=True,
        ):
            with StringIO() as output_string:
                device = TextConverter(rsrcmgr, output_string, codec=codec,
                                       laparams=laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                interpreter.process_page(page)
                yield output_string.getvalue()
   

