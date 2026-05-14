import fitz
import sys
from collections import Counter
import json

def dump_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    # just dump page 2 or so. Let's find "Communication skills"
    raw_blocks = []
    
    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict.get('blocks', []):
            if block['type'] == 0:  # Text block
                for line in block.get('lines', []):
                    line_text = "".join([span['text'] for span in line['spans']]).strip()
                    if not line_text: continue
                    font_sizes = [span['size'] for span in line['spans']]
                    font_size = max(font_sizes) if font_sizes else 12
                    is_bold = any(
                        (span.get('flags', 0) & 16) != 0 or 
                        any(x in span['font'].lower() for x in ['bold', 'black', 'heavy'])
                        for span in line['spans']
                    )
                    raw_blocks.append({
                        'text': line_text,
                        'font_size': font_size,
                        'bold': is_bold
                    })
    
    with open("pdf_blocks.json", "w", encoding="utf-8") as f:
        json.dump(raw_blocks, f, indent=2)

if __name__ == "__main__":
    pdf_path = "media/" + sys.argv[1] # wait, I don't know the file name
    # Let me just find any pdf in media/
