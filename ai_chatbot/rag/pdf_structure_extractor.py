import fitz  # PyMuPDF
import re
from typing import List, Dict
from collections import Counter

def extract_pdf_structure(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    
    # ---------------------------------------------------------
    # PASS 1: Identify Repeating Headers & Footers across pages
    # ---------------------------------------------------------
    line_frequencies = Counter()
    for page in doc:
        page_lines = set() # Count each line once per page
        for block in page.get_text("dict").get('blocks', []):
            if block['type'] == 0:
                for line in block.get('lines', []):
                    text = "".join([span['text'] for span in line['spans']]).strip()
                    if text:
                        page_lines.add(text)
        for text in page_lines:
            line_frequencies[text] += 1
            
    total_pages = len(doc)
    repeating_threshold = max(3, total_pages * 0.3) # If a line appears on >30% pages, it's a header/footer

    # ---------------------------------------------------------
    # PASS 2: Extract block text, strip noise completely
    # ---------------------------------------------------------
    raw_blocks = []
    for page in doc:
        page_dict = page.get_text("dict")
        page_height = page_dict.get("height", 0)
        
        for block in page_dict.get('blocks', []):
            if block['type'] == 0:  # Text block
                bbox = block['bbox']
                y0, y1 = bbox[1], bbox[3]
                
                for line in block.get('lines', []):
                    line_text = "".join([span['text'] for span in line['spans']]).strip()
                    
                    if not line_text:
                        continue
                        
                    # Ignore TOC lines with dotted leaders
                    if re.search(r'\.{4,}', line_text):
                        continue
                    
                    # Ignore Explicit Footers / Copyright (Catch "© Guardian Training...")
                    if '©' in line_text or "copyright" in line_text.lower() or "all rights reserved" in line_text.lower():
                        continue
                        
                    # Ignore Repeated Headers / Footers identified in Pass 1
                    # Only apply to top 15% or bottom 15% of page margins
                    is_margin = (y0 < page_height * 0.15) or (y1 > page_height * 0.85)
                    if is_margin and line_frequencies[line_text] >= repeating_threshold:
                        continue
                        
                    # Ignore plain page numbers in margins
                    if re.match(r'^\d+$', line_text) and is_margin:
                        continue
                    
                    # Store block typography info
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

    if not raw_blocks:
        return []

    # Get body font size to detect headings properly
    font_size_counts = Counter(b['font_size'] for b in raw_blocks)
    body_font_size = font_size_counts.most_common(1)[0][0]

    # ---------------------------------------------------------
    # PASS 3: Structure Generation and Hierarchy Building
    # ---------------------------------------------------------
    structure = []
    current_module = None
    current_subsection = None

    i = 0
    while i < len(raw_blocks):
        b = raw_blocks[i]
        text = b['text']
        
        # Rule 2: Detect Module ("SECTION X:")
        # Strict casing to avoid matching inline text like "Section 3 of the Criminal Law"
        section_match = re.match(r'^SECTION\s+\d+(?:[\s:-]+.*)?$', text)
        is_module = bool(section_match and len(text) < 100)
        
        if is_module:
            module_name = text
            # Smart join: If next line is a prominent ALL CAPS title (like "LAW AND LEGISLATION"), merge it
            if i + 1 < len(raw_blocks):
                next_b = raw_blocks[i+1]
                if next_b['text'].isupper() and len(next_b['text']) < 60:
                    module_name += f" - {next_b['text']}"
                    i += 1 # Consume that line so it doesn't become a subsection

            # Commit the previous module
            if current_module:
                if current_subsection and current_subsection['content'].strip():
                    current_module['subsections'].append(current_subsection)
                structure.append(current_module)
            
            current_module = {
                'module': module_name,
                'subsections': []
            }
            current_subsection = None
            i += 1
            continue

        # Prevent mapping orphan intro text if no module has started yet
        if not current_module:
            i += 1
            continue

        # Handle Inline Subsections
        # "Key principle: Close protection operatives..."
        inline_match = re.match(r'^((?:[A-Z][a-zA-Z\s]+?){1,6}):\s+([A-Z].*)$', text)
        if inline_match:
            heading_part, content_part = inline_match.groups()
            
            # Commit the previous subsection
            if current_subsection:
                if current_subsection['content'].strip():
                    current_module['subsections'].append(current_subsection)
            
            current_subsection = {
                'name': heading_part.strip(),
                'content': content_part
            }
            i += 1
            continue

        # Rule 3: Detect Subsections
        is_subsection = False
        ends_with_sentence_punctuation = text.endswith(('.', '?', '!', ';', ','))

        is_larger_font = b['font_size'] > body_font_size
        
        # Criterion 1: Bold or Larger Font than body, and does not end with sentence punctuation
        if (b['bold'] or is_larger_font) and not ends_with_sentence_punctuation:
            if len(text) < 120:  # Exclude massive bold paragraphs
                is_subsection = True
            
        # Criterion 2: ALL CAPS Titles
        elif text.isupper() and len(text) > 4 and not ends_with_sentence_punctuation:
            if len(text) < 120:
                is_subsection = True
            
        # Criterion 3: Colon headings ("Topic:", "Health and Safety:")
        elif text.endswith(':'):
            # Limit colon headings to label lengths, skipping full descriptive sentences ending in colon
            if len(text) < 45:
                is_subsection = True
            
        if is_subsection:
            # Commit the previous subsection
            if current_subsection:
                if current_subsection['content'].strip():
                    current_module['subsections'].append(current_subsection)
            
            current_subsection = {
                'name': text,
                'content': ''
            }
        else:
            # Add to content
            if not current_subsection:
                current_subsection = {
                    'name': 'Introduction',
                    'content': ''
                }
            
            if current_subsection['content']:
                prev_text = current_subsection['content'].strip()
                last_char = prev_text[-1] if prev_text else ''
                
                # Table/List cell detection:
                looks_like_cell = len(text) < 60 and not ends_with_sentence_punctuation and not text.startswith(('and ', 'but ', 'or ', 'to ', 'which '))

                if current_subsection['content'].endswith('-'):
                    current_subsection['content'] = current_subsection['content'][:-1] + text
                elif text.startswith(('•', '-', 'o ')) or re.match(r'^\d+\.', text):
                    # Explicit structured list or bullet leads to a newline
                    current_subsection['content'] += '\n' + text
                elif last_char in ['.', '!', '?', ':']:
                    # Proper end of a grammatical paragraph, add newline
                    current_subsection['content'] += '\n' + text
                elif looks_like_cell:
                    # Likely a table row or a list item without bullets, push to new line for clean preservation
                    current_subsection['content'] += '\n' + text
                else:
                    # Continues inside the same flowing paragraph, concatenate with a space
                    current_subsection['content'] += ' ' + text
            else:
                current_subsection['content'] = text
                
        i += 1

    # Finalize last dangling items
    if current_module:
        if current_subsection and current_subsection['content'].strip():
            current_module['subsections'].append(current_subsection)
        structure.append(current_module)

    print(f"[PDF Structure] Extracted {len(structure)} modules.")
    for m in structure:
        print(f"  Module: {m['module']} ({len(m['subsections'])} subsections)")
    return structure
