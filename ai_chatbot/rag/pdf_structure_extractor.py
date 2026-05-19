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

    # Helper: detect very short decorative labels/callouts (e.g., "Thought !", "Note")
    def _looks_like_decorative_callout(heading_text: str, next_text: str) -> bool:
        if not heading_text:
            return False
        h = heading_text.strip()
        # short by words and characters
        words = re.findall(r"\w+", h)
        if len(words) > 4 or len(h) > 40:
            return False
        # avoid true section-like titles
        lower_h = re.sub(r"[^\w\s]", "", h).strip().lower()
        section_terms = {"introduction", "overview", "summary", "section", "lesson", "topic", "key", "principle"}
        if lower_h in section_terms:
            return False

        # next block should look like a full sentence/callout body
        if not next_text or len(next_text.strip()) < 40:
            return False
        nxt = next_text.strip()
        if not nxt[0].isupper():
            return False
        if not nxt.endswith(('.', '?', '!')):
            return False

        # patterns that are strong signals for callouts
        callout_keywords = {"thought", "note", "warning", "caution", "important", "tip", "aside"}
        if lower_h in callout_keywords:
            return True
        # exclamatory short headings are often callouts: "Thought !" or "Note!"
        if h.endswith('!'):
            return True
        # single/two-word short headings that are not common section titles
        if len(words) <= 2:
            return True
        return False

    def _looks_like_sentence_heading(heading_text: str, next_text: str) -> bool:
        """Detect long sentence-like lines that were split across two blocks.

        These are not true subsection headings. Heuristic is conservative:
        - heading must be fairly long (many words or chars)
        - not ALL CAPS
        - contains many stopwords or common verbs
        - next_text should not be a bullet or explicit list marker
        """
        if not heading_text:
            return False
        h = heading_text.strip()
        words = re.findall(r"\w+", h)
        if not words:
            return False
        # Short lines are likely real headings
        if len(words) < 6 and len(h) < 60:
            return False
        # ALL CAPS headings are usually real headings
        if h.isupper():
            return False

        nxt = (next_text or "").lstrip()
        # If next line is a bullet or numbered list, this heading is likely real
        if nxt.startswith(('•', '-', '*')) or re.match(r'^\d+[\.)]', nxt):
            return False

        # stopword-heavy lines are sentence-like
        stopwords = {
            'the','a','an','and','or','but','if','then','so','because','that','which','who',
            'of','in','on','for','with','by','to','from','as','about','before','after',
            'is','are','was','were','be','have','has','do','does','did','will','would','should',
        }
        stop_count = sum(1 for w in words if w.lower() in stopwords)
        if float(stop_count) / float(len(words)) >= 0.35:
            return True

        # presence of verbs or modal verbs suggests a sentence/statement.
        # Use whole-word matching to avoid matching gerunds (e.g., 'understanding').
        verbs = ('is','are','should','must','may','can','could','would','identify','explain','define','apply','recognise','recognize','understand','protect','reduce','notice')
        first_word = words[0].lower()
        # If the heading starts with a gerund (e.g., 'Understanding ...') it's likely a heading,
        # so skip verb detection in that case. Otherwise require whole-word match.
        if not first_word.endswith('ing'):
            if any(re.search(r"\b" + re.escape(v) + r"\b", h.lower()) for v in verbs):
                return True

        return False

    i = 0
    while i < len(raw_blocks):
        b = raw_blocks[i]
        text = b['text']
        
        # Rule 2: Detect Module ("SECTION X:")
        # Strict casing to avoid matching inline text like "Section 3 of the Criminal Law"
        section_match = re.match(r'^SECTION\s+(\d+)(?:[\s:-]+.*)?$', text)
        is_module = bool(section_match and len(text) < 100)
        
        if is_module:
            section_number = section_match.group(1)
            module_name = text
            # Smart join: If next line is a prominent ALL CAPS title (like "LAW AND LEGISLATION"), merge it
            if i + 1 < len(raw_blocks):
                next_b = raw_blocks[i+1]
                if next_b['text'].isupper() and len(next_b['text']) < 60:
                    module_name = next_b['text']
                    i += 1 # Consume that line so it doesn't become a subsection

            # Commit the previous module
            if current_module:
                if current_subsection and current_subsection['content'].strip():
                    current_module['subsections'].append(current_subsection)
                structure.append(current_module)
            
            current_module = {
                'module': module_name,
                'section_number': section_number,
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
        
        # Demotion: short decorative callouts (e.g., "Thought !") should not become subsections
        if is_subsection:
            next_text = raw_blocks[i+1]['text'] if (i + 1) < len(raw_blocks) else ''
            if _looks_like_decorative_callout(text, next_text) or _looks_like_sentence_heading(text, next_text):
                is_subsection = False

        # Demote headings that end with explicit small callout words like 'Thought'
        if is_subsection:
            if re.search(r"\b(thought|note|warning|tip|aside|important)\b$", text, re.I):
                is_subsection = False

        # Demote when next line is lone punctuation (e.g., '!') and the following
        # line completes a sentence — this pattern indicates the punctuation and
        # first fragment were a callout, not a real heading.
        if is_subsection and (i + 2) < len(raw_blocks):
            nxt = raw_blocks[i+1]['text'].strip()
            nxt2 = raw_blocks[i+2]['text'].strip()
            if re.match(r'^[^\w\s]{1,2}$', nxt) and len(nxt2) > 20 and nxt2[0].isalpha():
                # e.g., heading_line, '!', 'A long sentence...'
                is_subsection = False

        # Additional conservative demotion: internal punctuation/verbs in a long
        # candidate heading (but only if next line is not a bullet/list).
        if is_subsection:
            next_text_peek = raw_blocks[i+1]['text'] if (i + 1) < len(raw_blocks) else ''
            next_is_list = next_text_peek.lstrip().startswith(('•', '-', '*')) or re.match(r'^\d+[\.)]', next_text_peek.lstrip())
            if not next_is_list:
                # internal colon/semicolon that is not an end-of-heading
                if (':' in text or ';' in text) and not text.strip().endswith(':') and len(text) > 40:
                    is_subsection = False
                # presence of common copula/modal verbs in long lines
                if re.search(r"\b(is|are|was|were|be|has|have|do|does|did|should|must|may|can|could)\b", text, re.I) and len(text) > 40:
                    is_subsection = False

        # Additional demotion: if the following line starts with a lowercase letter
        # and the current line is long (likely a wrapped sentence), do not treat
        # the current line as a subsection heading.
        if is_subsection and (i + 1) < len(raw_blocks):
            nxt = raw_blocks[i+1]['text'].lstrip()
            m = re.search(r'[A-Za-z]', nxt)
            if m:
                first_alpha = m.group(0)
                # If next line begins with lowercase and current line doesn't end
                # with explicit heading punctuation like ':' then it's likely a
                # sentence wrap rather than a heading.
                if first_alpha.islower() and not text.strip().endswith(':') and len(text) > 40:
                    is_subsection = False
            
        # Refinement: If we just created a heading and it has no content yet, 
        # do not allow consecutive headings. Demote this one to content.
        if is_subsection and current_subsection and not current_subsection['content'].strip():
            is_subsection = False

        if is_subsection:
            # Demotion checks already applied above. Commit the previous subsection
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
                section_name = f"Section {current_module.get('section_number', '')}" if current_module and 'section_number' in current_module else 'Introduction'
                current_subsection = {
                    'name': section_name.strip(),
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
