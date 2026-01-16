"""
MultiHeirtt Corpus Preprocessing Script

Chuyển đổi corpus MultiHeirtt sang format dễ hiểu hơn cho embedding models.

Usage:
    python preprocess_multiheirtt.py --mode linearized
    python preprocess_multiheirtt.py --mode augmented
    python preprocess_multiheirtt.py --mode row_chunks
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def detect_markdown_tables(text: str) -> List[Tuple[str, int, int]]:
    """
    Detect markdown tables in text.
    Returns: List of (table_text, start_line, end_line)
    """
    tables = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if '|' in line and line.count('|') >= 2:
            table_start = i
            table_lines = []
            
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            
            if len(table_lines) >= 2:
                table_text = '\n'.join(table_lines)
                tables.append((table_text, table_start, i))
            continue
        
        i += 1
    
    return tables


def parse_markdown_table(table_text: str) -> Optional[Dict]:
    """Parse markdown table into structured format."""
    lines = [l.strip() for l in table_text.split('\n') if l.strip()]
    
    if len(lines) < 2:
        return None
    
    rows = []
    for line in lines:
        if re.match(r'^\|?\s*[-:]+\s*\|', line):
            continue
        
        cells = [c.strip() for c in line.split('|')]
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        
        if cells:
            rows.append(cells)
    
    if not rows:
        return None
    
    header_rows = []
    data_rows = []
    
    for i, row in enumerate(rows):
        has_currency = any(re.search(r'[\$£€¥]', c) for c in row)
        has_numbers = any(re.search(r'\d{2,}', c) for c in row)
        is_year_row = any(re.match(r'^(19|20)\d{2}$', c.strip()) for c in row)
        is_unit_row = any('million' in c.lower() or 'thousand' in c.lower() for c in row)
        
        is_header = (
            i < 3 and 
            (is_year_row or is_unit_row or not (has_currency and has_numbers))
        )
        
        if is_header:
            header_rows.append(row)
        else:
            data_rows.append(row)
    
    if not header_rows and rows:
        header_rows = [rows[0]]
        data_rows = rows[1:]
    
    return {
        'headers': header_rows,
        'data': data_rows,
        'num_cols': len(rows[0]) if rows else 0,
        'num_rows': len(data_rows)
    }


def build_column_headers(header_rows: List[List[str]], num_cols: int) -> List[str]:
    """Build column headers from multiple header rows."""
    col_headers = ['' for _ in range(num_cols)]
    
    for row in header_rows:
        for i, cell in enumerate(row):
            if i < num_cols and cell.strip():
                if col_headers[i]:
                    col_headers[i] += ' - ' + cell.strip()
                else:
                    col_headers[i] = cell.strip()
    
    cleaned = []
    for h in col_headers:
        if re.match(r'^\s*\(.*\)\s*$', h):
            cleaned.append('')
        else:
            cleaned.append(h)
    
    return cleaned


def linearize_table(parsed: Dict) -> str:
    """Convert parsed table to natural language sentences."""
    if not parsed or not parsed['data']:
        return ""
    
    headers = parsed['headers']
    data = parsed['data']
    num_cols = parsed['num_cols']
    
    col_headers = build_column_headers(headers, num_cols)
    
    sentences = []
    
    for row in data:
        if not row:
            continue
        
        row_label = row[0].strip() if row else ""
        values = []
        
        for i, cell in enumerate(row[1:], 1):
            if i < len(col_headers) and col_headers[i] and cell.strip():
                values.append(f"{col_headers[i]}={cell.strip()}")
            elif cell.strip():
                values.append(cell.strip())
        
        if row_label and values:
            sentence = f"{row_label}: {', '.join(values)}"
            sentences.append(sentence)
    
    return ". ".join(sentences)


def linearize_row(row: List[str], col_headers: List[str]) -> str:
    """Convert a single table row to natural language."""
    if not row:
        return ""
    
    row_label = row[0].strip()
    values = []
    
    for i, cell in enumerate(row[1:], 1):
        if i < len(col_headers) and col_headers[i] and cell.strip():
            values.append(f"{col_headers[i]} = {cell.strip()}")
        elif cell.strip():
            values.append(cell.strip())
    
    if row_label and values:
        return f"{row_label}: {', '.join(values)}"
    return ""


def preprocess_document(doc_id: str, doc: Dict, mode: str = 'linearized') -> List[Dict]:
    """
    Preprocess a single document.
    
    Modes:
    - 'linearized': Replace tables with linearized text
    - 'augmented': Keep original + add linearized versions
    - 'row_chunks': Create separate documents for each table row
    """
    text = doc['text']
    title = doc.get('title', '')
    
    tables = detect_markdown_tables(text)
    
    if not tables:
        return [{
            '_id': doc_id,
            'title': title,
            'text': text,
            'type': 'text_only'
        }]
    
    results = []
    
    if mode == 'linearized':
        lines = text.split('\n')
        new_lines = lines.copy()
        
        # Process in reverse order to maintain line indices
        for table_text, start, end in reversed(tables):
            parsed = parse_markdown_table(table_text)
            if parsed:
                linearized = linearize_table(parsed)
                new_lines[start:end] = [linearized]
        
        results.append({
            '_id': doc_id,
            'title': title,
            'text': '\n'.join(new_lines),
            'type': 'linearized'
        })
    
    elif mode == 'augmented':
        results.append({
            '_id': doc_id,
            'title': title,
            'text': text,
            'type': 'original'
        })
        
        for i, (table_text, start, end) in enumerate(tables):
            parsed = parse_markdown_table(table_text)
            if parsed:
                linearized = linearize_table(parsed)
                if linearized:
                    results.append({
                        '_id': f"{doc_id}_table_{i}",
                        'title': title,
                        'text': linearized,
                        'type': 'linearized_table',
                        'parent_id': doc_id
                    })
    
    elif mode == 'row_chunks':
        results.append({
            '_id': doc_id,
            'title': title,
            'text': text,
            'type': 'original'
        })
        
        for i, (table_text, start, end) in enumerate(tables):
            parsed = parse_markdown_table(table_text)
            if parsed:
                col_headers = build_column_headers(
                    parsed['headers'], 
                    parsed['num_cols']
                )
                
                for j, row in enumerate(parsed['data']):
                    row_text = linearize_row(row, col_headers)
                    if row_text:
                        results.append({
                            '_id': f"{doc_id}_table_{i}_row_{j}",
                            'title': title,
                            'text': row_text,
                            'type': 'table_row',
                            'parent_id': doc_id
                        })
    
    return results


def load_corpus(filepath: Path) -> Dict[str, Dict]:
    """Load corpus from JSONL file."""
    corpus = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            corpus[data['_id']] = {
                'title': data.get('title', ''),
                'text': data.get('text', '')
            }
    return corpus


def preprocess_corpus(corpus: Dict, mode: str = 'linearized') -> List[Dict]:
    """Preprocess entire corpus."""
    all_docs = []
    stats = defaultdict(int)
    
    total = len(corpus)
    for i, (doc_id, doc) in enumerate(corpus.items()):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{total} documents...")
        
        results = preprocess_document(doc_id, doc, mode=mode)
        all_docs.extend(results)
        
        for r in results:
            stats[r['type']] += 1
    
    print(f"\n✅ Processed {len(corpus)} documents → {len(all_docs)} output documents")
    print(f"\n📊 Document types:")
    for doc_type, count in sorted(stats.items()):
        print(f"   {doc_type}: {count}")
    
    return all_docs


def save_corpus(docs: List[Dict], output_file: Path):
    """Save corpus to JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(f"\n✅ Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess MultiHeirtt corpus')
    parser.add_argument('--mode', type=str, default='linearized',
                        choices=['linearized', 'augmented', 'row_chunks'],
                        help='Preprocessing mode')
    parser.add_argument('--input', type=str, 
                        default='../../data/multiheirtt_corpus.jsonl/corpus.jsonl',
                        help='Input corpus file')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"📂 Input: {input_file}")
    print(f"📂 Output dir: {output_dir}")
    print(f"🔧 Mode: {args.mode}")
    print("=" * 60)
    
    # Load corpus
    print("\nLoading corpus...")
    corpus = load_corpus(input_file)
    print(f"✅ Loaded {len(corpus)} documents")
    
    # Process
    print(f"\nProcessing with mode '{args.mode}'...")
    processed = preprocess_corpus(corpus, mode=args.mode)
    
    # Save
    output_file = output_dir / f"multiheirtt_corpus_{args.mode}.jsonl"
    save_corpus(processed, output_file)
    
    print("\n" + "=" * 60)
    print("✅ Done!")


if __name__ == '__main__':
    main()
