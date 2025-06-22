import os
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from collections import defaultdict
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Heuristic keywords to identify BOM tables
BOM_KEYWORDS = ["item", "description", "qty", "quantity", "size", "material"]

def extract_bom_tables_from_pdf(pdf_path):
    bom_entries = []
    found_any_table = False
    poppler_path = r"C:\\poppler-24.08.0\\Library\\bin"  # Use double backslashes for Windows paths
    tesseract_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Path to tesseract executable
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if not tables:
                print(f"No tables found on page {page_num+1}.")
            else:
                found_any_table = True
                print(f"Tables found on page {page_num+1}:")
                for idx, table in enumerate(tables):
                    print(f"  Table {idx+1} (first 3 rows):")
                    for row in table[:3]:
                        print(f"    {row}")
            for table in tables:
                if is_bom_table(table):
                    bom_entries.extend(parse_bom_table(table, page_num+1))
    if not found_any_table:
        print("\nNo tables detected by pdfplumber. Trying OCR on each page...\n")
        try:
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
        except Exception as e:
            print("[ERROR] Could not convert PDF to images for OCR.\nReason:", e)
            print(f"\nIf you see a message about 'poppler', please check the poppler_path: {poppler_path}\nDownload from: https://github.com/oschwartz10612/poppler-windows/releases/")
            return bom_entries
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            print(f"--- OCR text from page {i+1} (first 20 lines) ---")
            lines = text.splitlines()
            for line in lines[:20]:
                print(line)
            print("...")
    return bom_entries

def is_bom_table(table):
    if not table or not table[0]:
        return False
    header = [str(cell).strip().lower() for cell in table[0] if cell]
    matches = sum(any(kw in h for kw in BOM_KEYWORDS) for h in header)
    return matches >= 2  # At least 2 BOM keywords in header

def parse_bom_table(table, page_num):
    header = [str(cell).strip().lower() for cell in table[0]]
    items = []
    for row in table[1:]:
        if any(row):
            item = {header[i]: row[i] for i in range(min(len(header), len(row)))}
            item['page'] = page_num
            items.append(item)
    return items

def aggregate_boms(bom_entries):
    agg = defaultdict(lambda: defaultdict(int))
    details = {}
    for entry in bom_entries:
        key = (entry.get('description','').strip().lower(), entry.get('size','').strip().lower())
        qty = entry.get('qty') or entry.get('quantity') or 1
        try:
            qty = int(str(qty).split()[0])
        except Exception:
            qty = 1
        agg[key]['qty'] += qty
        details[key] = entry
    result = []
    for key, val in agg.items():
        entry = details[key].copy()
        entry['total_qty'] = val['qty']
        result.append(entry)
    return result

def export_to_excel(bom_list, output_path):
    df = pd.DataFrame(bom_list)
    df.to_excel(output_path, index=False)

def strip_quotes(s):
    if s and ((s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'"))):
        return s[1:-1]
    return s

def main():
    pdf_path = strip_quotes(input("Enter path to PDF file: ").strip())
    output_path = strip_quotes(input("Enter output Excel file path: ").strip())
    bom_entries = extract_bom_tables_from_pdf(pdf_path)
    if not bom_entries:
        print("No BOM tables found.")
        return
    bom_list = aggregate_boms(bom_entries)
    export_to_excel(bom_list, output_path)
    print(f"Extracted BOMs exported to {output_path}")

if __name__ == "__main__":
    main()
