# BOMs Extraction Tool

A portable Python GUI tool for extracting structured Bill of Materials (BOM) data from piping isometric PDF drawings. Supports both rules-based and ML/AI-based extraction, region selection, column mapping, and output to Excel.

## Features
- Tkinter GUI for PDF region selection and extraction workflow
- Multiple extraction backends:
  - Rules-based (regex, user-guided column mapping)
  - ML/AI-based (Hugging Face NER models, ONNX/YOLO object detection)
  - Nanonets OCR (transformer-based OCR, pytesseract fallback)
- Excel-like preview and editing of column mapping
- Output to Excel (.xlsx)

## Quick Start
1. **Clone the repository:**
   ```sh
   git clone https://github.com/DucklingGod/BOMs_Extractor.git
   cd BOMs_Extractor
   ```
2. **Install dependencies:**
   - Python 3.8+
   - Install required packages:
     ```sh
     pip install -r requirements.txt
     ```
   - Install Poppler for PDF rendering (set `POPLER_PATH` in `select_bom_region.py`)
3. **Run the tool:**
   ```sh
   python select_bom_region.py
   ```

## Main Files
- `select_bom_region.py`: Main GUI and extraction logic
- `nanonets_ocr_integration.py`: ML/AI extraction helpers
- `train_bom_ner.py`: Fine-tune NER models
- `generate_bom_data.py`: Synthetic BOM data generator
- `bom_labeling_template.conll`: CoNLL-style labeling template
- `.gitignore`: Excludes models, outputs, and environment files
- `PROJECT_OVERVIEW.md`: In-depth project and developer notes

## Extraction Workflows
- **Rules-based:** Select region, provide example BOM line, map columns, extract
- **ML/AI-based:** Use fine-tuned NER models for field extraction
- **Object Detection:** Detect BOM rows as objects, crop, OCR
- **Nanonets OCR:** Transformer-based OCR or pytesseract fallback

## Model Directories (excluded from git)
- `bom_ner_model/`, `bom_ner_model_roberta/`

## License
MIT License

---
For more details, see `PROJECT_OVERVIEW.md`.
