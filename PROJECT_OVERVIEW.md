# BOMs Extraction Project - Project Overview and Developer Notes

## Project Purpose
This project provides a portable Python GUI tool for extracting structured Bill of Materials (BOM) data from piping isometric PDF drawings. It supports both rules-based and ML/AI-based extraction, region selection, column mapping, and output to Excel. The tool is designed for engineering workflows where BOMs are embedded in technical drawings and need to be digitized for further processing.

## Key Features
- **Tkinter GUI** for PDF region selection, page navigation, and extraction workflow.
- **Multiple Extraction Backends:**
  - Rules-based (regex, user-guided column mapping)
  - ML/AI-based (Hugging Face NER models, ONNX/YOLO object detection)
  - Nanonets OCR (transformer-based OCR, pytesseract fallback)
- **Excel-like preview and editing** of column mapping before extraction.
- **Output to Excel (.xlsx)** for easy integration with downstream systems.
- **Portable and open-source:** Designed for easy sharing and reproducibility (see `.gitignore`).

## Main Files
- `select_bom_region.py`: Main GUI and extraction logic. Handles PDF loading, region selection, extraction menu, and workflow.
- `nanonets_ocr_integration.py`: Helper functions for ML/AI extraction, including Hugging Face NER and transformer-based OCR.
- `train_bom_ner.py`: Script for fine-tuning NER models on BOM data.
- `generate_bom_data.py`: Synthetic BOM data generator for robust model training.
- `bom_labeling_template.conll`: CoNLL-style labeling template for NER model training.
- `.gitignore`: Excludes models, outputs, and environment files from version control.

## Extraction Workflows
- **Rules-based:** User selects region, provides example BOM line, and maps columns. Regex is inferred and applied to OCR output.
- **ML/AI-based:** Uses fine-tuned NER models to extract fields from OCR text.
- **Object Detection (ONNX/YOLO):** Detects BOM rows as objects, crops, and runs OCR on each.
- **Nanonets OCR:** Uses transformer-based OCR model or pytesseract fallback for text extraction.

## Model Directories
- `bom_ner_model/`, `bom_ner_model_roberta/`: Store fine-tuned NER models and checkpoints. Excluded from git.

## Usage Notes
- Requires Python 3.8+ and dependencies listed in the code (see import statements).
- Poppler is required for PDF rendering (set `POPLER_PATH` in `select_bom_region.py`).
- For ML/AI extraction, install Hugging Face Transformers, Torch, and ONNX Runtime as needed.
- For object detection, provide a compatible ONNX model.
- For Nanonets OCR, use the provided transformer model or pytesseract.

## GitHub
- Repository: https://github.com/DucklingGod/BOMs_Extractor.git
- `.gitignore` ensures only source code and essential files are tracked.

## Future Developer Guidance
- Extend extraction backends as new models/techniques become available.
- Improve error handling and user feedback in the GUI.
- Consider packaging as a standalone executable (e.g., with PyInstaller) for non-technical users.
- Keep model/data files out of version control; share via external storage if needed.

---
This file is intended as a knowledge base for future maintainers and AI agents. For detailed code logic, see the docstrings and comments in each Python file.
