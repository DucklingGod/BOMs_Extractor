# nanonets_ocr_integration.py
"""
Helper function to use a Hugging Face NER/token classification model for BOM line extraction.
Install requirements first:
    pip install transformers torch

Usage:
    fields = extract_bom_fields_with_ner(line, model_path_or_name)

Where model_path_or_name is your fine-tuned model directory or Hugging Face model hub name.
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoProcessor, AutoModelForImageTextToText
import torch
import re

def extract_bom_fields_with_ner(line, model_path_or_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForTokenClassification.from_pretrained(model_path_or_name)
    inputs = tokenizer(line, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]
    # Improved: group consecutive tokens for each label, detokenizing subwords
    fields = {}
    prev_label = None
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 'O' or token in tokenizer.all_special_tokens:
            prev_label = None
            continue
        # Remove RoBERTa's special space marker (\u0120 or 'Ġ')
        token_clean = token.replace('##', '').replace('\u0120', '').replace('Ġ', '')
        # Decide if we need a space before this token
        add_space = False
        if label in fields:
            # No space if subword, or if previous char is punctuation/number and current is number/punctuation
            if not token.startswith('##') and not (
                (fields[label][-1].isdigit() and token_clean.isdigit()) or
                (fields[label][-1] in '.-' and token_clean.isdigit()) or
                (fields[label][-1].isdigit() and token_clean in '.-')
            ):
                add_space = True
        if label not in fields:
            fields[label] = token_clean
        else:
            if add_space:
                fields[label] += ' '
            fields[label] += token_clean
        prev_label = label
    # Strip extra spaces
    for k in fields:
        fields[k] = fields[k].strip()
    # Post-processing for QTY and MESC fields
    if 'QTY' in fields:
        # Merge numbers and periods for QTY (e.g., '10 7' or '10 . 7' -> '10.7')
        qty = re.sub(r'\s*\.\s*', '.', fields['QTY'])
        qty = re.sub(r'\s+', '', qty) if '.' in qty else fields['QTY'].replace(' ', '')
        fields['QTY'] = qty
    if 'MESC' in fields:
        # Extract the last 8-12 digit number from the field
        match = re.search(r'(\d{10})$', fields['MESC'])
        if match:
            fields['MESC'] = match.group(1)
    return fields

def ocr_page_with_nanonets_s(pil_img, model, processor, max_new_tokens=4096):
    """
    Run Nanonets-OCR-s (image-to-text) on a PIL image using Hugging Face Transformers.
    Returns the generated text output.
    """
    prompt = (
        "Extract the text from the above document as if you were reading it naturally. "
        "Return the tables in html format. Return the equations in LaTeX representation. "
        "If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; "
        "otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. "
        "Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "
        "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
        "Prefer using ☐ and ☑ for check boxes."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt")
    import torch
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

if __name__ == "__main__":
    # Example usage for testing the Hugging Face BOM extractor model
    line = '1 10.7 MTR 2" PIPE, SMLS SCH.160 7430003073'
    model_path = "hsarfraz/eng-drawing-title-block-bill-of-material-extractor"  # Hugging Face BOM extractor model
    fields = extract_bom_fields_with_ner(line, model_path)
    print("Extracted fields:", fields)

    # Example usage for Nanonets OCR-s model
    try:
        from PIL import Image
        model_path_ocr = "nanonets/Nanonets-OCR-s"
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model_ocr = AutoModelForImageTextToText.from_pretrained(
            model_path_ocr,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        model_ocr.eval()
        processor_ocr = AutoProcessor.from_pretrained(model_path_ocr)
        pil_img = Image.open("Screenshot 2025-06-21 095926.png").convert("RGB")
        result = ocr_page_with_nanonets_s(pil_img, model_ocr, processor_ocr, max_new_tokens=15000)
        print("OCR result:\n", result)
    except Exception as e:
        print("OCR example failed:", e)
