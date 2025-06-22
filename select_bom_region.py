import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
from pdf2image import convert_from_path

# Set your poppler path here
POPLER_PATH = r"C:\\poppler-24.08.0\\Library\\bin"

class PDFRegionSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF BOM Region Selector")
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image = None
        self.tk_img = None
        self.pdf_images = []
        self.page_num = 0
        self.rect = None
        self.start_x = self.start_y = self.end_x = self.end_y = 0
        self.region_coords = None
        self.setup_menu()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open PDF", command=self.open_pdf)
        filemenu.add_command(label="Select Page", command=self.select_page)
        filemenu.add_command(label="Clear Selection", command=self.clear_selection)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        # Add Extract menu next to File
        extractmenu = tk.Menu(menubar, tearoff=0)
        extractmenu.add_command(label="Extract BOM from All Pages", command=self.extract_bom)
        extractmenu.add_command(label="Extract BOM (ML/AI)", command=self.extract_bom_ml)
        extractmenu.add_command(label="Detect BOM (Object Detection)", command=self.detect_bom_object_detection)
        extractmenu.add_command(label="Detect BOM (Nanonets OCR)", command=self.detect_bom_nanonets_ocr)
        menubar.add_cascade(label="Extract", menu=extractmenu)
        self.root.config(menu=menubar)
        # Remove the bottom button
        # (No need to add extract_btn at the bottom anymore)

    def open_pdf(self):
        pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not pdf_path:
            return
        self.pdf_images = convert_from_path(pdf_path, poppler_path=POPLER_PATH)
        self.page_num = 0
        self.show_page()

    def select_page(self):
        if not self.pdf_images:
            messagebox.showinfo("Info", "Open a PDF first.")
            return
        page = simpledialog.askinteger("Select Page", f"Enter page number (1-{len(self.pdf_images)}):", minvalue=1, maxvalue=len(self.pdf_images))
        if page:
            self.page_num = page - 1
            self.show_page()

    def show_page(self):
        img = self.pdf_images[self.page_num]
        self.image = img
        # Resize image to fit window while maintaining aspect ratio
        win_w = self.root.winfo_width() or 800
        win_h = self.root.winfo_height() or 600
        img_w, img_h = img.size
        scale = min(win_w / img_w, win_h / img_h, 1.0)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_img = img.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized_img)
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.region_coords = None
        self.rect = None
        self.scale = scale  # Save scale for coordinate conversion

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_drag(self, event):
        self.end_x, self.end_y = event.x, event.y
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)

    def on_release(self, event):
        self.end_x, self.end_y = event.x, event.y
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)
        # Convert region to original image coordinates
        scale = getattr(self, 'scale', 1.0)
        region = (
            int(min(self.start_x, self.end_x) / scale),
            int(min(self.start_y, self.end_y) / scale),
            int(max(self.start_x, self.end_x) / scale),
            int(max(self.start_y, self.end_y) / scale)
        )
        self.region_coords = region
        print(f"Selected region on page {self.page_num+1}: {self.region_coords} (original image coords)")
        messagebox.showinfo("Region Selected", f"Page: {self.page_num+1}\nRegion: {self.region_coords} (original image coords)")

    def clear_selection(self):
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
        self.region_coords = None

    def extract_bom(self):
        if not self.region_coords:
            messagebox.showwarning("No Region Selected", "Please select a region first.")
            return
        if not self.pdf_images:
            messagebox.showwarning("No PDF Loaded", "Please open a PDF first.")
            return
        # Ask for example BOM line
        example = simpledialog.askstring("Example BOM", "Paste or type an example BOM line as it appears in the PDF:\n\n(e.g. 1 10.7 MTR 2\" PIPE, SMLS SCH.160 7430003073)")
        if not example:
            messagebox.showwarning("No Example Provided", "Please provide an example BOM line.")
            return
        import re
        # Infer regex and columns from example
        def guess_regex(example):
            parts = example.split()
            regex_parts = []
            col_names = []
            for p in parts:
                if re.fullmatch(r"\d+", p):
                    regex_parts.append(r"(\\d+)")
                    col_names.append("item" if not col_names else "qty")
                elif re.fullmatch(r"\d+[.,]?\d*", p):
                    regex_parts.append(r"(\\d+[.,]?\\d*)")
                    col_names.append("qty")
                elif re.fullmatch(r"[A-Za-z\"'/°|]+", p):
                    regex_parts.append(r"([A-Za-z\"'/°|]+)")
                    col_names.append("unit")
                else:
                    regex_parts.append(r"(.+?)")
                    col_names.append("description")
            regex = r"^" + r"\\s+".join(regex_parts) + r"\\s*(.*)$"
            return regex, col_names
        bom_regex, col_names = guess_regex(example)
        # Preview: show how the example would be split
        preview_re = re.compile(bom_regex)
        m = preview_re.match(example)
        preview_values = []
        if m:
            for idx, name in enumerate(col_names):
                preview_values.append(m.group(idx+1))
            preview_values.append(m.group(len(col_names)+1))  # rest
        else:
            preview_values = [example] + [""] * (len(col_names))
        # Show Excel-like preview for editing
        self.show_excel_preview(col_names + ["rest"], preview_values)

    def show_excel_preview(self, col_names, values):
        preview_win = tk.Toplevel(self.root)
        preview_win.title("Preview/Edit BOM Columns")
        import tkinter.ttk as ttk
        frame = ttk.Frame(preview_win)
        frame.pack(fill=tk.BOTH, expand=True)
        col_entries = []
        val_entries = []
        def render_table():
            for widget in frame.winfo_children():
                widget.destroy()
            col_entries.clear()
            val_entries.clear()
            for i, col in enumerate(col_names):
                e = tk.Entry(frame)
                e.insert(0, col)
                e.grid(row=0, column=i, sticky="nsew", padx=1, pady=1)
                col_entries.append(e)
            for i, val in enumerate(values):
                e = tk.Entry(frame)
                e.insert(0, val)
                e.grid(row=1, column=i, sticky="nsew", padx=1, pady=1)
                val_entries.append(e)
            add_btn = tk.Button(frame, text="Add Column", command=add_column)
            add_btn.grid(row=2, column=0, pady=8, sticky="w")
            rem_btn = tk.Button(frame, text="Remove Column", command=remove_column)
            rem_btn.grid(row=2, column=1, pady=8, sticky="w")
            preview_btn = tk.Button(frame, text="Preview Output", command=preview_output)
            preview_btn.grid(row=2, column=2, pady=8, sticky="w")
            btn = tk.Button(frame, text="Confirm and Extract", command=on_confirm)
            btn.grid(row=2, column=3, columnspan=max(1, len(col_names)-3), pady=8, sticky="e")
        def add_column():
            col_names.append(f"col{len(col_names)+1}")
            values.append("")
            render_table()
        def remove_column():
            if len(col_names) > 1:
                col_names.pop()
                values.pop()
                render_table()
        def preview_output():
            # Editable preview of the output row as a table
            from tkinter import Toplevel, Label, Entry, Button
            preview_cols = [e.get().strip() for e in col_entries if e.get().strip()]
            preview_vals = [e.get() for e in val_entries]
            out_win = Toplevel(preview_win)
            out_win.title("Output Preview (Editable)")
            out_entries = []
            for i, col in enumerate(preview_cols):
                Label(out_win, text=col, relief="ridge", width=18).grid(row=0, column=i, sticky="nsew")
            for i, val in enumerate(preview_vals):
                e = Entry(out_win, width=18)
                e.insert(0, val)
                e.grid(row=1, column=i, sticky="nsew")
                out_entries.append(e)
            def apply_edits():
                # Copy edited values back to main preview
                for i, e in enumerate(out_entries):
                    if i < len(val_entries):
                        val_entries[i].delete(0, tk.END)
                        val_entries[i].insert(0, e.get())
                out_win.destroy()
            Button(out_win, text="Apply", command=apply_edits).grid(row=2, column=0, columnspan=len(preview_cols), pady=8)
            out_win.grab_set()
        def on_confirm():
            new_cols = [e.get().strip() for e in col_entries if e.get().strip()]
            new_vals = [e.get() for e in val_entries]
            preview_win.destroy()
            self.run_bom_extraction(new_cols, new_vals)
        render_table()
        preview_win.grab_set()
        preview_win.transient(self.root)
        preview_win.wait_window()

    def run_bom_extraction(self, col_names, preview_vals):
        from tkinter import filedialog
        output_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if not output_path:
            return
        import pytesseract
        import pandas as pd
        from PIL import Image
        import re
        # Build regex from col_names and preview_vals (try to keep original logic)
        def build_regex_from_example(cols, vals):
            regex_parts = []
            for v in vals[:-1]:
                if re.fullmatch(r"\d+", v):
                    regex_parts.append(r"(\\d+)")
                elif re.fullmatch(r"\d+[.,]?\d*", v):
                    regex_parts.append(r"(\\d+[.,]?\\d*)")
                elif re.fullmatch(r"[A-Za-z\"'/°|]+", v):
                    regex_parts.append(r"([A-Za-z\"'/°|]+)")
                else:
                    regex_parts.append(r"(.+?)")
            regex = r"^" + r"\\s+".join(regex_parts) + r"\\s*(.*)$"
            return regex
        bom_regex = build_regex_from_example(col_names, preview_vals)
        bom_entry_re = re.compile(bom_regex)
        def is_garbage(line):
            if len(line) < 5: return True
            if re.fullmatch(r"[=~\-\|oOaAuvnwl\s]+", line): return True
            if sum(c.isalpha() for c in line) < 2: return True
            return False
        def preprocess(text):
            lines = text.splitlines()
            cleaned = []
            for line in lines:
                l = line.strip()
                if not l: continue
                if re.search(r'(UNIT|DESCRIPTION|MESC|ITEM|QTY|REV\\.|No\\.|PIPE|FLANGE|GASKET|VALVE)', l, re.IGNORECASE) and len(l) < 30:
                    continue
                l = re.sub(r'\\s+', ' ', l)
                cleaned.append(l)
            return cleaned
        bom_data = []
        for i, img in enumerate(self.pdf_images):
            region_img = img.crop(self.region_coords)
            text = pytesseract.image_to_string(region_img)
            lines = preprocess(text)
            for line in lines:
                if is_garbage(line):
                    continue
                m = bom_entry_re.match(line)
                if m:
                    row = {'page': i+1}
                    for idx, name in enumerate(col_names):
                        row[name] = m.group(idx+1) if idx+1 <= m.lastindex else ''
                    row['raw'] = line
                    bom_data.append(row)
                else:
                    # Treat as continuation of previous row's last column
                    if bom_data and col_names:
                        last_col = col_names[-1]
                        bom_data[-1][last_col] += ' ' + line
                        bom_data[-1]['raw'] += ' | ' + line
                    else:
                        bom_data.append({'page': i+1, **{k:'' for k in col_names}, 'raw': line})
        if bom_data:
            import pandas as pd
            df = pd.DataFrame(bom_data)
            df.to_excel(output_path, index=False)
            messagebox.showinfo("Extraction Complete", f"Extracted BOMs saved to {output_path}")
        else:
            messagebox.showinfo("No BOMs Found", "No BOM text found in the selected region on any page.")

    def extract_bom_ml(self):
        if not self.region_coords:
            messagebox.showwarning("No Region Selected", "Please select a region first.")
            return
        if not self.pdf_images:
            messagebox.showwarning("No PDF Loaded", "Please open a PDF first.")
            return
        from nanonets_ocr_integration import extract_bom_fields_with_nanonets
        import pytesseract
        import pandas as pd
        from PIL import Image
        # Ask for output file
        output_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if not output_path:
            return
        def preprocess(text):
            lines = text.splitlines()
            cleaned = []
            for line in lines:
                l = line.strip()
                if not l: continue
                if len(l) < 5: continue
                cleaned.append(l)
            return cleaned
        bom_data = []
        all_fields = set()
        for i, img in enumerate(self.pdf_images):
            region_img = img.crop(self.region_coords)
            text = pytesseract.image_to_string(region_img)
            lines = preprocess(text)
            for line in lines:
                fields = extract_bom_fields_with_nanonets(line)
                if fields:
                    fields['page'] = i+1
                    fields['raw'] = line
                    bom_data.append(fields)
                    all_fields.update(fields.keys())
        if bom_data:
            # Ensure all columns are present
            all_fields = list(all_fields)
            df = pd.DataFrame(bom_data)
            for col in all_fields:
                if col not in df.columns:
                    df[col] = ''
            df = df[all_fields]  # reorder columns
            df.to_excel(output_path, index=False)
            messagebox.showinfo("Extraction Complete", f"Extracted BOMs (ML/AI) saved to {output_path}")
        else:
            messagebox.showinfo("No BOMs Found", "No BOM text found in the selected region on any page.")

    def detect_bom_object_detection(self):
        if not self.pdf_images:
            messagebox.showwarning("No PDF Loaded", "Please open a PDF first.")
            return
        # Ask if user wants to use full page
        use_full_page = messagebox.askyesno("Full Page Detection", "Run detection on the full page instead of the selected region?")
        if not use_full_page and not self.region_coords:
            messagebox.showwarning("No Region Selected", "Please select a region first or choose full page detection.")
            return
        import onnxruntime as ort
        import numpy as np
        import pytesseract
        import pandas as pd
        from PIL import Image
        from tkinter import filedialog
        # Ask for ONNX model path
        model_path = filedialog.askopenfilename(title="Select ONNX Model", filetypes=[("ONNX Model", "*.onnx")])
        if not model_path:
            return
        # Ask for output file
        output_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if not output_path:
            return
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        bom_data = []
        for i, img in enumerate(self.pdf_images):
            if use_full_page:
                region_img = img
            else:
                region_img = img.crop(self.region_coords)
            # Preprocess for model (assume 1024x1024, 3ch RGB, float32, 0-1)
            input_size = 1024
            img_resized = region_img.resize((input_size, input_size))
            img_np = np.array(img_resized.convert('RGB'), dtype=np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))[None, ...]  # (1,3,1024,1024)
            # Run ONNX model
            outputs = session.run(output_names, {input_name: img_np})
            # Postprocess: assume YOLO format [x1, y1, x2, y2, conf, class]
            dets = outputs[0][0] if isinstance(outputs[0], np.ndarray) else outputs[0]
            # Filter by confidence
            dets = [d for d in dets if d[4] > 0.1]
            for det in dets:
                x1, y1, x2, y2, conf, cls = det[:6]
                # Map back to region_img coordinates
                x1 = int(x1 / input_size * region_img.width)
                y1 = int(y1 / input_size * region_img.height)
                x2 = int(x2 / input_size * region_img.width)
                y2 = int(y2 / input_size * region_img.height)
                # Clip to image bounds
                x1 = max(0, min(region_img.width-1, x1))
                y1 = max(0, min(region_img.height-1, y1))
                x2 = max(0, min(region_img.width, x2))
                y2 = max(0, min(region_img.height, y2))
                # Only crop if box is valid
                if x2 > x1 and y2 > y1:
                    crop = region_img.crop((x1, y1, x2, y2))
                    text = pytesseract.image_to_string(crop)
                    bom_data.append({'page': i+1, 'bbox': f'{x1},{y1},{x2},{y2}', 'text': text.strip(), 'conf': conf})
        if bom_data:
            df = pd.DataFrame(bom_data)
            df.to_excel(output_path, index=False)
            messagebox.showinfo("Extraction Complete", f"Detected BOMs saved to {output_path}")
        else:
            messagebox.showinfo("No BOMs Found", "No BOMs detected in the selected region on any page.")

    def detect_bom_nanonets_ocr(self):
        if not self.pdf_images:
            messagebox.showwarning("No PDF Loaded", "Please open a PDF first.")
            return
        # Ask if user wants to use full page
        use_full_page = messagebox.askyesno("Full Page Detection", "Run detection on the full page instead of the selected region?")
        if not use_full_page and not self.region_coords:
            messagebox.showwarning("No Region Selected", "Please select a region first or choose full page detection.")
            return
        import pandas as pd
        from PIL import Image
        from tkinter import filedialog
        import pytesseract
        import io
        # Ask for output file
        output_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if not output_path:
            return
        bom_data = []
        for i, img in enumerate(self.pdf_images):
            if use_full_page:
                region_img = img
            else:
                region_img = img.crop(self.region_coords)
            # Run pytesseract OCR
            text = pytesseract.image_to_string(region_img)
            bom_data.append({'page': i+1, 'text': text})
        if bom_data:
            df = pd.DataFrame(bom_data)
            df.to_excel(output_path, index=False)
            messagebox.showinfo("Extraction Complete", f"Detected BOMs (pytesseract OCR) saved to {output_path}")
        else:
            messagebox.showinfo("No BOMs Found", "No BOMs detected in the selected region on any page.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFRegionSelector(root)
    root.mainloop()
