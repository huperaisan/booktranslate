import argparse
import base64
import json
import yaml
from pathlib import Path
import fitz  # PyMuPDF
from openai import OpenAI

class PDFHandler:
    """Handler for PDF file processing"""

    @staticmethod
    def generate_page_images(pdf_path, output_dir, max_height=2000, dpi=150):
        """
        Generate PNG images of each page in the PDF and save them to the specified directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        with fitz.open(pdf_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                
                # Calculate the scaling factor for the desired DPI
                scale = dpi / 72  # Default DPI in PyMuPDF is 72
                
                # Generate the pixmap with the scaling factor
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                
                # Resize image if height exceeds max_height
                if pix.height > max_height:
                    scale = max_height / pix.height
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                
                image_path = output_dir / f"page_{page_num}.png"
                pix.save(image_path)
                print(f"Saved image: {image_path}")

        print(f"All pages have been saved as images in {output_dir}")

    @staticmethod
    def read_config():
        config_file = 'config.yaml'
        try:
            with open(config_file, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            return config
        except FileNotFoundError:
            print(f"Error: {config_file} not found. Please create it with your OpenAI API key. [read_config]")
            print("Example config.yaml content:\nopenai:\n  api_key: 'your-api-key-here' [read_config]")
            exit(1)

    @staticmethod
    def encode_image(image_path):
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def sanitize_html_output(html_content):
        """Sanitize the HTML output by removing markdown code block markers"""
        lines = html_content.split('\n')
        if lines[0].strip() == "```html":
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        return '\n'.join(lines)

    @staticmethod
    def save_translated_pdf(translations, output_pdf_path):
        """Write a new PDF document from HTML pages"""
        output_pdf_path = Path(output_pdf_path)  # Ensure it's a Path object
        output_pdf_path.parent.mkdir(exist_ok=True)
        doc = fitz.open()
        page_width = 6 * 72  # 6 inches in points
        page_height = 9 * 72  # 9 inches in points
        margin = 1 * 72  # 1 inch in points

        for chunk_id in sorted(translations.keys(), key=lambda x: int(x.split('-')[1])):
            html_content = translations[chunk_id]
            page = doc.new_page(width=page_width, height=page_height)
            # Insert HTML content into the page with a bounding box
            page.insert_htmlbox(fitz.Rect(margin, margin, page_width - margin, page_height - margin), html_content)
        doc.save(output_pdf_path)
        print(f"PDF saved to {output_pdf_path}")

    @staticmethod
    def transcribe_pdf(input_pdf_path, dpi=150):
        input_pdf_path = Path(input_pdf_path)
        if not input_pdf_path.exists():
            print(f"Error: Input file not found: {input_pdf_path}")
            return

        # Create output directory
        output_dir = Path('./tempimg')
        output_dir.mkdir(exist_ok=True)

        # Generate images of PDF pages
        PDFHandler.generate_page_images(input_pdf_path, output_dir, dpi=dpi)

        config = PDFHandler.read_config()
        client = OpenAI(api_key=config['openai']['api_key'])

        all_chunks = []
        chapter_map = {}

        # Process each image
        for page_num, image_path in enumerate(sorted(output_dir.glob("*.png"))):
            base64_image = PDFHandler.encode_image(image_path)
            print(f"Transcribing {image_path}")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract the text from this image and produce an HTML document with formatting that reproduces the formatting of the input image. Do not use tables. Pay special attention to ITALICS and SUPERSCRIPT formatting. CRITICAL: If a block of text is smaller than other text in the image, use 'font-size: smaller' for it in your HTML. Use 'font-family: serif' and default line-height. If the image contains GREEK characters, recognise them letter-by-letter and do not try to make sense of the words. Return ONLY the HTML content, beginning with <!DOCTYPE html> and ending with </html>.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
            )

            html_content = response.choices[0].message.content
            sanitized_html_content = PDFHandler.sanitize_html_output(html_content)

            # Save HTML content to file
            html_file_path = output_dir / f"page_{page_num}.html"
            with open(html_file_path, "w", encoding="utf-8") as html_file:
                html_file.write(sanitized_html_content)
            
            print(f"Saved HTML: {html_file_path}")

            # Add to all_chunks and chapter_map
            chunk_id = f'chunk-{page_num}'
            all_chunks.append((chunk_id, sanitized_html_content))
            chapter_map[chunk_id] = {
                "item": f"page_{page_num}.html",
                "pos": 0
            }

        # Write all_chunks to all_chunks.json
        pdftemp_dir = Path('./pdftemp')
        pdftemp_dir.mkdir(exist_ok=True)
        with open(pdftemp_dir / 'all_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=4)

        # Write chapter_map to chapter_map.json
        with open(pdftemp_dir / 'chapter_map.json', 'w', encoding='utf-8') as f:
            json.dump(chapter_map, f, indent=4)

        print(f"Processing complete. HTML files and metadata saved to {pdftemp_dir}")

        return all_chunks, chapter_map

if __name__ == "__main__":
    PDFHandler.main()