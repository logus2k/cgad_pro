#!/usr/bin/env python3
"""
Convert Jupyter notebooks or Markdown documents to DOCX.
For notebooks: strip text outputs while keeping charts (images), then export to DOCX.
For Markdown: directly convert to DOCX.

Usage:
  python notebook_to_report.py path/to/input_file [options]

Input formats supported:
  - Jupyter notebooks (.ipynb): text outputs stripped, images preserved
  - Markdown documents (.md): converted directly to DOCX

Requirements:
  pip install nbformat nbconvert
  # For DOCX:
  pip install pypandoc            # required
  # OR install system pandoc: https://pandoc.org/install.html
  # For SVG images in DOCX:
  #   Linux: sudo apt install librsvg2-bin
  #   Windows: choco install rsvg-convert (or use Inkscape)

Options:
  --out report_basename    Base name for outputs (without extension)
  --template path.docx     Reference DOCX template for styling (page layout, fonts, etc.)
  --no-code                (Notebooks only) Hide code cells in the exported outputs
  --html                   (Notebooks only) Also export to HTML format
  --markdown               (Notebooks only) Also export to Markdown format

Template creation:
  To create a custom template with your preferred styles (A4 page, fonts, etc.):
    pandoc -o my_template.docx --print-default-data-file reference.docx
  Then open my_template.docx in Word/LibreOffice, customize page layout and styles
  (Heading 1, Heading 2, Normal, List Bullet, Table Grid, etc.), and save.

Examples:
  # Notebook to DOCX with code visible
  python notebook_to_report.py my_notebook.ipynb
  # Notebook to DOCX without code
  python notebook_to_report.py my_notebook.ipynb --no-code
  # Notebook to DOCX + HTML + Markdown
  python notebook_to_report.py my_notebook.ipynb --html --markdown
  # Markdown to DOCX
  python notebook_to_report.py my_document.md
  # Markdown to DOCX with custom template
  python notebook_to_report.py my_document.md --template my_template.docx
"""

import argparse
from pathlib import Path
import sys


def keep_only_image_outputs(nb):
    """Remove text-only outputs, keep image (PNG/SVG) and optional rich HTML outputs."""
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        new_outputs = []
        for out in cell.get("outputs", []):
            otype = out.get("output_type")
            data = out.get("data", {})
            has_img = any(k in data for k in ("image/png", "image/svg+xml"))
            keep_html = "text/html" in data  # set False if you don't want rich HTML
            if otype in {"display_data", "execute_result"} and (has_img or keep_html):
                new_outputs.append(out)
            # drop streams/errors/plain text-only execute_results
        cell["outputs"] = new_outputs
        cell["execution_count"] = None
    return nb


def export_html(nb, out_html, hide_code: bool):
    try:
        from traitlets.config import Config
        from nbconvert import HTMLExporter
    except ImportError:
        print("[error] nbformat/nbconvert not available. Install with: pip install nbformat nbconvert")
        sys.exit(1)
        
    c = Config()
    # Hide prompts; optionally hide code inputs entirely
    c.HTMLExporter.exclude_input_prompt = True
    c.HTMLExporter.exclude_output_prompt = True
    if hide_code:
        c.HTMLExporter.exclude_input = True
    html_exporter = HTMLExporter(config=c)
    body, _ = html_exporter.from_notebook_node(nb)
    Path(out_html).write_text(body, encoding="utf-8")


def export_markdown(nb, out_md, assets_dir: str, hide_code: bool):
    """
    Export Markdown and extract images to assets_dir (nbconvert writes files there).
    """
    try:
        from traitlets.config import Config
        from nbconvert import MarkdownExporter
    except ImportError:
        print("[error] nbformat/nbconvert not available. Install with: pip install nbformat nbconvert")
        sys.exit(1)
        
    c = Config()
    c.MarkdownExporter.exclude_input_prompt = True
    c.MarkdownExporter.exclude_output_prompt = True
    if hide_code:
        c.MarkdownExporter.exclude_input = True
    resources = {"output_files_dir": assets_dir}
    md_exporter = MarkdownExporter(config=c)
    body, resources = md_exporter.from_notebook_node(nb, resources=resources)
    Path(out_md).write_text(body, encoding="utf-8")

    # Extract images - nbconvert includes the assets_dir in the fname
    for fname, data in resources.get("outputs", {}).items():
        out_path = Path(out_md).parent / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)


def markdown_to_docx(md_path, docx_path, template_path=None):
    """Convert Markdown to DOCX using pypandoc if available; else shell pandoc."""
    extra_args = ["--standalone", "--resource-path=."]
    if template_path:
        extra_args.append(f"--reference-doc={template_path}")
    
    try:
        import pypandoc  # type: ignore
        pypandoc.convert_file(
            str(md_path),
            "docx",
            outputfile=str(docx_path),
            extra_args=extra_args,
        )
        return
    except Exception as e:
        print("[info] pypandoc not available or failed:", e)

    import subprocess
    cmd = ["pandoc", str(md_path), "-o", str(docx_path)] + extra_args
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        print(
            "[error] pandoc not found. Install it or `pip install pypandoc`.\n"
            "Download: https://pandoc.org/install.html  "
        )
        sys.exit(2)


def process_notebook(notebook_path, base_name, output_dir, args, template_path=None):
    """Process a Jupyter notebook file."""
    try:
        import nbformat as nbf
        from traitlets.config import Config
        from nbconvert import HTMLExporter, MarkdownExporter
    except ImportError:
        print("[error] nbformat/nbconvert not available. Install with: pip install nbformat nbconvert")
        sys.exit(1)
    
    out_html = output_dir / f"{base_name}_report.html"
    out_md = output_dir / f"{base_name}_report.md"
    assets_dir = f"{base_name}_report_files"
    out_docx = output_dir / f"{base_name}_report.docx"

    print("[1/3] Reading notebook...")
    nb = nbf.read(str(notebook_path), as_version=4)

    print("[2/3] Stripping text outputs, keeping charts...")
    nb_clean = keep_only_image_outputs(nb)

    # Export HTML if requested
    if args.html:
        print(f"[3a/3] Exporting HTML -> {out_html} (hide_code={args.no_code})")
        export_html(nb_clean, str(out_html), hide_code=args.no_code)

    # Always export Markdown (needed for DOCX), but only mention if explicitly requested
    if args.markdown:
        print(f"[3b/3] Exporting Markdown (and images) -> {out_md} (hide_code={args.no_code})")
    export_markdown(nb_clean, str(out_md), assets_dir=assets_dir, hide_code=args.no_code)

    # Always export DOCX
    print(f"[3/3] Converting to DOCX -> {out_docx}")
    markdown_to_docx(str(out_md), str(out_docx), template_path=template_path)

    # Clean up intermediate files if neither markdown nor html was requested
    if not args.markdown:
        print("[cleanup] Removing intermediate Markdown files...")
        out_md.unlink(missing_ok=True)
        import shutil
        assets_path = output_dir / assets_dir
        if assets_path.exists():
            shutil.rmtree(assets_path)

    return out_docx, out_html if args.html else None, out_md if args.markdown else None


def process_markdown(markdown_path, base_name, output_dir, template_path=None):
    """Process a Markdown file directly."""
    out_docx = output_dir / f"{base_name}.docx"
    
    print(f"[1/1] Converting Markdown to DOCX -> {out_docx}")
    markdown_to_docx(markdown_path, out_docx, template_path=template_path)
    
    return out_docx


def main():
    ap = argparse.ArgumentParser(
        description="Convert Jupyter notebooks or Markdown documents to DOCX."
    )
    ap.add_argument("input_file", help="Path to source .ipynb or .md file")
    ap.add_argument("--out", help="Base name for outputs (without extension)")
    ap.add_argument(
        "--template",
        help="Path to reference DOCX template for styling (page layout, fonts, etc.)",
    )
    ap.add_argument(
        "--no-code",
        action="store_true",
        help="(Notebooks only) Hide code cells in the exported outputs (markdown + charts only).",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="(Notebooks only) Also export to HTML format.",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="(Notebooks only) Also export to Markdown format.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_file).resolve()
    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}")
        sys.exit(1)

    # Determine file type and validate extensions
    if input_path.suffix.lower() not in ['.ipynb', '.md']:
        print(f"[error] Unsupported file type: {input_path.suffix}. Supported: .ipynb, .md")
        sys.exit(1)

    # Set up output base name and directory
    base = args.out if args.out else input_path.with_suffix("").name
    output_dir = input_path.parent

    # Validate template path if provided
    template_path = None
    if args.template:
        template_path = Path(args.template).resolve()
        if not template_path.exists():
            print(f"[error] Template file not found: {template_path}")
            sys.exit(1)
        print(f"[info] Using template: {template_path}")

    # Initialize variables to avoid Pylance warnings
    html_out = None
    md_out = None
    docx_out = None

    # Process based on file type
    if input_path.suffix.lower() == '.ipynb':
        # Notebook processing
        docx_out, html_out, md_out = process_notebook(input_path, base, output_dir, args, template_path=template_path)
    elif input_path.suffix.lower() == '.md':
        # Direct Markdown processing
        if args.no_code or args.html or args.markdown:
            print(f"[warning] Options --no-code, --html, --markdown are ignored for Markdown input")
        
        docx_out = process_markdown(input_path, base, output_dir, template_path=template_path)

    print("\nDone.")
    print("Outputs:")
    if html_out:
        print("  - HTML report     :", html_out)
    if md_out:
        assets_dir = f"{base}_report_files"
        print("  - Markdown        :", md_out, f"(assets in ./{assets_dir}/)")
    print("  - Word (.docx)    :", docx_out)


if __name__ == "__main__":
    main()
