#!/usr/bin/env python3
"""
Convert Markdown containing ECharts div placeholders to DOCX/PDF.

Charts are rendered to PNG images using Playwright (headless browser),
then the markdown is processed by pandoc to produce the final document.

Usage:
  python markdown_to_report.py path/to/document.md [--out basename] [--pdf] [--html] [--keep-md]

Requirements:
  pip install playwright
  playwright install chromium
  
  # For DOCX/PDF:
  pip install pypandoc
  # OR install system pandoc: https://pandoc.org/installing.html
  
  # For PDF without LaTeX:
  pip install weasyprint

Examples:
  # Basic: DOCX output
  python markdown_to_report.py report.md
  
  # DOCX + PDF
  python markdown_to_report.py report.md --pdf
  
  # All formats, keep intermediate markdown
  python markdown_to_report.py report.md --pdf --html --keep-md
  
  # Specify base path for resolving relative images
  python markdown_to_report.py docs/report.md --base-path /project/root
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path


# ECharts HTML template for rendering charts
ECHARTS_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{ width: {width}px; height: {height}px; background: {background}; overflow: hidden; }}
        #chart {{ width: {width}px; height: {height}px; }}
    </style>
</head>
<body>
    <div id="chart"></div>
    <script>
        const config = {config_json};
        const chart = echarts.init(document.getElementById('chart'), null, {{ renderer: 'svg' }});
        chart.setOption(buildChartOption(config));
        
        function buildChartOption(config) {{
            const THEME = {{
                text: '#333',
                textLight: '#666',
                title: '#1a5a7a',
                axisLine: '#999',
                splitLine: '#ddd',
                background: 'transparent',
                pieBorder: '#ffffff'
            }};
            const colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4'];
            
            const baseOption = {{
                backgroundColor: '{background}',
                textStyle: {{ color: THEME.text }},
                title: {{
                    text: config.title || '',
                    left: 'center',
                    top: 10,
                    textStyle: {{ color: THEME.title, fontSize: 14, fontWeight: 'bold' }}
                }},
                tooltip: {{ trigger: config.type === 'pie' ? 'item' : 'axis' }},
                color: colors
            }};

            if (config.type === 'pie') {{
                return {{
                    ...baseOption,
                    tooltip: {{ trigger: 'item', formatter: '{{b}}: {{c}}% ({{d}}%)' }},
                    legend: {{
                        orient: 'vertical',
                        right: 10,
                        top: 'middle',
                        textStyle: {{ color: THEME.text }}
                    }},
                    series: [{{
                        type: 'pie',
                        radius: ['30%', '55%'],
                        center: ['35%', '55%'],
                        avoidLabelOverlap: true,
                        itemStyle: {{ borderRadius: 4, borderColor: THEME.pieBorder, borderWidth: 2 }},
                        label: {{ 
                            show: true, 
                            formatter: '{{b}}: {{c}}%', 
                            color: THEME.text, 
                            fontSize: 11
                        }},
                        labelLine: {{
                            show: true,
                            lineStyle: {{ color: THEME.textLight, width: 1 }},
                            smooth: 0.2, length: 8, length2: 12
                        }},
                        data: config.data
                    }}]
                }};
            }}

            if (config.type === 'bar') {{
                return {{
                    ...baseOption,
                    legend: {{ data: config.series.map(s => s.name), bottom: 10, textStyle: {{ color: THEME.text }} }},
                    grid: {{ left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true }},
                    xAxis: {{
                        type: 'category',
                        data: config.categories,
                        axisLabel: {{ color: THEME.text, rotate: 30 }},
                        axisLine: {{ lineStyle: {{ color: THEME.axisLine }} }}
                    }},
                    yAxis: {{
                        type: 'value',
                        name: config.yAxisName || '',
                        nameTextStyle: {{ color: THEME.text }},
                        axisLabel: {{ color: THEME.text }},
                        axisLine: {{ lineStyle: {{ color: THEME.axisLine }} }},
                        splitLine: {{ lineStyle: {{ color: THEME.splitLine }} }}
                    }},
                    series: config.series.map(s => ({{ name: s.name, type: 'bar', data: s.data, barGap: '10%' }}))
                }};
            }}

            if (config.type === 'stacked-bar') {{
                return {{
                    ...baseOption,
                    legend: {{ data: config.series.map(s => s.name), bottom: 10, textStyle: {{ color: THEME.text }} }},
                    grid: {{ left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true }},
                    xAxis: {{
                        type: 'category',
                        data: config.categories,
                        axisLabel: {{ color: THEME.text, rotate: 30 }},
                        axisLine: {{ lineStyle: {{ color: THEME.axisLine }} }}
                    }},
                    yAxis: {{
                        type: 'value',
                        name: config.yAxisName || '',
                        nameTextStyle: {{ color: THEME.text }},
                        axisLabel: {{ color: THEME.text }},
                        axisLine: {{ lineStyle: {{ color: THEME.axisLine }} }},
                        splitLine: {{ lineStyle: {{ color: THEME.splitLine }} }}
                    }},
                    series: config.series.map(s => ({{
                        name: s.name, type: 'bar', stack: 'total',
                        emphasis: {{ focus: 'series' }}, data: s.data
                    }}))
                }};
            }}

            if (config.type === 'line') {{
                return {{
                    ...baseOption,
                    legend: {{ data: config.series.map(s => s.name), bottom: 10, textStyle: {{ color: THEME.text }} }},
                    grid: {{ left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true }},
                    xAxis: {{
                        type: 'category',
                        name: config.xAxisName || '',
                        nameTextStyle: {{ color: THEME.text }},
                        data: config.categories,
                        axisLabel: {{ color: THEME.text }},
                        axisLine: {{ lineStyle: {{ color: THEME.axisLine }} }}
                    }},
                    yAxis: {{
                        type: 'value',
                        name: config.yAxisName || '',
                        nameTextStyle: {{ color: THEME.text }},
                        axisLabel: {{ color: THEME.text }},
                        axisLine: {{ lineStyle: {{ color: THEME.axisLine }} }},
                        splitLine: {{ lineStyle: {{ color: THEME.splitLine }} }}
                    }},
                    series: config.series.map(s => ({{ name: s.name, type: 'line', smooth: true, data: s.data }}))
                }};
            }}

            return baseOption;
        }}
    </script>
</body>
</html>
"""

# Regex to find chart divs - handles JSON with internal double quotes
# Pattern for data-chart='...' (single quotes around JSON)
CHART_DIV_PATTERN_SINGLE = re.compile(
    r'<div\s+class="echart-container[^"]*"[^>]*data-chart=\'(\{[^\']+\})\'[^>]*>\s*</div>',
    re.IGNORECASE | re.DOTALL
)

# Pattern for data-chart="..." (double quotes around JSON, entities encoded)
CHART_DIV_PATTERN_DOUBLE = re.compile(
    r'<div\s+class="echart-container[^"]*"[^>]*data-chart="(\{[^"]*\})"[^>]*>\s*</div>',
    re.IGNORECASE | re.DOTALL
)

# Pattern with height extraction (single quotes)
CHART_DIV_PATTERN_HEIGHT = re.compile(
    r'<div\s+class="echart-container[^"]*"\s+id="[^"]*"\s+style="[^"]*height:\s*(\d+)px[^"]*"\s+data-chart=\'(\{[^\']+\})\'[^>]*>\s*</div>',
    re.IGNORECASE | re.DOTALL
)


def extract_charts_from_markdown(markdown_content: str, debug: bool = False) -> list:
    """
    Extract chart configurations from markdown content.
    
    Returns list of dicts: [{match, config, height, chart_id}, ...]
    """
    charts = []
    
    if debug:
        # Check if there's any echart-container in the file
        if 'echart-container' in markdown_content:
            print("    [debug] Found 'echart-container' in content")
            # Show first occurrence
            idx = markdown_content.find('data-chart')
            if idx > 0:
                snippet = markdown_content[idx:idx+100]
                print(f"    [debug] data-chart snippet: {snippet}...")
        else:
            print("    [debug] No 'echart-container' found in content")
        
        if 'data-chart' in markdown_content:
            print("    [debug] Found 'data-chart' in content")
        else:
            print("    [debug] No 'data-chart' found in content")
    
    # Try pattern with height extraction first (single quotes)
    for match in CHART_DIV_PATTERN_HEIGHT.finditer(markdown_content):
        height = int(match.group(1))
        config_str = match.group(2)
        if debug:
            print(f"    [debug] HEIGHT pattern matched, config starts: {config_str[:50]}...")
        try:
            config = json.loads(config_str)
            charts.append({
                'match': match.group(0),
                'config': config,
                'height': height,
                'chart_id': config.get('title', f'chart_{len(charts)}').replace(' ', '_')
            })
        except json.JSONDecodeError as e:
            if debug:
                print(f"    [debug] JSON parse error: {e}, string: {config_str[:100]}")
            continue
    
    # Try single-quote pattern
    if not charts:
        for match in CHART_DIV_PATTERN_SINGLE.finditer(markdown_content):
            config_str = match.group(1)
            if debug:
                print(f"    [debug] SINGLE pattern matched, config starts: {config_str[:50]}...")
            try:
                config = json.loads(config_str)
                charts.append({
                    'match': match.group(0),
                    'config': config,
                    'height': 300,
                    'chart_id': config.get('title', f'chart_{len(charts)}').replace(' ', '_')
                })
            except json.JSONDecodeError as e:
                if debug:
                    print(f"    [debug] JSON parse error: {e}, string: {config_str[:100]}")
                continue
    
    # Try double-quote pattern (HTML entities encoded)
    if not charts:
        for match in CHART_DIV_PATTERN_DOUBLE.finditer(markdown_content):
            config_str = match.group(1)
            # Decode HTML entities
            config_str = config_str.replace('&quot;', '"').replace('&amp;', '&')
            if debug:
                print(f"    [debug] DOUBLE pattern matched, config starts: {config_str[:50]}...")
            try:
                config = json.loads(config_str)
                charts.append({
                    'match': match.group(0),
                    'config': config,
                    'height': 300,
                    'chart_id': config.get('title', f'chart_{len(charts)}').replace(' ', '_')
                })
            except json.JSONDecodeError as e:
                if debug:
                    print(f"    [debug] JSON parse error: {e}, string: {config_str[:100]}")
                continue
    
    return charts


def render_chart_to_png(config: dict, output_path: Path, width: int = 600, height: int = 300, background: str = '#ffffff'):
    """
    Render an ECharts configuration to a PNG file using Playwright.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[error] Playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(2)
    
    # Generate HTML for this chart
    config_json = json.dumps(config)
    html_content = ECHARTS_HTML_TEMPLATE.format(
        config_json=config_json,
        width=width,
        height=height,
        background=background
    )
    
    # Create temp HTML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_html = Path(f.name)
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': width, 'height': height})
            page.goto(f'file://{temp_html}')
            
            # Wait for ECharts to render fully
            page.wait_for_timeout(1000)
            
            # Take full page screenshot
            page.screenshot(path=str(output_path), full_page=False)
            
            browser.close()
    finally:
        temp_html.unlink(missing_ok=True)


def render_all_charts(charts: list, assets_dir: Path, assets_rel_path: str = None, width: int = 600) -> dict:
    """
    Render all charts to PNG files.
    
    Args:
        charts: List of chart dicts from extract_charts_from_markdown
        assets_dir: Directory to save PNG files
        assets_rel_path: Relative path to use in markdown (default: assets_dir.name)
        width: Chart width in pixels
    
    Returns dict mapping original match string to image markdown.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    replacements = {}
    
    # Path to use in markdown image references
    rel_prefix = str(assets_rel_path) if assets_rel_path else assets_dir.name
    
    for i, chart in enumerate(charts):
        filename = f"chart_{i}.png"
        output_path = assets_dir / filename
        
        title = chart['config'].get('title', f'Chart {i+1}')
        height = chart.get('height', 300)
        chart_type = chart['config'].get('type', '')
        
        # Pie charts need extra width for the side legend and height for labels
        if chart_type == 'pie':
            chart_width = width + 250
            chart_height = max(height, 350)
        else:
            chart_width = width
            chart_height = height
        
        print(f"    Rendering: {title} ({chart_type}, {chart_width}x{chart_height})")
        render_chart_to_png(
            config=chart['config'],
            output_path=output_path,
            width=chart_width,
            height=chart_height,
            background='#ffffff'
        )
        
        # Create markdown image reference
        # Use simple alt text to avoid duplicate title in Word (chart already has title)
        img_path = f"{rel_prefix}/{filename}"
        chart_type_label = chart_type.replace('-', ' ').title() if chart_type else 'Chart'
        img_markdown = f"![{chart_type_label}]({img_path})"
        
        replacements[chart['match']] = img_markdown
    
    return replacements


def process_markdown(markdown_content: str, replacements: dict) -> str:
    """
    Replace chart divs with image references in markdown content.
    """
    result = markdown_content
    for original, replacement in replacements.items():
        result = result.replace(original, replacement)
    return result


def convert_to_docx(md_path: Path, docx_path: Path, resource_path: Path = None):
    """Convert Markdown to DOCX using pypandoc or system pandoc."""
    res_path = resource_path or md_path.parent
    try:
        import pypandoc
        pypandoc.convert_file(
            str(md_path),
            'docx',
            outputfile=str(docx_path),
            extra_args=['--standalone', f'--resource-path={res_path}']
        )
        return True
    except Exception as e:
        print(f"[info] pypandoc failed: {e}, trying system pandoc...")
    
    import subprocess
    try:
        subprocess.check_call([
            'pandoc', str(md_path), '-o', str(docx_path),
            '--standalone', f'--resource-path={res_path}'
        ])
        return True
    except FileNotFoundError:
        print("[error] pandoc not found. Install it or `pip install pypandoc`.")
        print("Download: https://pandoc.org/installing.html")
        return False


def convert_to_pdf(md_path: Path, pdf_path: Path, resource_path: Path = None):
    """
    Convert Markdown to PDF.
    
    Tries multiple methods in order:
    1. pandoc with xelatex
    2. pandoc with pdflatex
    3. weasyprint (via HTML intermediate)
    """
    import subprocess
    res_path = resource_path or md_path.parent
    
    # Try pandoc with different PDF engines
    pdf_engines = ['xelatex', 'pdflatex', 'lualatex']
    
    for engine in pdf_engines:
        try:
            import pypandoc
            pypandoc.convert_file(
                str(md_path),
                'pdf',
                outputfile=str(pdf_path),
                extra_args=[
                    '--standalone',
                    f'--resource-path={res_path}',
                    f'--pdf-engine={engine}',
                    '-V', 'geometry:margin=1in'
                ]
            )
            print(f"    PDF generated using {engine}")
            return True
        except Exception:
            pass
        
        try:
            subprocess.check_call([
                'pandoc', str(md_path), '-o', str(pdf_path),
                '--standalone', f'--resource-path={res_path}',
                f'--pdf-engine={engine}',
                '-V', 'geometry:margin=1in'
            ], stderr=subprocess.DEVNULL)
            print(f"    PDF generated using {engine}")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    
    # Fallback: weasyprint (HTML to PDF)
    print("[info] LaTeX not available, trying weasyprint...")
    try:
        from weasyprint import HTML, CSS
        
        # First convert MD to HTML
        html_temp = md_path.with_suffix('.temp.html')
        try:
            import pypandoc
            pypandoc.convert_file(
                str(md_path),
                'html',
                outputfile=str(html_temp),
                extra_args=[
                    '--standalone',
                    f'--resource-path={res_path}',
                    '--metadata', 'title=Report'
                ]
            )
        except Exception:
            subprocess.check_call([
                'pandoc', str(md_path), '-o', str(html_temp),
                '--standalone', f'--resource-path={res_path}',
                '--metadata', 'title=Report'
            ])
        
        # Convert HTML to PDF with weasyprint
        HTML(filename=str(html_temp), base_url=str(res_path)).write_pdf(
            str(pdf_path),
            stylesheets=[CSS(string='body { font-family: sans-serif; margin: 1in; }')]
        )
        html_temp.unlink(missing_ok=True)
        print("    PDF generated using weasyprint")
        return True
        
    except ImportError:
        print("[error] weasyprint not installed. Run: pip install weasyprint")
    except Exception as e:
        print(f"[error] weasyprint failed: {e}")
    
    print("[error] PDF conversion failed. Install one of:")
    print("    - texlive-xetex (apt install texlive-xetex)")
    print("    - weasyprint (pip install weasyprint)")
    return False


def convert_to_html(md_path: Path, html_path: Path, resource_path: Path = None):
    """Convert Markdown to standalone HTML."""
    res_path = resource_path or md_path.parent
    try:
        import pypandoc
        pypandoc.convert_file(
            str(md_path),
            'html',
            outputfile=str(html_path),
            extra_args=[
                '--standalone',
                '--self-contained',
                f'--resource-path={res_path}',
                '--metadata', 'title=Report'
            ]
        )
        return True
    except Exception as e:
        print(f"[info] pypandoc HTML failed: {e}, trying system pandoc...")
    
    import subprocess
    try:
        subprocess.check_call([
            'pandoc', str(md_path), '-o', str(html_path),
            '--standalone', '--self-contained',
            f'--resource-path={res_path}',
            '--metadata', 'title=Report'
        ])
        return True
    except FileNotFoundError:
        print("[error] pandoc not found for HTML conversion.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert Markdown with ECharts to DOCX/PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('markdown', help='Path to source Markdown file')
    parser.add_argument('--out', help='Base name for outputs (without extension)')
    parser.add_argument('--out-dir', help='Output directory for generated files (default: same as input)')
    parser.add_argument('--assets-dir', help='Directory for chart images (default: {out}_assets)')
    parser.add_argument('--assets-rel', help='Relative path to use in markdown for images (default: assets directory name)')
    parser.add_argument('--pdf', action='store_true', help='Also export to PDF')
    parser.add_argument('--html', action='store_true', help='Also export to HTML')
    parser.add_argument('--keep-md', action='store_true', help='Keep processed Markdown file')
    parser.add_argument('--width', type=int, default=600, help='Chart image width in pixels (default: 600)')
    parser.add_argument('--base-path', help='Base path for resolving relative image paths (default: markdown file directory)')
    parser.add_argument('--debug', action='store_true', help='Show debug info for chart detection')
    args = parser.parse_args()
    
    src_md = Path(args.markdown).resolve()
    if not src_md.exists():
        print(f"[error] Markdown file not found: {src_md}")
        sys.exit(1)
    
    # Setup output paths
    base = args.out if args.out else src_md.stem
    output_dir = Path(args.out_dir).resolve() if args.out_dir else src_md.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Assets directory - where chart PNGs are saved
    if args.assets_dir:
        assets_dir = Path(args.assets_dir).resolve()
    else:
        assets_dir = output_dir / f"{base}_assets"
    
    # Relative path for image references in markdown
    # This is what goes in ![title](HERE/chart.png)
    if args.assets_rel:
        assets_rel_path = args.assets_rel
    else:
        # Try to make it relative to output_dir
        try:
            assets_rel_path = assets_dir.relative_to(output_dir)
        except ValueError:
            # assets_dir is not under output_dir, use absolute
            assets_rel_path = assets_dir
    
    processed_md = output_dir / f"{base}_processed.md"
    out_docx = output_dir / f"{base}.docx"
    out_pdf = output_dir / f"{base}.pdf"
    out_html = output_dir / f"{base}.html"
    
    # Resource path for resolving relative images
    resource_path = Path(args.base_path).resolve() if args.base_path else src_md.parent
    
    # Read source markdown
    print(f"[1/4] Reading markdown: {src_md.name}")
    markdown_content = src_md.read_text(encoding='utf-8')
    
    # Extract charts
    print("[2/4] Extracting chart configurations...")
    charts = extract_charts_from_markdown(markdown_content, debug=args.debug)
    print(f"    Found {len(charts)} chart(s)")
    
    # Render charts to PNG
    if charts:
        print(f"[3/4] Rendering charts to PNG ({args.width}px width)...")
        print(f"    Assets directory: {assets_dir}")
        print(f"    Image path in markdown: {assets_rel_path}/")
        replacements = render_all_charts(charts, assets_dir, assets_rel_path=assets_rel_path, width=args.width)
        
        # Process markdown
        processed_content = process_markdown(markdown_content, replacements)
    else:
        print("[3/4] No charts found, using original markdown")
        processed_content = markdown_content
    
    # Write processed markdown
    processed_md.write_text(processed_content, encoding='utf-8')
    
    # Convert to output formats
    print("[4/4] Converting to output formats...")
    print(f"    Resource path: {resource_path}")
    
    success_docx = convert_to_docx(processed_md, out_docx, resource_path)
    success_pdf = False
    success_html = False
    
    if args.pdf:
        success_pdf = convert_to_pdf(processed_md, out_pdf, resource_path)
    
    if args.html:
        success_html = convert_to_html(processed_md, out_html, resource_path)
    
    # Cleanup
    if not args.keep_md:
        print("[cleanup] Removing intermediate files...")
        processed_md.unlink(missing_ok=True)
        if assets_dir.exists() and not args.keep_md:
            # Only remove assets if not keeping markdown
            # Actually keep assets for DOCX/PDF to reference
            pass
    
    # Summary
    print("\nDone!")
    print("Outputs:")
    if success_docx:
        print(f"  - Word (.docx): {out_docx}")
    if args.pdf and success_pdf:
        print(f"  - PDF: {out_pdf}")
    if args.html and success_html:
        print(f"  - HTML: {out_html}")
    if args.keep_md:
        print(f"  - Processed Markdown: {processed_md}")
    if charts:
        print(f"  - Chart images: {assets_dir}/")


if __name__ == '__main__':
    main()
