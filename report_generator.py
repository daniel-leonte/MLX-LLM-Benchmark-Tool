"""
Benchmark Report Generator
Handles HTML report generation using template files
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


class BenchmarkReportGenerator:
    """Generates HTML reports from benchmark data using template files"""
    
    def __init__(self, templates_dir: str = "templates", results_dir: str = "results"):
        self.templates_dir = Path(templates_dir)
        self.results_dir = Path(results_dir)
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        # Template file paths
        self.html_template = self.templates_dir / "report.html"
        self.css_template = self.templates_dir / "report.css"
        self.js_template = self.templates_dir / "report.js"
        
        # Validate template files exist
        self._validate_templates()
    
    def _validate_templates(self) -> None:
        """Validate that all required template files exist"""
        required_files = [self.html_template, self.css_template, self.js_template]
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing template files: {missing_files}")
    
    def _read_template(self, template_path: Path) -> str:
        """Read template file content"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading template {template_path}: {e}")
    
    def _copy_static_files(self) -> None:
        """Copy CSS and JS files to results directory"""
        try:
            # Copy CSS file
            shutil.copy2(self.css_template, self.results_dir / "report.css")
            
            # Copy JS file
            shutil.copy2(self.js_template, self.results_dir / "report.js")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not copy static files: {e}")
    
    def _generate_runs_badges(self, runs_info: Dict[str, Any]) -> str:
        """Generate HTML for run badges"""
        if not runs_info:
            return ""
        
        badges_html = '<p><strong>Runs:</strong> '
        for run_id, info in runs_info.items():
            timestamp = info['timestamp'][:19] if len(info['timestamp']) > 19 else info['timestamp']
            badges_html += f'<span class="run-badge">{timestamp} ({info["count"]} models)</span>'
        badges_html += '</p>'
        
        return badges_html
    
    def _generate_table_rows(self, df: pd.DataFrame) -> str:
        """Generate HTML table rows for the summary table"""
        rows_html = ""
        
        for idx, (_, row) in enumerate(df.iterrows()):
            error_display = row.get('error', '') if not pd.isna(row.get('error', '')) else ''
            run_date = row.get('run_timestamp', 'Unknown')[:19] if pd.notna(row.get('run_timestamp', '')) else 'Unknown'
            run_id = row.get('run_id', 'unknown')
            
            rows_html += f"""
            <tr class="model-row" id="table-row-{idx}" data-model-name="{row['model_name']}" data-run-id="{run_id}">
                <td>{row['model_name']}</td>
                <td>{row['framework']}</td>
                <td>{row['load_time']:.2f}</td>
                <td>{row['peak_memory_mb']:.1f}</td>
                <td>{row['tokens_per_sec']:.2f}</td>
                <td>{row['similarity_score']:.3f}</td>
                <td>{run_date}</td>
                <td>{error_display}</td>
                <td>
                    <button class="table-remove-btn" onclick="removeModel('{idx}', '{row['model_name']}', '{run_id}')" title="Remove this model result">×</button>
                </td>
            </tr>"""
        
        return rows_html
    
    def _generate_empty_state_row(self, df: pd.DataFrame) -> str:
        """Generate empty state row when no results are available"""
        if len(df) == 0:
            return '''
            <tr>
                <td colspan="9" style="text-align: center; padding: 40px; color: #666; font-style: italic;">
                    📭 No benchmark results found. Run some benchmarks to see results here!
                </td>
            </tr>'''
        return ""
    
    def _generate_detailed_sections(self, df: pd.DataFrame) -> str:
        """Generate HTML for detailed model sections"""
        sections_html = ""
        
        for idx, (_, row) in enumerate(df.iterrows()):
            output_file = row.get('output_file', '')
            run_date = row.get('run_timestamp', 'Unknown')[:19] if pd.notna(row.get('run_timestamp', '')) else 'Unknown'
            run_id = row.get('run_id', 'unknown')
            
            # Handle NaN values
            if pd.isna(output_file):
                output_file = ''
            
            if output_file and isinstance(output_file, str) and os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract just the response part
                        if "FULL RESPONSE:" in content:
                            response = content.split("FULL RESPONSE:\n" + "-" * 80 + "\n")[1]
                        else:
                            response = content
                    
                    sections_html += f"""
    <div class="model-section" id="model-{idx}" data-model-name="{row['model_name']}" data-run-id="{run_id}">
        <div class="model-header">
            {row['model_name']}
            <div class="run-info">Run: {run_date}</div>
            <button class="remove-btn" onclick="removeModel('{idx}', '{row['model_name']}', '{run_id}')" title="Remove this model result">×</button>
        </div>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{row['tokens_per_sec']:.1f}</div>
                <div class="metric-label">Tokens/sec</div>
            </div>
            <div class="metric">
                <div class="metric-value">{row['tokens_generated']}</div>
                <div class="metric-label">Tokens</div>
            </div>
            <div class="metric">
                <div class="metric-value">{row['load_time']:.2f}s</div>
                <div class="metric-label">Load Time</div>
            </div>
            <div class="metric">
                <div class="metric-value">{row['peak_memory_mb']:.0f}MB</div>
                <div class="metric-label">Memory</div>
            </div>
            <div class="metric">
                <div class="metric-value">{row['similarity_score']:.3f}</div>
                <div class="metric-label">Similarity</div>
            </div>
        </div>
        <div class="output">{response}</div>
    </div>"""
                except Exception as e:
                    print(f"⚠️  Could not read {output_file}: {e}")
        
        return sections_html
    
    def _generate_empty_state_details(self, df: pd.DataFrame) -> str:
        """Generate empty state message for detailed sections when no results are available"""
        if len(df) == 0:
            return '''
    <div style="text-align: center; padding: 60px; color: #666; background-color: #f9f9f9; border-radius: 8px; margin: 20px 0;">
        <h3 style="color: #999; margin-bottom: 20px;">📭 No Detailed Results Available</h3>
        <p style="font-size: 1.1em; margin-bottom: 15px;">All benchmark results have been removed.</p>
        <p style="font-size: 0.9em; color: #888;">Run <code>python benchmark.py --categories medium</code> to generate new results.</p>
    </div>'''
        return ""
    
    def _collect_runs_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Collect information about benchmark runs"""
        runs_info = {}
        
        if 'run_timestamp' in df.columns:
            for _, row in df.iterrows():
                run_id = row.get('run_id', 'unknown')
                if run_id not in runs_info:
                    runs_info[run_id] = {
                        'timestamp': row.get('run_timestamp', 'Unknown'),
                        'count': 0
                    }
                runs_info[run_id]['count'] += 1
        
        return runs_info
    
    def generate_report(self, df: pd.DataFrame, custom_prompt: str, 
                       system_prompt_description: str = "PromptCraft Architect (Technical Prompt Refinement)") -> str:
        """Generate the HTML report from benchmark data"""
        
        # Collect run information
        runs_info = self._collect_runs_info(df)
        
        # Prepare template data
        template_data = {
            'TITLE': 'MLX LLM Benchmark Results',
            'PROMPT': custom_prompt,
            'SYSTEM_PROMPT_DESCRIPTION': system_prompt_description,
            'TOTAL_RESULTS': str(len(df)),
            'RUNS_COUNT': str(len(runs_info)),
            'RUNS_BADGES': self._generate_runs_badges(runs_info),
            'RESULTS_TABLE_ROWS': self._generate_table_rows(df),
            'EMPTY_STATE_ROW': self._generate_empty_state_row(df),
            'DETAILED_SECTIONS': self._generate_detailed_sections(df),
            'EMPTY_STATE_DETAILS': self._generate_empty_state_details(df)
        }
        
        # Read HTML template
        html_content = self._read_template(self.html_template)
        
        # Replace placeholders with actual data
        for placeholder, value in template_data.items():
            html_content = html_content.replace(f'{{{{{placeholder}}}}}', str(value))
        
        # Copy static files (CSS and JS)
        self._copy_static_files()
        
        # Write final HTML file
        output_file = self.results_dir / "benchmark_report.html"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"🌐 HTML report generated: {output_file}")
            return str(output_file)
        except Exception as e:
            raise RuntimeError(f"Error writing HTML report: {e}")
    
    def cleanup_old_reports(self) -> None:
        """Clean up old report files"""
        old_files = [
            self.results_dir / "benchmark_report.html",
            self.results_dir / "report.css",
            self.results_dir / "report.js"
        ]
        
        for file in old_files:
            if file.exists():
                try:
                    file.unlink()
                    print(f"🗑️  Removed old report file: {file}")
                except Exception as e:
                    print(f"⚠️  Could not remove {file}: {e}")


def create_report_generator(templates_dir: str = "templates", results_dir: str = "results") -> BenchmarkReportGenerator:
    """Factory function to create a report generator instance"""
    return BenchmarkReportGenerator(templates_dir, results_dir) 