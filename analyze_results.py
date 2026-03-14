"""
The Interrupt Effect - Results Analyzer
Measures code quality metrics across conditions and models.

Metrics:
1. Output length (chars, lines)
2. Code structure (classes, functions, imports)
3. Python syntax validity (ast.parse)
4. Test count
5. Docstring coverage
6. Type hint usage
7. Error handling (try/except blocks)
8. Code duplication (within file)
9. Edit distance between runs (intra-condition variance)
10. Edit distance between conditions (inter-condition divergence)
"""

import os
import re
import ast
import json
import sys
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from statistics import mean, stdev, median

RESULTS_DIR = Path(__file__).parent / "results"
ANALYSIS_DIR = Path(__file__).parent / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


def extract_python_code(text: str) -> str:
    """Extract Python code blocks from markdown-formatted output."""
    # Try to find ```python ... ``` blocks
    blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return '\n\n'.join(blocks)
    
    # Try ``` ... ``` blocks
    blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return '\n\n'.join(blocks)
    
    # Fallback: return everything that looks like Python
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ', 'class ', 'def ', 'async ', '@', 'if ', 'for ', 'while ', 'try:', 'except', 'with ')):
            in_code = True
        if in_code:
            code_lines.append(line)
        if stripped == '' and in_code and len(code_lines) > 3:
            # Keep going through blank lines within code
            pass
    
    return '\n'.join(code_lines) if code_lines else text


def analyze_code(code: str) -> dict:
    """Analyze a piece of Python code and return metrics."""
    metrics = {}
    
    # Basic metrics
    lines = code.split('\n')
    metrics['total_chars'] = len(code)
    metrics['total_lines'] = len(lines)
    metrics['non_empty_lines'] = sum(1 for l in lines if l.strip())
    metrics['comment_lines'] = sum(1 for l in lines if l.strip().startswith('#'))
    
    # Syntax validity
    try:
        tree = ast.parse(code)
        metrics['syntax_valid'] = True
        metrics['parse_error'] = None
    except SyntaxError as e:
        metrics['syntax_valid'] = False
        metrics['parse_error'] = str(e)
        # Still try to extract what we can
        tree = None
    
    # AST-based metrics (only if parseable)
    if tree:
        metrics['num_classes'] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        metrics['num_functions'] = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        metrics['num_imports'] = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
        
        # Docstrings
        docstring_count = 0
        total_defs = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                total_defs += 1
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                    docstring_count += 1
        metrics['docstring_count'] = docstring_count
        metrics['total_definitions'] = total_defs
        metrics['docstring_ratio'] = docstring_count / total_defs if total_defs > 0 else 0
        
        # Type hints
        type_hint_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns:
                    type_hint_count += 1
                for arg in node.args.args:
                    if arg.annotation:
                        type_hint_count += 1
        metrics['type_hint_count'] = type_hint_count
        
        # Error handling
        metrics['try_except_blocks'] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))
        
        # Async usage
        metrics['async_functions'] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef))
        metrics['await_expressions'] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Await))
        
        # Assertions (test-related)
        metrics['assert_count'] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))
        
        # Decorators
        decorator_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorator_count += len(node.decorator_list)
        metrics['decorator_count'] = decorator_count
    else:
        for key in ['num_classes', 'num_functions', 'num_imports', 'docstring_count',
                     'total_definitions', 'docstring_ratio', 'type_hint_count',
                     'try_except_blocks', 'async_functions', 'await_expressions',
                     'assert_count', 'decorator_count']:
            metrics[key] = 0
    
    # Regex-based metrics (work even if AST fails)
    metrics['pytest_markers'] = len(re.findall(r'@pytest\.mark\.', code))
    metrics['test_functions'] = len(re.findall(r'(?:def|async def)\s+test_', code))
    metrics['dataclass_usage'] = len(re.findall(r'@dataclass', code))
    metrics['enum_usage'] = len(re.findall(r'class\s+\w+\(.*Enum.*\)', code))
    
    # Code duplication (self-similarity of code blocks)
    if metrics['non_empty_lines'] > 10:
        chunks = ['\n'.join(lines[i:i+5]) for i in range(0, len(lines)-5, 5)]
        if len(chunks) > 2:
            similarities = []
            for i in range(len(chunks)):
                for j in range(i+1, len(chunks)):
                    sim = SequenceMatcher(None, chunks[i], chunks[j]).ratio()
                    if sim > 0.8:  # High similarity threshold
                        similarities.append(sim)
            metrics['duplicate_chunks'] = len(similarities)
            metrics['duplication_ratio'] = len(similarities) / (len(chunks) * (len(chunks)-1) / 2) if len(chunks) > 1 else 0
        else:
            metrics['duplicate_chunks'] = 0
            metrics['duplication_ratio'] = 0
    else:
        metrics['duplicate_chunks'] = 0
        metrics['duplication_ratio'] = 0
    
    return metrics


def compute_similarity(code1: str, code2: str) -> float:
    """Compute similarity between two code outputs."""
    return SequenceMatcher(None, code1, code2).ratio()


def load_results() -> dict:
    """Load all experiment results organized by model/condition."""
    results = defaultdict(lambda: defaultdict(list))
    
    for dir_entry in RESULTS_DIR.iterdir():
        if not dir_entry.is_dir():
            continue
        
        dir_name = dir_entry.name  # e.g., "opus_complete", "sonnet_interrupted"
        parts = dir_name.split('_', 1)
        if len(parts) != 2:
            continue
        
        model_short = parts[0]
        condition = parts[1]
        
        for txt_file in sorted(dir_entry.glob("run_*.txt")):
            if "_meta" in txt_file.name or "_fragment" in txt_file.name:
                continue
            
            raw_output = txt_file.read_text(encoding='utf-8', errors='replace')
            code = extract_python_code(raw_output)
            
            # Load metadata if exists
            meta_file = txt_file.with_name(txt_file.stem + "_meta.json")
            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding='utf-8'))
                except:
                    pass
            
            run_index = re.search(r'run_(\d+)', txt_file.name)
            run_idx = int(run_index.group(1)) if run_index else 0
            
            results[model_short][condition].append({
                'run_index': run_idx,
                'raw_output': raw_output,
                'code': code,
                'meta': meta,
                'file': str(txt_file)
            })
    
    return dict(results)


def analyze_all(results: dict) -> dict:
    """Run full analysis on all results."""
    analysis = {}
    
    for model, conditions in results.items():
        analysis[model] = {}
        
        for condition, runs in conditions.items():
            if not runs:
                continue
            
            # Analyze each run
            run_metrics = []
            codes = []
            for run in runs:
                metrics = analyze_code(run['code'])
                metrics['run_index'] = run['run_index']
                if run['meta']:
                    metrics['duration_seconds'] = run['meta'].get('duration_seconds', 0)
                run_metrics.append(metrics)
                codes.append(run['code'])
            
            # Aggregate metrics
            agg = {}
            numeric_keys = [k for k in run_metrics[0].keys() 
                          if isinstance(run_metrics[0][k], (int, float)) and k != 'run_index']
            
            for key in numeric_keys:
                values = [m[key] for m in run_metrics if m[key] is not None]
                if values:
                    agg[key] = {
                        'mean': round(mean(values), 3),
                        'median': round(median(values), 3),
                        'stdev': round(stdev(values), 3) if len(values) > 1 else 0,
                        'min': round(min(values), 3),
                        'max': round(max(values), 3),
                        'values': [round(v, 3) for v in values]
                    }
            
            # Validity rate
            valid_count = sum(1 for m in run_metrics if m.get('syntax_valid', False))
            agg['syntax_valid_rate'] = round(valid_count / len(run_metrics), 3) if run_metrics else 0
            
            # Intra-condition variance (how similar are runs within the same condition)
            if len(codes) > 1:
                sims = []
                for i in range(len(codes)):
                    for j in range(i+1, len(codes)):
                        sims.append(compute_similarity(codes[i], codes[j]))
                agg['intra_condition_similarity'] = {
                    'mean': round(mean(sims), 4),
                    'stdev': round(stdev(sims), 4) if len(sims) > 1 else 0,
                    'min': round(min(sims), 4),
                    'max': round(max(sims), 4)
                }
            
            analysis[model][condition] = {
                'n_runs': len(runs),
                'individual_runs': run_metrics,
                'aggregated': agg
            }
        
        # Cross-condition comparisons
        condition_codes = {}
        for condition, runs in conditions.items():
            condition_codes[condition] = [r['code'] for r in runs]
        
        cross = {}
        cond_list = list(condition_codes.keys())
        for i in range(len(cond_list)):
            for j in range(i+1, len(cond_list)):
                c1, c2 = cond_list[i], cond_list[j]
                sims = []
                for code1 in condition_codes[c1]:
                    for code2 in condition_codes[c2]:
                        sims.append(compute_similarity(code1, code2))
                if sims:
                    cross[f"{c1}_vs_{c2}"] = {
                        'mean_similarity': round(mean(sims), 4),
                        'stdev': round(stdev(sims), 4) if len(sims) > 1 else 0,
                        'min': round(min(sims), 4),
                        'max': round(max(sims), 4)
                    }
        
        analysis[model]['cross_condition_similarity'] = cross
    
    return analysis


def generate_report(analysis: dict) -> str:
    """Generate a human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("THE INTERRUPT EFFECT - Analysis Report")
    lines.append("=" * 70)
    lines.append("")
    
    for model in sorted(analysis.keys()):
        if model == 'cross_condition_similarity':
            continue
        
        lines.append(f"\n{'='*50}")
        lines.append(f"MODEL: {model.upper()}")
        lines.append(f"{'='*50}")
        
        model_data = analysis[model]
        conditions = [c for c in model_data.keys() if c != 'cross_condition_similarity']
        
        # Summary table
        lines.append(f"\n{'Metric':<30} ", )
        header = f"{'Metric':<30}"
        for c in sorted(conditions):
            header += f" | {c:>20}"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Key metrics to compare
        key_metrics = [
            'total_lines', 'non_empty_lines', 'num_classes', 'num_functions',
            'num_imports', 'test_functions', 'docstring_ratio', 'type_hint_count',
            'try_except_blocks', 'async_functions', 'syntax_valid_rate',
            'duplicate_chunks', 'duration_seconds'
        ]
        
        for metric in key_metrics:
            row = f"{metric:<30}"
            for c in sorted(conditions):
                cond_data = model_data.get(c, {})
                agg = cond_data.get('aggregated', {})
                
                if metric == 'syntax_valid_rate':
                    val = agg.get(metric, 'N/A')
                    row += f" | {val:>20}"
                elif metric in agg:
                    m = agg[metric]
                    val = f"{m['mean']:.1f} (+/-{m['stdev']:.1f})"
                    row += f" | {val:>20}"
                else:
                    row += f" | {'N/A':>20}"
            lines.append(row)
        
        # Cross-condition similarities
        cross = model_data.get('cross_condition_similarity', {})
        if cross:
            lines.append(f"\nCross-Condition Similarity:")
            for comp, data in cross.items():
                lines.append(f"  {comp}: {data['mean_similarity']:.4f} (stdev: {data['stdev']:.4f})")
        
        # Intra-condition similarities
        lines.append(f"\nIntra-Condition Similarity (how consistent is each condition):")
        for c in sorted(conditions):
            cond_data = model_data.get(c, {})
            agg = cond_data.get('aggregated', {})
            intra = agg.get('intra_condition_similarity', {})
            if intra:
                lines.append(f"  {c}: {intra['mean']:.4f} (stdev: {intra['stdev']:.4f})")
    
    # Key findings
    lines.append(f"\n\n{'='*70}")
    lines.append("KEY FINDINGS")
    lines.append(f"{'='*70}\n")
    
    for model in sorted(analysis.keys()):
        model_data = analysis[model]
        conditions = [c for c in model_data.keys() if c != 'cross_condition_similarity']
        
        lines.append(f"\n--- {model.upper()} ---")
        
        # Compare complete vs interrupted
        complete = model_data.get('complete', {}).get('aggregated', {})
        interrupted = model_data.get('interrupted', {}).get('aggregated', {})
        fragment = model_data.get('fragment_continue', {}).get('aggregated', {})
        
        if complete and interrupted:
            lines.append("\nCOMPLETE vs INTERRUPTED (re-prompted from scratch):")
            for metric in ['total_lines', 'num_functions', 'test_functions', 'type_hint_count', 'docstring_ratio']:
                c_val = complete.get(metric, {}).get('mean', 0)
                i_val = interrupted.get(metric, {}).get('mean', 0)
                if c_val > 0:
                    diff_pct = ((i_val - c_val) / c_val) * 100
                    direction = "MORE" if diff_pct > 0 else "LESS"
                    lines.append(f"  {metric}: {c_val:.1f} -> {i_val:.1f} ({diff_pct:+.1f}% {direction})")
        
        if complete and fragment:
            lines.append("\nCOMPLETE vs FRAGMENT+CONTINUE:")
            for metric in ['total_lines', 'num_functions', 'test_functions', 'syntax_valid_rate']:
                c_val = complete.get(metric, {}).get('mean', 0) if metric != 'syntax_valid_rate' else complete.get(metric, 0)
                f_val = fragment.get(metric, {}).get('mean', 0) if metric != 'syntax_valid_rate' else fragment.get(metric, 0)
                if isinstance(c_val, dict):
                    c_val = c_val.get('mean', 0)
                if isinstance(f_val, dict):
                    f_val = f_val.get('mean', 0)
                lines.append(f"  {metric}: {c_val:.3f} -> {f_val:.3f}")
    
    return '\n'.join(lines)


def main():
    print("Loading results...")
    results = load_results()
    
    if not results:
        print("ERROR: No results found in", RESULTS_DIR)
        print("Run the experiment first: .\\run_experiment.ps1")
        sys.exit(1)
    
    print(f"Found data for models: {list(results.keys())}")
    for model, conditions in results.items():
        for cond, runs in conditions.items():
            print(f"  {model}/{cond}: {len(runs)} runs")
    
    print("\nAnalyzing...")
    analysis = analyze_all(results)
    
    # Save raw analysis
    analysis_file = ANALYSIS_DIR / "full_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Full analysis saved to: {analysis_file}")
    
    # Generate report
    report = generate_report(analysis)
    report_file = ANALYSIS_DIR / "report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_file}")
    
    # Print report
    print("\n")
    print(report)


if __name__ == "__main__":
    main()
