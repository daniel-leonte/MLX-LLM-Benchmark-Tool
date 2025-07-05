#!/usr/bin/env python3
"""
LLM Benchmarking Script for MacBook M1 Pro
Supports MPS/Metal, ONNX, and llama.cpp acceleration methods
"""

import os
import sys
import time
import gc
import psutil
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

# Set environment variables for optimal performance
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore")

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration
SYSTEM_PROMPT = """You are PromptCraft Architect, an AI specializing in refactoring user inputs into precise, structured, plain-text prompts for other advanced LLMs. Your focus is on a wide range of technical tasks for developers, from creating entire applications to fixing small bugs.

Core Mission: Convert any developer request into a superior, plain-text prompt that clarifies intent and elicits the best possible technical response from an LLM, without inventing details.

Guiding Principles:
Precision is Paramount: Eliminate ambiguity. Be explicit.
Context is Key, Not Assumed: Structure the user's provided context. Do not invent a tech stack or add technical details the user did not provide or clearly imply.
Structure for Clarity: Use capitalized headers, lists, and line breaks to create a logical, easy-to-follow request.
Adapt to Scope: Your output structure must fit the task, whether it's an end-to-end solution for an application or feature, a single function, or a debugging request.

Execution Workflow

1. Input Interpretation:
Treat the entire user input as a draft prompt to rewrite.
NEVER engage conversationally. Your sole function is prompt refinement.

2. The Refactoring Blueprint:
Construct the optimized prompt using these steps:

A. ESTABLISH PERSONA:
Begin with "Act as..." defining a relevant technical expert. If the tech stack is ambiguous, use a generalist persona like "Senior Software Engineer".

B. CLARIFY SCOPE & CONTEXT:
Analyze the input for what is known and what is missing.
Explicitly state the known technologies. If a critical detail like programming language is missing, frame the request to be language-agnostic or use a placeholder like [Specify Language] to guide the end-user.
Crucially, do not add assumptions. If the user asks for a "database script" without specifying the database, do not add "PostgreSQL." Frame the prompt around "a generic SQL script."

C. ENFORCE ACTIONABLE STRUCTURE:
Transform the request into a direct set of instructions or requirements.
For creation tasks, detail what needs to be built.
For debugging/refactoring tasks, clearly present the problematic code and the desired change or outcome.

D. ADD GENERAL BEST PRACTICES:
Where appropriate, incorporate general, non-stack-specific constraints like "Ensure the code is well-commented," "Consider security best practices," or "Optimize for readability."

E. DEFINE CONCRETE GOAL:
Conclude with GOAL: - a clear, one-sentence summary of the user's intended outcome.

3. Output Rules (Non-Negotiable):
Your output MUST BE the optimized prompt exclusively.
The entire output prompt must be plain text. Do not use markdown characters.
NO preambles, apologies, or meta-commentary."""

CUSTOM_PROMPT = "build a react app like facebook"

GOLD_ANSWER = """Act as a Senior Software Engineer specializing in React.

CONTEXT:
The user wants to create a React application similar to Facebook.

INSTRUCTIONS:
Develop a React application with the following features:
  - User authentication (registration, login, logout).
  - User profiles (displaying user information).
  - Friend requests and management.
  - Posting and viewing of text-based status updates.
  - Basic news feed displaying posts from friends.

Ensure the application is responsive and has a clean UI.
Consider state management (e.g., using Context, Redux, or a similar library).
Implement a basic backend (can be mocked or a simple API) for user data, posts, and friend connections.

GOAL: Create a functional React application that mimics key features of Facebook."""

MAX_NEW_TOKENS = 800  # Allow full technical responses for complex prompts

# MLX Community model configurations: (model_name, special_params)
# No need for manual chat templates - MLX models have built-in templates
MODELS = [
    ("mlx-community/Phi-3-mini-4k-instruct-4bit", {"repetition_penalty": 1.1}),
    ("mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit", {}),
    ("mlx-community/Meta-Llama-3-8B-Instruct-4bit", {}),
    # Add any MLX model without worrying about chat templates!
    ("mlx-community/Qwen2-1.5B-Instruct-4bit", {}),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", {}),
]

def install_dependencies():
    """Install required dependencies if not available"""
    required_packages = [
        "mlx-lm",
        "sentence-transformers",
        "pandas",
        "numpy",
        "psutil",
        "tqdm"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "mlx-lm":
                import mlx_lm
            elif package == "sentence-transformers":
                import sentence_transformers
            elif package == "pandas":
                import pandas
            elif package == "numpy":
                import numpy
            elif package == "psutil":
                import psutil
            elif package == "tqdm":
                import tqdm
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install dependencies first
install_dependencies()

# Import after installation
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
import mlx.core as mx

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def clear_memory():
    """Clear memory and force garbage collection"""
    gc.collect()
    # MLX automatically manages memory on Apple Silicon

def compute_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Compute cosine similarity between two texts"""
    try:
        embeddings = model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0

def load_mlx_model(model_name: str) -> Tuple[Any, Any, str]:
    """Load MLX model and tokenizer"""
    try:
        print(f"Loading MLX model: {model_name}")
        model, tokenizer = load(model_name)
        return model, tokenizer, "mlx"
    except Exception as e:
        print(f"Error loading MLX model {model_name}: {e}")
        return None, None, "mlx"

def benchmark_model(model_name: str, special_params: Dict[str, Any], similarity_model: Optional[SentenceTransformer] = None) -> Dict[str, Any]:
    """Benchmark a single MLX model"""
    print(f"\nBenchmarking {model_name} (MLX)")
    
    results = {
        "model_name": model_name,
        "framework": "mlx",
        "load_time": 0.0,
        "peak_memory_mb": 0.0,
        "inference_time": 0.0,
        "tokens_generated": 0,
        "tokens_per_sec": 0.0,
        "similarity_score": 0.0,
        "generated_text": "",
        "output_file": "",
        "error": None
    }
    
    try:
        # Measure initial memory
        initial_memory = get_memory_usage()
        
        # Load MLX model
        start_time = time.perf_counter()
        
        model, tokenizer, _ = load_mlx_model(model_name)
        
        if model is None:
            results["error"] = "Failed to load model"
            return results
        
        load_time = time.perf_counter() - start_time
        peak_memory = get_memory_usage()
        
        results["load_time"] = load_time
        results["peak_memory_mb"] = peak_memory - initial_memory
        
        # Run inference
        start_time = time.perf_counter()
        
        # Use tokenizer's built-in chat template for consistent formatting
        try:
            # Mistral models prefer user-first conversations without separate system roles
            if "mistral" in model_name.lower():
                # Combine system prompt with user prompt for Mistral models
                combined_prompt = f"{SYSTEM_PROMPT}\n\n{CUSTOM_PROMPT}"
                messages = [
                    {"role": "user", "content": combined_prompt}
                ]
                print(f"🔧 Using Mistral-optimized format (user-first)")
            else:
                # Standard system + user format for other models
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": CUSTOM_PROMPT}
                ]
            
            # Apply the model's chat template
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                print(f"✅ Using model's built-in chat template")
            else:
                # Fallback to simple format if no chat template
                formatted_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {CUSTOM_PROMPT}\n\nAssistant:"
                print(f"⚠️  No chat template found, using fallback format")
        
        except Exception as e:
            # Ultimate fallback
            formatted_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {CUSTOM_PROMPT}\n\nAssistant:"
            print(f"⚠️  Chat template error ({e}), using fallback format")
        
        # MLX inference - using simpler parameters
        response = generate(
            model, 
            tokenizer, 
            formatted_prompt,
            max_tokens=MAX_NEW_TOKENS,
            verbose=False
        )
        
        # Clean response by removing the input prompt
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):].strip()
        else:
            # If the response doesn't start with our prompt, it might be already clean
            # This can happen with some chat templates
            response = response.strip()
        
        # Count tokens in the response
        tokens_generated = len(tokenizer.encode(response)) if response else 0
        
        inference_time = time.perf_counter() - start_time
        tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0
        
        results["inference_time"] = inference_time
        results["tokens_generated"] = tokens_generated
        results["tokens_per_sec"] = tokens_per_sec
        # Save full response to individual file and store truncated version in CSV
        output_filename = RESULTS_DIR / f"output_{model_name.replace('/', '_').replace('-', '_')}.txt"
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Prompt: {CUSTOM_PROMPT}\n")
                f.write(f"Tokens Generated: {tokens_generated}\n")
                f.write(f"Tokens/sec: {tokens_per_sec:.2f}\n")
                f.write(f"Inference Time: {inference_time:.2f}s\n")
                f.write("-" * 80 + "\n")
                f.write("FULL RESPONSE:\n")
                f.write("-" * 80 + "\n")
                f.write(response)
            print(f"📝 Full output saved to: {output_filename}")
        except Exception as e:
            print(f"⚠️  Could not save output file: {e}")
        
        results["generated_text"] = response[:200]  # Keep CSV manageable
        results["output_file"] = str(output_filename)
        
        # Check for meaningful output
        if tokens_generated < 5:
            print(f"⚠️  Warning: Only {tokens_generated} tokens generated - may indicate prompt formatting issue")
        
        # Compute similarity if available
        if similarity_model and response and len(response.strip()) > 0:
            similarity_score = compute_similarity(response, GOLD_ANSWER, similarity_model)
            results["similarity_score"] = similarity_score
        
        print(f"Generated: {response[:200]}...")
        print(f"Tokens/sec: {tokens_per_sec:.2f}, Tokens: {tokens_generated}")
        if len(response) > 200:
            print(f"[Response continues for {len(response)} total characters]")
        
        # Clean up model
        del model
        if tokenizer:
            del tokenizer
        clear_memory()
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        results["error"] = str(e)
        clear_memory()
    
    return results

def generate_html_report(df: pd.DataFrame, output_files: List[str]):
    """Generate an HTML report with full outputs"""
    
    # Create HTML header with embedded CSS
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>MLX LLM Benchmark Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .model-section {{ margin: 30px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .model-header {{ background-color: #e8f4f8; padding: 15px; font-weight: bold; }}
        .metrics {{ display: flex; gap: 20px; padding: 15px; background-color: #f9f9f9; }}
        .metric {{ text-align: center; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #2196F3; }}
        .metric-label {{ font-size: 0.9em; color: #666; }}
        .output {{ padding: 20px; font-family: monospace; background-color: #f8f8f8; 
                 white-space: pre-wrap; border-top: 1px solid #ddd; }}
        .summary-table {{ width: 100%%; border-collapse: collapse; margin: 20px 0; }}
        .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .summary-table th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 MLX LLM Benchmark Results</h1>
        <p><strong>🍎 Apple Silicon Optimized</strong> | PromptCraft Architect Test</p>
        <p><strong>Prompt:</strong> "{CUSTOM_PROMPT}"</p>
        <p><strong>System Prompt:</strong> PromptCraft Architect (Technical Prompt Refinement)</p>
    </div>
    
    <h2>📊 Summary Table</h2>
    {df.to_html(classes="summary-table", escape=False, index=False)}
    
    <h2>📝 Detailed Outputs</h2>
"""
    
    # Add detailed sections for each model
    for _, row in df.iterrows():
        output_file = row.get('output_file', '')
        # Handle NaN values (pandas converts None to NaN which is a float)
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
                
                html_content += f"""
    <div class="model-section">
        <div class="model-header">{row['model_name']}</div>
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
    </div>
"""
            except Exception as e:
                print(f"Could not read {output_file}: {e}")
    
    html_content += """
</body>
</html>
"""
    
    html_file = RESULTS_DIR / "benchmark_report.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"🌐 HTML report generated: {html_file}")

def main():
    """Main benchmarking function"""
    print("🚀 Starting MLX LLM Benchmarking on MacBook M1 Pro")
    print("🍎 Using MLX (Apple Silicon optimized)")
    print(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # Initialize similarity model
    print("\nLoading similarity model...")
    try:
        similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Similarity model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load similarity model: {e}")
        similarity_model = None
    
    # Run benchmarks
    all_results = []
    
    for i, (model_name, special_params) in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Starting benchmark...")
        results = benchmark_model(model_name, special_params, similarity_model)
        all_results.append(results)
        
        # Print intermediate results
        if results.get("error") is None:
            print(f"✅ {model_name}: {results['tokens_per_sec']:.2f} tokens/sec, "
                  f"Load: {results['load_time']:.2f}s, "
                  f"Memory: {results['peak_memory_mb']:.1f}MB")
        else:
            print(f"❌ {model_name}: {results['error']}")
    
    # Create DataFrame and save results
    df = pd.DataFrame(all_results)
    
    # Display results
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    # Create a summary table for display
    display_columns = [
        'model_name', 'framework', 'load_time', 
        'peak_memory_mb', 'tokens_per_sec', 'similarity_score', 'error'
    ]
    
    display_df = df[display_columns].copy()
    display_df['load_time'] = display_df['load_time'].round(2)
    display_df['peak_memory_mb'] = display_df['peak_memory_mb'].round(1)
    display_df['tokens_per_sec'] = display_df['tokens_per_sec'].round(2)
    display_df['similarity_score'] = display_df['similarity_score'].round(3)
    
    print(display_df.to_string(index=False))
    
    # Save to CSV
    output_file = RESULTS_DIR / "benchmark_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n📊 Results saved to {output_file}")
    
    # Generate HTML report
    output_files = [row.get('output_file', '') if not pd.isna(row.get('output_file', '')) else '' for _, row in df.iterrows()]
    generate_html_report(df, output_files)
    
    # Summary statistics
    successful_runs = df[df['error'].isna()]
    if not successful_runs.empty:
        print(f"\n📈 SUMMARY:")
        print(f"Successful runs: {len(successful_runs)}/{len(df)}")
        print(f"Average tokens/sec: {successful_runs['tokens_per_sec'].mean():.2f}")
        
        best_model = successful_runs.loc[successful_runs['tokens_per_sec'].idxmax()]
        print(f"Best performing: {best_model['model_name']} ({best_model['tokens_per_sec']:.2f} tokens/sec)")
        
        print(f"Average load time: {successful_runs['load_time'].mean():.2f}s")
        print(f"Average memory usage: {successful_runs['peak_memory_mb'].mean():.1f}MB")
        
        if similarity_model:
            print(f"Average similarity: {successful_runs['similarity_score'].mean():.3f}")
    else:
        print("\n❌ No successful benchmark runs completed")
    
    print(f"\n🏁 Benchmarking complete!")
    print(f"📊 CSV: {output_file}")
    print(f"🌐 HTML Report: {RESULTS_DIR / 'benchmark_report.html'}")
    print(f"📝 Individual outputs: {RESULTS_DIR / 'output_*.txt'} files")

if __name__ == "__main__":
    main() 