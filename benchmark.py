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
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import json
import yaml
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser

# Set environment variables for optimal performance
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration loading
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"📋 Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file {config_path} not found. Using default values.")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        print("Using default values.")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Return default configuration if config file is not available"""
    return {
        "prompts": {
            "system_prompt": """You are PromptCraft Architect, an AI specializing in refactoring user inputs into precise, structured, plain-text prompts for other advanced LLMs. Your focus is on a wide range of technical tasks for developers, from creating entire applications to fixing small bugs.

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
NO preambles, apologies, or meta-commentary.""",
            "custom_prompt": "build a react app like facebook",
            "gold_answer": """Act as a Senior Software Engineer specializing in React.

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

GOAL: Create a functional React application that mimics key features of Facebook.""",
            "max_new_tokens": 800
        },
        "model_config": {
            "small": {
                "memory_limit_gb": 2.0,
                "models": []
            },
            "medium": {
                "memory_limit_gb": 4.0,
                "models": [
                    {"name": "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "params": {}},
                ]
            },
            "large": {
                "memory_limit_gb": 6.0,
                "models": [
                    {"name": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit", "params": {}},
                ]
            },
            "experimental": {
                "memory_limit_gb": 5.0,
                "models": []
            }
        }
    }

# Configuration will be loaded after argument parsing
CONFIG = None

def transform_model_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Transform YAML model config to the format expected by the benchmark script"""
    transformed = {}
    for category, category_config in config_dict.items():
        # Handle case where models is None (empty YAML section with only comments)
        models_list = category_config.get("models") or []
        transformed[category] = {
            "memory_limit_gb": category_config["memory_limit_gb"],
            "models": [
                (model["name"], model["params"]) 
                for model in models_list
            ]
        }
    return transformed

def reload_configuration(config_path: str = "config.yaml") -> None:
    """Reload configuration and update global variables"""
    global CONFIG, SYSTEM_PROMPT, CUSTOM_PROMPT, GOLD_ANSWER, MAX_NEW_TOKENS, MODEL_CONFIG
    
    CONFIG = load_config(config_path)
    
    # Update prompt configuration
    SYSTEM_PROMPT = CONFIG["prompts"]["system_prompt"]
    CUSTOM_PROMPT = CONFIG["prompts"]["custom_prompt"]
    GOLD_ANSWER = CONFIG["prompts"]["gold_answer"]
    MAX_NEW_TOKENS = CONFIG["prompts"]["max_new_tokens"]
    
    # Update model configuration
    MODEL_CONFIG = transform_model_config(CONFIG["model_config"])

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Results persistence configuration
RESULTS_DATABASE = RESULTS_DIR / "benchmark_history.json"
APPEND_RESULTS = True  # Set to False to overwrite instead of append

def parse_arguments_early():
    """Early parse to get config file path only"""
    parser = argparse.ArgumentParser(description="MLX LLM Benchmark Tool", add_help=False)
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    # Parse only known args to get config path, ignore everything else
    args, _ = parser.parse_known_args()
    return args.config

def parse_arguments(available_categories: List[str]):
    """Parse command line arguments with dynamic category choices"""
    # Add 'all' to available categories for convenience
    all_categories = available_categories + ["all"]
    
    parser = argparse.ArgumentParser(description="MLX LLM Benchmark Tool")
    parser.add_argument(
        "--categories", 
        nargs="+", 
        default=["small", "medium"] if "small" in available_categories and "medium" in available_categories else available_categories[:2],
        choices=all_categories,
        help=f"Model categories to run. Available: {', '.join(available_categories)} (default: auto-detected)"
    )
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="List all available models by category"
    )
    parser.add_argument(
        "--memory-check", 
        action="store_true",
        help="Check memory requirements for each category"
    )
    parser.add_argument(
        "--safety-margin", 
        type=float, 
        default=2.0,
        help="Memory safety margin in GB (default: 2.0)"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite existing results instead of appending (default: append)"
    )
    parser.add_argument(
        "--serve", 
        action="store_true",
        help="Start web server to view interactive report (enables persistent deletion)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port for web server (default: 8000, used with --serve)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    return parser.parse_args()

def cleanup_old_files():
    """Remove old result files from the main directory"""
    old_files = [
        "benchmark_results.csv",
        "benchmark_report.html"
    ]
    
    # Remove old output files
    for file in Path(".").glob("output_*.txt"):
        if file.exists():
            file.unlink()
            print(f"🗑️  Removed old file: {file}")
    
    # Remove old CSV and HTML files
    for file in old_files:
        if Path(file).exists():
            Path(file).unlink()
            print(f"🗑️  Removed old file: {file}")

def list_models_by_category():
    """List all available models organized by category"""
    print("\n📋 Available Models by Category:")
    print("=" * 60)
    
    for category, config in MODEL_CONFIG.items():
        memory_req = config["memory_limit_gb"]
        available = get_available_memory_gb()
        can_run = can_run_model(category)
        status = "✅ CAN RUN" if can_run else "❌ INSUFFICIENT MEMORY"
        
        print(f"\n📂 {category.upper()} ({memory_req}GB required) - {status}")
        print(f"   Available: {available:.1f}GB")
        print("-" * 50)
        
        for i, (model_name, params) in enumerate(config["models"], 1):
            print(f"   {i}. {model_name}")
            if params:
                print(f"      Params: {params}")
    
    print("\n" + "=" * 60)

def check_memory_requirements():
    """Check memory requirements for all categories"""
    print("\n💾 Memory Requirements Analysis:")
    print("=" * 60)
    
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_gb = get_available_memory_gb()
    
    print(f"Total System Memory: {total_memory_gb:.1f}GB")
    print(f"Available Memory: {available_gb:.1f}GB")
    print(f"Safety Margin: {MEMORY_SAFETY_MARGIN_GB:.1f}GB")
    print(f"Usable Memory: {available_gb - MEMORY_SAFETY_MARGIN_GB:.1f}GB")
    
    print("\nCategory Analysis:")
    print("-" * 50)
    
    for category, config in MODEL_CONFIG.items():
        required_gb = config["memory_limit_gb"]
        total_needed = required_gb + MEMORY_SAFETY_MARGIN_GB
        can_run = can_run_model(category)
        
        status = "✅ YES" if can_run else "❌ NO"
        print(f"{category:12} | Req: {required_gb:4.1f}GB | Total: {total_needed:4.1f}GB | Can Run: {status}")
    
    print("\n" + "=" * 60)

# Configuration placeholders (will be loaded after argument parsing)
SYSTEM_PROMPT = ""
CUSTOM_PROMPT = ""
GOLD_ANSWER = ""
MAX_NEW_TOKENS = 800
MODEL_CONFIG = {}

# Runtime configuration
MEMORY_SAFETY_MARGIN_GB = 0.0  # Keep 0GB free for system
MAX_CONCURRENT_MODELS = 1  # Process one model at a time
SKIP_ON_MEMORY_ERROR = True  # Skip models that fail due to memory
CATEGORIES_TO_RUN = ["small", "medium"]  # Which categories to run by default

def install_dependencies():
    """Install required dependencies if not available"""
    required_packages = [
        "mlx-lm",
        "sentence-transformers", 
        "pandas",
        "numpy",
        "psutil",
        "tqdm",
        "pyyaml"
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
from report_generator import create_report_generator

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def get_available_memory_gb() -> float:
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def can_run_model(category: str) -> bool:
    """Check if we have enough memory to run models in this category"""
    available_gb = get_available_memory_gb()
    required_gb = MODEL_CONFIG[category]["memory_limit_gb"]
    return available_gb >= (required_gb + MEMORY_SAFETY_MARGIN_GB)

def get_models_to_run() -> List[Tuple[str, str, Dict]]:
    """Get list of models to run based on memory constraints and configuration"""
    models_to_run = []
    
    for category in CATEGORIES_TO_RUN:
        if category not in MODEL_CONFIG:
            print(f"⚠️  Category '{category}' not found in configuration")
            continue
            
        if not can_run_model(category):
            available = get_available_memory_gb()
            required = MODEL_CONFIG[category]["memory_limit_gb"] + MEMORY_SAFETY_MARGIN_GB
            print(f"⚠️  Skipping '{category}' category - need {required:.1f}GB, have {available:.1f}GB")
            continue
            
        print(f"✅ Category '{category}' approved - {len(MODEL_CONFIG[category]['models'])} models")
        
        for model_name, special_params in MODEL_CONFIG[category]["models"]:
            models_to_run.append((category, model_name, special_params))
    
    return models_to_run

def clear_memory():
    """Clear memory and force garbage collection"""
    gc.collect()
    # MLX automatically manages memory on Apple Silicon
    
    # Force additional cleanup
    mx.clear_cache()  # Clear MLX cache

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

def load_existing_results() -> List[Dict[str, Any]]:
    """Load existing benchmark results from JSON database"""
    if not RESULTS_DATABASE.exists():
        return []
    
    try:
        with open(RESULTS_DATABASE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"📖 Loaded {len(data)} existing benchmark results")
            return data
    except Exception as e:
        print(f"⚠️  Error loading existing results: {e}")
        return []

def save_results_database(results: List[Dict[str, Any]]) -> None:
    """Save benchmark results to JSON database"""
    try:
        with open(RESULTS_DATABASE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Saved {len(results)} results to database")
    except Exception as e:
        print(f"❌ Error saving results database: {e}")

def remove_result_from_database(model_name: str, run_id: str) -> bool:
    """Remove a specific result from the JSON database"""
    try:
        results = load_existing_results()
        
        # Find and remove the matching result
        original_count = len(results)
        results = [r for r in results if not (r.get('model_name') == model_name and r.get('run_id') == run_id)]
        
        if len(results) < original_count:
            save_results_database(results)
            print(f"🗑️  Removed {model_name} (run: {run_id}) from database")
            return True
        else:
            print(f"⚠️  No matching result found for {model_name} (run: {run_id})")
            return False
    except Exception as e:
        print(f"❌ Error removing result from database: {e}")
        return False

def regenerate_html_report() -> None:
    """Regenerate the HTML report from the current database"""
    try:
        results = load_existing_results()
        if results:
            df = pd.DataFrame(results)
            output_files = [r.get('output_file', '') for r in results]
            generate_html_report(df, output_files)
            print("🔄 HTML report regenerated")
        else:
            # Create empty DataFrame for empty database
            df = pd.DataFrame()
            generate_html_report(df, [])
            print("🔄 HTML report regenerated (empty database)")
    except Exception as e:
        print(f"❌ Error regenerating HTML report: {e}")

def add_run_metadata(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add run metadata to distinguish between different benchmark sessions"""
    import datetime
    
    timestamp = datetime.datetime.now().isoformat()
    run_id = timestamp.replace(':', '-').replace('.', '-')  # Safe filename
    
    for result in results:
        result['run_timestamp'] = timestamp
        result['run_id'] = run_id
    
    return results

def merge_results(existing_results: List[Dict[str, Any]], new_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge new results with existing ones, handling duplicates intelligently"""
    if not APPEND_RESULTS:
        print("🔄 Overwrite mode: replacing all existing results")
        return new_results
    
    if not existing_results:
        print("✨ No existing results found, starting fresh")
        return new_results
    
    # Create a set of existing model names for deduplication
    existing_models = {(r['model_name'], r.get('run_id', '')) for r in existing_results}
    
    # Add new results, avoiding duplicates from the same run
    merged_results = existing_results.copy()
    new_count = 0
    
    for result in new_results:
        model_key = (result['model_name'], result.get('run_id', ''))
        if model_key not in existing_models:
            merged_results.append(result)
            new_count += 1
        else:
            print(f"⚠️  Skipping duplicate: {result['model_name']} from same run")
    
    print(f"🔗 Merged results: {len(existing_results)} existing + {new_count} new = {len(merged_results)} total")
    return merged_results

def generate_html_report(df: pd.DataFrame, output_files: List[str]):
    """Generate an HTML report using the template-based report generator"""
    try:
        # Create report generator instance
        report_generator = create_report_generator()
        
        # Generate the report
        report_generator.generate_report(df, CUSTOM_PROMPT)
        
    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")
        print("Falling back to basic report generation...")
        
        # Fallback: create a basic HTML file if template system fails
        html_file = RESULTS_DIR / "benchmark_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head><title>MLX LLM Benchmark Results</title></head>
<body>
    <h1>MLX LLM Benchmark Results</h1>
    <p>Report generation failed. Please check the console for errors.</p>
    <p>Error: {e}</p>
</body>
</html>""")
        print(f"📄 Basic HTML report generated: {html_file}")

def main():
    """Main benchmarking function"""
    global MEMORY_SAFETY_MARGIN_GB, CATEGORIES_TO_RUN, APPEND_RESULTS
    
    # Two-stage argument parsing:
    # 1. Early parse to get config file path
    config_path = parse_arguments_early()
    
    # 2. Load configuration to get available categories
    reload_configuration(config_path)
    
    # 3. Parse arguments again with dynamic category choices
    available_categories = list(MODEL_CONFIG.keys())
    args = parse_arguments(available_categories)
    
    # Update global configuration based on arguments
    MEMORY_SAFETY_MARGIN_GB = args.safety_margin
    APPEND_RESULTS = not args.overwrite  # Invert the flag
    
    if "all" in args.categories:
        CATEGORIES_TO_RUN = list(MODEL_CONFIG.keys())
    else:
        CATEGORIES_TO_RUN = args.categories
    
    print("🚀 Starting MLX LLM Benchmarking on MacBook M1 Pro")
    print("🍎 Using MLX (Apple Silicon optimized)")
    print(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Selected Categories: {', '.join(CATEGORIES_TO_RUN)}")
    
    # Handle utility commands
    if args.list_models:
        list_models_by_category()
        return
        
    if args.memory_check:
        check_memory_requirements()
        return
    
    if args.serve:
        # Start web server mode
        html_file = RESULTS_DIR / "benchmark_report.html"
        if not html_file.exists():
            print("⚠️  No HTML report found. Run benchmarks first to generate a report.")
            print("Example: python benchmark.py --categories medium")
            return
        
        print(f"🌐 Starting web server mode...")
        start_benchmark_server(args.port)
        return
    
    # Initialize similarity model
    print("\nLoading similarity model...")
    try:
        similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Similarity model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load similarity model: {e}")
        similarity_model = None
    
    # Get models to run based on memory constraints
    models_to_run = get_models_to_run()
    
    if not models_to_run:
        print("❌ No models can be run with current memory constraints")
        return
    
    print(f"📋 Planning to run {len(models_to_run)} models")
    
    # Run benchmarks
    all_results = []
    
    for i, (category, model_name, special_params) in enumerate(models_to_run, 1):
        print(f"\n[{i}/{len(models_to_run)}] Starting benchmark...")
        print(f"📂 Category: {category}")
        print(f"🔍 Model: {model_name}")
        print(f"💾 Available memory: {get_available_memory_gb():.1f}GB")
        
        results = benchmark_model(model_name, special_params, similarity_model)
        all_results.append(results)
        
        # Print intermediate results
        if results.get("error") is None:
            print(f"✅ {model_name}: {results['tokens_per_sec']:.2f} tokens/sec, "
                  f"Load: {results['load_time']:.2f}s, "
                  f"Memory: {results['peak_memory_mb']:.1f}MB")
        else:
            print(f"❌ {model_name}: {results['error']}")
            
        # Force cleanup between models
        clear_memory()
        time.sleep(1)  # Brief pause to let system stabilize
    
    # Add run metadata to new results
    all_results = add_run_metadata(all_results)
    
    # Load existing results and merge with new ones
    existing_results = load_existing_results()
    merged_results = merge_results(existing_results, all_results)
    
    # Save merged results to database
    save_results_database(merged_results)
    
    # Create DataFrame from merged results
    df = pd.DataFrame(merged_results)
    
    # Display results
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    # Create a summary table for display (show recent results first)
    display_columns = [
        'model_name', 'framework', 'load_time', 
        'peak_memory_mb', 'tokens_per_sec', 'similarity_score', 'run_timestamp', 'error'
    ]
    
    # Filter to show columns that exist in the DataFrame
    existing_columns = [col for col in display_columns if col in df.columns]
    display_df = df[existing_columns].copy()
    
    # Sort by run_timestamp descending to show newest results first
    if 'run_timestamp' in display_df.columns:
        display_df = display_df.sort_values('run_timestamp', ascending=False)
    
    # Round numeric columns
    for col in ['load_time', 'peak_memory_mb', 'tokens_per_sec', 'similarity_score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    print(display_df.to_string(index=False))
    
    # Save to CSV (complete history)
    output_file = RESULTS_DIR / "benchmark_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n📊 Complete results saved to {output_file}")
    
    # Generate HTML report with all results
    output_files = [row.get('output_file', '') if not pd.isna(row.get('output_file', '')) else '' for _, row in df.iterrows()]
    generate_html_report(df, output_files)
    
    # Summary statistics
    successful_runs = df[df['error'].isna()]
    current_run_results = [r for r in all_results if r.get('error') is None]
    
    if not successful_runs.empty:
        print(f"\n📈 SUMMARY:")
        print(f"Total successful runs: {len(successful_runs)}/{len(df)} (all time)")
        print(f"Current session: {len(current_run_results)}/{len(all_results)} successful")
        
        if len(current_run_results) > 0:
            current_tokens_per_sec = [r['tokens_per_sec'] for r in current_run_results]
            print(f"Current session avg tokens/sec: {sum(current_tokens_per_sec)/len(current_tokens_per_sec):.2f}")
        
        print(f"All-time average tokens/sec: {successful_runs['tokens_per_sec'].mean():.2f}")
        
        best_model = successful_runs.loc[successful_runs['tokens_per_sec'].idxmax()]
        print(f"Best performing (all-time): {best_model['model_name']} ({best_model['tokens_per_sec']:.2f} tokens/sec)")
        
        print(f"All-time average load time: {successful_runs['load_time'].mean():.2f}s")
        print(f"All-time average memory usage: {successful_runs['peak_memory_mb'].mean():.1f}MB")
        
        if similarity_model and len(successful_runs) > 0:
            print(f"All-time average similarity: {successful_runs['similarity_score'].mean():.3f}")
    else:
        print("\n❌ No successful benchmark runs completed")
    
    print(f"\n🏁 Benchmarking complete!")
    print(f"📊 CSV: {output_file}")
    print(f"🌐 HTML Report: {RESULTS_DIR / 'benchmark_report.html'}")
    print(f"🗄️  Database: {RESULTS_DATABASE}")
    print(f"📝 Individual outputs: {RESULTS_DIR / 'output_*.txt'} files")
    
    print(f"\n🌟 Interactive Mode:")
    print(f"   Run: python benchmark.py --serve")
    print(f"   → Start web server for interactive report with persistent deletion")
    print(f"   → Deletions will be saved to the database permanently")
    
    if APPEND_RESULTS:
        print(f"\n💡 Tip: Use --overwrite to replace existing results instead of appending")
    else:
        print(f"\n💡 Tip: Remove --overwrite to append to existing results")

class BenchmarkHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the benchmark server"""
    
    def log_message(self, format, *args):
        """Override to reduce server log noise"""
        return
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/' or parsed_path.path == '/report':
            # Serve the HTML report
            html_file = RESULTS_DIR / "benchmark_report.html"
            if html_file.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                with open(html_file, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                self.send_error(404, "Report not found")
        elif parsed_path.path == '/report.css':
            # Serve the CSS file
            css_file = RESULTS_DIR / "report.css"
            if css_file.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/css')
                self.end_headers()
                with open(css_file, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                self.send_error(404, "CSS not found")
        elif parsed_path.path == '/report.js':
            # Serve the JS file
            js_file = RESULTS_DIR / "report.js"
            if js_file.exists():
                self.send_response(200)
                self.send_header('Content-type', 'application/javascript')
                self.end_headers()
                with open(js_file, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                self.send_error(404, "JS not found")
        else:
            self.send_error(404, "Not found")
    
    def do_DELETE(self):
        """Handle DELETE requests to remove results"""
        try:
            # Parse query parameters
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)
            
            if parsed_path.path == '/api/remove-result':
                model_name = query_params.get('model_name', [None])[0]
                run_id = query_params.get('run_id', [None])[0]
                
                if model_name and run_id:
                    success = remove_result_from_database(model_name, run_id)
                    if success:
                        # Regenerate HTML report
                        regenerate_html_report()
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        response = {'success': True, 'message': 'Result removed successfully'}
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                    else:
                        self.send_error(400, "Failed to remove result")
                else:
                    self.send_error(400, "Missing model_name or run_id")
            else:
                self.send_error(404, "Not found")
        except Exception as e:
            print(f"❌ Error processing DELETE request: {e}")
            self.send_error(500, "Internal server error")
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def start_benchmark_server(port: int = 8000) -> None:
    """Start the benchmark HTTP server"""
    try:
        server_address = ('', port)
        httpd = HTTPServer(server_address, BenchmarkHTTPRequestHandler)
        
        print(f"🌐 Starting benchmark server on http://localhost:{port}")
        print(f"📊 Access your report at: http://localhost:{port}/report")
        print(f"🛑 Press Ctrl+C to stop the server")
        
        # Open browser automatically
        webbrowser.open(f"http://localhost:{port}/report")
        
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 