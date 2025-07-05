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

# Configuration
CUSTOM_PROMPT = "Explain the Pythagorean theorem concisely."
GOLD_ANSWER = "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c²."
MAX_NEW_TOKENS = 50

# MLX Community model configurations: (model_name, prompt_template, special_params)
MODELS = [
    ("mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit", "Question: {prompt}\nAnswer:", {}),
    ("mlx-community/Phi-3-mini-4k-instruct-4bit", "Question: {prompt}\nAnswer:", {"repetition_penalty": 1.1}),
    ("mlx-community/Meta-Llama-3-8B-Instruct-4bit", "Question: {prompt}\nAnswer:", {}),
    # Add more models as needed - expand the list with verified MLX models
    # ("mlx-community/gemma-2b-it-4bit", "Question: {prompt}\nAnswer:", {}),
    # ("mlx-community/Qwen2-1.5B-Instruct-4bit", "Question: {prompt}\nAnswer:", {}),
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

def benchmark_model(model_name: str, prompt_template: str, special_params: Dict[str, Any], similarity_model: Optional[SentenceTransformer] = None) -> Dict[str, Any]:
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
        
        # Format prompt using template
        formatted_prompt = prompt_template.format(prompt=CUSTOM_PROMPT)
        
        # MLX inference - using simpler parameters
        response = generate(
            model, 
            tokenizer, 
            formatted_prompt,
            max_tokens=MAX_NEW_TOKENS,
            verbose=False
        )
        
        # Extract only the generated part (remove the prompt)
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):].strip()
        else:
            # Fallback: try to find where the answer starts
            if "Answer:" in response:
                response = response.split("Answer:", 1)[1].strip()
            elif "Bot:" in response:
                response = response.split("Bot:", 1)[1].strip()
            elif "[/INST]" in response:
                response = response.split("[/INST]", 1)[1].strip()
        
        # Count tokens in the response
        tokens_generated = len(tokenizer.encode(response)) if response else 0
        
        inference_time = time.perf_counter() - start_time
        tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0
        
        results["inference_time"] = inference_time
        results["tokens_generated"] = tokens_generated
        results["tokens_per_sec"] = tokens_per_sec
        results["generated_text"] = response[:200]  # Truncate for display
        
        # Check for meaningful output
        if tokens_generated < 5:
            print(f"⚠️  Warning: Only {tokens_generated} tokens generated - may indicate prompt formatting issue")
        
        # Compute similarity if available
        if similarity_model and response and len(response.strip()) > 0:
            similarity_score = compute_similarity(response, GOLD_ANSWER, similarity_model)
            results["similarity_score"] = similarity_score
        
        print(f"Generated: {response[:100]}...")
        print(f"Tokens/sec: {tokens_per_sec:.2f}, Tokens: {tokens_generated}")
        
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
    
    for i, (model_name, prompt_template, special_params) in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Starting benchmark...")
        results = benchmark_model(model_name, prompt_template, special_params, similarity_model)
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
    output_file = "benchmark_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n📊 Results saved to {output_file}")
    
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
    
    print(f"\n🏁 Benchmarking complete! Results saved to {output_file}")

if __name__ == "__main__":
    main() 