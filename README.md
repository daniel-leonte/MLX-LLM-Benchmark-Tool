# MLX LLM Benchmark Tool

An intelligent benchmarking tool for MLX-optimized language models on Apple Silicon Macs. Features automatic memory management, categorized model organization, and comprehensive reporting.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark with default settings (small + medium models)
python benchmark.py

# Check what models are available
python benchmark.py --list-models

# Check memory requirements
python benchmark.py --memory-check
```

## 📊 Features

- **🧠 Intelligent Memory Management**: Automatically prevents OOM errors
- **📁 Categorized Models**: Organized by memory requirements and model size
- **🔧 Flexible Configuration**: Easy to add new models and categories
- **📈 Comprehensive Reporting**: HTML, CSV, and individual text outputs
- **🎯 Command Line Interface**: Full control over benchmark execution

## 🛠️ Usage

### Basic Commands

```bash
# Run specific categories
python benchmark.py --categories small medium

# Run all available models (if memory permits)
python benchmark.py --categories all

# Run only large models
python benchmark.py --categories large

# Adjust memory safety margin
python benchmark.py --safety-margin 1.5

# Ignore safety margin
python benchmark.py --categories all --safety-margin -69
```

### Utility Commands

```bash
# List all available models by category
python benchmark.py --list-models

# Check memory requirements vs available memory
python benchmark.py --memory-check
```

## 📂 Model Categories

### 🟢 Small Models (≤2GB)
- **DeepSeek-R1-Distill-Qwen-1.5B**: Fast, efficient for basic tasks
- **Memory Required**: ~2GB

### 🟡 Medium Models (≤4GB)
- **Mistral-7B-Instruct**: Balanced performance and efficiency
- **Meta-Llama-3.1-8B-Instruct**: High-quality general purpose
- **DeepSeek variants**: Specialized for different use cases
- **Memory Required**: ~4GB

### 🔴 Large Models (≤6GB)
- **Gemma-2-9B**: Advanced reasoning capabilities
- **Qwen3-8B**: Multilingual support
- **DeepSeek-Coder**: Specialized for code generation
- **Memory Required**: ~6GB

### 🧪 Experimental Models
- **Cutting-edge variants**: Latest model architectures
- **Memory Required**: ~5GB

## 🎯 Adding New Models

Simply edit the `MODEL_CONFIG` in `benchmark.py`:

```python
MODEL_CONFIG = {
    "your_category": {
        "memory_limit_gb": 3.0,
        "models": [
            ("mlx-community/your-model-name", {}),
            ("mlx-community/another-model", {"param": "value"}),
        ]
    }
}
```

## 📊 Output Files

All results are saved to the `results/` directory:

- **benchmark_results.csv**: Complete data table
- **benchmark_report.html**: Interactive HTML report
- **output_*.txt**: Individual model responses

## 🔧 Configuration

Key settings you can modify:

```python
# Memory safety margin (GB to keep free)
MEMORY_SAFETY_MARGIN_GB = 2.0

# Categories to run by default
CATEGORIES_TO_RUN = ["small", "medium"]

# Maximum tokens to generate
MAX_NEW_TOKENS = 800
```

## 🚨 Memory Management

The tool automatically:
- ✅ Checks available memory before loading models
- ✅ Skips models that won't fit in memory
- ✅ Cleans up models after benchmarking
- ✅ Maintains safety margins to prevent system crashes

## 🤝 Best Practices

1. **Start Small**: Begin with `--categories small` to test your system
2. **Monitor Memory**: Use `--memory-check` to understand requirements
3. **Close Applications**: Free up memory before running large models
4. **Regular Cleanup**: The tool automatically cleans up old files

## 📈 Performance Tips

- **Close other applications** to free up memory
- **Use smaller categories** if you have limited memory
- **Run overnight** for comprehensive benchmarks
- **Monitor system temperature** during long runs

## 🔍 Troubleshooting

**Memory Issues:**
```bash
# Check what you can run
python benchmark.py --memory-check

# Reduce safety margin if needed
python benchmark.py --safety-margin 1.0
```

**Model Loading Errors:**
- Some models may not be publicly available
- Check your internet connection
- Verify model names are correct

## 📝 Example Workflow

```bash
# 1. Check your system
python benchmark.py --memory-check

# 2. See what models are available
python benchmark.py --list-models

# 3. Run a conservative benchmark
python benchmark.py --categories small

# 4. If successful, try medium models
python benchmark.py --categories medium

# 5. Generate comprehensive report
python benchmark.py --categories all
```

## 🎉 Results

The tool provides:
- **Performance metrics** (tokens/second, load time, memory usage)
- **Quality assessment** (similarity to reference answers)
- **Detailed outputs** for manual inspection
- **Comparative analysis** across models

Perfect for researchers, developers, and anyone interested in comparing LLM performance on Apple Silicon! 