# MLX LLM Benchmark Tool

Benchmark and compare language models optimized for Apple Silicon Macs. Features intelligent memory management, interactive reports, and configurable model categories.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark (default: small + medium models)
python benchmark.py

# View interactive report with removal capabilities
python benchmark.py --serve
```

![Benchmark Results Example](image.png)
*Interactive benchmark report showing model performance metrics, detailed outputs, and removal capabilities*

## 📋 Essential Commands

```bash
# Choose specific model categories
python benchmark.py --categories medium large

# See available models and memory requirements
python benchmark.py --list-models
python benchmark.py --memory-check

# Use custom configuration
python benchmark.py --config my-config.yaml

# Reduce memory safety margin if needed
python benchmark.py --categories large --safety-margin 0.5
```

## 🎯 Key Features

- **🧠 Smart Memory Management**: Prevents crashes, skips models that won't fit
- **🌐 Interactive Reports**: Web interface with persistent result deletion
- **📊 Multiple Outputs**: HTML reports, CSV data, individual text files
- **⚙️ Fully Configurable**: YAML-based prompts, models, and categories
- **📱 Real-time Metrics**: Tokens/sec, memory usage, similarity scores

## 📁 Model Categories

- **small** (≤2GB): Fast, lightweight models
- **medium** (≤4GB): Balanced performance and efficiency  
- **large** (≤6GB): High-quality, resource-intensive models
- **experimental** (≤5GB): Latest cutting-edge variants

## ⚙️ Configuration

Create custom `config.yaml` files to customize:

```yaml
prompts:
  system_prompt: "Your custom system prompt..."
  custom_prompt: "Your test prompt"
  max_new_tokens: 800

model_config:
  my_category:
    memory_limit_gb: 3.0
    models:
      - name: "mlx-community/your-model"
        params: {}
```

## 🌐 Interactive Mode

```bash
# Start web server for interactive reports
python benchmark.py --serve

# Custom port
python benchmark.py --serve --port 8001
```

Access at `http://localhost:8000` to view reports and permanently delete results.

## 📊 Advanced Options

```bash
# Run all available models
python benchmark.py --categories all

# Replace existing results instead of appending
python benchmark.py --overwrite

# Ignore safety margins (use carefully)
python benchmark.py --safety-margin 0

# Multiple categories and custom config
python benchmark.py --config test.yaml --categories test experimental
```

## 📁 Output Files

Results saved to `results/` directory:
- `benchmark_report.html` - Interactive web report
- `benchmark_results.csv` - Complete data table  
- `benchmark_history.json` - Persistent database
- `output_*.txt` - Individual model responses

## 💡 Tips

1. **Start small**: Use `--memory-check` to see what fits
2. **Monitor memory**: Close other apps before running large models
3. **Custom configs**: Create specialized test configurations
4. **Interactive mode**: Use `--serve` for easy result management

## 🔧 Troubleshooting

**Out of memory?**
```bash
python benchmark.py --memory-check
python benchmark.py --categories small --safety-margin 0.5
```

**Model not loading?** Check internet connection and model availability on HuggingFace.

Perfect for researchers, developers, and anyone comparing LLM performance on Apple Silicon! 🍎 