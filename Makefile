# ============================================================================
# 🚀 FinanceRAG Makefile - Simplified Commands
# ============================================================================
# Usage: make <command>
# Example: make docker-up

.PHONY: help docker-up docker-down docker-logs docker-rebuild clean install jupyter test

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# 📖 Help
# ============================================================================
help: ## Show this help message
	@echo "========================================"
	@echo "🚀 FinanceRAG Makefile Commands"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# 🐳 Docker Commands
# ============================================================================
docker-up: ## Start Docker containers (CPU only)
	docker-compose up -d
	@echo "✅ Container started! Open http://localhost:8888"

docker-up-gpu: ## Start Docker containers with GPU support
	docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
	@echo "✅ Container started with GPU! Open http://localhost:8888"

docker-down: ## Stop Docker containers
	docker-compose down
	@echo "✅ Containers stopped"

docker-logs: ## View Docker logs (follow mode)
	docker-compose logs -f

docker-restart: ## Restart Docker containers
	docker-compose restart
	@echo "✅ Containers restarted"

docker-rebuild: ## Rebuild and restart containers
	docker-compose down
	docker-compose up -d --build
	@echo "✅ Containers rebuilt and started"

docker-clean: ## Remove containers, volumes, and images
	docker-compose down -v --rmi all
	@echo "✅ All Docker resources cleaned"

docker-shell: ## Open bash shell in container
	docker exec -it financerag-notebook bash

# ============================================================================
# 💻 Local Development
# ============================================================================
install: ## Install Python dependencies (local)
	pip install --upgrade pip setuptools wheel
	pip install -r requirements_compatible.txt
	@echo "✅ Dependencies installed"

install-dev: ## Install development dependencies
	pip install --upgrade pip setuptools wheel
	pip install -r requirements_compatible.txt
	pip install jupyter jupyterlab ipywidgets
	@echo "✅ Dev dependencies installed"

jupyter: ## Start Jupyter Lab (local)
	jupyter lab --port=8888 --no-browser

jupyter-notebook: ## Start Jupyter Notebook classic (local)
	jupyter notebook --port=8888 --no-browser

# ============================================================================
# 🧪 Testing & Validation
# ============================================================================
test-imports: ## Test if all packages import correctly
	@echo "Testing imports..."
	@python -c "import torch; print('✅ PyTorch:', torch.__version__)"
	@python -c "import sentence_transformers; print('✅ sentence-transformers OK')"
	@python -c "import transformers; print('✅ transformers OK')"
	@python -c "import faiss; print('✅ faiss OK')"
	@python -c "from FlagEmbedding import FlagReranker; print('✅ FlagEmbedding OK')"
	@echo "✅ All imports successful!"

test-gpu: ## Test GPU availability
	@python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

verify-data: ## Verify data files exist
	@echo "Checking data files..."
	@python -c "import os; datasets = ['convfinqa', 'financebench', 'finder', 'finqa', 'finqabench', 'multiheirtt', 'tatqa']; missing = [d for d in datasets if not os.path.exists(f'data/{d}_corpus.jsonl')]; print(f'✅ Found {len(datasets)-len(missing)}/{len(datasets)} datasets'); [print(f'❌ Missing: {d}') for d in missing] if missing else None"

# ============================================================================
# 🧹 Cleaning
# ============================================================================
clean: ## Clean Python cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Python cache cleaned"

clean-outputs: ## Clean notebook outputs
	jupyter nbconvert --clear-output --inplace notebook/**/*.ipynb
	@echo "✅ Notebook outputs cleared"

clean-models: ## Clean downloaded model cache
	rm -rf ~/.cache/huggingface/hub/*
	@echo "✅ Model cache cleaned"

clean-all: clean clean-outputs ## Clean everything
	@echo "✅ Everything cleaned"

# ============================================================================
# 📊 Notebooks
# ============================================================================
run-baseline: ## Run baseline notebook (notebook 1)
	jupyter nbconvert --execute --to notebook --inplace \
		notebook/1_baseline/1.\ baseline.ipynb

run-quickwins: ## Run quick wins notebook (notebook 2)
	jupyter nbconvert --execute --to notebook --inplace \
		notebook/2_quick_wins/2.\ quick_wins_notebook.ipynb

run-chunking-eval: ## Run chunking evaluation (notebook 3)
	jupyter nbconvert --execute --to notebook --inplace \
		notebook/3_chunking_evaluation/3.\ chunking_evaluation.ipynb

run-production: ## Run production pipeline (notebook 4)
	jupyter nbconvert --execute --to notebook --inplace \
		notebook/4_improved_chunking/4.\ improved_chunking_pipeline.ipynb

# ============================================================================
# 📦 Data Management
# ============================================================================
download-models: ## Pre-download models (run in container or local)
	@echo "Downloading embedding model..."
	@python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
	@echo "Downloading reranker model..."
	@python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3')"
	@echo "✅ Models downloaded to cache"

# ============================================================================
# 📊 Status & Info
# ============================================================================
status: ## Show project status
	@echo "========================================"
	@echo "📊 FinanceRAG Project Status"
	@echo "========================================"
	@echo "Docker containers:"
	@docker ps -a --filter name=financerag || echo "  No Docker containers"
	@echo ""
	@echo "Data files:"
	@make verify-data
	@echo ""
	@echo "Python environment:"
	@which python || echo "  No Python in PATH"
	@python --version 2>/dev/null || echo "  Python not available"
	@echo ""

info: ## Show system information
	@echo "========================================"
	@echo "💻 System Information"
	@echo "========================================"
	@echo "OS: $$(uname -s)"
	@echo "Python: $$(python --version 2>&1)"
	@echo "Docker: $$(docker --version 2>&1 || echo 'Not installed')"
	@echo "GPU: $$(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'Not available')"
	@echo "Memory: $$(free -h 2>/dev/null | grep Mem | awk '{print $$2}' || echo 'N/A')"
	@echo ""

# ============================================================================
# 🎯 Quick Commands
# ============================================================================
quick-start: docker-up ## Quick start with Docker (recommended)
	@echo "🚀 Quick start complete!"
	@echo "📖 Open: http://localhost:8888"
	@echo "📂 Navigate to: notebook/4_improved_chunking/"

quick-start-local: install jupyter ## Quick start locally
	@echo "🚀 Local environment ready!"
	@echo "📖 Jupyter Lab starting..."
