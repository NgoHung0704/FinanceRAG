# ⚡ Quick Start Guide - FinanceRAG

## 🎯 3 Bước để chạy project

### Bước 1: Clone và Setup ⏱️ 5 phút

```bash
# Clone repository
git clone <your-repo-url>
cd FinanceRAG
```

### Bước 2: Chọn phương thức setup

#### 🐳 **Option A: Docker (RECOMMENDED)** - Dễ nhất, không cần config

```bash
# Start container
docker-compose up -d

# Wait ~2 minutes for setup
# Then open: http://localhost:8888
```

**✅ Ưu điểm:**
- Không cần cài Python, pip, packages
- Môi trường đồng nhất trên mọi máy
- Tự động download models khi cần

**❌ Nhược điểm:**
- Cần Docker Desktop (8GB RAM)
- Download image lần đầu ~3-5 phút

---

#### 💻 **Option B: Local Python** - Linh hoạt hơn

**Windows:**
```bash
# Tạo virtual environment
python -m venv financerag_env
financerag_env\Scripts\activate

# Install packages
pip install --upgrade pip
pip install -r requirements_compatible.txt

# Start Jupyter
jupyter lab
```

**Linux/Mac:**
```bash
# Tạo virtual environment
python3 -m venv financerag_env
source financerag_env/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements_compatible.txt

# Start Jupyter
jupyter lab
```

**✅ Ưu điểm:**
- Không cần Docker
- Dùng GPU local (nếu có)
- Customize được environment

**❌ Nhược điểm:**
- Phải cài Python 3.10
- Có thể gặp dependency conflicts
- Setup phức tạp hơn

---

### Bước 3: Chạy Notebook 🚀

1. Mở Jupyter Lab (tự động mở browser hoặc copy link từ terminal)
2. Navigate to: `notebook/4_improved_chunking/`
3. Mở: `4. improved_chunking_pipeline.ipynb`
4. Run All Cells: `Ctrl+Shift+Enter` (hoặc `Cmd+Shift+Enter` trên Mac)

**Kết quả:**
- File `submission_optimal_chunking.csv` được tạo
- NDCG@10 evaluation hiển thị trong notebook
- Ready to submit!

---

## 🎮 Cheat Sheet - Docker Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Restart after code changes
docker-compose restart

# Rebuild from scratch
docker-compose down -v
docker-compose up -d --build

# Access container shell
docker exec -it financerag-notebook bash

# Stop and remove everything (clean slate)
docker-compose down -v --rmi all
```

---

## 🎮 Cheat Sheet - Jupyter Commands

```bash
# Start Jupyter Lab
jupyter lab

# Start Jupyter Notebook (classic)
jupyter notebook

# Run notebook from command line
jupyter nbconvert --execute --to notebook \
  notebook/4_improved_chunking/4.\ improved_chunking_pipeline.ipynb

# Clear all outputs
jupyter nbconvert --clear-output --inplace notebook/*.ipynb
```

---

## 📊 Expected Runtime

| Notebook | CPU Time | GPU Time | Output |
|----------|----------|----------|--------|
| 1. Baseline | ~30 min | ~10 min | submission.csv |
| 2. Quick Wins | ~45 min | ~15 min | submission_improved.csv |
| 3. Chunking Eval | ~2 hours | ~30 min | Optimal configs |
| 4. Production | ~60 min | ~20 min | submission_optimal.csv |

**Tip:** Notebook 4 có thể load pre-chunked data → nhanh hơn!

---

## ⚠️ Troubleshooting

### Docker Issues

**"Cannot connect to Docker daemon"**
```bash
# Windows: Start Docker Desktop
# Linux: sudo systemctl start docker
```

**"Port 8888 already in use"**
```bash
# Change port in docker-compose.yml:
ports:
  - "8889:8888"  # Use 8889 instead
```

**"Container keeps restarting"**
```bash
# Check logs
docker-compose logs financerag

# Common cause: Out of memory
# Solution: Increase Docker memory in Docker Desktop settings
```

### Python Issues

**"No module named 'sentence_transformers'"**
```bash
# Make sure virtual environment is activated
# Windows: financerag_env\Scripts\activate
# Linux/Mac: source financerag_env/bin/activate

# Reinstall
pip install -r requirements_compatible.txt
```

**"CUDA out of memory"**
```python
# In config.py, reduce batch sizes:
CONFIG = {
    'embed_batch_size': 8,   # was 16
    'rerank_batch_size': 8,  # was 16
}
```

**"ModuleNotFoundError: No module named 'financerag'"**
```bash
# Make sure PYTHONPATH is set
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or add to notebook first cell:
import sys
sys.path.insert(0, '..')
```

---

## 🎯 Next Steps

1. ✅ Setup completed → Start with notebook 4 (production pipeline)
2. 📊 Review evaluation report: `data/chunked_corpus/dataset_specific_evaluation_report.txt`
3. 🔧 Customize config: `notebook/4_improved_chunking/config.py`
4. 🚀 Generate submission and compete!

---

## 📞 Need Help?

- Check [README.md](README.md) for full documentation
- Review notebook comments - they're detailed!
- Common issues? See Troubleshooting section above

**Happy Coding! 🚀**
