# Klasifikasi Teks Komentar Instagram (Tugas Besar Pengganti UTS)

Perbandingan minimal 3 algoritma (SVM, Naive Bayes, Ensemble Tree, ANN) untuk klasifikasi teks
pada komentar Instagram tentang pemberitaan survei kepuasan publik.

## Struktur
- `data/raw/` dataset asli (`comments_cleaned.csv`: post_id, comment_id, text_clean)
- `data/labeled/` dataset berlabel (belum ada, sedang dilabel tim)
- `src/` skrip preprocessing, training per algoritma, evaluasi
- `notebooks/` EDA, pelabelan, eksperimen, analisis hasil
- `results/` metrik dan gambar
- `paper/` draft paper SINTA 3/4
- `docs/` pembagian tugas dan catatan eksperimen

## Menjalankan proyek (virtual environment)

Semua paket Python dipasang di dalam **virtual environment (venv)** di folder `.venv/`
supaya versi paket seragam di semua anggota tim dan tidak mengotori Python bawaan komputer.
Folder `.venv/` **tidak** ikut di-commit (sudah ada di `.gitignore`), jadi setiap anggota membuatnya sendiri.

**Prasyarat:** Python 3.10 atau lebih baru (proyek ini dibuat dengan 3.12) dan Git.

### 1. Ambil kode
```bash
git clone https://github.com/bimyusufk/mbg-survey-comments-classifier.git
cd mbg-survey-comments-classifier
```

### 2. Buat dan aktifkan venv
Cukup dibuat sekali. Aktifkan setiap kali membuka terminal baru.

| Terminal | Buat (sekali) | Aktifkan |
|---|---|---|
| Windows PowerShell | `python -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |
| Windows CMD | `python -m venv .venv` | `.venv\Scripts\activate.bat` |
| Git Bash (Windows) | `python -m venv .venv` | `source .venv/Scripts/activate` |
| macOS / Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |

Venv aktif jika prompt diawali `(.venv)`. Untuk keluar: `deactivate`.

### 3. Pasang paket
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Pasang PyTorch (pilih sesuai komputer)
PyTorch sengaja dipisah dari `requirements.txt`, karena di Windows `pip install torch` biasa hanya memasang versi **CPU**.

```bash
# GPU NVIDIA (termasuk seri RTX 50, wajib CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Tanpa GPU (CPU saja)
pip install torch
```

Cek hasilnya:
```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```
Pada komputer dengan GPU, keluarannya harus `... CUDA: True`. Kalau `False`, kemungkinan yang terpasang versi CPU. Ulangi perintah GPU di atas dengan tambahan `--force-reinstall`.

### 5. Buka JupyterLab
```bash
jupyter lab
```
Di **VS Code**, buka folder proyek, lalu pilih interpreter `.venv` (`Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`) dan pilih kernel `.venv` di notebook.

### Tips dan masalah umum
- **PowerShell menolak `Activate.ps1`** ("running scripts is disabled"): jalankan sekali `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, lalu coba lagi.
- **`python` tidak dikenali di Windows:** coba `py -3.12 -m venv .venv`.
- **Selalu aktifkan venv sebelum `pip install` atau menjalankan notebook.** Jangan memasang paket ke Python global.
- **Menambah paket baru:** `pip install nama-paket`, lalu catat di `requirements.txt` dan commit, supaya anggota lain bisa menjalankan `pip install -r requirements.txt` lagi.
- **Tidak punya GPU?** Semua yang ada di `notebooks/01_eda.ipynb` dan algoritma klasik (SVM, Naive Bayes, Random Forest/XGBoost) berjalan di CPU. GPU hanya terasa untuk model ANN.

## Status
- Dataset `data/raw/comments_cleaned.csv`: 6.566 komentar dari 4 post, **belum berlabel**.
- EDA tersedia di `notebooks/01_eda.ipynb` (grafik di `results/figures/`), berisi rekomendasi skema pelabelan.
- Pembagian tugas: lihat `docs/pembagian_tugas.md`.
