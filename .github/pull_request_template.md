## Apa yang berubah
<!-- 1-3 kalimat. Tautkan issue: "Closes #12" -->

## Jenis perubahan
- [ ] Data / preprocessing
- [ ] Model / eksperimen
- [ ] Analisis / grafik
- [ ] Paper
- [ ] Infrastruktur / lainnya

## Eksperimen (jika ada)
<!-- Nama run di MLflow + metrik utama. JANGAN tempel link trycloudflare di sini. -->
| Run | Model | macro-F1 | Akurasi |
|---|---|---|---|
| | | | |

## Cek sebelum minta review
- [ ] Notebook sudah disinkron (`jupytext --sync notebooks/*.ipynb`), file `notebooks/py/*.py` ikut di-commit
- [ ] Tidak ada password, token, `.env`, atau link tunnel di commit
- [ ] Kode bisa dijalankan dari venv bersih (`pip install -r requirements.txt`)
- [ ] Bagian yang dibantu AI dicatat di `docs/deklarasi_ai.md`
