import nbformat as nbf
import os


NOTEBOOK_03 = '03_Deep_Learning_Model_Experimentation.ipynb'
NOTEBOOK_04 = '04_Final_Evaluation_and_TFLite_Export.ipynb'


def add_guidance_cell(nb):
	md = nbf.v4.new_markdown_cell(
		"""
### 🚦 Petunjuk Eksekusi (Run All)

- Jalankan notebook ini dari atas ke bawah tanpa melewati cell.
- Pastikan dependensi sudah terpasang sesuai `requirements.txt`.
- Preprocessing diseragamkan: Tokenizer Keras, padding/truncating 'post', MAX_LEN=256.
- Untuk deployment mobile, gunakan ekspor TFLite bawaan (builtin-only) dan tokenizer JSON.
"""
	)
	# Sisipkan setelah cell pertama agar terlihat di awal
	nb.cells.insert(1, md)


def add_mobile_export_cells(nb):
	md = nbf.v4.new_markdown_cell(
		"""
## 📱 Mobile Export (Builtin-only)

Menjalankan pipeline ekspor untuk:
- Konversi model terbaik ke TFLite builtin-only bila memungkinkan.
- Pemilihan model mobile terbaik berdasarkan metrik evaluasi.
- Penyalinan `model.tflite` ke `src/assets`.
- Ekspor `tokenizer.json` agar kompatibel dengan aplikasi mobile.
"""
	)
	code = nbf.v4.new_code_cell(
		"""
# 📱 Mobile Export: jalankan pipeline
try:
	from mobile_compat_pipeline import main as mobile_export_main
	mobile_export_main()
except Exception as e:
	print('⚠️ Mobile export gagal:', e)
"""
	)
	nb.cells.append(md)
	nb.cells.append(code)


def add_tokenizer_export_cells(nb):
	md = nbf.v4.new_markdown_cell(
		"""
## 🔑 Tokenizer JSON Export

Mengekspor tokenizer Keras (pickle) ke format JSON yang kompatibel untuk
aplikasi mobile (TS/JS), termasuk `word_index`, `max_len`, `max_words`, dan `oov_token`.
"""
	)
	code = nbf.v4.new_code_cell(
		"""
# 🔑 Ekspor Tokenizer ke JSON (mobile-compatible)
try:
	from mobile_compat_pipeline import load_tokenizer, export_tokenizer_json
	 tok = load_tokenizer()
	 max_words = getattr(tok, 'num_words', None) or len(tok.word_index)
	 out = export_tokenizer_json(tok, 256, max_words)
	 print('✅ Tokenizer JSON diekspor:', out)
except Exception as e:
	print('⚠️ Ekspor tokenizer JSON gagal:', e)
"""
	)
	nb.cells.append(md)
	nb.cells.append(code)


def process_notebook(path: str, add_mobile: bool = False, add_tokenizer: bool = False):
	if not os.path.exists(path):
		raise FileNotFoundError(f'Notebook tidak ditemukan: {path}')
	with open(path, 'r', encoding='utf-8') as f:
		nb = nbf.read(f, as_version=4)

	# Tambah guidance cell
	add_guidance_cell(nb)

	# Tambahan sesuai permintaan
	if add_mobile:
		add_mobile_export_cells(nb)
	if add_tokenizer:
		add_tokenizer_export_cells(nb)

	with open(path, 'w', encoding='utf-8') as f:
		nbf.write(nb, f)


def main():
	# Notebook 04: tambah guidance + mobile export
	process_notebook(NOTEBOOK_04, add_mobile=True, add_tokenizer=False)
	print(f'✅ Updated: {NOTEBOOK_04} (guidance + mobile export cells)')

	# Notebook 03: tambah guidance + tokenizer export
	process_notebook(NOTEBOOK_03, add_mobile=False, add_tokenizer=True)
	print(f'✅ Updated: {NOTEBOOK_03} (guidance + tokenizer export cells)')


if __name__ == '__main__':
	main()

