import os
import json
import pickle
from typing import Optional, Tuple

import pandas as pd
import tensorflow as tf


def ensure_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def load_tokenizer(num_words_default: int = 10000) -> tf.keras.preprocessing.text.Tokenizer:
	"""
	Load existing tokenizer from models/tokenizer.pkl. If not found, rebuild
	from cleaned_dataset.csv and save the pickle for future runs.

	Returns a Keras Tokenizer.
	"""
	tokenizer_path = os.path.join('models', 'tokenizer.pkl')
	if os.path.exists(tokenizer_path):
		with open(tokenizer_path, 'rb') as f:
			tokenizer = pickle.load(f)
		return tokenizer

	# Rebuild tokenizer from cleaned_dataset.csv
	data_path = 'cleaned_dataset.csv'
	if not os.path.exists(data_path):
		raise FileNotFoundError(
			'cleaned_dataset.csv tidak ditemukan, tidak bisa membangun tokenizer baru.'
		)
	print('⚠️ Tokenizer tidak ditemukan, membangun ulang dari cleaned_dataset.csv...')
	df = pd.read_csv(data_path)
	texts = df['text'].astype(str).values.tolist()
	tokenizer = tf.keras.preprocessing.text.Tokenizer(
		num_words=num_words_default,
		oov_token='<OOV>'
	)
	tokenizer.fit_on_texts(texts)
	# Simpan untuk reuse
	ensure_dir('models')
	with open(tokenizer_path, 'wb') as f:
		pickle.dump(tokenizer, f)
	print('✅ Tokenizer baru dibuat dan disimpan ke models/tokenizer.pkl')
	return tokenizer


def export_tokenizer_json(
	tokenizer: tf.keras.preprocessing.text.Tokenizer,
	max_len: int,
	max_words: Optional[int] = None,
	output_dir: str = 'src/assets'
) -> str:
	"""
	Export tokenizer ke JSON agar bisa dipakai di TS/JS mobile app.

	Menggunakan word_index yang dipotong sampai max_words jika diberikan.
	"""
	ensure_dir(output_dir)
	if max_words is None:
		# Gunakan konfigurasi tokenizer jika ada, fallback ke panjang vocab
		max_words = getattr(tokenizer, 'num_words', None) or len(tokenizer.word_index)

	# Filter word_index hingga batas max_words
	word_index = {k: v for k, v in tokenizer.word_index.items() if v < max_words}
	tokenizer_json = {
		'word_index': word_index,
		'max_len': int(max_len),
		'max_words': int(max_words)
	}
	output_path = os.path.join(output_dir, 'tokenizer.json')
	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(tokenizer_json, f)
	print(f"✅ Tokenizer JSON diekspor: {output_path} (vocab={len(word_index)}, max_len={max_len})")
	# Simpan juga salinan di models untuk arsip
	models_copy = os.path.join('models', 'tokenizer.json')
	with open(models_copy, 'w', encoding='utf-8') as f:
		json.dump(tokenizer_json, f)
	print(f"📦 Salinan tokenizer JSON disimpan: {models_copy}")
	return output_path


def try_convert_tflite_builtins_only(model: tf.keras.Model) -> Tuple[Optional[bytes], Optional[str]]:
	"""
	Konversi model Keras ke TFLite dengan hanya builtin ops (tanpa SELECT_TF_OPS).
	Mengembalikan tuple (tflite_model_bytes, error_message).
	"""
	try:
		converter = tf.lite.TFLiteConverter.from_keras_model(model)
		# Secara default hanya builtin ops, jangan set SELECT_TF_OPS.
		tflite_model = converter.convert()
		return tflite_model, None
	except Exception as e:
		return None, str(e)


def load_best_model_candidates(padding_size: int = 256) -> list:
	"""
	Mengembalikan daftar kandidat model .h5 untuk diekspor ke TFLite.
	Urutan prioritas: hybrid_cnn_gru, cnn1d, fasttext, gru, lstm.
	"""
	model_names = [
		f'hybrid_cnn_gru_best_{padding_size}.h5',
		f'cnn1d_best_{padding_size}.h5',
		f'fasttext_best_{padding_size}.h5',
		f'gru_best_{padding_size}.h5',
		f'lstm_best_{padding_size}.h5'
	]
	paths = [os.path.join('models', name) for name in model_names]
	return [p for p in paths if os.path.exists(p)]


def export_mobile_tflite(
	padding_size: int = 256,
	output_dir: str = 'src/assets'
) -> Optional[str]:
	"""
	Coba ekspor model terbaik ke TFLite dengan builtin ops saja.
	Jika kandidat pertama gagal (mis. perlu SELECT_TF_OPS), fallback ke kandidat
	berikutnya. Menyimpan hasil ke src/assets/model.tflite.
	"""
	ensure_dir(output_dir)
	candidates = load_best_model_candidates(padding_size)
	if not candidates:
		print('❌ Tidak menemukan model kandidat di folder models/.')
		return None

	for model_path in candidates:
		try:
			print(f"🔄 Mencoba konversi (builtin-only): {model_path}")
			model = tf.keras.models.load_model(model_path)
			tflite_model, err = try_convert_tflite_builtins_only(model)
			if tflite_model is not None:
				out_path = os.path.join(output_dir, 'model.tflite')
				with open(out_path, 'wb') as f:
					f.write(tflite_model)
				size_mb = os.path.getsize(out_path) / (1024 * 1024)
				print(f"✅ TFLite berhasil diekspor: {out_path} ({size_mb:.2f} MB)")
				return out_path
			else:
				print(f"⚠️ Gagal konversi {model_path} tanpa SELECT_TF_OPS: {err}")
		except Exception as e:
			print(f"⚠️ Error memuat/konversi model {model_path}: {e}")

	print('❌ Semua kandidat gagal dikonversi tanpa SELECT_TF_OPS.')
	return None


def main():
	# Parameter umum yang konsisten dengan notebook: max_len 256, padding post
	MAX_LEN = 256
	# Gunakan batas vocab yang lebih kecil agar model/JSON ringan untuk mobile
	# Jika tokenizer punya num_words, kita pakai itu; jika tidak, fallback ke 5000.
	MAX_WORDS_DEFAULT = 5000

	print('📦 Menyiapkan asset untuk mobile (TS/JS) ...')
	ensure_dir('src/assets')

	# Tokenizer
	tokenizer = load_tokenizer(num_words_default=MAX_WORDS_DEFAULT)
	max_words = getattr(tokenizer, 'num_words', None) or MAX_WORDS_DEFAULT
	export_tokenizer_json(tokenizer, max_len=MAX_LEN, max_words=max_words, output_dir='src/assets')

	# Model TFLite (builtin-only)
	_ = export_mobile_tflite(padding_size=MAX_LEN, output_dir='src/assets')

	print('🎯 Selesai mengekspor asset untuk mobile: src/assets/tokenizer.json dan src/assets/model.tflite')


if __name__ == '__main__':
	main()
