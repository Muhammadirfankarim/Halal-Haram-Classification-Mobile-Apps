import os
import json
import pickle
import importlib
from typing import Dict, List, Optional, Tuple

import tensorflow as tf


def ensure_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def _load_tokenizer_pickle_resilient(f):
    """Load Tokenizer pickle lintas versi Keras/TF.

    Memetakan path modul yang berubah (mis. 'keras.src.preprocessing.text')
    ke modul yang tersedia ('tensorflow.keras.preprocessing.text') agar
    deserialisasi tidak gagal.
    """

    class ResilientUnpickler(pickle.Unpickler):
        MODULE_MAP = {
            ('keras.src.preprocessing.text', 'Tokenizer'):
                ('tensorflow.keras.preprocessing.text', 'Tokenizer'),
            ('keras.preprocessing.text', 'Tokenizer'):
                ('tensorflow.keras.preprocessing.text', 'Tokenizer'),
            ('keras_preprocessing.text', 'Tokenizer'):
                ('tensorflow.keras.preprocessing.text', 'Tokenizer'),
        }

        def find_class(self, module, name):
            mapped = self.MODULE_MAP.get((module, name))
            if mapped:
                module, name = mapped
            try:
                mod = importlib.import_module(module)
                return getattr(mod, name)
            except Exception:
                return super().find_class(module, name)

    try:
        return ResilientUnpickler(f).load()
    except Exception:
        f.seek(0)
        return pickle.load(f)


def load_tokenizer() -> tf.keras.preprocessing.text.Tokenizer:
    path = os.path.join('models', 'tokenizer.pkl')
    with open(path, 'rb') as f:
        return _load_tokenizer_pickle_resilient(f)


def export_tokenizer_json(tokenizer, max_len: int, max_words: Optional[int] = None) -> str:
	ensure_dir('src/assets')
	if max_words is None:
		max_words = getattr(tokenizer, 'num_words', None) or len(tokenizer.word_index)
	word_index = {k: v for k, v in tokenizer.word_index.items() if v < max_words}
	payload = {
		'word_index': word_index,
		'max_len': int(max_len),
		'max_words': int(max_words),
		'oov_token': getattr(tokenizer, 'oov_token', None)
	}
	out = os.path.join('src', 'assets', 'tokenizer.json')
	with open(out, 'w', encoding='utf-8') as f:
		json.dump(payload, f)
	return out


def try_convert_builtin_only(model: tf.keras.Model) -> Tuple[Optional[bytes], Optional[str]]:
	try:
		converter = tf.lite.TFLiteConverter.from_keras_model(model)
		# Hanya builtin ops, jangan set SELECT_TF_OPS
		return converter.convert(), None
	except Exception as e:
		return None, str(e)


def candidate_models(padding_size: int = 256) -> Dict[str, str]:
	"""
	Map nama model -> path .h5 jika ada.
	"""
	base = 'models'
	order = [
		f'hybrid_cnn_gru_best_{padding_size}.h5',
		f'cnn1d_best_{padding_size}.h5',
		f'fasttext_best_{padding_size}.h5',
		f'gru_best_{padding_size}.h5',
		f'lstm_best_{padding_size}.h5',
	]
	result = {}
	for fname in order:
		path = os.path.join(base, fname)
		if os.path.exists(path):
			name = fname.replace(f'_best_{padding_size}.h5', '')
			result[name] = path
	return result


def mobile_support_map() -> Dict[str, bool]:
	# Perkiraan dukungan builtin-only
	return {
		'fasttext': True,   # Embedding + AvgPool + Dense
		'cnn1d': True,      # Conv1D + Pooling + Dense
		'lstm': True,       # TFLite punya builtin LSTM (tergantung konfigurasi)
		'gru': False,       # Sering perlu SELECT_TF_OPS
		'hybrid_cnn_gru': False,
	}


def export_all_builtin(padding_size: int = 256) -> Dict[str, Dict[str, Optional[str]]]:
	"""
	Coba konversi semua kandidat ke builtin-only.
	Return dict: {model_name: {status, out_path, error}}
	"""
	ensure_dir(os.path.join('models', 'mobile_builtins'))
	summary = {}
	for name, path in candidate_models(padding_size).items():
		entry = {'status': None, 'out_path': None, 'error': None}
		try:
			model = tf.keras.models.load_model(path)
			blob, err = try_convert_builtin_only(model)
			if blob:
				out_name = f'{name}_builtin.tflite'
				out_path = os.path.join('models', 'mobile_builtins', out_name)
				with open(out_path, 'wb') as f:
					f.write(blob)
				entry['status'] = 'success'
				entry['out_path'] = out_path
			else:
				entry['status'] = 'failed'
				entry['error'] = err
		except Exception as e:
			entry['status'] = 'error'
			entry['error'] = str(e)
		summary[name] = entry
	return summary


def load_metrics(padding_size: int = 256) -> Dict[str, Dict]:
	"""
	Muat metrik per model dari experiment_results/deep_learning_results_{padding_size}.json
	Mengembalikan dict dengan kunci model canonical: fasttext, cnn1d, lstm, gru, hybrid_cnn_gru
	"""
	metrics = {}
	path = os.path.join('experiment_results', f'deep_learning_results_{padding_size}.json')
	if os.path.exists(path):
		try:
			with open(path, 'r', encoding='utf-8') as f:
				raw = json.load(f)
			# Pemetaan nama dari file ke canonical
			name_map = {
				'FastText': 'fasttext',
				'CNN1D': 'cnn1d',
				'LSTM': 'lstm',
				'GRU': 'gru',
				'Hybrid_CNN_GRU': 'hybrid_cnn_gru',
			}
			for k, v in raw.items():
				canon = name_map.get(k)
				if canon:
					metrics[canon] = v
		except Exception:
			pass
	return metrics


def choose_best_mobile(summary: Dict[str, Dict[str, Optional[str]]], padding_size: int = 256) -> Optional[str]:
	"""
	Pilih model terbaik yang berhasil konversi builtin-only.
	Kita lihat experiment_results/final_evaluation_results_256.json jika ada,
	memilih berdasarkan 'f1' atau 'accuracy'. Jika tidak ada, pilih urutan
	prioritas: cnn1d > fasttext > lstm.
	"""
	metrics = load_metrics(padding_size)

	# Kandidat yang berhasil
	success = [name for name, info in summary.items() if info.get('status') == 'success']
	if not success:
		return None

	# Jika ada metrics, pilih berdasarkan f1 lalu accuracy
	def score(name: str) -> float:
		m = metrics.get(name, {})
		# gunakan test_f1 lalu test_accuracy jika tersedia
		return float(m.get('test_f1', m.get('test_accuracy', 0.0)))

	priority = ['cnn1d', 'fasttext', 'lstm']
	best = sorted(success, key=lambda n: (score(n), n in priority), reverse=True)[0]
	return best


def write_mobile_summary(summary: Dict[str, Dict[str, Optional[str]]], best_mobile: Optional[str], padding_size: int = 256) -> str:
	ensure_dir('experiment_results')
	payload = {
		'padding_size': padding_size,
		'models': summary,
		'best_mobile': best_mobile
	}
	out = os.path.join('experiment_results', f'mobile_compatibility_summary_{padding_size}.json')
	with open(out, 'w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2)
	return out


def main():
	PADDING = 256
	print('🔧 Menyiapkan ekspor TFLite builtin-only untuk 5 model...')
	# Ekspor tokenizer JSON
	try:
		tok = load_tokenizer()
		max_words = getattr(tok, 'num_words', None) or len(tok.word_index)
		path_tok = export_tokenizer_json(tok, max_len=PADDING, max_words=max_words)
		print(f'✅ Tokenizer JSON: {path_tok}')
	except Exception as e:
		print(f'⚠️ Gagal mengekspor tokenizer JSON: {e}')

	# Konversi semua kandidat ke builtin-only
	summary = export_all_builtin(padding_size=PADDING)
	for name, info in summary.items():
		print(f" - {name}: {info['status']} {info.get('out_path') or info.get('error')}")

	best_mobile = choose_best_mobile(summary, padding_size=PADDING)
	path_summary = write_mobile_summary(summary, best_mobile, padding_size=PADDING)
	print(f'📄 Ringkasan kompatibilitas: {path_summary}')

	# Jika ada best mobile, salin ke src/assets/model.tflite
	if best_mobile and summary[best_mobile]['out_path']:
		ensure_dir(os.path.join('src', 'assets'))
		final_out = os.path.join('src', 'assets', 'model.tflite')
		with open(summary[best_mobile]['out_path'], 'rb') as fsrc, open(final_out, 'wb') as fdst:
			fdst.write(fsrc.read())
		print(f'🎯 Model mobile terbaik: {best_mobile} -> {final_out}')
		# Simpan info best ke file
		best_info_path = os.path.join('models', 'mobile_builtins', 'best_mobile_model.txt')
		with open(best_info_path, 'w', encoding='utf-8') as f:
			f.write(best_mobile)
		print(f'📝 Best mobile info saved: {best_info_path}')
	else:
		print('❌ Tidak ada model yang berhasil dikonversi builtin-only. Lihat ringkasan untuk detail.')


if __name__ == '__main__':
	main()
