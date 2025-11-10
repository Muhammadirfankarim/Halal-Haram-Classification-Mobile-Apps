import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
from pathlib import Path
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Halal/Haram Food Classifier",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .halal-result {
        background: linear-gradient(135deg, #06A77D 0%, #05d69e 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .haram-result {
        background: linear-gradient(135deg, #C73E1D 0%, #e74c3c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .confidence-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .ingredient-box {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-family: monospace;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2E86AB 0%, #1a5f7a 100%);
        color: white;
        font-size: 1.2rem;
        padding: 0.8rem;
        border: none;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1a5f7a 0%, #0d3b4d 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MANUAL PAD SEQUENCES (Fix for keras.preprocessing issue)
# ============================================================================

def pad_sequences_manual(sequences, maxlen=256, padding='post', truncating='post', value=0):
    """Manual implementation of pad_sequences to avoid keras.preprocessing import issues"""
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        seq = list(seq)
        
        if len(seq) > maxlen:
            # Truncate
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        
        if len(seq) < maxlen:
            # Pad
            if padding == 'post':
                padded[i, :len(seq)] = seq
            else:  # 'pre'
                padded[i, -len(seq):] = seq
        else:
            padded[i] = seq
    
    return padded

# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================

@st.cache_resource
def load_model(model_path):
    """Load Keras or TFLite model"""
    try:
        if model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path)
            return model, 'keras'
        elif model_path.endswith('.tflite'):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter, 'tflite'
        else:
            st.error(f"❌ Unsupported model format: {model_path}")
            return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

@st.cache_resource
def load_tokenizer(tokenizer_path):
    """Load tokenizer from pickle or JSON"""
    try:
        if tokenizer_path.endswith('.pkl'):
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            return tokenizer, 'pickle'
        elif tokenizer_path.endswith('.json'):
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_json = json.load(f)
            return tokenizer_json, 'json'
        else:
            st.error(f"❌ Unsupported tokenizer format: {tokenizer_path}")
            return None, None
    except Exception as e:
        st.error(f"❌ Error loading tokenizer: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_text(text, tokenizer, tokenizer_type, max_length=256):
    """Preprocess text for model input"""
    
    try:
        if tokenizer_type == 'pickle':
            # Keras tokenizer - use manual padding
            sequences = tokenizer.texts_to_sequences([text])
            
            # Use manual pad_sequences to avoid keras.preprocessing issues
            padded = pad_sequences_manual(
                sequences, 
                maxlen=max_length, 
                padding='post', 
                truncating='post'
            )
            
            return padded.astype(np.int32)
        
        elif tokenizer_type == 'json':
            # Manual tokenization from JSON
            word_index = tokenizer['word_index']
            config = tokenizer.get('config', {})
            
            # Lowercase if needed
            if config.get('lower', True):
                text = text.lower()
            
            # Split
            words = text.split(config.get('split', ' '))
            
            # Map to indices
            sequence = []
            for word in words:
                if word in word_index:
                    sequence.append(word_index[word])
                elif config.get('oov_token') and config['oov_token'] in word_index:
                    sequence.append(word_index[config['oov_token']])
            
            # Pad manually
            if len(sequence) < max_length:
                sequence = sequence + [0] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
            
            return np.array([sequence], dtype=np.int32)
    
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================================================
# PREDICTION
# ============================================================================

def predict(text, model, model_type, tokenizer, tokenizer_type, max_length=256):
    """Make prediction"""
    
    try:
        # Preprocess
        input_data = preprocess_text(text, tokenizer, tokenizer_type, max_length)
        
        if input_data is None:
            return None
        
        # Predict
        if model_type == 'keras':
            prediction = model.predict(input_data, verbose=0)
            probability = float(prediction[0][0])
        
        elif model_type == 'tflite':
            # Get input/output details
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            # Set input
            model.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            model.invoke()
            
            # Get output
            output = model.get_tensor(output_details[0]['index'])
            probability = float(output[0][0])
        
        # Classify
        is_haram = probability > 0.5
        confidence = probability if is_haram else (1 - probability)
        
        return {
            'is_haram': is_haram,
            'probability': probability,
            'confidence': confidence,
            'label': 'HARAM' if is_haram else 'HALAL'
        }
    
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🍔 Halal/Haram Food Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Bilingual Deep Learning Model for Food Ingredient Classification</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("📦 Model")
        
        # Check if models directory exists
        models_dir = Path('models')
        if not models_dir.exists():
            st.error("❌ 'models/' directory not found!")
            st.info("Please create a 'models/' folder and add your model files")
            return
        
        model_files = list(models_dir.glob('*.h5')) + list(models_dir.glob('*.tflite'))
        
        if not model_files:
            st.error("❌ No models found in 'models/' directory!")
            st.info("Please add your .h5 or .tflite model files to the 'models/' folder")
            return
        
        model_file = st.selectbox(
            "Select Model",
            model_files,
            format_func=lambda x: x.name
        )
        
        # Tokenizer selection
        st.subheader("📝 Tokenizer")
        tokenizer_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('tokenizer*.json'))
        
        if not tokenizer_files:
            st.error("❌ No tokenizer found!")
            st.info("Please add tokenizer.pkl or tokenizer.json to the 'models/' folder")
            return
        
        tokenizer_file = st.selectbox(
            "Select Tokenizer",
            tokenizer_files,
            format_func=lambda x: x.name
        )
        
        # Max length
        max_length = st.slider("Max Sequence Length", 64, 512, 256, 64)
        
        # Load button
        if st.button("🔄 Load Model & Tokenizer"):
            with st.spinner("Loading model and tokenizer..."):
                st.session_state.model, st.session_state.model_type = load_model(str(model_file))
                st.session_state.tokenizer, st.session_state.tokenizer_type = load_tokenizer(str(tokenizer_file))
                st.session_state.max_length = max_length
                
                if st.session_state.model is not None and st.session_state.tokenizer is not None:
                    st.success(f"✅ Model loaded: {model_file.name}")
                    st.success(f"✅ Tokenizer loaded: {tokenizer_file.name}")
        
        st.markdown("---")
        
        # Model info
        if 'model' in st.session_state and st.session_state.model is not None:
            st.subheader("📊 Model Info")
            st.info(f"""
            **Type:** {st.session_state.model_type.upper()}
            **Max Length:** {st.session_state.max_length}
            **Tokenizer:** {st.session_state.tokenizer_type}
            """)
        
        # About
        st.subheader("ℹ️ About")
        st.info("""
        **How it works:**
        1. Enter food ingredients
        2. Model analyzes text
        3. Get halal/haram prediction
        
        **Supports:**
        - English & Indonesian
        - Multiple ingredients
        - Real-time prediction
        """)
        
        # Stats
        if 'prediction_count' in st.session_state:
            st.metric("Predictions Made", st.session_state.prediction_count)
    
    # Main content
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ Please load a model from the sidebar first!")
        
        st.markdown("### 📚 Quick Start Guide")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1️⃣ Prepare Files")
            st.code("""
models/
  ├── your_model.h5
  └── tokenizer.pkl
            """)
        
        with col2:
            st.markdown("#### 2️⃣ Load Model")
            st.write("Select model and tokenizer from sidebar")
        
        with col3:
            st.markdown("#### 3️⃣ Start Testing")
            st.write("Enter ingredients and get predictions!")
        
        return
    
    # Input methods
    st.header("📝 Input Ingredients")
    
    tab1, tab2, tab3 = st.tabs(["✍️ Manual Input", "📄 Examples", "🔢 Batch Testing"])
    
    with tab1:
        st.subheader("Enter Food Ingredients")
        
        # Text input
        user_input = st.text_area(
            "Ingredients (comma-separated):",
            placeholder="Example: chicken, salt, pepper, onion, garlic\nContoh: ayam, garam, lada, bawang, bawang putih",
            height=150,
            help="Enter ingredients in English or Indonesian, separated by commas"
        )
        
        # Language hint
        col1, col2 = st.columns(2)
        with col1:
            st.caption("🇬🇧 English: chicken, beef, pork, wine, gelatin")
        with col2:
            st.caption("🇮🇩 Indonesian: ayam, sapi, babi, arak, gelatin")
        
        # Predict button
        if st.button("🔮 Predict", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing ingredients..."):
                    # Predict
                    result = predict(
                        user_input,
                        st.session_state.model,
                        st.session_state.model_type,
                        st.session_state.tokenizer,
                        st.session_state.tokenizer_type,
                        st.session_state.max_length
                    )
                    
                    if result is None:
                        st.error("❌ Prediction failed. Please check the error messages above.")
                        return
                    
                    # Update counter
                    if 'prediction_count' not in st.session_state:
                        st.session_state.prediction_count = 0
                    st.session_state.prediction_count += 1
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    # Result box
                    if result['is_haram']:
                        st.markdown(f'<div class="haram-result">⚠️ HARAM</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="halal-result">✅ HALAL</div>', unsafe_allow_html=True)
                    
                    # Details
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                    
                    with col2:
                        st.metric("Probability Score", f"{result['probability']:.4f}")
                    
                    with col3:
                        status = "🔴 High Risk" if result['is_haram'] else "🟢 Safe"
                        st.metric("Status", status)
                    
                    # Ingredients display
                    st.markdown("### 🧾 Analyzed Ingredients:")
                    ingredients = [i.strip() for i in user_input.split(',') if i.strip()]
                    
                    cols = st.columns(min(3, len(ingredients)))
                    for idx, ingredient in enumerate(ingredients):
                        with cols[idx % len(cols)]:
                            st.markdown(f'<div class="ingredient-box">• {ingredient}</div>', unsafe_allow_html=True)
                    
                    # Interpretation
                    st.markdown("### 💡 Interpretation")
                    if result['is_haram']:
                        st.error("""
                        **⚠️ This product may contain haram ingredients.**
                        
                        The model detected patterns associated with non-halal ingredients. 
                        Please verify the ingredient list carefully before consumption.
                        """)
                    else:
                        st.success("""
                        **✅ This product appears to be halal.**
                        
                        The model did not detect any haram ingredients. However, 
                        always check for halal certification for complete assurance.
                        """)
            else:
                st.warning("⚠️ Please enter some ingredients first!")
    
    with tab2:
        st.subheader("📄 Example Ingredients")
        
        examples = {
            "✅ Halal Example 1": "chicken breast, salt, black pepper, olive oil, garlic",
            "✅ Halal Example 2": "ayam, garam, lada hitam, minyak zaitun, bawang putih",
            "✅ Halal Example 3": "beef, potato, carrot, onion, tomato sauce",
            "⚠️ Haram Example 1": "pork sausage, bacon, ham, lard",
            "⚠️ Haram Example 2": "wine, beer, alcohol, rum extract",
            "⚠️ Haram Example 3": "gelatin from pork, bacon bits, lard, pork fat",
        }
        
        for name, ingredients in examples.items():
            with st.expander(name):
                st.code(ingredients)
                if st.button(f"Test This Example", key=f"test_{name}"):
                    with st.spinner("Analyzing..."):
                        result = predict(
                            ingredients,
                            st.session_state.model,
                            st.session_state.model_type,
                            st.session_state.tokenizer,
                            st.session_state.tokenizer_type,
                            st.session_state.max_length
                        )
                        
                        if result:
                            if result['is_haram']:
                                st.markdown(f'<div class="haram-result">⚠️ HARAM</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="halal-result">✅ HALAL</div>', unsafe_allow_html=True)
                            
                            st.metric("Confidence", f"{result['confidence']*100:.2f}%")
    
    # Ganti bagian tab3 (Batch Testing) dengan kode berikut:

    with tab3:
        st.subheader("🔢 Batch Testing")
        
        st.info("Upload a CSV file with 'ingredients' column for batch prediction")
        
        # CSV format info
        with st.expander("📋 CSV Format Guide"):
            st.markdown("""
            **Required Column:**
            - `ingredients` - Food ingredients text
            
            **Optional Columns:**
            - `category` - Product category
            - `expected_label` - Expected result (for validation)
            - `reason` - Reason for classification
            
            **Supported Delimiters:**
            - Comma (`,`)
            - Semicolon (`;`)
            - Tab (`\\t`)
            
            **Example CSV:**
    ```
            ingredients;expected_label
            chicken breast salt pepper;halal
            pork bacon wine;haram
    ```
            """)
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file:
            import pandas as pd
            
            try:
                # Try to detect delimiter
                first_line = uploaded_file.readline().decode('utf-8')
                uploaded_file.seek(0)  # Reset file pointer
                
                # Detect delimiter
                if ';' in first_line:
                    delimiter = ';'
                    st.info(f"✅ Detected delimiter: semicolon (`;`)")
                elif '\t' in first_line:
                    delimiter = '\t'
                    st.info(f"✅ Detected delimiter: tab")
                else:
                    delimiter = ','
                    st.info(f"✅ Detected delimiter: comma (`,`)")
                
                # Read CSV with detected delimiter
                df = pd.read_csv(uploaded_file, delimiter=delimiter)
                
                st.success(f"✅ CSV loaded successfully! {len(df)} rows found")
                
                # Show columns
                st.write(f"**Columns found:** {', '.join(df.columns.tolist())}")
                
                # Preview
                st.write("**Preview:**")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Check for ingredients column
                if 'ingredients' not in df.columns:
                    st.error("❌ CSV must have 'ingredients' column!")
                    
                    # Suggest column mapping
                    st.warning("**Available columns:**")
                    for col in df.columns:
                        st.write(f"- `{col}`")
                    
                    st.info("💡 Tip: Rename your ingredients column to 'ingredients' or select it below:")
                    
                    ingredient_col = st.selectbox(
                        "Select the column containing ingredients:",
                        df.columns.tolist()
                    )
                    
                    if st.button("✅ Use This Column"):
                        df = df.rename(columns={ingredient_col: 'ingredients'})
                        st.success(f"✅ Column '{ingredient_col}' mapped to 'ingredients'")
                        st.rerun()
                
                else:
                    # Batch prediction
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        include_expected = st.checkbox(
                            "Compare with expected labels",
                            value='expected_label' in df.columns
                        )
                    
                    with col2:
                        show_confidence = st.checkbox("Show confidence scores", value=True)
                    
                    if st.button("🚀 Run Batch Prediction", type="primary"):
                        with st.spinner("Processing batch..."):
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, row in df.iterrows():
                                status_text.text(f"Processing {idx + 1}/{len(df)}...")
                                
                                result = predict(
                                    str(row['ingredients']),
                                    st.session_state.model,
                                    st.session_state.model_type,
                                    st.session_state.tokenizer,
                                    st.session_state.tokenizer_type,
                                    st.session_state.max_length
                                )
                                
                                if result:
                                    results.append(result)
                                else:
                                    results.append({
                                        'label': 'ERROR',
                                        'confidence': 0,
                                        'is_haram': False,
                                        'probability': 0
                                    })
                                
                                progress_bar.progress((idx + 1) / len(df))
                            
                            status_text.empty()
                            
                            # Add results to dataframe
                            df['prediction'] = [r['label'] for r in results]
                            
                            if show_confidence:
                                df['confidence'] = [f"{r['confidence']*100:.2f}%" for r in results]
                                df['probability'] = [f"{r['probability']:.4f}" for r in results]
                            
                            # Add correctness column if expected_label exists
                            if include_expected and 'expected_label' in df.columns:
                                df['correct'] = df.apply(
                                    lambda row: '✅' if row['prediction'].lower() == str(row['expected_label']).lower() else '❌',
                                    axis=1
                                )
                                
                                # Calculate accuracy
                                accuracy = (df['correct'] == '✅').sum() / len(df) * 100
                                
                                st.success(f"✅ Batch prediction complete! Accuracy: {accuracy:.2f}%")
                                
                                # Show confusion matrix
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    correct_count = (df['correct'] == '✅').sum()
                                    st.metric("Correct Predictions", f"{correct_count}/{len(df)}")
                                
                                with col2:
                                    halal_count = (df['prediction'] == 'HALAL').sum()
                                    st.metric("Predicted HALAL", halal_count)
                                
                                with col3:
                                    haram_count = (df['prediction'] == 'HARAM').sum()
                                    st.metric("Predicted HARAM", haram_count)
                            
                            else:
                                st.success("✅ Batch prediction complete!")
                                
                                # Summary statistics
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    halal_count = (df['prediction'] == 'HALAL').sum()
                                    st.metric("Predicted HALAL", halal_count)
                                
                                with col2:
                                    haram_count = (df['prediction'] == 'HARAM').sum()
                                    st.metric("Predicted HARAM", haram_count)
                            
                            # Display results
                            st.dataframe(df, use_container_width=True, height=400)
                            
                            # Download results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                csv = df.to_csv(index=False, sep=';').encode('utf-8')
                                st.download_button(
                                    "📥 Download Results (CSV with ;)",
                                    csv,
                                    "batch_predictions.csv",
                                    "text/csv"
                                )
                            
                            with col2:
                                csv_comma = df.to_csv(index=False, sep=',').encode('utf-8')
                                st.download_button(
                                    "📥 Download Results (CSV with ,)",
                                    csv_comma,
                                    "batch_predictions_comma.csv",
                                    "text/csv"
                                )
                            
                            # Detailed analysis
                            if include_expected and 'expected_label' in df.columns:
                                st.markdown("---")
                                st.subheader("📊 Detailed Analysis")
                                
                                # Show errors
                                errors = df[df['correct'] == '❌']
                                if len(errors) > 0:
                                    st.error(f"❌ {len(errors)} incorrect predictions found:")
                                    st.dataframe(
                                        errors[['ingredients', 'expected_label', 'prediction', 'confidence']],
                                        use_container_width=True
                                    )
                                else:
                                    st.success("🎉 Perfect predictions! All results match expected labels!")
            
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")
                
                st.info("💡 **Troubleshooting:**")
                st.markdown("""
                1. Make sure your CSV is properly formatted
                2. Check that you have an 'ingredients' column
                3. Try saving your CSV with different delimiter (`;` or `,`)
                4. Ensure the file is UTF-8 encoded
                """)
                
                # Show detailed error
                with st.expander("🔍 Show detailed error"):
                    import traceback
                    st.code(traceback.format_exc())