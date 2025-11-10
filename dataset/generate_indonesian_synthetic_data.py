import pandas as pd
import numpy as np
import re
from typing import List, Dict
import random

# Comprehensive ingredient translation dictionary (English -> Indonesian)
INGREDIENT_TRANSLATIONS = {
    # Meat & Poultry
    'pork': 'babi', 'bacon': 'bacon babi', 'ham': 'ham babi', 
    'pork fat': 'lemak babi', 'lard': 'lemak babi', 
    'pork sausage': 'sosis babi', 'pork belly': 'perut babi',
    'prosciutto': 'prosciutto babi', 'pancetta': 'pancetta babi',
    
    'beef': 'daging sapi', 'chicken': 'ayam', 'turkey': 'kalkun',
    'lamb': 'daging kambing', 'goat': 'kambing', 'duck': 'bebek',
    'veal': 'daging sapi muda', 'venison': 'daging rusa',
    
    # Seafood
    'clam': 'kerang', 'clams': 'kerang', 'oyster': 'tiram', 
    'oysters': 'tiram', 'mussel': 'remis', 'mussels': 'remis',
    'shrimp': 'udang', 'prawns': 'udang', 'crab': 'kepiting',
    'lobster': 'lobster', 'fish': 'ikan', 'salmon': 'salmon',
    'tuna': 'tuna', 'cod': 'ikan kod', 'anchovy': 'ikan teri',
    'squid': 'cumi-cumi', 'octopus': 'gurita', 'scallop': 'kerang kampak',
    
    # Dairy
    'milk': 'susu', 'cream': 'krim', 'butter': 'mentega',
    'cheese': 'keju', 'cheddar': 'keju cheddar', 'mozzarella': 'keju mozzarella',
    'parmesan': 'keju parmesan', 'yogurt': 'yogurt', 'whey': 'whey',
    'sour cream': 'krim asam', 'buttermilk': 'susu mentega',
    'condensed milk': 'susu kental manis', 'evaporated milk': 'susu evaporasi',
    'whole milk': 'susu murni', 'skim milk': 'susu skim',
    'nonfat milk': 'susu tanpa lemak', 'cultured milk': 'susu kultur',
    
    # Alcohol
    'wine': 'anggur', 'beer': 'bir', 'alcohol': 'alkohol',
    'rum': 'rum', 'whiskey': 'wiski', 'vodka': 'vodka',
    'sake': 'sake', 'brandy': 'brendi', 'liqueur': 'likeur',
    'champagne': 'sampanye', 'cooking wine': 'anggur masak',
    'rice wine': 'arak beras', 'mirin': 'mirin',
    
    # Vegetables
    'carrot': 'wortel', 'carrots': 'wortel', 'celery': 'seledri',
    'onion': 'bawang bombay', 'onions': 'bawang bombay', 
    'garlic': 'bawang putih', 'potato': 'kentang', 'potatoes': 'kentang',
    'tomato': 'tomat', 'tomatoes': 'tomat', 'broccoli': 'brokoli',
    'spinach': 'bayam', 'lettuce': 'selada', 'cabbage': 'kubis',
    'bell pepper': 'paprika', 'red bell pepper': 'paprika merah',
    'green bell pepper': 'paprika hijau', 'chili': 'cabai',
    'mushroom': 'jamur', 'mushrooms': 'jamur', 'corn': 'jagung',
    'peas': 'kacang polong', 'beans': 'kacang', 'green beans': 'buncis',
    'navy beans': 'kacang navy', 'chick peas': 'kacang arab',
    'cucumber': 'mentimun', 'eggplant': 'terong', 'zucchini': 'zukini',
    
    # Fruits
    'apple': 'apel', 'apples': 'apel', 'orange': 'jeruk',
    'lemon': 'lemon', 'lime': 'jeruk nipis', 'peach': 'persik',
    'peaches': 'persik', 'strawberry': 'stroberi', 'blueberry': 'bluberi',
    'raspberry': 'raspberry', 'grape': 'anggur', 'banana': 'pisang',
    'mango': 'mangga', 'pineapple': 'nanas', 'watermelon': 'semangka',
    
    # Grains & Flour
    'wheat': 'gandum', 'wheat flour': 'tepung gandum', 'flour': 'tepung',
    'enriched wheat flour': 'tepung gandum diperkaya',
    'whole wheat flour': 'tepung gandum utuh',
    'rice': 'beras', 'rice flour': 'tepung beras',
    'corn': 'jagung', 'cornmeal': 'tepung jagung', 'cornstarch': 'tepung maizena',
    'oat': 'oat', 'oats': 'oat', 'barley': 'jelai',
    'malted barley flour': 'tepung jelai malt',
    'pasta': 'pasta', 'noodle': 'mi', 'bread': 'roti',
    
    # Oils & Fats
    'vegetable oil': 'minyak sayur', 'oil': 'minyak',
    'olive oil': 'minyak zaitun', 'extra virgin olive oil': 'minyak zaitun extra virgin',
    'canola oil': 'minyak kanola', 'soybean oil': 'minyak kedelai',
    'palm oil': 'minyak kelapa sawit', 'coconut oil': 'minyak kelapa',
    'sunflower oil': 'minyak bunga matahari', 'sesame oil': 'minyak wijen',
    'corn oil': 'minyak jagung', 'cottonseed oil': 'minyak biji kapas',
    'margarine': 'margarin', 'shortening': 'shortening',
    'hydrogenated': 'terhidrogenasi',
    
    # Sweeteners
    'sugar': 'gula', 'brown sugar': 'gula merah', 'cane sugar': 'gula tebu',
    'high fructose corn syrup': 'sirup jagung fruktosa tinggi',
    'corn syrup': 'sirup jagung', 'honey': 'madu', 'molasses': 'molase',
    'maple syrup': 'sirup maple', 'invert sugar': 'gula invert',
    'dextrose': 'dekstrosa', 'glucose': 'glukosa', 'fructose': 'fruktosa',
    
    # Spices & Seasonings
    'salt': 'garam', 'pepper': 'merica', 'black pepper': 'merica hitam',
    'white pepper': 'merica putih', 'paprika': 'bubuk paprika',
    'turmeric': 'kunyit', 'ginger': 'jahe', 'cinnamon': 'kayu manis',
    'clove': 'cengkeh', 'nutmeg': 'pala', 'cumin': 'jintan',
    'coriander': 'ketumbar', 'basil': 'kemangi', 'oregano': 'oregano',
    'thyme': 'thyme', 'rosemary': 'rosemary', 'parsley': 'peterseli',
    'bay leaf': 'daun salam', 'vanilla': 'vanila',
    'mustard': 'mustard', 'mustard seed': 'biji mustard',
    'soy sauce': 'kecap asin', 'vinegar': 'cuka',
    
    # Additives & Preservatives
    'gelatin': 'gelatin', 'gelatine': 'gelatin',
    'sodium nitrite': 'natrium nitrit', 'sodium nitrate': 'natrium nitrat',
    'monosodium glutamate': 'monosodium glutamat', 'msg': 'msg',
    'yeast': 'ragi', 'yeast extract': 'ekstrak ragi',
    'baking soda': 'soda kue', 'baking powder': 'baking powder',
    'citric acid': 'asam sitrat', 'ascorbic acid': 'asam askorbat',
    'vitamin c': 'vitamin c', 'vitamin a': 'vitamin a',
    'natural flavor': 'perisa alami', 'natural flavoring': 'perisa alami',
    'artificial flavor': 'perisa buatan',
    'food coloring': 'pewarna makanan', 'caramel color': 'pewarna karamel',
    'annatto': 'annatto', 'beta carotene': 'beta karoten',
    'lecithin': 'lesitin', 'soy lecithin': 'lesitin kedelai',
    
    # Stocks & Broths
    'stock': 'kaldu', 'broth': 'kaldu', 'chicken stock': 'kaldu ayam',
    'beef stock': 'kaldu sapi', 'vegetable stock': 'kaldu sayur',
    'pork stock': 'kaldu babi', 'bone broth': 'kaldu tulang',
    
    # Miscellaneous
    'water': 'air', 'ice': 'es', 'egg': 'telur', 'eggs': 'telur',
    'chocolate': 'coklat', 'cocoa': 'kakao', 'cocoa powder': 'bubuk kakao',
    'coffee': 'kopi', 'tea': 'teh', 'tamarind': 'asam jawa',
    'starch': 'pati', 'modified food starch': 'pati makanan termodifikasi',
    'modified corn starch': 'pati jagung termodifikasi',
    'contains': 'mengandung', 'less than': 'kurang dari',
    'soybean': 'kedelai', 'soybeans': 'kedelai',
    'wheat': 'gandum', 'gluten': 'gluten', 'wheat gluten': 'gluten gandum',
    'sesame': 'wijen', 'sesame seed': 'biji wijen',
    'extract': 'ekstrak', 'concentrate': 'konsentrat',
    'powder': 'bubuk', 'dried': 'kering', 'dehydrated': 'dehidrasi',
    'smoke flavor': 'perisa asap', 'smoke flavoring': 'perisa asap',
    'frankfurter': 'sosis frankfurter', 'sausage': 'sosis',
}

# Common phrases translation
PHRASE_TRANSLATIONS = {
    'contains less than': 'mengandung kurang dari',
    'contains or less of': 'mengandung atau kurang dari',
    'made with': 'dibuat dengan',
    'prepared': 'disiapkan',
    'enriched': 'diperkaya',
    'cultured': 'kultur',
    'processed with': 'diproses dengan',
    'water added': 'ditambah air',
    'no nitrates or nitrites added': 'tanpa nitrat atau nitrit tambahan',
    'except for those naturally occurring': 'kecuali yang alami terjadi',
    'for color': 'untuk pewarna',
    'vitamin': 'vitamin',
    'added': 'ditambahkan',
    'natural': 'alami',
}

def translate_ingredient(text: str) -> str:
    """
    Translate English ingredient text to Indonesian.
    Uses comprehensive dictionary and maintains food safety context.
    """
    text_lower = text.lower()
    translated = text_lower
    
    # First, translate common phrases
    for eng_phrase, indo_phrase in PHRASE_TRANSLATIONS.items():
        translated = translated.replace(eng_phrase, indo_phrase)
    
    # Then translate individual ingredients
    # Sort by length (longest first) to avoid partial matches
    sorted_ingredients = sorted(INGREDIENT_TRANSLATIONS.items(), 
                               key=lambda x: len(x[0]), 
                               reverse=True)
    
    for eng_ingredient, indo_ingredient in sorted_ingredients:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(eng_ingredient) + r'\b'
        translated = re.sub(pattern, indo_ingredient, translated, flags=re.IGNORECASE)
    
    return translated

def augment_text(text: str, method: str = 'synonym') -> str:
    """
    Apply text augmentation techniques to increase diversity.
    """
    if method == 'synonym':
        # Add slight variations in phrasing
        synonyms = {
            'mengandung': ['terdiri dari', 'berisi', 'memiliki'],
            'dan': ['serta', 'dengan', 'beserta'],
            'atau': ['ataupun', 'maupun'],
        }
        for original, replacements in synonyms.items():
            if original in text and random.random() > 0.7:
                text = text.replace(original, random.choice(replacements), 1)
    
    elif method == 'reorder':
        # Slightly reorder some comma-separated items (10% chance)
        if ',' in text and random.random() > 0.9:
            parts = text.split(',')
            if len(parts) > 2:
                # Swap two random adjacent items
                idx = random.randint(0, len(parts) - 2)
                parts[idx], parts[idx + 1] = parts[idx + 1], parts[idx]
                text = ','.join(parts)
    
    return text

def create_hybrid_text(text_en: str, text_id: str) -> str:
    """
    Create hybrid text mixing English and Indonesian (common in Indonesia).
    """
    # Split into components
    parts_en = [p.strip() for p in text_en.split(',') if p.strip()]
    parts_id = [p.strip() for p in text_id.split(',') if p.strip()]
    
    if len(parts_en) != len(parts_id) or len(parts_en) < 2:
        return text_id
    
    # Randomly mix English and Indonesian parts
    hybrid_parts = []
    for en, ind in zip(parts_en, parts_id):
        # 30% chance to keep English term, 70% Indonesian
        if random.random() < 0.3:
            hybrid_parts.append(en)
        else:
            hybrid_parts.append(ind)
    
    return ', '.join(hybrid_parts)

def generate_indonesian_synthetic_data(input_csv: str, output_csv: str):
    """
    Generate comprehensive Indonesian synthetic data from English dataset.
    """
    print("="*70)
    print("GENERATING INDONESIAN SYNTHETIC DATA")
    print("="*70)
    
    # Load original data
    print(f"\n📂 Loading data from: {input_csv}")
    df_original = pd.read_csv(input_csv)
    print(f"✅ Loaded {len(df_original):,} original samples")
    print(f"   - Halal: {sum(df_original['label'] == 'halal'):,}")
    print(f"   - Haram: {sum(df_original['label'] == 'haram'):,}")
    
    # Generate Indonesian translations
    print(f"\n🔄 Translating to Indonesian...")
    indonesian_data = []
    
    for idx, row in df_original.iterrows():
        if idx % 5000 == 0:
            print(f"   Progress: {idx:,}/{len(df_original):,} ({idx/len(df_original)*100:.1f}%)")
        
        original_text = row['text']
        label = row['label']
        
        # 1. Pure Indonesian translation
        indonesian_text = translate_ingredient(original_text)
        indonesian_data.append({
            'text': indonesian_text,
            'label': label,
            'source': 'id_translation'
        })
        
        # 2. Augmented Indonesian (with synonyms)
        if random.random() > 0.5:  # 50% chance
            augmented_text = augment_text(indonesian_text, method='synonym')
            indonesian_data.append({
                'text': augmented_text,
                'label': label,
                'source': 'id_augmented'
            })
        
        # 3. Hybrid English-Indonesian (common in Indonesia)
        if random.random() > 0.6:  # 40% chance
            hybrid_text = create_hybrid_text(original_text, indonesian_text)
            indonesian_data.append({
                'text': hybrid_text,
                'label': label,
                'source': 'hybrid'
            })
        
        # 4. Word order variation
        if random.random() > 0.8:  # 20% chance
            reordered_text = augment_text(indonesian_text, method='reorder')
            indonesian_data.append({
                'text': reordered_text,
                'label': label,
                'source': 'id_reordered'
            })
    
    # Create DataFrame
    df_indonesian = pd.DataFrame(indonesian_data)
    print(f"\n✅ Generated {len(df_indonesian):,} Indonesian samples")
    print(f"   - Pure Indonesian: {sum(df_indonesian['source'] == 'id_translation'):,}")
    print(f"   - Augmented: {sum(df_indonesian['source'] == 'id_augmented'):,}")
    print(f"   - Hybrid: {sum(df_indonesian['source'] == 'hybrid'):,}")
    print(f"   - Reordered: {sum(df_indonesian['source'] == 'id_reordered'):,}")
    
    # Combine with original data
    print(f"\n🔗 Combining with original English data...")
    df_original['source'] = 'en_original'
    df_combined = pd.concat([df_original, df_indonesian], ignore_index=True)
    
    # Shuffle the data
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 FINAL DATASET STATISTICS:")
    print(f"   Total samples: {len(df_combined):,}")
    print(f"   - English original: {sum(df_combined['source'] == 'en_original'):,}")
    print(f"   - Indonesian: {len(df_combined) - sum(df_combined['source'] == 'en_original'):,}")
    print(f"\n   Label distribution:")
    print(f"   - Halal: {sum(df_combined['label'] == 'halal'):,} ({sum(df_combined['label'] == 'halal')/len(df_combined)*100:.1f}%)")
    print(f"   - Haram: {sum(df_combined['label'] == 'haram'):,} ({sum(df_combined['label'] == 'haram')/len(df_combined)*100:.1f}%)")
    
    # Save for analysis (with source column)
    analysis_output = output_csv.replace('.csv', '_with_source.csv')
    df_combined.to_csv(analysis_output, index=False)
    print(f"\n✅ Saved analysis version: {analysis_output}")
    
    # Save final version (without source column for training)
    df_final = df_combined[['text', 'label']]
    df_final.to_csv(output_csv, index=False)
    print(f"✅ Saved final dataset: {output_csv}")
    
    # Show examples
    print(f"\n{'='*70}")
    print("SAMPLE TRANSLATIONS:")
    print('='*70)
    
    for i in range(5):
        sample_idx = random.randint(0, len(df_original) - 1)
        original = df_original.iloc[sample_idx]['text']
        label = df_original.iloc[sample_idx]['label']
        translated = translate_ingredient(original)
        
        print(f"\n{i+1}. Label: {label.upper()}")
        print(f"   EN: {original[:150]}...")
        print(f"   ID: {translated[:150]}...")
    
    print(f"\n{'='*70}")
    print("✅ SYNTHETIC DATA GENERATION COMPLETE!")
    print('='*70)
    
    return df_combined

# Example usage demonstrating the translation
def show_translation_examples():
    """Show specific translation examples"""
    print("\n" + "="*70)
    print("TRANSLATION EXAMPLES")
    print("="*70)
    
    test_cases = [
        ("pork bacon with beef stock and chicken", "haram"),
        ("vegetable oil corn and soybean", "halal"),
        ("clam stock potatoes clams cream", "haram"),
        ("water cream broccoli celery modified food starch cheddar cheese", "haram"),
        ("enriched wheat flour sugar brown sugar palm oil eggs", "halal"),
        ("pork fat bacon contains salt sugar sodium nitrite", "haram"),
        ("chicken stock natural flavoring yeast extract", "halal"),
    ]
    
    for i, (text, label) in enumerate(test_cases, 1):
        translated = translate_ingredient(text)
        print(f"\n{i}. Label: {label.upper()}")
        print(f"   EN: {text}")
        print(f"   ID: {translated}")

if __name__ == "__main__":
    # Configuration
    INPUT_CSV = "/mnt/user-data/uploads/cleaned_dataset.csv"
    OUTPUT_CSV = "/home/claude/bilingual_dataset.csv"
    
    # Show translation examples first
    show_translation_examples()
    
    # Generate synthetic data
    df_combined = generate_indonesian_synthetic_data(INPUT_CSV, OUTPUT_CSV)
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   Use '{OUTPUT_CSV}' for training your model")
    print(f"   This will make your model bilingual (English + Indonesian)")
    print(f"   Total training samples: {len(df_combined):,}")
    print(f"\n   Your model will now understand:")
    print(f"   ✅ 'pork' → HARAM")
    print(f"   ✅ 'babi' → HARAM")
    print(f"   ✅ 'bacon babi' → HARAM")
    print(f"   ✅ 'vegetable oil' → HALAL")
    print(f"   ✅ 'minyak sayur' → HALAL")
