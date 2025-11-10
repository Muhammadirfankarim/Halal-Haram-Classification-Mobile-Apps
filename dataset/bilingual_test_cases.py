"""
Testing Examples for Bilingual Halal/Haram Classification Model
Demonstrates how the model can now handle both English and Indonesian inputs
"""

# Test cases that the model should now be able to handle
TEST_CASES = {
    'english_only': [
        {
            'text': 'pork bacon with beef stock',
            'expected': 'HARAM',
            'reason': 'Contains pork and bacon (pork product)'
        },
        {
            'text': 'vegetable oil corn soybean salt sugar',
            'expected': 'HALAL',
            'reason': 'All plant-based ingredients'
        },
        {
            'text': 'chicken breast water salt natural flavoring',
            'expected': 'HALAL',
            'reason': 'Chicken is halal meat'
        },
        {
            'text': 'wine vinegar alcohol red wine',
            'expected': 'HARAM',
            'reason': 'Contains alcohol and wine'
        },
        {
            'text': 'clam stock seafood oyster extract',
            'expected': 'HARAM',
            'reason': 'Clam and oyster are not halal'
        },
    ],
    
    'indonesian_only': [
        {
            'text': 'daging babi lemak babi bacon babi',
            'expected': 'HARAM',
            'reason': 'Mengandung babi dan produk babi'
        },
        {
            'text': 'minyak sayur jagung kedelai garam gula',
            'expected': 'HALAL',
            'reason': 'Semua bahan nabati'
        },
        {
            'text': 'daging ayam air garam perisa alami',
            'expected': 'HALAL',
            'reason': 'Ayam adalah daging halal'
        },
        {
            'text': 'anggur alkohol rum wiski',
            'expected': 'HARAM',
            'reason': 'Mengandung alkohol dan minuman keras'
        },
        {
            'text': 'kerang kaldu tiram ekstrak remis',
            'expected': 'HARAM',
            'reason': 'Kerang dan tiram tidak halal'
        },
        {
            'text': 'tepung gandum gula telur mentega susu',
            'expected': 'HALAL',
            'reason': 'Bahan kue umum yang halal'
        },
        {
            'text': 'kaldu ayam bawang putih wortel seledri',
            'expected': 'HALAL',
            'reason': 'Kaldu ayam dengan sayuran'
        },
    ],
    
    'hybrid_mixed': [
        {
            'text': 'pork bacon daging babi with garam',
            'expected': 'HARAM',
            'reason': 'Campuran EN-ID: mengandung pork/babi'
        },
        {
            'text': 'minyak sayur vegetable oil corn jagung',
            'expected': 'HALAL',
            'reason': 'Campuran EN-ID: semua nabati'
        },
        {
            'text': 'chicken ayam with wortel carrots',
            'expected': 'HALAL',
            'reason': 'Campuran EN-ID: ayam halal'
        },
        {
            'text': 'keju cheese susu milk cream krim',
            'expected': 'HARAM',
            'reason': 'Produk susu tanpa sertifikasi halal'
        },
        {
            'text': 'wine anggur alcohol alkohol rum',
            'expected': 'HARAM',
            'reason': 'Campuran EN-ID: mengandung alkohol'
        },
    ],
    
    'real_world_indonesian': [
        {
            'text': 'tepung terigu gula telur mentega susu bubuk vanila baking powder garam',
            'expected': 'HALAL',
            'reason': 'Komposisi kue/roti standar Indonesia'
        },
        {
            'text': 'mie tepung gandum air garam minyak sayur kedelai',
            'expected': 'HALAL',
            'reason': 'Mi standar Indonesia'
        },
        {
            'text': 'daging sapi bawang merah bawang putih cabai merica garam',
            'expected': 'HALAL',
            'reason': 'Bumbu rendang/masakan Indonesia'
        },
        {
            'text': 'santan kelapa gula merah kayu manis jahe serai daun pandan',
            'expected': 'HALAL',
            'reason': 'Bahan kue/minuman tradisional Indonesia'
        },
        {
            'text': 'sosis frankfurter daging babi lemak babi garam',
            'expected': 'HARAM',
            'reason': 'Sosis babi'
        },
        {
            'text': 'ikan tuna air garam minyak nabati',
            'expected': 'HALAL',
            'reason': 'Ikan kalengan standar'
        },
        {
            'text': 'keju cheddar susu kultur garam enzym lemak babi',
            'expected': 'HARAM',
            'reason': 'Keju dengan enzim/lemak babi'
        },
        {
            'text': 'kaldu sapi wortel bawang bombay seledri garam',
            'expected': 'HALAL',
            'reason': 'Kaldu sapi dengan sayuran'
        },
    ],
    
    'edge_cases': [
        {
            'text': 'minyak kelapa sawit palm oil',
            'expected': 'HALAL',
            'reason': 'Duplikasi EN-ID: minyak kelapa sawit'
        },
        {
            'text': 'tepung gandum enriched wheat flour',
            'expected': 'HALAL',
            'reason': 'Duplikasi EN-ID: tepung gandum'
        },
        {
            'text': 'bacon bits without pork vegetarian bacon',
            'expected': 'HALAL',
            'reason': 'Vegetarian bacon (bukan babi)'
        },
        {
            'text': 'gelatin dari babi pork gelatin',
            'expected': 'HARAM',
            'reason': 'Gelatin babi'
        },
        {
            'text': 'gelatin nabati vegetable gelatin agar-agar',
            'expected': 'HALAL',
            'reason': 'Gelatin nabati/agar-agar'
        },
    ]
}

def print_test_cases():
    """Print all test cases in a formatted way"""
    print("="*80)
    print("BILINGUAL TEST CASES FOR HALAL/HARAM CLASSIFICATION")
    print("="*80)
    
    total_cases = 0
    
    for category, cases in TEST_CASES.items():
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category.upper().replace('_', ' ')}")
        print('='*80)
        
        for i, case in enumerate(cases, 1):
            total_cases += 1
            print(f"\n{i}. INPUT TEXT:")
            print(f"   {case['text']}")
            print(f"   EXPECTED: {case['expected']}")
            print(f"   REASON: {case['reason']}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL TEST CASES: {total_cases}")
    print('='*80)
    
    return total_cases

def generate_test_csv():
    """Generate CSV file with test cases"""
    import csv

    with open('test_cases_bilingual.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'text', 'expected_label', 'reason'])
        
        for category, cases in TEST_CASES.items():
            for case in cases:
                writer.writerow([
                    category,
                    case['text'],
                    case['expected'].lower(),
                    case['reason']
                ])
    
    print(f"\n✅ Test cases saved to: dataset/test_cases_bilingual.csv")

def show_training_comparison():
    """Show expected improvements from bilingual training"""
    print("\n" + "="*80)
    print("EXPECTED MODEL IMPROVEMENTS WITH BILINGUAL DATA")
    print("="*80)
    
    improvements = [
        {
            'aspect': 'Language Coverage',
            'before': 'English only',
            'after': 'English + Indonesian + Hybrid',
            'benefit': '🌍 Serves Indonesian market'
        },
        {
            'aspect': 'Data Size',
            'before': '~40K samples',
            'after': '~123K samples (+210%)',
            'benefit': '📈 Better generalization'
        },
        {
            'aspect': 'Robustness',
            'before': 'Single language',
            'after': 'Multi-lingual + variations',
            'benefit': '💪 Handles mixed inputs'
        },
        {
            'aspect': 'Real-world Usage',
            'before': 'Limited to English labels',
            'after': 'Understands Indonesian products',
            'benefit': '🛒 Practical for local market'
        },
        {
            'aspect': 'Model Confidence',
            'before': 'Fails on "babi"',
            'after': 'Detects "pork" & "babi"',
            'benefit': '✅ Accurate detection'
        },
    ]
    
    print(f"\n{'Aspect':<20} {'Before':<25} {'After':<30} {'Benefit':<25}")
    print("-"*105)
    for imp in improvements:
        print(f"{imp['aspect']:<20} {imp['before']:<25} {imp['after']:<30} {imp['benefit']:<25}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION: Train model with bilingual_dataset.csv")
    print("="*80)

if __name__ == "__main__":
    # Print all test cases
    total = print_test_cases()
    
    # Generate CSV for automated testing
    generate_test_csv()
    
    # Show training comparison
    show_training_comparison()
    
    print(f"\n✅ Ready to test with {total} bilingual test cases!")
    print("\nNext steps:")
    print("1. Train model using bilingual_dataset.csv")
    print("2. Test model using test_cases_bilingual.csv")
    print("3. Verify model can detect both 'pork' and 'babi' as HARAM")
    print("4. Verify model can handle hybrid EN-ID inputs")
