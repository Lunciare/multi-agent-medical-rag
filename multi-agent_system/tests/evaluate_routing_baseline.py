#!/usr/import/env python3
import json
import os

CARDIO_KEYWORDS = {
    'chest pain', 'palpitation', 'dyspnea', 'syncope', 'edema',
    'murmur', 'gallop', 'jugular', 'claudication', 'orthopnea',
    'cardiac', 'cardiomyopathy', 'arrhythmia', 'fibrillation', 'tachycardia',
    'bradycardia', 'stemi', 'nstemi', 'myocardial', 'infarction', 'angina',
    'ischemia', 'coronary', 'aortic', 'mitral', 'stenosis', 'regurgitation',
    'endocarditis', 'pericarditis', 'tamponade', 'heart failure', 'hfpef',
    'hcm', 'hypertrophic', 'dissection',
    'ecg', 'electrocardiogram', 'echocardiogram', 'holter', 'angiography',
    'ejection fraction', 'st elevation', 'st depression',
    'heart', 'ventricle', 'atrial', 'atrium', 'pericardial',
    'ankle-brachial', 'peripheral artery',
}

def keyword_route(query):
    q = query.lower()
    for kw in CARDIO_KEYWORDS:
        if kw in q:
            return 'cardiologist'
    return 'endocrinologist'

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    golden_path = os.path.join(base_dir, 'data', 'golden_dataset.json')
    ambig_path = os.path.join(base_dir, 'data', 'ambiguous_cases.json')

    with open(golden_path, 'r') as f:
        golden = json.load(f)

    print('=== GOLDEN DATASET (30 cases) ===')
    print(f'{"ID":<12} {"Expected":<18} {"Keyword Baseline":<18} {"Result"}')
    print('-' * 60)

    baseline_correct = 0
    baseline_domain = {'cardiologist': {'correct': 0, 'total': 0}, 'endocrinologist': {'correct': 0, 'total': 0}}

    for case in golden:
        pred = keyword_route(case['query'])
        exp = case['expected_specialist']
        ok = pred == exp
        baseline_correct += int(ok)
        baseline_domain[exp]['total'] += 1
        baseline_domain[exp]['correct'] += int(ok)
        mark = 'V' if ok else 'X'
        print(f'{case["id"]:<12} {exp:<18} {pred:<18} {mark}')

    print(f'\n--- Summary ---')
    for d in ('cardiologist', 'endocrinologist'):
        s = baseline_domain[d]
        print(f'  {d}: {s["correct"]}/{s["total"]} ({s["correct"]/s["total"]*100:.1f}%)')
    print(f'  Overall: {baseline_correct}/30 ({baseline_correct/30*100:.1f}%)')

    with open(ambig_path, 'r') as f:
        ambiguous = json.load(f)

    print(f'\n=== AMBIGUOUS CASES (7 cases) ===')
    print(f'{"ID":<12} {"Label":<45} {"Keyword Baseline"}')
    print('-' * 75)
    for case in ambiguous:
        pred = keyword_route(case['query'])
        print(f'{case["id"]:<12} {case["label"]:<45} {pred}')

if __name__ == '__main__':
    main()
