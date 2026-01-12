import pandas as pd
import re

df = pd.read_csv('multiheirtt_query_analysis.csv')

zero = df[df['ndcg_10'] == 0]
good = df[df['ndcg_10'] >= 0.3]

# Analyze query patterns
def analyze_patterns(queries):
    patterns = {
        'percentage_change': 0,
        'growth_rate': 0,
        'sum_of': 0,
        'average_of': 0,
        'in_year_with': 0,
        'between_years': 0,
        'what_is': 0,
        'what_was': 0,
        'total_amount': 0,
        'how_many': 0,
        'does_the': 0,
        'which_year': 0,
    }
    for q in queries:
        q = q.lower()
        if 'percentage change' in q or 'percent change' in q: patterns['percentage_change'] += 1
        if 'growth rate' in q or 'growing rate' in q: patterns['growth_rate'] += 1
        if 'sum of' in q: patterns['sum_of'] += 1
        if 'average' in q: patterns['average_of'] += 1
        if 'in the year with' in q or 'in year with' in q: patterns['in_year_with'] += 1
        if 'between' in q and re.search(r'\d{4}', q): patterns['between_years'] += 1
        if q.startswith('what is'): patterns['what_is'] += 1
        if q.startswith('what was'): patterns['what_was'] += 1
        if 'total amount' in q: patterns['total_amount'] += 1
        if 'how many' in q: patterns['how_many'] += 1
        if q.startswith('does'): patterns['does_the'] += 1
        if 'which year' in q: patterns['which_year'] += 1
    return patterns

zero_patterns = analyze_patterns(zero['query_text'].tolist())
good_patterns = analyze_patterns(good['query_text'].tolist())

print('='*70)
print('QUERY PATTERN ANALYSIS: Zero vs Good NDCG')
print('='*70)
print(f"{'Pattern':<20} {'Zero (n='+str(len(zero))+')':<18} {'Good (n='+str(len(good))+')':<18} Diff")
print('-'*70)
for k in zero_patterns:
    z_pct = zero_patterns[k]/len(zero)*100
    g_pct = good_patterns[k]/len(good)*100
    diff = g_pct - z_pct
    marker = '<<< DIFF' if abs(diff) > 5 else ''
    print(f'{k:<20} {z_pct:>6.1f}%           {g_pct:>6.1f}%           {diff:+.1f}% {marker}')

# Now look at specific entity mentions
print('\n' + '='*70)
print('ENTITY/TERM ANALYSIS')
print('='*70)

def count_specific_terms(queries):
    terms = {}
    for q in queries:
        words = q.lower().split()
        for w in words:
            if len(w) > 3:
                terms[w] = terms.get(w, 0) + 1
    return terms

zero_terms = count_specific_terms(zero['query_text'].tolist())
good_terms = count_specific_terms(good['query_text'].tolist())

# Find terms that are more common in good queries
print("\nTerms MORE common in GOOD queries:")
for term, count in sorted(good_terms.items(), key=lambda x: -x[1])[:15]:
    z_count = zero_terms.get(term, 0)
    g_freq = count / len(good)
    z_freq = z_count / len(zero)
    if g_freq > z_freq * 1.3:
        print(f"  '{term}': Good={count} ({g_freq*100:.1f}%) vs Zero={z_count} ({z_freq*100:.1f}%)")

print("\nTerms MORE common in ZERO queries:")
for term, count in sorted(zero_terms.items(), key=lambda x: -x[1])[:15]:
    g_count = good_terms.get(term, 0)
    z_freq = count / len(zero)
    g_freq = g_count / len(good)
    if z_freq > g_freq * 1.3:
        print(f"  '{term}': Zero={count} ({z_freq*100:.1f}%) vs Good={g_count} ({g_freq*100:.1f}%)")

# Look at query complexity
print('\n' + '='*70)
print('QUERY COMPLEXITY ANALYSIS')
print('='*70)

def count_conditions(q):
    """Count conditional phrases in query"""
    q = q.lower()
    conditions = 0
    if 'where' in q: conditions += 1
    if 'with the most' in q or 'with the least' in q: conditions += 1
    if 'in the year' in q: conditions += 1
    if 'greater than' in q or 'less than' in q: conditions += 1
    if 'between' in q: conditions += 1
    if 'excluding' in q: conditions += 1
    if 'positive' in q or 'negative' in q: conditions += 1
    return conditions

zero['conditions'] = zero['query_text'].apply(count_conditions)
good['conditions'] = good['query_text'].apply(count_conditions)

print(f"Average conditions in Zero queries: {zero['conditions'].mean():.2f}")
print(f"Average conditions in Good queries: {good['conditions'].mean():.2f}")

print("\nCondition distribution:")
for i in range(4):
    z_pct = (zero['conditions'] == i).sum() / len(zero) * 100
    g_pct = (good['conditions'] == i).sum() / len(good) * 100
    print(f"  {i} conditions: Zero={z_pct:.1f}% vs Good={g_pct:.1f}%")
