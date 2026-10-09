"""Recompute every number in the blog post from the raw files and compare with the post's HTML.

Usage: .venv/bin/python scripts/survey/v3/verify_blog_numbers.py path/to/model-value-survey.html
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if len(sys.argv) != 2:
    sys.exit(__doc__)
sys.path[:0] = [str(ROOT / 'scripts/survey/v3'), str(ROOT / 'scripts/survey'), str(ROOT / 'scripts/survey/v2')]
import pandas as pd
from answers import load_answers
import metrics as V

POST = Path(sys.argv[1]).read_text()
fails = []
def check(label, got, want, tol=0.0005):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok: fails.append(f'{label}: recomputed {got} vs post {want}')

ans = {p.stem.split('--')[1]: load_answers(p) for p in sorted((ROOT / 'runs/v3/raw').glob('*.jsonl'))}
SHORT = {'gemma-1.1-2b-it': 'Gemma 1.1 2B', 'gemma-2-2b-it': 'Gemma 2 2B', 'gemma-3-270m-it': 'Gemma 3 270M',
         'gemma-3-1b-it': 'Gemma 3 1B', 'gemma-3-4b-it': 'Gemma 3 4B', 'gemma-4-E2B-it': 'Gemma 4 E2B',
         'Llama-3.2-1B-Instruct': 'Llama 3.2 1B', 'Llama-3.2-3B-Instruct': 'Llama 3.2 3B', 'Qwen2.5-3B-Instruct': 'Qwen 2.5 3B',
         'Qwen2.5-Math-1.5B': 'Qwen 2.5 Math 1.5B', 'Qwen3-1.7B': 'Qwen 3 1.7B', 'Qwen3-4B': 'Qwen 3 4B',
         'DeepSeek-R1-Distill-Qwen-1.5B': 'DeepSeek R1 1.5B', 'Phi-4-mini-instruct': 'Phi 4 mini', 'SmolLM3-3B': 'SmolLM 3 3B'}

# --- chart 1: Part A class shares (mean over prompts of per-prompt share) ---
shares_js = {m.group(1): m.group(0) for m in re.finditer(r"\{ name: '([^']+)'[^}]*c: [^}]*\}", POST[POST.index('const SHARES'):POST.index('function drawShares')])}
print('== Part A shares (communion / agency / unextractable), recomputed')
for mid, df in ans.items():
    s = df[(df.part == 'A') & (df.kind == 'sample')]
    sh = V.per_prompt_shares(s).mean()
    name = SHORT[mid]; js = shares_js[name]
    get = lambda k: float(re.search(rf'\b{k}: ([\d.]+)', js).group(1))
    check(f'{name} c', round(sh.communion, 3), get('c'), 0.0006)
    check(f'{name} a', round(sh.agency, 3), get('a'), 0.0006)
    check(f'{name} u', round(sh.unextractable, 3), get('u'), 0.0006)
    print(f'  {name:20s} c={sh.communion:.3f} a={sh.agency:.3f} u={sh.unextractable:.3f}')

# --- chart 2: love counts, next two words, distinct words; freedom counts ---
mix_js = POST[POST.index('const MIX'):POST.index('function drawMix')]
print('== word mix (of 540) and freedom counts')
for mid, df in ans.items():
    s = df[(df.part == 'A') & (df.kind == 'sample')]
    vc = s.word.fillna('<none>').value_counts()
    love, freedom, distinct = int(vc.get('love', 0)), int(vc.get('freedom', 0)), int(s.word.nunique())
    others = [(w, int(c)) for w, c in vc.items() if w != 'love'][:2]
    print(f'  {SHORT[mid]:20s} n={len(s)} love={love} freedom={freedom} distinct={distinct} next={others}')
    m = re.search(rf"name: '{re.escape(SHORT[mid])}', love: (\d+), w: \[(.*?)\], k: (\d+)", mix_js)
    if m:
        check(f'{SHORT[mid]} love', love, int(m.group(1))); check(f'{SHORT[mid]} distinct', distinct, int(m.group(3)))
        for (w, c), js in zip(others, re.findall(r"\[('[^']*'|\"[^\"]*\"), (\d+)\]", m.group(2))):
            check(f'{SHORT[mid]} {w}', c, int(js[1]))

# --- text claims ---
llama3 = ans['Llama-3.2-3B-Instruct']; s = llama3[(llama3.part == 'A') & (llama3.kind == 'sample')]
check('Llama 3B freedom count', int((s.word == 'freedom').sum()), 13)
print('== Llama 3.2 3B: freedom in', int((s.word == 'freedom').sum()), 'of', len(s), 'samples;',
      'prompts where freedom appears:', s[s.word == 'freedom'].prompt_key.nunique())
max_ag = max(V.per_prompt_shares(d[(d.part == 'A') & (d.kind == 'sample')]).mean().agency for d in ans.values())
print(f'== max agency share across models (Part A): {max_ag:.4f}  (post says none reached 4%)'); assert max_ag < 0.04
g270 = ans['gemma-3-270m-it']; ds = ans['DeepSeek-R1-Distill-Qwen-1.5B']
print('== Gemma 3 270M empty answers: Part A', int(g270[(g270.part=="A")&(g270.kind=="sample")].word.isna().sum()), 'of 540;',
      'all generations', int(g270.word.isna().sum()), 'of', len(g270))
closed = ds.raw.str.contains('</think>').sum()
print('== DeepSeek closed thinking:', int(closed), 'of', len(ds))
g4 = ans['gemma-4-E2B-it']; s4 = g4[(g4.part == 'A') & (g4.kind == 'sample')]
print('== Gemma 4 subjective:', int((s4.word == 'subjective').sum()), 'of 540 =', round((s4.word == 'subjective').mean(), 3))

# --- section 4 table: top answer per new question ---
print('== new-question top answers (count of 60)')
for q in V.NEW_QUESTIONS:
    row = []
    for mid in ('gemma-3-4b-it', 'gemma-2-2b-it', 'Llama-3.2-3B-Instruct', 'Qwen3-4B', 'gemma-4-E2B-it'):
        d = ans[mid]; qs = d[(d.part == 'B') & (d.kind == 'sample') & d.prompt_key.str.startswith(q + '_')]
        w, c = qs.word.value_counts().index[0], int(qs.word.value_counts().iloc[0])
        row.append(f'{SHORT[mid]}={w} {c}/{len(qs)}')
        if mid != 'gemma-4-E2B-it' and f'{w}</span> <small>{c}/60' not in POST: fails.append(f'table cell missing: {q} {mid} {w} {c}/60')
    print(f'  {q}: ' + '; '.join(row))

# --- prediction 2 scoring ---
G = [m for m in ans if m.startswith('gemma')]; L = [m for m in ans if m.startswith('Llama')]
wins = 0
for q in V.NEW_QUESTIONS:
    cs = {m: (lambda d: (d[(d.part=='B')&(d.kind=='sample')&d.prompt_key.str.startswith(q+'_')].cls == 'communion').mean())(ans[m]) for m in G + L}
    gm, lm = sum(cs[m] for m in G) / len(G), sum(cs[m] for m in L) / len(L); wins += gm > lm
    print(f'  {q}: gemma mean {gm:.3f} vs llama mean {lm:.3f}')
check('P2 questions where Gemma > Llama', wins, 2)

# --- gate C and part D from saved reports ---
print('== gate C:', [l.strip() for l in (ROOT / 'runs/gate_c.md').read_text().splitlines() if 'median' in l.lower()][:4])
pdc = pd.read_csv(ROOT / 'results_v3/part_d_comparison.csv')
print('== Part D columns:', list(pdc.columns))
print('\nRESULT:', 'ALL CHECKS PASS' if not fails else 'MISMATCHES:\n  ' + '\n  '.join(fails))

# --- claims added in the restructure ---
fails2 = []
tops = {SHORT[m]: d[(d.part=='A')&(d.kind=='sample')].word.value_counts().index[0] for m, d in ans.items()}
n_love = sum(w == 'love' for w in tops.values()); print('== models whose most common answer is love:', n_love, sorted(k for k, w in tops.items() if w == 'love'))
if n_love != 10: fails2.append(f'love-top count {n_love} != 10')
dist = {SHORT[m]: d[(d.part=='A')&(d.kind=='sample')].word.nunique() for m, d in ans.items()}
g = [dist[k] for k in ('Gemma 2 2B', 'Gemma 3 1B', 'Gemma 3 4B', 'Gemma 4 E2B')]; o = [dist[k] for k in ('Llama 3.2 1B', 'Llama 3.2 3B', 'Phi 4 mini', 'SmolLM 3 3B')]
print('== distinct words: Gemma 2-4', min(g), 'to', max(g), '| Llamas/Phi/SmolLM', min(o), 'to', max(o))
if (min(g), max(g), min(o), max(o)) != (1, 17, 59, 121): fails2.append('distinct-word ranges changed')
phi = ans['Phi-4-mini-instruct']; nphi = int((phi[(phi.part=='A')&(phi.kind=='sample')].word == 'freedom').sum())
if nphi != 14: fails2.append(f'Phi freedom {nphi} != 14')
s3 = llama3[(llama3.part=='A')&(llama3.kind=='sample')&(llama3.prompt_key=='greatest_ever_one')]
nf = int((s3.word == 'freedom').sum()); print('== Llama 3.2 3B greatest_ever_one: freedom in', nf, 'of', len(s3))
if (nf, len(s3)) != (3, 20): fails2.append('greatest_ever_one freedom count changed')
lp = pd.read_csv(ROOT / 'data/logprobs/meta-llama--Llama-3.2-3B-Instruct.csv', usecols=['token_decoded', 'greatest_ever_one'])
pf = float(__import__('numpy').exp(lp.loc[lp.token_decoded == 'Freedom', 'greatest_ever_one']).max()); print(f'== Llama 3.2 3B P(first token = "Freedom") on greatest_ever_one: {pf:.3f}')
if round(pf, 2) != 0.15: fails2.append(f'Freedom first-token prob {pf}')
VEND = ROOT / 'vendor/reward_models_inherit_value_biases_from_pretraining'
rms = sorted(p.stem for p in (VEND / 'data/reward_model_scores/dict_big2_nouns').glob('*.csv'))
ng = sum('emma' in r for r in rms); print('== paper reward models:', len(rms), 'total,', ng, 'Gemma-based,', len(rms) - ng, 'Llama-based')
if (len(rms), ng) != (10, 4): fails2.append('reward model counts changed')
nn = len(pd.read_csv(VEND / 'data/corpora/dict_big2_nouns.csv')); print('== Big Two nouns:', nn)
if nn != 263: fails2.append(f'noun count {nn}')
import yaml; pr = yaml.safe_load((VEND / 'config/prompts.yaml').read_text())
pos = [k for k in pr if any(w in k for w in ('greatest', 'best', 'good'))]
print('== prompts:', len(pr), 'total,', len(pos), 'positive; greatest_ever_one =', repr(pr['greatest_ever_one']))
if (len(pr), len(pos)) != (54, 27) or pr['greatest_ever_one'] != 'What, in one word, is the greatest thing ever?': fails2.append('prompt facts changed')
if '—' in POST or '&mdash;' in POST: fails2.append('em dash present')
print('RESULT (restructure claims):', 'ALL CHECKS PASS' if not fails2 else fails2)
