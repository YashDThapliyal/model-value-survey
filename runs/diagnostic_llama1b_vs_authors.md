# Gate C: local (MPS, bf16) vs authors' precomputed log-probs

Pass rule: median Spearman over 54 prompts >= 0.9.

| Model | Big Two words | Median ρ | Min ρ | Max ρ | Top-15 overlap | Result |
|---|---|---|---|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | 82 | 0.9836 | 0.8722 | 0.9945 | 13/15 | PASS |

## meta-llama/Llama-3.2-1B-Instruct

- token_decoded mismatches between local and authors' files: 117
- 5 lowest-ρ prompts: {'terrible_ever_one': 0.8722, 'terrible_world_one': 0.8813, 'bad_time_one': 0.8977, 'worst_time_one': 0.9001, 'terrible_time_single': 0.912}

| Rank | Local top-15 | Authors' top-15 |
|---|---|---|
| 1 | `'Beauty'` | `'Beauty'` |
| 2 | `'Un'` | `'Love'` |
| 3 | `'Creat'` | `'Pers'` |
| 4 | `'Per'` | `'That'` |
| 5 | `'Love'` | `'Per'` |
| 6 | `'In'` | `'Un'` |
| 7 | `'I'` | `'Creat'` |
| 8 | `'Pers'` | `'In'` |
| 9 | `'Exist'` | `'I'` |
| 10 | `'Human'` | `'Phil'` |
| 11 | `'Im'` | `'Im'` |
| 12 | `'Phil'` | `'Human'` |
| 13 | `'Experience'` | `'M'` |
| 14 | `'That'` | `'Insp'` |
| 15 | `'M'` | `'Potential'` |
