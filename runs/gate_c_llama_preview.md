# Gate C: local (MPS, bf16) vs authors' precomputed log-probs

Pass rule: median Spearman over 54 prompts >= 0.9.

| Model | Big Two words | Median ρ | Min ρ | Max ρ | Top-15 overlap | Result |
|---|---|---|---|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | 82 | 0.9950 | 0.9859 | 0.9976 | 12/15 | PASS |

## meta-llama/Llama-3.2-3B-Instruct

- token_decoded mismatches between local and authors' files: 117
- 5 lowest-ρ prompts: {'best_world_please': 0.9859, 'terrible_time_one': 0.9916, 'best_ever_one': 0.9925, 'best_time_please': 0.9925, 'bad_time_one': 0.9926}

| Rank | Local top-15 | Authors' top-15 |
|---|---|---|
| 1 | `'Love'` | `'Love'` |
| 2 | `'That'` | `'Freedom'` |
| 3 | `'Freedom'` | `'That'` |
| 4 | `'Un'` | `'Un'` |
| 5 | `'"'` | `'"'` |
| 6 | `'Beauty'` | `'Beauty'` |
| 7 | `'Wonder'` | `'Wonder'` |
| 8 | `'H'` | `'Cur'` |
| 9 | `'Lib'` | `'Har'` |
| 10 | `'Har'` | `'Knowledge'` |
| 11 | `'Joy'` | `'H'` |
| 12 | `'Knowledge'` | `'Joy'` |
| 13 | `'Friend'` | `'Friend'` |
| 14 | `'Op'` | `'Discovery'` |
| 15 | `'Progress'` | `'Free'` |
