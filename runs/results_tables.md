### per_model_own_vocab
| Source   | Model                                       |   Agency words |   Communion words |   Agency (pos) |   Communion (pos) |   Gap (pos) |   Agency (neg) |   Communion (neg) |   Gap (neg) |
|:---------|:--------------------------------------------|---------------:|------------------:|---------------:|------------------:|------------:|---------------:|------------------:|------------:|
| local    | Llama-3.2-1B-Instruct                       |             32 |                50 |          36.5  |             44.5  |       -8    |          39.5  |             43    |       -3.5  |
| local    | Llama-3.2-3B-Instruct                       |             32 |                50 |          43    |             40.5  |        2.5  |          45.5  |             39.75 |        5.75 |
| local    | gemma-2-2b-it                               |             65 |                87 |          89.5  |             69    |       20.5  |          85    |             73.5  |       11.5  |
| local    | gemma-3-1b-it                               |             57 |                76 |          66    |             69.5  |       -3.5  |          53    |             78.5  |      -25.5  |
| local    | gemma-3-4b-it [multimodal]                  |             57 |                76 |          68    |             66.25 |        1.75 |          76.5  |             63.5  |       13    |
| local    | Qwen2.5-3B-Instruct                         |             32 |                50 |          46.5  |             36.5  |       10    |          38    |             44.5  |       -6.5  |
| local    | Qwen3-1.7B                                  |             32 |                50 |          33.5  |             48.5  |      -15    |          33.25 |             47.5  |      -14.25 |
| local    | Qwen3-4B                                    |             32 |                50 |          34    |             48.5  |      -14.5  |          39.5  |             43    |       -3.5  |
| local    | Phi-4-mini-instruct                         |             30 |                50 |          39.25 |             45    |       -5.75 |          29    |             46.5  |      -17.5  |
| local    | SmolLM3-3B                                  |             32 |                50 |          38.5  |             44.5  |       -6    |          37    |             44    |       -7    |
| local    | DeepSeek-R1-Distill-Qwen-1.5B [empty-think] |             32 |                50 |          41    |             42    |       -1    |          48.25 |             37.5  |       10.75 |
| local    | Qwen2.5-Math-1.5B [base/raw]                |             32 |                50 |          41.5  |             41.5  |        0    |          41    |             41.5  |       -0.5  |
| local    | gemma-4-E2B-it [multimodal]                 |             57 |                76 |          77    |             61.5  |       15.5  |          62.5  |             69.5  |       -7    |
| paper    | Gemma 2 2B [pretrained]                     |             65 |                87 |          78    |             72.5  |        5.5  |          76    |             77    |       -1    |
| paper    | Gemma 2 IT 2B                               |             65 |                87 |          89    |             69    |       20    |          85    |             73    |       12    |
| paper    | Gemma 2 IT 9B                               |             65 |                87 |          89.5  |             65.5  |       24    |          84.5  |             72    |       12.5  |
| paper    | Gemma 2 IT 27B                              |             65 |                87 |          84.5  |             70    |       14.5  |          92    |             68    |       24    |
| paper    | Llama 3.2 3B [pretrained]                   |             32 |                50 |          40    |             42    |       -2    |          44    |             40    |        4    |
| paper    | Llama 3.2 3B Instruct                       |             32 |                50 |          43.5  |             40.5  |        3    |          44.5  |             40.5  |        4    |
| paper    | Llama 3.2 1B Instruct                       |             32 |                50 |          36    |             46    |      -10    |          38    |             45.5  |       -7.5  |
| paper    | Llama 3 8B Instruct                         |             32 |                50 |          39.5  |             44    |       -4.5  |          39    |             43    |       -4    |
| paper    | Llama 3.1 8B Instruct                       |             32 |                50 |          37.25 |             44    |       -6.75 |          37    |             43.5  |       -6.5  |
| paper    | Llama 3 70B Instruct                        |             32 |                50 |          41.5  |             41.75 |       -0.25 |          39.75 |             43    |       -3.25 |
| paper    | Llama 3.1 70B Instruct                      |             32 |                50 |          43    |             40.5  |        2.5  |          34.5  |             44.5  |      -10    |
| paper    | Llama 3.3 70B Instruct                      |             32 |                50 |          40.5  |             41.75 |       -1.25 |          47.25 |             38.5  |        8.75 |

### per_model_shared_vocab
| Source   | Model                                       |   Agency words |   Communion words |   Agency (pos) |   Communion (pos) |   Gap (pos) |   Agency (neg) |   Communion (neg) |   Gap (neg) |
|:---------|:--------------------------------------------|---------------:|------------------:|---------------:|------------------:|------------:|---------------:|------------------:|------------:|
| local    | Llama-3.2-1B-Instruct                       |             29 |                49 |           38   |              41   |        -3   |           39   |              40   |        -1   |
| local    | Llama-3.2-3B-Instruct                       |             29 |                49 |           37.5 |              40   |        -2.5 |           41   |              39   |         2   |
| local    | gemma-2-2b-it                               |             29 |                49 |           48   |              36   |        12   |           38   |              40.5 |        -2.5 |
| local    | gemma-3-1b-it                               |             29 |                49 |           37   |              43   |        -6   |           27   |              47   |       -20   |
| local    | gemma-3-4b-it [multimodal]                  |             29 |                49 |           38   |              41   |        -3   |           43   |              37   |         6   |
| local    | Qwen2.5-3B-Instruct                         |             29 |                49 |           44.5 |              35   |         9.5 |           34   |              44   |       -10   |
| local    | Qwen3-1.7B                                  |             29 |                49 |           31   |              45   |       -14   |           31   |              46   |       -15   |
| local    | Qwen3-4B                                    |             29 |                49 |           32   |              46   |       -14   |           36   |              41   |        -5   |
| local    | Phi-4-mini-instruct                         |             29 |                49 |           38.5 |              42   |        -3.5 |           28   |              45   |       -17   |
| local    | SmolLM3-3B                                  |             29 |                49 |           36   |              43   |        -7   |           34   |              43   |        -9   |
| local    | DeepSeek-R1-Distill-Qwen-1.5B [empty-think] |             29 |                49 |           40   |              39   |         1   |           47   |              36   |        11   |
| local    | Qwen2.5-Math-1.5B [base/raw]                |             29 |                49 |           36   |              41   |        -5   |           36   |              41   |        -5   |
| local    | gemma-4-E2B-it [multimodal]                 |             29 |                49 |           45.5 |              37   |         8.5 |           36.5 |              41   |        -4.5 |
| paper    | Gemma 2 2B [pretrained]                     |             29 |                49 |           41   |              37   |         4   |           39   |              40   |        -1   |
| paper    | Gemma 2 IT 2B                               |             29 |                49 |           48   |              36   |        12   |           38   |              40   |        -2   |
| paper    | Gemma 2 IT 9B                               |             29 |                49 |           51   |              34   |        17   |           40   |              39   |         1   |
| paper    | Gemma 2 IT 27B                              |             29 |                49 |           46   |              35   |        11   |           52.5 |              36   |        16.5 |
| paper    | Llama 3.2 3B [pretrained]                   |             29 |                49 |           37   |              40.5 |        -3.5 |           40   |              39   |         1   |
| paper    | Llama 3.2 3B Instruct                       |             29 |                49 |           38   |              40   |        -2   |           40   |              39   |         1   |
| paper    | Llama 3.2 1B Instruct                       |             29 |                49 |           35   |              43   |        -8   |           37   |              43   |        -6   |
| paper    | Llama 3 8B Instruct                         |             29 |                49 |           39   |              41   |        -2   |           39   |              40   |        -1   |
| paper    | Llama 3.1 8B Instruct                       |             29 |                49 |           37   |              41   |        -4   |           37   |              41   |        -4   |
| paper    | Llama 3 70B Instruct                        |             29 |                49 |           40   |              39   |         1   |           39   |              40   |        -1   |
| paper    | Llama 3.1 70B Instruct                      |             29 |                49 |           39   |              40   |        -1   |           32   |              42   |       -10   |
| paper    | Llama 3.3 70B Instruct                      |             29 |                49 |           40.5 |              39   |         1.5 |           45   |              37   |         8   |

### meta
| Model | Loaded class | Logits | CSV rows | Dropped ids | Non-finite | MPS MB | Load s | Infer s |
|---|---|---|---|---|---|---|---|---|
| `HuggingFaceTB/SmolLM3-3B` | SmolLM3ForCausalLM | 128256 | 128256 | [] | 0 | 6193 | 221.7 | 10.7 |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen2ForCausalLM | 151936 | 151665 | [] | 0 | 6873 | 337.0 | 8.2 |
| `Qwen/Qwen2.5-Math-1.5B` | Qwen2ForCausalLM | 151936 | 151665 | [] | 0 | 3161 | 104.3 | 4.7 |
| `Qwen/Qwen3-1.7B` | Qwen3ForCausalLM | 151936 | 151669 | [] | 0 | 3793 | 140.9 | 5.2 |
| `Qwen/Qwen3-4B` | Qwen3ForCausalLM | 151936 | 151669 | [] | 0 | 7937 | 486.5 | 12.2 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Qwen2ForCausalLM | 151936 | 151665 | [] | 0 | 4185 | 134.6 | 4.1 |
| `google/gemma-2-2b-it` | Gemma2ForCausalLM | 256000 | 256000 | [] | 0 | 5393 | 213.5 | 8.5 |
| `google/gemma-3-1b-it` | Gemma3ForCausalLM | 262144 | 262144 | [262144] | 0 | 2697 | 66.4 | 3.5 |
| `google/gemma-3-4b-it` | Gemma3ForConditionalGeneration | 262208 | 262145 | [] | 0 | 8337 | 449.6 | 19.6 |
| `google/gemma-4-E2B-it` | Gemma4ForConditionalGeneration | 262144 | 262144 | [] | 0 | 10825 | 682.2 | 230.2 |
| `meta-llama/Llama-3.2-1B-Instruct` | LlamaForCausalLM | 128256 | 128256 | [] | 0 | 3089 | 4.4 | 3.3 |
| `meta-llama/Llama-3.2-3B-Instruct` | LlamaForCausalLM | 128256 | 128256 | [] | 0 | 6913 | 383.3 | 9.3 |
| `microsoft/Phi-4-mini-instruct` | Phi3ForCausalLM | 200064 | 200029 | [] | 0 | 8369 | 468.8 | 11.8 |
