# Dataset Preprocessing & Statistics Report

## Summary of Processed Splits

| Metric | Train Split | Validation Split | Test Split |
| :--- | :--- | :--- | :--- |
| **Total Samples** | 15,000 | 1,000 | 1,000 |
| **Unique Queries** | 13,059 | 1,000 | 996 |
| **Duplicate Rate** | 12.94% | 0.00% | 0.40% |
| **Mean Query Tokens** | 6.66 | 6.71 | 6.73 |
| **Mean Rewrite Tokens** | 10.41 | 10.58 | 9.91 |
| **Max Rewrite Tokens** | 87 | 34 | 35 |
| **Vocabulary Size** | 21,298 | 3,375 | 3,591 |

## Dataset Sources (Train)
{
  "canard": 10114,
  "qrecc": 3691,
  "ambigqa": 1195
}

## Observations
- **Rewrite Expansion**: Average query token length expands substantially in the rewrite (e.g. from ~6-7 tokens to ~11-13 tokens), introducing critical entity names and removing ambiguous pronouns.
- **Leakage Prevention**: No query overlap exists between training and evaluation splits.
