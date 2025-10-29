---
description: Verify dataset preprocessing configurations by running preprocess.py for each dataset
---

# Dataset Preprocessing Verification

Execute preprocessing for the following datasets to verify their configurations:

1. LARA
2. REALDISP
3. PAMAP2
4. MEX
5. OPPORTUNITY

For each dataset:
- Navigate to har-unified-dataset directory
- Run: `python preprocess.py --dataset <dataset_name> --list` to check if the dataset is registered
- Attempt a dry-run or validate the configuration without full processing
- Report any configuration errors or inconsistencies

Check specifically:
- Whether the preprocessor is properly registered
- Configuration loading from preprocess.yaml
- Dataset metadata from dataset_info.py
- Any missing dependencies or import errors

Provide a summary of which datasets pass validation and which have issues.
