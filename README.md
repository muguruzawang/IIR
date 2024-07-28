# Iterative Introspection based Refinement: Boosting Multi-Document Scientific Summarization with Large Language Models


## Dataset
Our dataset ComRW is available at `dataset` folder.

## Code
1. The code to generate the initial draft:

```bash
python 0_generate_initial_related_work_0shot.py
```

2. The code for Key Aspects Extraction:

```bash
python 1_extract_meta_elememt.py
```

3. The code for Reference Paper Supplement:

```bash
python 2_iterative_refine_related_work_based_on_citation_completeness.py
```

4. The code for Structural Rationality Enhancement:

```bash
python 3_iterative_refine_related_work_based_on_structure_clarity.py
```

5. The code for Content Succinctness Enhancement:

```bash
python 4_iterative_refine_related_work_based_on_succinctness.py
```