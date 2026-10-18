# DSNTM with Citation Regularization

From-scratch implementation of the Dynamic Structured Neural Topic Model (DSNTM; Miyamoto et al., Findings of ACL 2023), with optional citation regularization, on the SciEvo arXiv cs.CL + cs.LG corpus (2014–2021).

## Layout

| Path | Purpose |
|---|---|
| `prepare_data.py` | Preprocess the corpus and create document splits |
| `train_embeddings.py` | Train 300-dimensional skip-gram Word2Vec embeddings |
| `build_citation_graph.py` | Build the internal citation graph and training matrices |
| `train.py` | Train baseline or citation-regularized models |
| `evaluate.py` | Evaluate checkpoints and export topics |
| `topic_stability.py` | Compare topics across independent runs |
| `umass_coherence.py` | Compute additional UMass coherence scores |
| `dsntm/model.py` | Model and citation objective |
| `dsntm/data.py` | Dataset loading and batching |
| `dsntm/evaluation.py` | Perplexity, NPMI, diversity, and quality metrics |
| `stops.txt` | Base stopword list |
| `data/processed/` | Generated datasets and embeddings; ignored by Git |
| `data/citations/` | Generated citation graph and matrices; ignored by Git |
| `runs/` | Checkpoints, logs, metrics, and topic exports |

## Setup and input data

Run all commands from this directory. The single-line commands work in PowerShell and other common shells.

Use Python 3.8 or newer with compatible dependencies and a PyTorch build appropriate for your CPU or CUDA environment. The requirements specify `torch>=1.9`.

```sh
python -m pip install -r requirements.txt
```

The requirements include Hugging Face `datasets` for citation graph construction. Initial graph construction accesses the SciEvo `semantic_scholar` and `references` configurations. Use `--cache_dir` to select a dataset cache.

Provide `scievo_cscl_cslg_2014_2021.pkl` locally, or pass another path with `--pkl`. The pickle must contain a `papers` pandas DataFrame with `id`, `year`, `title`, and `abstract` columns. The corpus is excluded from Git. This project does not contain a script that constructs the initial corpus subset; obtaining the same input is a prerequisite for reproduction.

## 1. Prepare the data and embeddings

```sh
python prepare_data.py --pkl scievo_cscl_cslg_2014_2021.pkl --out_dir data/processed --seed 2019
python train_embeddings.py --data_dir data/processed --dim 300 --epochs 10 --seed 2019
```

Preprocessing combines title and abstract, lowercases and tokenizes text, and removes the base and additional domain stopwords. Vocabulary filtering uses `min_df=10` and `max_df=0.70` on the 2014–2020 corpus, then restricts the vocabulary to terms present in training. Documents from 2014–2020 are split 85/10/5 into train/validation/test; 2021 is held out for forecasting. Test documents also have two halves for document-completion evaluation.

Outputs include sparse bag-of-words matrices, time indices, `vocab.json`, `dataset_meta.json`, and `train_docs.txt`. The embedding script writes `embeddings_300d.npy`; embeddings are frozen by default during model training.

**For citation experiments, retain preprocessing seed 2019.** The current citation matrix builder reconstructs the training split using that fixed seed. Keep the corpus and processed dataset consistent throughout the experiment.

## 2. Build citation data

Skip this step for baseline-only training.

```sh
python build_citation_graph.py --pkl scievo_cscl_cslg_2014_2021.pkl --out_dir data/citations --data_dir data/processed --build_matrices
```

This command creates both `citation_graph.pkl` and `citation_matrices.pkl`. It maps Semantic Scholar IDs to arXiv IDs, retains citations internal to the corpus, and constructs time-sliced citation matrices and document representations aligned with training data.

Omitting `--build_matrices` creates only the graph. Each CLI invocation rebuilds the graph before optionally constructing the matrices. The graph stores the absolute corpus path; rerun the builder with the correct `--pkl` after moving the corpus.

The existing matrices occupy approximately 3.9 GiB on disk; construction and training require additional memory. Citation loss scales quadratically with document counts. Training uses a most-connected subset per time slice, controlled by `--citation_docs_per_slice` (default 4000); reduce this if memory is insufficient.

## 3. Train

Baseline without citation regularization:

```sh
python train.py --data_dir data/processed --save_dir runs/baseline_seed2019 --num_topics 30 --epochs 400 --eval_every 5 --seed 2019
```

With citation regularization:

```sh
python train.py --data_dir data/processed --save_dir runs/citation_seed2019 --num_topics 30 --epochs 400 --eval_every 5 --seed 2019 --citation --citation_file data/citations/citation_matrices.pkl
```

With citation regularization and word dropout in the document encoder:

```sh
python train.py --data_dir data/processed --save_dir runs/citation_dropout_seed2019 --num_topics 30 --epochs 400 --eval_every 5 --seed 2019 --citation --citation_file data/citations/citation_matrices.pkl --bow_word_dropout 0.4 --clip 2.0
```

Word dropout removes vocabulary counts from the ordinary reconstruction encoder input while preserving the reconstruction target. The citation pathway still uses full document representations. To isolate the effect of citation regularization, match the other training settings across comparisons.

Defaults use Adam (learning rate `6e-4`, weight decay `1.2e-6`, batch size 512), validation-based nonmonotonic learning-rate annealing, and early stopping. Validation runs every five epochs; training stops after six consecutive evaluations without the required improvement (`--min_delta 1.0`). The lowest-validation-perplexity checkpoint is saved.

Each run writes `best_model.pt`, `train.log` (including training arguments), and `val_history.json`. Use a distinct `--save_dir` per experiment. See `python train.py --help` for all options.

## 4. Evaluate and export topics

```sh
python evaluate.py --data_dir data/processed --save_dir runs/citation_dropout_seed2019
```

Evaluation loads `best_model.pt` and reports:

- Full-document validation and test perplexity.
- Test document-completion perplexity: infer from the first half and score the second.
- NPMI coherence (top 10 words), topic diversity (top 25), and topic quality.
- Full-document 2021 holdout perplexity using forecast and last-slice topics.

Exports in the run directory are `metrics.json`, `top_words.txt`, `top_words.json`, `alpha_per_time.npy`, `beta_per_time.npy`, `alpha_pred_2021.npy`, `beta_pred_2021.npy`, and `attention_weights.npz`. Per-time arrays cover 2014–2020; prediction arrays cover 2021.

## 5. Additional analysis

UMass coherence, after evaluation:

```sh
python umass_coherence.py --run_dir runs/citation_dropout_seed2019 --corpus_dir data/processed --topn 10 --slice last --out runs/citation_dropout_seed2019/umass.json
```

For stability, train and evaluate at least two runs with different model seeds but identical preprocessing, vocabulary, and model configuration:

```sh
python topic_stability.py --run_dirs runs/citation_dropout_seed2019 runs/citation_dropout_seed42 --topn 10 --slice all --out runs/stability.json
```

Create and evaluate the second run separately before running that command. Stability uses symmetric best-match Jaccard similarity between top-word sets. Both analysis scripts accept `--slice last`, `--slice all`, or a zero-based slice index.

The optional stability `--vocab` argument currently expects a pickle, whereas preprocessing produces JSON; omit it as shown above. All compared runs must use the same word-index mapping.

## Git tracking and reproducibility

Commit source code, README, requirements, stopwords, and `.gitignore`. Ignore rules exclude Python caches, virtual environments, the root corpus pickle, generated data directories, and run checkpoints and binary arrays. Small run reports (`metrics.json`, `val_history.json`, `top_words.txt`, `top_words.json`, and `train.log`) remain eligible for Git so experiment settings and results can be preserved.

Keep local or separately backed-up copies of the corpus and checkpoints. Ignoring files does not delete them. Evaluation exports can be regenerated from checkpoints and matching data; lost checkpoints require retraining.

Record input provenance, dependency versions, commands, seeds, and hardware for published experiments. Multiworker Word2Vec training and GPU execution may vary despite fixed seeds. Existing run names do not fully describe their settings; consult their training arguments before grouping results.
