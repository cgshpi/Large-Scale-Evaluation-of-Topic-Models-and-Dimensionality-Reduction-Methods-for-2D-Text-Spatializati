This is the repository for a current research project. Details will be announced after final acceptance.

# Benchmark Specs

## Datasets

* `20_newsgroups`
* `reuters`
* `github_projects`
* `emails`
* `seven_categories`

## Topic Models

* `VSM`
* `VSM` + Tfidf-Weighting
* `LSI`
* `LSI` + Linear Combined
* `LSI` + Tfidf-Weighting
* `LSI` + Tfidf-Weighting + Linear Combined
* `NMF`
* `NMF` + Linear Combined
* `NMF` + Tfidf-Weighting
* `NMF` + Tfidf-Weighting + Linear Combined
* `LDA`
* `LDA` + Linear Combined

## Dimension Reduction Techniques

* `MDS`
* `SOM`
* `t-SNE`
* `UMAP`

## Quality Metrics

Local Metrics
* Trustworthiness
* Continuity
* Shephard Diagram Correlation
* 7-Neighborhood Hit

Cluster-based Metrics
* Calinski-Harabasz-Index
* Silhouette Coefficient
* Davies-Bouldin-Index

Perception Metric
* Distance consistency

## Analysis

* Heatmaps
* Statistical Measures
* Correlation Tests

# Dev Setup (Ubuntu)

## Dependencies

* openjdk-19-jdk
* ant
* python3-minimal
* python3.10-full
* python3-pip
* git

## Setup

```bash
> pip3 install -r requirements.txt
> python3 -m spacy download en_core_web_sm
> python3 -m spacy download en_core_web_lg
```

## Run

### Parameter Generator

```bash
> python3 parameter_generator.py > parameters.csv
```

### ML Processing

Repeated calls to main.py using a wide range of parameters (see parameter generator) like this call:
```bash
> python3 main.py --perplexity_tsne 40 --n_iter_tsne 6000 --dataset_name reuters --res_file_name ./results/reuters/results_perplexity_tsne_40_n_iter_tsne_6000_dataset_name_reuters.csv
```

### Analysis

After finishing your runs, it is recommended to run parameter_generator.py again to see which job did finish and which not. The results_files then are copied to a directory called res_files_only, where the results can be collected. Thereafter, the results can be collected with postprocessing.py. Then, statistic.py can perform some standard statistical tests on the resulting `full_res` files.

So the standard workflow is:

```bash
> python3 parameter_generator.py
> python3 postprocessing.py
```

The analysis scripts can also be called directly on a results directory by using the `res_dir_path` flag.

# Docker Setup

## Build

```bash
> docker build . -t python-ml_batch:latest projections_benchmark --build-arg PLATFORM=amd64
```

## Run

```bash
> docker run python-ml_batch python3 main.py --perplexity_tsne 40 --n_iter_tsne 6000 --dataset_name reuters --res_file_name ./results/reuters/results_perplexity_tsne_40_n_iter_tsne_6000_dataset_name_reuters.csv
```
Additionally, mounts and workdir need to be set accordingly.

## Batch Run

```bash
> ./batch.sh
> ./batch_big.sh
```

# Topic Models

We uploaded the used topic models in two parts to Zenodo.

* `Part 1:` [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8113828.svg)](https://doi.org/10.5281/zenodo.8113828) contains all topic models for `20-newsgroups`, `emails`, `reuters`, `seven_categories`, and some models for the `github_projects` dataset
* `Part 2:` [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8114601.svg)](https://doi.org/10.5281/zenodo.8114601) contains the remaining topic models for the `github_projects` dataset
