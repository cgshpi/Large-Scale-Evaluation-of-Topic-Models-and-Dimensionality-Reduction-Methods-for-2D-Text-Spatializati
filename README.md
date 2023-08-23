This is the repository for the publication:
* Daniel Atzberger*, Tim Cech*, Matthias Trapp, Rico Richter, Jürgen Döllner, Tobias Schreck: "Large-Scale Evaluation of Topic Models and Dimensionality Reduction Methods for 2D Text Spatialization", accepted for publication at IEEE VIS 2023.
\*Both authors contributed equally to this work

![image](https://github.com/hpicgs/Topic-Models-and-Dimensionality-Reduction-Benchmark/assets/27726055/0c857118-74af-4028-b73b-f9d450b4cdd0)

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

We have written our code on a Ubuntu 22.04 system. 

## Dependencies

* openjdk-19-jdk
* ant
* python3-minimal
* python3.10-full
* python3-pip
* git
* RScript from r-base-core

Please install this via

```bash
> sudo apt install openjdk-19-jdk ant python3-minimal python3.10-full python3-pip git r-base-core
```

## Setup

```bash
> pip3 install numpy==1.23.5
> pip3 install -r requirements.txt
> python3 -m spacy download en_core_web_sm
> python3 -m spacy download en_core_web_lg
```

Please note that the results reported and the BERT model linked below were trained with hdbscan 0.8.28. Due to compatability issues from Cython 3 with hdbscan, we followed the advise of the developers of hdbscan and updated our requirement to hdbscan 0.8.33 to avoid having to rely on Conda or suppressing the usage of Cython 3 (compare to https://github.com/scikit-learn-contrib/hdbscan/releases).

For postprocessing we also need ggplot2. Please install it via executing:

```bash
> R
> > install.packages("ggplot2")
```
and answering yes at every prompt.

## Run

### Parameter Generator

```bash
> python3 parameter_generator.py > parameters.csv
```

### ML Processing

Repeated calls to main.py using a wide range of parameters (see parameter generator) like this call:
```bash
> python3 main.py --perplexity_tsne 30 --n_iter_tsne 1000 --learning_rate auto --n_neighbors_umap 15 --min_dist_umap 0.1 --max_iter_mds 300 --dataset_name 20_newsgroups --topic_model lsi_tfidf --res_file_name ./results/20_newsgroups/results_perplexity_tsne_30_n_iter_tsne_1000_learning_rate_auto_n_neighbors_umap_15_min_dist_umap_0.1_max_iter_mds_300_dataset_name_20_newsgroups_topic_model_lsi_tfidf.csv
```
*For replication, we recommend you to (first) test a command like above. For running the full benchmark you will most probably need a computer cluster and about two weeks. Further calls can be produced by parameter_generator.py. See above.*

### Analysis

After finishing your runs, it is recommended to run parameter_generator.py again to see which job did finish and which not. The results_files then are copied to a directory called res_files_only, where the results can be collected. Thereafter, the results can be collected with postprocessing.py. Then, statistic.py can perform some standard statistical tests on the resulting `full_res` files.

So the standard workflow is:

```bash
> python3 parameter_generator.py
> python3 postprocessing.py
```

The analysis scripts can also be called directly on a results directory by using the `res_dir_path` flag.

### Postprocessing

To execute this step you need ggplot2. Please refer to "Setup" for instructions how to install this package.
For aesthetics we used ggplot2 for postprocessing our scatter plots in the paper. This is done via the get_r_pdf_plots.py script. If you executed the call under "ML Processing", the according postprocessing call would be:

```bash
> python3 get_r_pdf_plots.py --base_path results/20_newsgroups --dataset_name 20_newsgroups
```

Afterwards, you can find the new scatter plots in the "Analysis_Visualization" directory.

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

# Precomputed resources

Gathering the raw data and training the topic models requires a lot of time and resources. Therefore, we provide these precomputed artefacts

## Data and NLTK corpora data

We uploaded our raw data and our version of nltk corpora data under:

* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8238920.svg)](https://doi.org/10.5281/zenodo.8238920)

Since some data may be subject to copyright, we provide restricted access to these precomputed files.
We will make this data available iff you affirm that you plan to scientifically reproduce our benchmark.

## Topic Models

We uploaded the used topic models in two parts to Zenodo.

* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8113828.svg)](https://doi.org/10.5281/zenodo.8113828) contains all topic models for `20-newsgroups`, `emails`, `reuters`, `seven_categories`, and some models for the `github_projects` dataset
* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8114601.svg)](https://doi.org/10.5281/zenodo.8114601) contains the remaining topic models for the `github_projects` dataset

For replication we recommend to download the models under both links and put them in a single [project_home]/models directory.

## Results

All of our results obtained by the method described above can be found in Results/data/full_res_[dataset_name].csv

## Project Structure

After downloading and cloning all files you should end up with the following project structure:

[project_home]
  - Analysis_Visualization
  - Results
  - Mallet
  - models
    - 20_newsgroups
    - emails
    - github_projects
    - reuters
    - seven_categories 
  - data
    - 20_newsgroups
    - emails
    - github_projects
    - reuters
    - seven_categories 
  - corpora
    - stopwords
    - words
    - reuters.zip
    - stopwords.zip
    - words.zip
  - [remaining_source_code_files_from_this_repository] 
