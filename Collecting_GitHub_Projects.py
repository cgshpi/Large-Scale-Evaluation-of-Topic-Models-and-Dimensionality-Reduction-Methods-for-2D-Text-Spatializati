#!/usr/bin/env python
# coding: utf-8

# # Collecting a Corpus of Representative GitHub Projects

# This script collects the first 100 most relevant projects for the concepts Cryptocurrency, Data Visualization,
# Machine Learning, Frontend, Server, Database, Shell, 3d according to their stars.


from github import Github
import pandas as pd
from tqdm.notebook import tqdm

ACCESS_TOKEN = 'SECRET'
g = Github(ACCESS_TOKEN)


def main():
    concepts = ["cryptocurrency", "data-visualization", "machine-learning", "frontend", "database", "shell", "server",
                "3d"]
    frames = [get_concept_dataframe(concept) for concept in concepts]
    complete_df = pd.concat(frames, ignore_index=True)
    complete_df.to_csv("Github_projects.csv", index=False)


def get_concept_dataframe(concept):
    query = "topic:" + concept
    result = g.search_repositories(query, 'stars', 'desc')
    print(f'Found {result.totalCount} repo(s)')
    project_list = []
    stars_list = []
    i = 0
    for repo in tqdm(result):
        if i < 100:
            project_list.append(repo.clone_url)
            stars_list.append(repo.stargazers_count)
            i += 1
    df_cryptocurrency = pd.DataFrame({"project": project_list, "stars": stars_list})
    print(len(project_list))
    print(len(stars_list))
    df_cryptocurrency["concept"] = concept
    return df_cryptocurrency


if __name__ == "__main__":
    main()




