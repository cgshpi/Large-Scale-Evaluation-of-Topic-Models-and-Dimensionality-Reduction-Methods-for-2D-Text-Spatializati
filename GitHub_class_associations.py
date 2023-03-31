import os
import pandas as pd
import pickle


def main():
    df = pd.read_csv("Github_projects.csv", index_col=0)
    project_list = df["project"].to_list()
    project_list = [project.split("/")[-1].replace(".git", "") for project in project_list]
    concept_list = df["concept"].to_list()
    project_unique = []
    for project in project_list:
        if project not in os.listdir(os.path.join("data", "github_projects")):
            continue
        if project in project_unique:
            print("Duplicated project: " + project)
        else:
            project_unique.append(project)
    project_concept_mapping = {project_list[i]: concept_list[i] for i in range(len(project_list))}
    print(len(project_concept_mapping))
    with open(os.path.join("data", "github_projects", "project_class_association.pkl"), "wb+") as out_file:
        pickle.dump(project_concept_mapping, out_file)


if __name__ == "__main__":
    main()
