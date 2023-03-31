import os
import shutil
import subprocess


def main():
    base_path = os.path.join(os.getcwd(), "optimal_results")
    r_base_path = os.path.join(os.getcwd(), "Analysis_Visualization")
    r_script = "Scatterplot.R"

    get_pdfs_from_csv_files(base_path, r_base_path, r_script)


def get_pdfs_from_csv_files(base_path, r_base_path, r_script):
    for cur_dir, dirs, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".csv"):
                continue

            r_path = os.path.join(r_base_path, file)
            shutil.copy(os.path.join(cur_dir, file), r_path)
            subprocess.check_call(["Rscript", r_script, r_path], cwd=r_base_path)


if __name__ == "__main__":
    main()
