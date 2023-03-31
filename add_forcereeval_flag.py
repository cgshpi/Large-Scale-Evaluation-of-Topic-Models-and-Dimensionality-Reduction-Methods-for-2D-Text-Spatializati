import pandas as pd

parameter_file = "parameters_big.csv"


def main():
    df = pd.read_csv(parameter_file, header=None)
    parameters = df[df.columns[1]].to_numpy()
    df[df.columns[1]] = [el.strip() + " --force_reeval" for el in parameters]
    df.to_csv(parameter_file, header=None, index=None)


if __name__ == "__main__":
    main()
