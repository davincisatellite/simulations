import pandas as pd

def read_files():
            """Returns specific case data as arrays. Specifically for Q2.
            """
            filename = f"dependent_data.csv"
            dependents = pd.read_csv(
                "./data/"+filename, delimiter="\t", header= None)
            dependent_arr = dependents.to_numpy()

            filename = f"state_data.csv"
            states = pd.read_csv(
                "./data/"+filename, delimiter="\t", header= None)
            state_array = states.to_numpy()

            return state_array, dependent_arr


