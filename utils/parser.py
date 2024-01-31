import os
import numpy as np
import json
import pandas as pd
import pickle
import random
from collections import Counter

class Parser:

    # -----------------------------------------------------
    def __init__(self,
                 factors=None,
                 keys=None):

        if keys is None:
            keys = {
                'no_info': -1,
                'same': 0,
                'different': 1
            }
            print(f'Parser initialized with the keys {keys}')
        if factors is None:
            factors = ['angle', 'style', 'similar', 'background', 'main_object']
            print(f'Parser initialized with the factors {factors}')

        self.keys = keys
        self.factors = factors
    # -----------------------------------------------------

    # -----------------------------------------------------
    def raw_to_df(self, paths, overlap=3):
        print(f'Overlapping across annotators is {overlap}')
        def aggregate(inp):
            if overlap == 1:
                return inp[0]

            inp = list(inp)
            max_val = max(Counter(inp).values())
            if max_val == 1:
                return -1
            else:
                return max(set(inp), key=inp.count)

        # Collect all raw annotations into pandas df
        df_raw = pd.DataFrame()
        for path in paths:
            with open(f"{path}", "r") as io_str:
                data = json.load(io_str)
                df_path = pd.DataFrame.from_dict(data)
                df_raw = pd.concat([df_raw, df_path])

        df_raw['idx'] = range(0, len(df_raw))
        df_raw = df_raw.set_index('idx')

        # To easy to use pandas
        total_dict = {}
        for i in range(len(df_raw)):
            raw_input = df_raw['inputValues'][i]
            for key in raw_input.keys():
                try:
                    total_dict[key].append(raw_input[key])
                except KeyError:
                    total_dict[key] = []
                    total_dict[key].append(raw_input[key])

            raw_output = df_raw['outputValues'][i]
            for key in raw_output.keys():
                try:
                    total_dict[key].append(self.keys[raw_output[key]])
                except KeyError:
                    total_dict[key] = []
                    total_dict[key].append(self.keys[raw_output[key]])

        # Aggregating across annotators
        new_dict = {}
        for key in total_dict.keys():
            old_values = total_dict[key]
            splitted_values = np.array_split(old_values, len(df_raw) / overlap)
            new_values = [aggregate(items) for items in splitted_values]
            new_dict[key] = new_values
        df_final = pd.DataFrame.from_dict(new_dict)

        return df_final
    # -----------------------------------------------------

    # -----------------------------------------------------
    def aggregate(self, df):
        assert isinstance(df, pd.DataFrame), "Aggregation requires parsed dataframe"

        # Aggregation function
        def calculate(inp):
            res = {}
            average_factors = []
            for factor in self.factors:
                factor_values = inp[factor]
                values = [elem for elem in factor_values if elem != -1]
                mean_factor_df = np.mean(values)
                average_factors.append(mean_factor_df)
                res[factor] = mean_factor_df
            res['score'] = np.mean(average_factors)
            return res

        # Bootstrapped estimation
        bootstrap_means = np.zeros(1000)
        for i in range(1000):
            bootstrap_sample = df.sample(n=len(df), replace=True)
            bootstrap_means[i] = calculate(bootstrap_sample)['score']
        confidence_interval = np.percentile(bootstrap_means, [0.5, 99.5])

        res = calculate(df)
        res['confidence_interval_99'] = confidence_interval

        return res
    # -----------------------------------------------------
