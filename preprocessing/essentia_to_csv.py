import os
import json
import config

import pandas as pd

def dict_to_row(dictionary):
    if isinstance(dictionary, dict):
        cols = []
        values = []
        for key in dictionary.keys():
            sub_columns, sub_values = dict_to_row(dictionary[key])
            if sub_columns and sub_values:
                sub_columns = [str(key) + '_' + col for col in sub_columns]
                cols += sub_columns
                values += sub_values

        return cols, values
    else:
        if isinstance(dictionary, int) or isinstance(dictionary, float):
            return [''], [dictionary]

        return None, None

def load_json_to_dict(feature_index, json_filepath, features_values_dictionary):
    with open(json_filepath, 'r') as f:
        features = json.load(f)

    # Get only select features - not metadata
    selected_features = {}
    for select_feature in config.ESSENTIA_FEATURES:
        selected_features[select_feature] = features[select_feature]

    cols = ['track_id']
    values = [feature_index]

    new_cols, new_values = dict_to_row(selected_features)
    new_cols = [new_col[:-1] for new_col in new_cols]

    cols = cols + new_cols
    values = values + new_values

    # Update Dictionary with values
    if not features_values_dictionary:
        for i, col in enumerate(cols):
            features_values_dictionary[col] = [values[i]]
    else:
        for col in cols:
            if col not in features_values_dictionary.keys():
                return

        for key in features_values_dictionary.keys():
            if key not in cols:
                return 

        for i, col in enumerate(cols):
            features_values_dictionary[col] += [values[i]]

    return
        

def load_all_essentia_features_in_dir(subdir_path, features_values_dictionary):
    for feature_file in os.listdir(subdir_path):
        feature_index = int(feature_file[:feature_file.index('.')])
        feature_path = os.path.join(subdir_path, feature_file)

        load_json_to_dict(feature_index, feature_path, features_values_dictionary)

if __name__ == "__main__":
    features_values_dictionary = {}

    for subdir in os.listdir(config.ESS_DIR):
        subdir_path = os.path.join(config.ESS_DIR, subdir)
        if os.path.isdir(subdir_path):
            load_all_essentia_features_in_dir(subdir_path, features_values_dictionary)

    print(features_values_dictionary.keys())
    print(len(features_values_dictionary['tonal_tuning_frequency']))

    df = pd.DataFrame.from_dict(features_values_dictionary)
    df.to_csv('essentia_features.csv')