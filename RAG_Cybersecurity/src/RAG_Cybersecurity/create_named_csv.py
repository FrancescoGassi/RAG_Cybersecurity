import csv
from src.feature_names import BODMASFeatureNames

feature_names = BODMASFeatureNames.get_all_feature_names()

input_files = [
    ('DatasetPE/BODMAS_features.csv', 'DatasetPE/BODMAS_features_named.csv'),
    ('DatasetPE/test.csv', 'DatasetPE/test_named.csv')
]

for input_path, output_path in input_files:
    print(f"Elaborazione di {input_path}...")
    with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        old_header = next(reader)
        n_features = len(old_header) - 1
        if n_features != len(feature_names):
            print(f"Attenzione: numero di feature nel file ({n_features}) diverso da {len(feature_names)}. Uso i primi {n_features} nomi.")
            feature_names_to_use = feature_names[:n_features]
        else:
            feature_names_to_use = feature_names
        new_header = feature_names_to_use + ['label']
        writer.writerow(new_header)
        for row in reader:
            writer.writerow(row)
    print(f"Creato {output_path}\n")
print("Operazione completata.")