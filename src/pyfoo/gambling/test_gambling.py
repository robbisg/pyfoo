import pandas as pd
import numpy as np

dataframe = pd.read_excel("/media/robbis/DATA/meg/gambling/gambling.xlsx")


dummies = {
    "Full_Pers._Drug_Use/abuse": {
        '0': 'Drug_None',
        '1': 'Drug_Amphetamine',
        '2': 'Drug_Cocaine',
        '3': 'Drug_Opioid',
        '4': 'Drug_Benzo',
        '5': 'Drug_Cannabis',
    },
    "Fam_Hx_Gamble_Full": {
        '0': 'Gamble_None',
        '1': 'Gamble_Father',
        '2': 'Gamble_Mother',
        '3': 'Gamble_Sibling',
        '4': 'Gamble_Uncle',
        '5': 'Gamble_Child',
    },
    "Full_Fam_Hx_Alc_Probs": {
        '0': 'Alc_None',
        '1': 'Alc_Father',
        '2': 'Alc_Mother',
        '3': 'Alc_Sibling',
        '4': 'Alc_Uncle',
        '5': 'Alc_Child',
    },
}

for keyword, mapping in dummies.items():
    # Assuming your data is in a DataFrame called 'dataframe' and the column is named keyword
    dataframe[keyword] = dataframe[keyword].astype(str)  # Convert everything to strings for consistency
    dataframe[keyword] = dataframe[keyword].replace('nan', np.nan)  # Replace 'nan' strings with NaN values

    # Split combined categories and create a list of all unique categories
    all_categories = set()
    for categories in dataframe[keyword]:
        if pd.notna(categories):  # Skip NaN values
            for category in categories.split(','):
                all_categories.add(category.strip())  # Strip leading/trailing spaces

    # Create a new DataFrame with dummy variables
    columns = list(mapping.values())
    dataframe_dummies = pd.DataFrame(0, index=dataframe.index, columns=columns)

    # Fill in the dummy variables
    for i, categories in enumerate(dataframe[keyword]):
        if pd.notna(categories):
            for category in categories.split(','):
                dataframe_dummies.at[i, mapping[category.strip()]] = 1

    # Combine the original DataFrame with the dummy variables
    dataframe_final = pd.concat([dataframe, dataframe_dummies], axis=1)
    dataframe_final = dataframe_final.drop(columns=[keyword])