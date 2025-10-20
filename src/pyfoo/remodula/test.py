import pandas as pd
import numpy as np
import os

from skrub import TableReport, Cleaner

path = "/media/robbis/DATA/meg/remodula/"
fname = "remodula.xlsx"

raw_dataframe = pd.read_excel(os.path.join(path, fname))

# Change column names to remove spaces, dots, and uppercase
raw_dataframe.columns = raw_dataframe.columns.str.replace(r'[.\n\s]+', '_', regex=True).str.lower()

columns_to_drop = [
    'nomi', 'ssri_attuale_dic', 'snri_attuale_dic', 'altriantidep_attuale_dic',
    'ssri_att', 'snri_att', 'altriantidep_att', 'antipsy_att', 'bdz_att', 'moodstab_att', 
    'altreterapie_att', 'ssri_prev', 'snri_prev', 'altriantidep_prev', 'antipsy_prev', 
    'bdz_prev', 'moodstab_prev', 'altro_prev', 'bdz_sost', 'coc_sost', 
    'alcol_sost', 'nicot_sost', 'thc_sost', 'opi_sost', 'amph_sost', 'nps_sost',
    'patologieorganichecomorbili', 'tmta_t0', 'tmtb_t0', 'tmta_t1', 'tmtb_t1',
    'tmta_t2', 'tmtb_t2', 'tmta_t3', 'tmtb_t3', 
    'ctqea_cutoff', 'ctqpa_cutoff', 'ctqen_cutoff', 'ctqpn_cutoff', 'ctqsa_cutoff', 'ctqde_cutoff', 
    'deltat0t1', 'deltat0t2', 'deltat0t2_dicotomico'
]

dataframe = raw_dataframe.drop(columns=columns_to_drop)

# Exclude rows using column 'escludere' and then drop the column
dataframe = dataframe[dataframe['escludere'] != 1].drop(columns=['escludere'])
dataframe['madrs_t1'] = dataframe['madrs_t1'].fillna(0)


# Select columns with t0, t1 and count missing values on a row basis
t0_columns = [col for col in dataframe.columns if col.endswith('_t0')]
t1_columns = [col for col in dataframe.columns if col.endswith('_t1')]
t2_columns = [col for col in dataframe.columns if col.endswith('_t2')]
t3_columns = [col for col in dataframe.columns if col.endswith('_t3')]

# Remove madrs_t1 from t1_columns
t1_columns.remove('madrs_t1')

num_missing_t0 = dataframe[t0_columns].isnull().sum(axis=1) / len(t0_columns)
num_missing_t1 = dataframe[t1_columns].isnull().sum(axis=1) / len(t1_columns)

# Exclude t1, t2, t3 columns
dataframe = dataframe.drop(columns=t1_columns + t2_columns + t3_columns)

# Fill missing values in madrs_t1 with zero
dataframe = dataframe[~np.isnan(dataframe['madrs_t0'])]

# Clean the dataframe using skrub
cleaner = Cleaner()
cleaned_dataframe = cleaner.fit_transform(dataframe)

# Generate a report
report = TableReport(cleaned_dataframe)
#report.open()

# How many missing values are there in each column?
missing_values = cleaned_dataframe.isnull().sum()
percentage_missing = missing_values / len(cleaned_dataframe)
columns_with_too_many_missing = percentage_missing[percentage_missing > 0.45]
names_of_missing = columns_with_too_many_missing.index.tolist()
columns_with_too_many_missing = columns_with_too_many_missing.index
cleaned_dataframe = cleaned_dataframe.drop(columns=columns_with_too_many_missing)

# How many rows have missing values in each row?
missing_rows = cleaned_dataframe.isnull().sum(axis=1)
percentage_missing_rows = missing_rows / len(cleaned_dataframe.T)
subjects_with_missing = percentage_missing_rows > 0.45
subjects_codes_with_missing = cleaned_dataframe[subjects_with_missing]['codicepaziente'].values

cleaned_dataframe = cleaned_dataframe[~subjects_with_missing]

X = cleaned_dataframe.drop(columns=['codicepaziente', 'madrs_t1', 'madrs_t0'])
y = (cleaned_dataframe['madrs_t1'] / cleaned_dataframe['madrs_t0']) <= 0.5

####################################################################################
# Imputing missing values with the mean of each column

from skrub import tabular_pipeline, TableVectorizer, StringEncoder, ToCategorical
from sklearn.model_selection import cross_validate
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import itertools


model = tabular_pipeline("classification")
results = cross_validate(model, X=X, y=y, cv=5)

for strategy in ['median', 'mean', 'most_frequent']:
    
    model = Pipeline(steps=[
        ('balancer', RandomOverSampler()),
        ('table_vectorizer', TableVectorizer(high_cardinality=StringEncoder(), 
                                             low_cardinality=ToCategorical())),
        ('imputer', SimpleImputer(strategy=strategy)),
        ('classifier', RandomForestClassifier())
    ])
    
    results = cross_validate(model, X=X, y=y, cv=StratifiedShuffleSplit(n_splits=15, test_size=0.2, random_state=42))
    accuracy = results['test_score'].mean()
    print(f"Imputation strategy: {strategy} - Accuracy: {accuracy:.3f}")

# Want to try different balancing strategies, different models, and different imputing strategies
# Balancing strategies: undersampling, oversampling
# Models: RandomForest, HistGradientBoosting
# Imputing strategies: mean, median, most_frequent




classifiers = [RandomForestClassifier(), 
               HistGradientBoostingClassifier(), 
               SVC(kernel='rbf'), 
               LinearSVC()]

balancers = [RandomUnderSampler(), 
             RandomOverSampler()]

imputers = [SimpleImputer(strategy='mean'), 
            SimpleImputer(strategy='median'), 
            SimpleImputer(strategy='most_frequent'),
            KNNImputer(n_neighbors=5)
            ]


options = itertools.product(classifiers, balancers, imputers)


df_scores = []
for clf, balancer, imputer in tqdm(options):
    scores = []
    for n in tqdm(range(50)):
        
        model = Pipeline(steps=[
            ('balancer', balancer),
            ('table_vectorizer', TableVectorizer(high_cardinality=StringEncoder(), 
                                                 low_cardinality=ToCategorical())),
            ('imputer', imputer),
            ('classifier', clf)
        ])
        
        cv = StratifiedShuffleSplit(n_splits=50, test_size=0.25, random_state=n)
        results = cross_validate(model, X=X, y=y, cv=cv, 
                                 scoring='balanced_accuracy', n_jobs=-1)
        accuracy = results['test_score'].mean()
        scores.append(accuracy)

    if imputer.__class__.__name__ == 'SimpleImputer':
        imputer_name = f"SimpleImputer({imputer.strategy})"
    else:
        imputer_name = imputer.__class__.__name__

    df_scores.append({
        'balancer': balancer.__class__.__name__,
        'imputer': imputer_name,
        'classifier': clf.__class__.__name__,
        'accuracy': np.mean(scores),
        'std': np.std(scores)
    })

df_scores = pd.DataFrame(df_scores)
print(df_scores.sort_values(by='accuracy', ascending=False))
df_scores.to_csv(os.path.join(path, "remodula_classification_scores.csv"), index=False)
