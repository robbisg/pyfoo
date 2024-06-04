import numpy as np
from tqdm import tqdm
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit


def analysis(model, data, y, n_resampling=100):
    scores = []
    importances = []
    for i in tqdm(range(n_resampling)):
        cv = cross_validate(model, data, y, 
                            n_jobs=-1, 
                            scoring='balanced_accuracy',
                            cv=StratifiedShuffleSplit(n_splits=150), 
                            return_estimator=True)
        scores.append(cv['test_score'].mean())
    
        cv_importances = np.array([est['randomforestclassifier'].feature_importances_ \
            for est in cv['estimator']])
        importances.append(cv_importances.mean(0))
    
    return cv, scores, importances