import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import re
import os

def case_categoriser(grp, include_diagnosis_and_cause_of_death):
    """Helper function to encode the logic assiging groups of rows having a unique id to the composite outcome variable
    
    Args: 
        grp (pandas DataFrameGroupBy): Rows from source data groupbed by id
        include_diagnosis_and_cause_of_death (bool): Whether the clinician diagnosis or cause of death values are included in outcome
        
    Returns: 
        pandas DataFrameGroupBy with outcome assigned to `bc_positive_or_diagnosis_or_cause_of_death` column
    """
    if include_diagnosis_and_cause_of_death and (any(grp['eons_diagnosis']) or any(grp['eons_cause_of_death'])):
        grp['bc_positive_or_diagnosis_or_cause_of_death'] = True
        grp['description'] = 'diagnosis_or_death'
    else:
        valid_grp = grp.loc[
            (~pd.isna(grp['Neolab_finalbcresult'])) &
            (grp['age_at_test'] >= 0) &
            (grp['diff_between_test_and_admission'] <= 48)
        ]
        if len(valid_grp) == 0:
            grp['bc_positive_or_diagnosis_or_cause_of_death'] = False
            n_tests_removed = (~pd.isna(grp['Neolab_finalbcresult'])).sum() - len(valid_grp)
            grp['description'] = 'no_tests_taken' if n_tests_removed == 0 else 'all_tests_excluded'
        else:
            definitive_result_found = False
            for idx, row in valid_grp.iterrows():
                if row['Neolab_finalbcresult'] in ['Pos', 'Positive']:
                    grp['bc_positive_or_diagnosis_or_cause_of_death'] = True
                    definitive_result_found = True
                    grp['description'] = 'pos_result_found'
            if not definitive_result_found:
                for idx, row in valid_grp.iterrows():
                    if row['Neolab_finalbcresult'] in ['Neg', 'Negative (FINAL)']:
                        grp['bc_positive_or_diagnosis_or_cause_of_death'] = False
                        definitive_result_found = True
                        grp['description'] = 'neg_result_found'
            if not definitive_result_found:
                for idx, row in valid_grp.iterrows():
                    if row['Neolab_finalbcresult'] in ['PosP', 'Positive (Preliminary awaiting identification and AST)']:
                        grp['bc_positive_or_diagnosis_or_cause_of_death'] = True
                        definitive_result_found = True
                        grp['description'] = 'pos_result_found'
            if not definitive_result_found:
                for idx, row in valid_grp.iterrows():
                    if row['Neolab_finalbcresult'] in ['NegP', 'Negative (PRELIMINARY)']:
                        grp['bc_positive_or_diagnosis_or_cause_of_death'] = False
                        definitive_result_found = True
                        grp['description'] = 'neg_result_found'
            if not definitive_result_found:
                grp['bc_positive_or_diagnosis_or_cause_of_death'] = False
                grp['description'] = 'confusing'
    return grp
        
        

class DataManager(): 
    """Convenient wrapper around source data.
    """
    def __init__(self, filepath, scale='all', dummies=False, drop_first=True, reduce_cardinality=True, include_diagnosis_and_cause_of_death=True, hours_threshold=72): 
        """Create instance of class.
        
        Args: 
            filepath (str): Where to find the data
            scale (str): Whether to scale continuous features only (`numeric`), or both continuous and categorical variables (`all`), using scikit-learn's StandardScaler. Defaults to `all`
            dummies (bool): Whether to convert categorcal variable to dummy variables or use as is. Defaults to True
            drop_first (bool): If converting to dummies, whether to drop the first dummy. Defaults to True
            reduce_cardinality (bool): Whether to group features with high cardinality into buckets. Defaults to True
            include_diagnosis_and_cause_of_death (bool): Whether the clinician diagnosis or cause of death values are included in outcome. Defaults to True
        """
        self.filepath = filepath
        self.reduce_cardinality = reduce_cardinality
        self.include_diagnosis_and_cause_of_death = include_diagnosis_and_cause_of_death
        self.hours_threshold = hours_threshold
        self.df = self._load_data()
        self.scale = scale
        self.dummies = dummies
        self.drop_first = drop_first
        
    def _load_data(self): 
        """Load data and perform various type-related tasks and some feature engineering
        
        Returns: 
            Tidied pandas DataFrame
        """
        df = pd.read_csv(self.filepath)
        df = df.assign(Neolab_datebct    = pd.to_datetime(df['Neolab_datebct'],    utc=True),
                       Neolab_datebcr    = pd.to_datetime(df['Neolab_datebcr'],    utc=True),
                       Datetimeadmission = pd.to_datetime(df['Datetimeadmission'], utc=True),
                       Datetimedeath     = pd.to_datetime(df['Datetimedeath'],     utc=True))
        df['id'] = df['Uid'].str.cat(df['Datetimeadmission'].dt.strftime('%Y-%m-%dT%H:%M:%S'), sep='_')
        df = df.loc[~pd.isna(df['Age'])]
        df['Age'] = df['Age'].str.replace(',','').astype(float)
        df['Birthweight'] = df['Birthweight'].str.replace(',','').astype(float)
        df = df.loc[(df['Age'] <= self.hours_threshold) & (df['Age'] >= 0)].reset_index()
        # Age at test is age at admission plus difference between time that test was taken and time of admission:
        df['age_at_test'] = df['Age'] + (df['Neolab_datebct'] - df['Datetimeadmission']) / pd.Timedelta(hours=1)
        df['diff_between_test_and_admission'] = (df['Neolab_datebct'] - df['Datetimeadmission']) / pd.Timedelta(hours=1)
        # If tested, was keep only those records where age at test was less than seven days:
        # df = df.loc[(pd.isna(df['Neolab_datebct'])) | (df['age_at_test'] < 24 * 7)].reset_index()
        # If there was a diagnosis at discharge or cause of death listed, did they contain any of the labels relating to EONS:
        df['eons_diagnosis'] = (~pd.isna(df['Diagdis1'])) & (df['Diagdis1'].str.contains('SEPS|Sepsis|EONS'))
        df['eons_cause_of_death'] = (~pd.isna(df['Causedeath'])) & (df['Causedeath'].str.contains('SEPS|Sepsis|EONS'))
        n_cases = len(df['id'].unique())
        print(n_cases)
        df = pd.concat([case_categoriser(grp, self.include_diagnosis_and_cause_of_death) for _, grp in df.groupby('id')])
        # Ensure we haven't lost any cases:
        print(len(df['id'].unique()))
        assert len(df['id'].unique()) == n_cases
        if self.reduce_cardinality:
            def simplify_vomiting(x):
                if pd.isna(x):
                    return None
                elif x in ['Poss', 'No']:
                    return x
                else:
                    return 'Yes'
            df['Vomiting'] = df['Vomiting'].apply(simplify_vomiting)
            df['Vomiting'] = pd.Categorical(df['Vomiting'].fillna('missing'), categories=['missing', 'No', 'Poss', 'Yes'])
            def simplify_norm_values(x):
                if pd.isna(x):
                    return None
                elif x == 'Norm':
                    return x
                else:
                    return 'Abnormal'
            df['Umbilicus'] = df['Umbilicus'].apply(simplify_norm_values)
            df['Umbilicus'] = pd.Categorical(df['Umbilicus'].fillna('missing'), categories=['missing', 'Norm', 'Abnormal'])
            df['Abdomen'] = df['Abdomen'].apply(simplify_norm_values)
            df['Abdomen'] = pd.Categorical(df['Abdomen'].fillna('missing'), categories=['missing', 'Norm', 'Abnormal'])
            def simplify_signs(x):
                if pd.isna(x):
                    return None
                elif x == 'None':
                    return 'no_signs'
                else:
                    return 'signs_present'
            df['Signsrd'] = df['Signsrd'].apply(simplify_signs)
            df['Signsrd'] = pd.Categorical(df['Signsrd'].fillna('missing'), categories=['missing', 'signs_present'])
            df['Dangersigns'] = df['Dangersigns'].apply(simplify_signs)
            df['Dangersigns'] = pd.Categorical(df['Dangersigns'].fillna('missing'), categories=['missing', 'signs_present'])
            def simplify_typebirth(x):
                if 'Tr' in x:
                    return 'triplet'
                elif 'Tw' in x:
                    return 'twin'
                elif 'S' in x:
                    return 'single'
                else:
                    raise ValueError('Unexpected value in typebirth column')
            df['Typebirth'] = df['Typebirth'].apply(simplify_typebirth)
            df['Typebirth'] = pd.Categorical(df['Typebirth'].fillna('missing'), categories=['missing', 'single', 'twin', 'triplet'])
            def simplify_romlength(x):
                if pd.isna(x):
                    return 'missing'
                elif x in ['NOPROM', '< 18 hours']:
                    return 'noprom'
                elif x in ['PROM', '> 18 hours']:
                    return 'prom'
                else:
                    raise ValueError('Unexpected value in Romlength column')
            df['Romlength'] = df['Romlength'].apply(simplify_romlength)
            df['Romlength'] = pd.Categorical(df['Romlength'], categories=['missing', 'noprom', 'prom'])
            def simplify_skin(x):
                if pd.isna(x):
                    return 'missing'
                elif x in ['Rash', 'PUST', 'BOIL', 'Mong', 'Folds', '{Rash,BOIL}']:
                    return 'condition_present'
                else:
                    raise ValueError('Unexpected value in Skin column')
            df['Skin'] = df['Skin'].apply(simplify_skin)
            df['Skin'] = pd.Categorical(df['Skin'], categories=['missing', 'condition_present'])
            def simplify_wob(x):
                if pd.isna(x):
                    return 'missing'
                elif x in ['Sev', 'Severe']: 
                    return 'severe'
                elif x in ['Mod', 'Moderate']:
                    return 'moderate'
                elif x == 'Mild':
                    return 'mild'
                else:
                    raise ValueError('Unexpected value in wob column')
            df['Wob'] = df['Wob'].apply(simplify_wob)
            df['Wob'] = pd.Categorical(df['Wob'], categories=['missing', 'mild', 'moderate', 'severe'])
            def simplify_activity(x):
                if pd.isna(x):
                    return 'missing'
                elif x in ['Alert', 'Alert, active, appropriate']: 
                    return 'alert'
                elif x in ['Irrit', 'Irritable']:
                    return 'irritable'
                elif x in ['Leth', 'Lethargic, quiet, decreased activity']:
                    return 'lethargic'
                elif x in ['Conv', 'Convulsions', 'Seizures, convulsions, or twitchings ']:
                    return 'convulsions'
                elif x in ['Coma', 'Coma (unresponsive)']:
                    return 'coma'
                else:
                    print(f'The problematic value is "{x}"')
                    raise ValueError('Unexpected value in Activity column')
            df['Activity'] = df['Activity'].apply(simplify_activity)
            df['Activity'] = pd.Categorical(df['Activity'], categories=['alert', 'irritable', 'lethargic', 'convulsions', 'coma'])
            df['Colour'] = pd.Categorical(df['Colour'].fillna('missing'), categories=[
                'missing', 'Pink', 'Blue', '{Pink,White}', 'White', '{Yell,White}', 'Yell'
            ])
            def simplify_fontanelle(x):
                if pd.isna(x):
                    return 'missing'
                elif x in ['Flat', 'Flat, Not Tense (normal)']: 
                    return 'flat'
                elif x in ['Bulg', 'Bulging']:
                    return 'bulging'
                elif x in ['Sunk', 'Sunken']:
                    return 'sunken'
                else:
                    raise ValueError('Unexpected value in Fontanelle column')
            df['Fontanelle'] = df['Fontanelle'].apply(simplify_fontanelle)
            df['Fontanelle'] = pd.Categorical(df['Fontanelle'], categories=['missing', 'flat', 'bulging', 'sunken'])
            def simplify_gender(x):
                if x in ['M', 'Male']: 
                    return 'male'
                elif x in ['F', 'Female']:
                    return 'female'
                elif x in ['NS', 'Not Sure']:
                    return 'not_sure'
                else:
                    raise ValueError('Unexpected value in Gender column')
            df['Gender'] = df['Gender'].apply(simplify_gender)
            df['Gender'] = pd.Categorical(df['Gender'], categories=['not_sure', 'male', 'female'])
            
        return df
    
    def get_benchmark(self, df):
        return None
#         def get_p(row):
#             logit_p = -39.4
#             logit_p += 0.99 * row['Temperature']
#             logit_p += 0.06 * row['Rr'] / 5
#             logit_p += 1.44 * row['is_mf']
#             logit_p += 0.54 * row['is_ol']
#             logit_p += 0.36 * row['is_prom']
#             logit_p += 0.59 * row['is_leth']
#             logit_p += 0.84 * row['is_irrit_conv_coma']
#             logit_p += 0.41 * row['is_chest_retractions']
#             logit_p += 0.18 * row['is_grunt']
#             return np.exp(logit_p) / (1 + np.exp(logit_p))

#         base_dir = './imputed-data-n40-k20'
#         rng = np.random.default_rng(seed=2024)
#         imputed_df_idx = rng.integers(40)
#         imputed_df_filename = [filename for filename in os.listdir(base_dir) if 'csv' in filename][imputed_df_idx]
#         imputed_df = pd.read_csv(os.path.join(base_dir, imputed_df_filename))
#         values = []
#         for idx, (_, row) in enumerate(imputed_df.iterrows()):
#             expected_temp = df['Temperature'].iloc[idx]
#             assert row['Temperature'] == expected_temp
#             expected_rr = df['Rr'].iloc[idx]
#             if not pd.isna(expected_rr):
#                 assert row['Rr'] == expected_rr
#             values.append(get_p(row))
#         return values
        
    def remove_outliers(self):
        """Remove cases where any value for selected variables appears to be an outlier"""
        scaler = StandardScaler()
        scaled_df = pd.DataFrame()
        selected_variables = ['Age', 'Satsair', 'Temperature', 'Rr', 'Hr']
        scaled_df[selected_variables] = scaler.fit_transform(self.df[selected_variables])
        
        def is_not_outlier(row): 
            """Build a filter requiring that every value from the selected columns is less than five standard deviations away from the mean"""
            return all((abs(row) < 5) | (pd.isna(row)))

        not_outlier_mask = scaled_df.apply(is_not_outlier, axis=1)
        self.df = self.df.reset_index(drop=True)[not_outlier_mask]
        
    def remove_duplicate_predictors(self, X_cols, y_label):
        """All cases should now have a unique set of predictors and composite outcome variable"""
        self.df = self.df.drop_duplicates(X_cols + ['id', y_label])
        assert len(self.df.drop_duplicates(X_cols + ['id', y_label])) == len(self.df['id'].unique())
        
    def get_X_y(self, X_cols, seed, y_label='bc_positive_or_diagnosis_or_cause_of_death'): 
        """Return features and target according to preferences set out in instantiation of class""" 
        # Get a perfectly balanced target:
        # df_pos = self.df[self.df[y_label]]
        # df_neg = self.df[~self.df[y_label]]
        # self.df = pd.concat([resample(df_neg, replace=False, n_samples=len(df_pos)), df_pos])
        X = self.df[X_cols].copy()
        description = self.df['description']
        y = self.df[y_label]
        
        continuous_colnames = X.select_dtypes(['int64', 'float64']).columns
        
        if self.dummies: 
            X = pd.get_dummies(X, dummy_na=True, drop_first=self.drop_first)
            X = X.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]', '', x))
            self.na_mask = ~X.isnull().any(axis=1)
            X = X.loc[self.na_mask]
            description = description[self.na_mask]
            y = y[self.na_mask]
        
        else: 
            display_X = X.copy()
            for colname in X.columns.difference(continuous_colnames): 
                if X[colname].dtype.name != 'category':
                    X[colname] = pd.Categorical(X[colname]).codes
                    display_X[colname] = pd.Categorical(display_X[colname])
                else:
                    X[colname] = X[colname].cat.codes
                    display_X[colname] = display_X[colname]
            self.display_X = display_X
            
            if self.scale == 'all':
                self.na_mask = ~X.isnull().any(axis=1)
                X = X.loc[self.na_mask]
                description = description[self.na_mask]
                y = y[self.na_mask]
                
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=description, random_state=seed)
        self.y_train = y_train
        self.y_test = y_test
        
        if self.scale == 'numeric':
            X_train_continuous = X_train[continuous_colnames]
            X_test_continuous = X_test[continuous_colnames]
            X_train_categorical = X_train[X_train.columns.difference(continuous_colnames)]
            X_test_categorical = X_test[X_test.columns.difference(continuous_colnames)]

            if len(continuous_colnames) > 0:
                scaler = StandardScaler()
                scaler.fit(X_train_continuous)
                X_train_continuous = pd.DataFrame(scaler.transform(X_train_continuous), columns=scaler.feature_names_in_)
                X_test_continuous = pd.DataFrame(scaler.transform(X_test_continuous), columns=scaler.feature_names_in_)

            X_train = pd.concat([X_train_continuous.reset_index(drop=True), X_train_categorical.reset_index(drop=True)], axis=1)
            X_test = pd.concat([X_test_continuous.reset_index(drop=True), X_test_categorical.reset_index(drop=True)], axis=1)
        
        elif self.scale == 'all':
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), columns=scaler.feature_names_in_)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=scaler.feature_names_in_)
        
        elif self.scale != 'none':
            raise ValueError('DataManager scale parameter must be either `numeric`, `all` or `none`')
        
        return X_train, X_test, y_train, y_test
        
        