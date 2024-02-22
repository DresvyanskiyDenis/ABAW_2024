import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, normalize, PowerTransformer
from sklearn.svm import SVC, SVR
from sklearn.metrics import balanced_accuracy_score, make_scorer, mean_absolute_error, mean_squared_error, recall_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, Lasso
from scipy.stats import zscore
from sklearn.utils.estimator_checks import check_estimator
from elm_kernel import ELM
import math
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns


def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator

# with open('SBERT_LIWC_STATS_train.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     SBERT_train = stored_data['SBERT']
#     LIWC_train = stored_data['LIWC']
#     STATS_train = stored_data['STATS']
    
# with open('SBERT_LIWC_STATS_dev.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     SBERT_dev = stored_data['SBERT']
#     LIWC_dev = stored_data['LIWC']
#     STATS_dev = stored_data['STATS']
    
# with open('SBERT_LIWC_STATS_test.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     SBERT_test = stored_data['SBERT']
#     LIWC_test = stored_data['LIWC']
#     STATS_test = stored_data['STATS']

# Load session-level pdem wav2vec embedding

# Load feature map file
feature_mapped_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/index_sort_filter.csv'

# Load label file
Train_label_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/train_index.csv'
Dev_label_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/validation_index.csv'

# Load feature file
PDEM_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/PDEM_outputs/PDEM/PDEM.pkl'
VAD_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/PDEM_outputs/VAD/VAD.pkl'



def get_chosen_features(feature_mapped_file, label_file, embedding_file, scores_file, percentage, feature_index, ascending_order):
    # Read CSV file
    df_csv = pd.read_csv(feature_mapped_file)
    df_label_csv = pd.read_csv(label_file)

    # Read wav2vec embedding pickle file
    with open(embedding_file, 'rb') as f:
        data = pickle.load(f)
        embedding_data = data.to_numpy()

    # Read pickl file including arousal、valence、dominance
    with open(scores_file, 'rb') as f:
        scores_data = pickle.load(f)

    df_scores = pd.DataFrame(scores_data, columns=['arousal', 'valence', 'dominance'])
    df_scores.set_index(df_csv['Segment Path'], inplace=True)

    # choose 'Segment Path','Original Audio' and 'Duration' column
    df_combined = pd.DataFrame(df_csv, columns=['Segment Path','Original Audio','Duration'])
    # df_combined['Original Audio'] = df_csv['Original Audio']
    df_combined['embedding_feature'] = [np.array(embedding) for embedding in embedding_data]
    
    # Merge df_scores and df_combined by Segment Path
    df_combined = df_combined.merge(df_scores, left_on='Segment Path', right_index=True)
    # df_combined['Absolute_btw_arousal_valence'] = abs(df_combined['arousal'] - df_combined['valence'])
    
    df_combined_label = df_combined.merge(df_label_csv, left_on='Segment Path', right_index=True)

    # Removing chunks that duration less than 0.5 second.
    df_combined_gt_1s = df_combined[df_combined['Duration'] >= 0.5]

    # Calculate mean value of 'arousal' and 'valence' column
    mean_arousal_valence = df_combined_gt_1s.groupby('Original Audio')[['arousal', 'valence']].transform('mean')

    # Adding new column 'arousal_diff' and 'valence_diff'
    df_combined_gt_1s['arousal_diff'] = abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    df_combined_gt_1s['valence_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence'])

    # Adding new column 'arousal_valence_diff'
    df_combined_gt_1s['arousal_valence_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence']) + abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])

    # Sorting data by Original Audio and chosen feature (feature_index)
    df_combined_sorted = df_combined_gt_1s.sort_values(by=['Original Audio', feature_index], ascending=[True, ascending_order])

    # df_combined_sorted = df_combined.sort_values(by=['Original Audio'], ascending=[True])

    print(df_combined_sorted)
    # df_combined_sorted.to_csv('tmp.txt',index=False, sep='\t')
    # Get top percentage data to form session-level acoustic feature
    df_top_percentage = df_combined_sorted.groupby('Original Audio').apply(lambda x: x.head(math.ceil(x.shape[0] * percentage / 100))).drop('Original Audio', axis=1)
    # df_top_percentage = df_top_percentage.drop('Original Audio', axis=1, inplace=True)
    
    # Calculating embedding and scores mean value
    avg_embedding = df_top_percentage.groupby('Original Audio')['embedding_feature'].apply(lambda x: np.mean(x.values, axis=0))
    avg_scores = df_top_percentage.groupby('Original Audio')[['arousal', 'valence', 'dominance']].mean()
    # Calculating embedding and scores standard deviation value
    std_embedding = df_top_percentage.groupby('Original Audio')['embedding_feature'].apply(lambda x: np.std(x.values, axis=0, ddof=0))
    ## For Pandas pkg, ddof default value of std() is 0
    std_scores = df_top_percentage.groupby('Original Audio')[['arousal', 'valence', 'dominance']].std(ddof=0)


    ave_embedding_np = np.vstack(avg_embedding.to_numpy())  # to two-dimensional matrix
    avg_scores_np = np.vstack(avg_scores.to_numpy())

    std_embedding_np = np.vstack(std_embedding.to_numpy())  # to two-dimensional matrix
    std_scores_np = np.vstack(std_scores.to_numpy())

    return ave_embedding_np, avg_scores_np, std_embedding_np, std_scores_np



Data_use_top_percentage = 100

feature_index = 'valence_diff'
ascending_order = False
C_range = [1,2,3,4,5,6,7,8,9,10] # before
kernels = ['rbf','linear', 'sigmoid', 'poly']

dev_avg_embedding, dev_avg_scores, dev_std_embedding, dev_std_scores = get_chosen_features(feature_mapped_file, Dev_label_file, PDEM_file, VAD_file, Data_use_top_percentage, feature_index, ascending_order)
train_avg_embedding, train_avg_scores, train_std_embedding, train_std_scores = get_chosen_features(feature_mapped_file, Train_label_file, PDEM_file, VAD_file, Data_use_top_percentage, feature_index, ascending_order)

## To do: concatenate features selected by arousal and valence
# feature_index = 'arousal_diff'
# dev_avg_embedding_as, dev_avg_scores, dev_std_embedding_as, dev_std_scores = get_chosen_features(dev_mapped_label_file, PDEM_w2v2_dev_file, VAD_dev_file, Data_use_top_percentage, feature_index, ascending_order)
# train_avg_embedding_as, train_avg_scores, train_std_embedding_as, train_std_scores = get_chosen_features(train_mapped_label_file, PDEM_w2v2_train_file, VAD_train_file, Data_use_top_percentage, feature_index, ascending_order)

# feature_index = 'valence_diff'
# dev_avg_embedding_vl, dev_avg_scores, dev_std_embedding_vl, dev_std_scores = get_chosen_features(dev_mapped_label_file, PDEM_w2v2_dev_file, VAD_dev_file, Data_use_top_percentage, feature_index, ascending_order)
# train_avg_embedding_vl, train_avg_scores, train_std_embedding_vl, train_std_scores = get_chosen_features(train_mapped_label_file, PDEM_w2v2_train_file, VAD_train_file, Data_use_top_percentage, feature_index, ascending_order)

# X_train = np.concatenate((train_avg_embedding_as, train_avg_embedding_vl), axis=1)
# X_dev = np.concatenate((dev_avg_embedding_as, dev_avg_embedding_vl), axis=1)

# X_train = pdem_w2v2_trai.to_numpy() # Works well!
# X_dev = pdem_w2v2_dev.to_numpy()

# X_train = vad_w2v2_train
# X_dev = vad_w2v2_dev

# X_train = train_avg_embedding
# X_dev = dev_avg_embedding

## The following feature combination (PDEM mean + PDEM std + VAD std + SBERT) got the best performance!
# X_train = train_avg_embedding
# X_dev = dev_avg_embedding

X_train = np.concatenate((train_avg_embedding, train_std_embedding), axis=1)
X_dev = np.concatenate((dev_avg_embedding, dev_std_embedding), axis=1)

# X_train = np.concatenate((train_avg_embedding, train_std_embedding, SBERT_train), axis=1)
# X_dev = np.concatenate((dev_avg_embedding, dev_std_embedding, SBERT_dev), axis=1)


# For debug, only using small part of the training and dev dataset
# X_train = pdem_w2v2_train[:2000, :]
# X_dev = pdem_w2v2_dev[:200, :]

# X_train = LIWC_train # WORKS okay without any norm
# X_dev = LIWC_dev

# X_train = STATS_train
# X_dev = STATS_dev

# X_train = np.concatenate((SBERT_train, LIWC_train), axis = 1)
# X_dev = np.concatenate((SBERT_dev, LIWC_dev), axis = 1)

# X_train = np.concatenate((SBERT_train, STATS_train), axis = 1)
# X_dev = np.concatenate((SBERT_dev, STATS_dev), axis = 1)

# X_train = np.concatenate((STATS_train, LIWC_train), axis = 1)
# X_dev = np.concatenate((STATS_dev, LIWC_dev), axis = 1)

# X_train = np.concatenate((SBERT_train, STATS_train, LIWC_train), axis = 1)
# X_dev = np.concatenate((SBERT_dev, STATS_dev, LIWC_dev), axis = 1)

### Todo: clean feature and code.

scaler = StandardScaler().fit(X_train)

X_train_scale = scaler.transform(X_train)
X_dev_scale = scaler.transform(X_dev)

# X_train_scale = np.multiply(np.sign(X_train_scale), np.power(np.abs(X_train_scale), 0.5)) # FOR ELM
# X_dev_scale = np.multiply(np.sign(X_dev_scale), np.power(np.abs(X_dev_scale), 0.5)) # FOR ELM

X_train_scale = normalize(X_train_scale)
X_dev_scale = normalize(X_dev_scale)


symptoms_data = pd.read_csv('Detailed_PHQ8_Labels_mapped.csv', sep = ';')
PHQ_sev_train = symptoms_data.loc[56:, 'PHQ_8Total']
PHQ_sev_dev = symptoms_data.loc[:55, 'PHQ_8Total']

PHQ_symptoms = symptoms_data.drop(columns = ['Participant_ID', 'Original Audio', 'PHQ_8Total']).to_numpy()

PHQ_symptoms_train = PHQ_symptoms[56:]
PHQ_symptoms_dev = PHQ_symptoms[:56]

y_train_symp_unscaled = PHQ_symptoms_train
y_dev_symp = PHQ_symptoms_dev

y_train_sev = np.array(PHQ_sev_train)
y_dev_sev = np.array(PHQ_sev_dev)

# overall_mean = np.mean(y_train_symp_unscaled, axis = None)
# overall_sd = np.std(y_train_symp_unscaled, axis = None)


symptom_means_list = []
y_train_symp = np.zeros((len(y_train_symp_unscaled), 8))
for i in range(8):
    mean_i = np.mean(y_train_symp_unscaled[:,i])
    sd_i = np.std(y_train_symp_unscaled[:,i])
    # mean_i = 0
    # sd_i = 1
    # mean_i = overall_mean
    # sd_i = overall_sd
    symptom_means_list.append([mean_i, sd_i])
    y_train_symp[:,i] = (y_train_symp_unscaled[:,i] - mean_i) / sd_i

symptom_means_list = np.array(symptom_means_list)

def best_params_elm(C_list, Kernel_list, folds):
    
    val_pred_C_values = []
    for C in C_list:
        
        val_pred_kernel_values = []
        for Kernel in Kernel_list:
                
            elm = ELM(c = C, kernel = Kernel, is_classification = False, weighted = True)
            # why don't use scaled training data?                
            elm.fit(X_train, y_train_symp)
            symps_pred_dev = elm.predict(X_dev_scale)
            
            for symp in range(8):
                symps_pred_dev[:,symp] = (symps_pred_dev[:,symp] * symptom_means_list[symp, 1]) + symptom_means_list[symp,0]
            
            # Sanitizing symptoms:
            symps_pred_dev[symps_pred_dev < 0] = 0
            symps_pred_dev[symps_pred_dev > 3] = 3
            
            # Rounding symptoms:
            symps_pred_dev = np.rint(symps_pred_dev)
            
            score_list = []
            score_list_RMSE = []
            score_list_UAR = []
            for i in range(8):
                # UAR:
                try:                                      
                    score_UAR_i = recall_score(y_dev_symp[:,i], symps_pred_dev[:,i], average = 'macro')
                except ValueError:
                    score_UAR_i = 0
                score_list_UAR.append(score_UAR_i)

                # RMSE:
                score_RMSE_i = mean_squared_error(y_dev_symp[:,i], symps_pred_dev[:,i], squared = False)
                score_list_RMSE.append(score_RMSE_i)
                
                # CCC:
                score_i = concordance_correlation_coefficient(y_dev_symp[:,i], symps_pred_dev[:,i])
                score_list.append(score_i)
            
            # print(symps_pred_dev)
             
            pred_dev_summation = np.sum(symps_pred_dev, axis = 1)
            
            # Sanitizing final pred:
            pred_dev_summation[pred_dev_summation < 0] = 0
            pred_dev_summation[pred_dev_summation > 24] = 24
            
            summation_pred_CCC = concordance_correlation_coefficient(y_dev_sev, pred_dev_summation)
            summation_pred_RMSE = mean_squared_error(y_dev_sev, pred_dev_summation, squared = False)
            
            
            score_average = np.mean(score_list)
            RMSE_average = np.mean(score_list_RMSE)
            UAR_average = np.mean(score_list_UAR)
            
            val_pred_kernel_values.append([score_average, score_list, 
                                           RMSE_average, score_list_RMSE,
                                           summation_pred_CCC, summation_pred_RMSE,
                                           UAR_average, score_list_UAR])
            
        val_pred_C_values.append(val_pred_kernel_values)
    # Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. 
    # If you meant to do this, you must specify 'dtype=object' when creating the ndarray. 
    dev_score_table = np.array(val_pred_C_values, dtype=object)
    
    dev_mean_score_table = dev_score_table[:,:,0]
    dev_per_symptoms_score_table = dev_score_table[:,:,1]
    
    dev_mean_RMSE_score_table = dev_score_table[:,:,2]
    dev_per_symptoms_RMSE_score_table = dev_score_table[:,:,3]
        
    dev_summation_CCC_table = dev_score_table[:,:,4]
    dev_summation_RMSE_table = dev_score_table[:,:,5]
      
    dev_mean_UAR_score_table = dev_score_table[:,:,6]
    dev_per_symptoms_UAR_score_table = dev_score_table[:,:,7]
    
    
    ### Optimization Metric ###
    
    # ## For mean Symptom RMSE:
    # best_C_idx = np.where(dev_mean_RMSE_score_table == np.nanmin(dev_mean_RMSE_score_table))[0][0]
    # best_kernel_idx = np.where(dev_mean_RMSE_score_table == np.nanmin(dev_mean_RMSE_score_table))[1][0] 
    
    # ## For mean Symptom UAR:
    # best_C_idx = np.where(dev_mean_UAR_score_table == np.nanmax(dev_mean_UAR_score_table))[0][0]
    # best_kernel_idx = np.where(dev_mean_UAR_score_table == np.nanmax(dev_mean_UAR_score_table))[1][0] 
        
    # For mean Symptom CCC:
    best_C_idx = np.where(dev_mean_score_table == np.nanmax(dev_mean_score_table))[0][0]
    best_kernel_idx = np.where(dev_mean_score_table == np.nanmax(dev_mean_score_table))[1][0]
    
            
    # # For Summation RMSE:
    # best_C_idx = np.where(dev_summation_RMSE_table == np.nanmin(dev_summation_RMSE_table))[0][0]
    # best_kernel_idx = np.where(dev_summation_RMSE_table == np.nanmin(dev_summation_RMSE_table))[1][0]
    
    # # For Summation CCC:
    # best_C_idx = np.where(dev_summation_CCC_table == np.nanmax(dev_summation_CCC_table))[0][0]
    # best_kernel_idx = np.where(dev_summation_CCC_table == np.nanmax(dev_summation_CCC_table))[1][0]
    
    
    
    
    
    
    dev_best_score = np.round_(dev_mean_score_table[best_C_idx, best_kernel_idx], decimals = 4)
    dev_best_per_symps = np.round_(dev_per_symptoms_score_table[best_C_idx, best_kernel_idx], decimals = 4)
    
    dev_best_score_RMSE = np.round_(dev_mean_RMSE_score_table[best_C_idx, best_kernel_idx], decimals = 4)
    dev_best_RMSE_per_symps = np.round_(dev_per_symptoms_RMSE_score_table[best_C_idx, best_kernel_idx], decimals = 4)
    
    dev_best_summation_CCC = np.round_(dev_summation_CCC_table[best_C_idx, best_kernel_idx], decimals = 4)
    dev_best_summation_RMSE = np.round_(dev_summation_RMSE_table[best_C_idx, best_kernel_idx], decimals = 4)
    
    dev_best_score_UAR = np.round_(dev_mean_UAR_score_table[best_C_idx, best_kernel_idx], decimals = 4)
    dev_best_UAR_per_symps = np.round_(dev_per_symptoms_UAR_score_table[best_C_idx, best_kernel_idx], decimals = 4)
    
    
    return C_list[best_C_idx], Kernel_list[best_kernel_idx], dev_best_score, dev_mean_score_table, dev_best_per_symps.reshape(1,-1), dev_best_score_RMSE, dev_best_RMSE_per_symps, dev_best_summation_CCC, dev_best_summation_RMSE, dev_best_score_UAR, dev_best_UAR_per_symps
# C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 0.0003, 0.003, 0.03, 0.3, 3, 30, 300]
# C_range = np.arange(1,10,1)

# C_range = [0.01, 0.1, 0.5, 1,2,3,4,5,6,7,8,9,10,20,40,80,100,200,300,600,1024,2048]

# Todo: trying C=5 in testing!
# C_range = [7]


# kernels = ['linear']

best_C, best_kernel, dev_best_score, dev_scores, dev_best_per_symps, dev_best_score_RMSE, dev_best_RMSE_per_symps, dev_best_summation_CCC, dev_best_summation_RMSE, dev_best_score_UAR, dev_best_UAR_per_symps = best_params_elm(C_list = C_range, 
                                                                          Kernel_list = kernels, 
                                                                          folds = 5)

def get_preds(C, Kernel):
    elm = ELM(c = C, kernel = Kernel, is_classification = False, weighted = True)
                                
    elm.fit(X_train, y_train_symp)
    symps_pred_dev = elm.predict(X_dev_scale)
    
    for symp in range(8):
        symps_pred_dev[:,symp] = (symps_pred_dev[:,symp] * symptom_means_list[symp, 1]) + symptom_means_list[symp,0]
    
    # Sanitizing symptoms:
    symps_pred_dev[symps_pred_dev < 0] = 0
    symps_pred_dev[symps_pred_dev > 3] = 3
    
    # Rounding symptoms:
    symps_pred_dev = np.rint(symps_pred_dev)
    
    score_list = []
    score_list_RMSE = []
    for i in range(8):
        # # UAR:
        # score_i = recall_score(y_dev_symp[:,i], symps_pred_dev[:,i], average = 'macro')
        # score_list.append(score_i)
        # RMSE:
        score_RMSE_i = mean_squared_error(y_dev_symp[:,i], symps_pred_dev[:,i], squared = False)
        score_list_RMSE.append(score_RMSE_i)
        # CCC:
        score_i = concordance_correlation_coefficient(y_dev_symp[:,i], symps_pred_dev[:,i])
        score_list.append(score_i)
    
    print(np.mean(score_list))
    print(np.mean(score_list_RMSE))
     
    pred_dev_summation = np.sum(symps_pred_dev, axis = 1)
    
    # Sanitizing final pred:
    pred_dev_summation[pred_dev_summation < 0] = 0
    pred_dev_summation[pred_dev_summation > 24] = 24

    return symps_pred_dev, pred_dev_summation

dev_symp_pred, dev_summation_pred = get_preds(best_C, best_kernel)
# verify_summation_pred_CCC = concordance_correlation_coefficient(y_dev_sev, dev_summation_pred)
# verify_summation_pred_RMSE = mean_squared_error(y_dev_sev, dev_summation_pred, squared = False)

summation_pred_CCC = np.round_(concordance_correlation_coefficient(y_dev_sev, dev_summation_pred), decimals = 4)
summation_pred_MAE = np.round_(mean_absolute_error(y_dev_sev, dev_summation_pred), decimals = 4)
summation_pred_RMSE = np.round_(mean_squared_error(y_dev_sev, dev_summation_pred, squared = False), decimals = 4)

print('Top data use percentage:', Data_use_top_percentage)
print('Best C is:', best_C)
print('Is ascending order?', ascending_order)
print('summation_pred_CCC:', summation_pred_CCC)
print('summation_pred_RMSE', summation_pred_RMSE)

df_pred = pd.DataFrame({'Actual PHQ8 Score': y_dev_sev, 'Predicted Score': dev_summation_pred})
sns.set(font_scale = 1.2)
g = sns.relplot(data = df_pred, x = 'Actual PHQ8 Score',  y = 'Predicted Score', color = 'b')
g.ax.axline(xy1=(0, 0), slope=1, dashes=(3, 2))
g.ax.set(ylim=(-1, 25), xlim = (-1,25))

# df_dev_pred_symps_regression = pd.DataFrame(dev_symp_pred)
# df_dev_pred_symps_regression.to_csv('Multi-task regression symptom predictions DEV.csv')

plt.scatter(y_dev_sev, dev_summation_pred)
plt.show()