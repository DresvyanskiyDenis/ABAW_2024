from typing import Tuple, List, Dict, Union

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score


def __interpolate_to_100_fps(predictions:np.ndarray, predictions_timesteps:np.ndarray)->\
        Tuple[np.ndarray, np.ndarray]:
    """ Interpolates the predictions from whatever FPS to 100 FPS. To do so, you should pass
    the predictions and corresponding timesteps with the shapes (N, M) and (N,) respectively.
    N - the length of the sequence (the number of timesteps), M - the number of "labels" (class probabilities or
    valence and arousal values). The function will interpolate the labels values using linear interpolation.

    :param predictions: np.ndarray
        The predictions with the shape (N, M), where N is the number of timesteps and M is the number of labels.
    :param predictions_timesteps: np.ndarray
        The timesteps corresponding to the predictions with the shape (N,).
    :return: Tuple[np.ndarray, np.ndarray]
        The interpolated predictions and corresponding timesteps. The predictions will have the shape (N', M),
        where N' is the number of timesteps after interpolation.
    """
    # if the first timestep is not 0.0 seconds, than add it to the timesteps and first row of the predictions
    if predictions_timesteps[0] != 0.0:
        predictions_timesteps = np.concatenate([[0.0], predictions_timesteps])
        predictions = np.concatenate([np.expand_dims(predictions[0].copy(), axis=0), predictions], axis=0)
    # create dataframe with predictions
    predictions_df = pd.DataFrame(predictions, columns=[f'pred_{i}' for i in range(predictions.shape[-1])])
    predictions_df['timestep'] = predictions_timesteps
    predictions_df["milliseconds"] = predictions_df["timestep"] * 1000
    # set timestep as index
    predictions_df = predictions_df.set_index('milliseconds')
    # convert index to TimeDeltaIndex
    predictions_df.index = pd.to_timedelta(predictions_df.index, unit='ms')
    # Resample the DataFrame to 100 FPS (0.01 seconds interval)
    predictions_df = predictions_df.resample('10ms').asfreq()

    # Interpolate the missing values
    predictions_df = predictions_df.interpolate(method='linear')

    # Reset the index to get the timesteps back as a column
    predictions_df.reset_index(inplace=True)
    predictions_df['timestep'] = predictions_df['milliseconds'].dt.total_seconds().apply("float64")

    # Get the predictions and timesteps
    predictions = predictions_df[[f'pred_{i}' for i in range(predictions.shape[-1])]].values
    predictions_timesteps = predictions_df['timestep'].values

    # round timestep to 2 decimal places
    predictions_timesteps = np.round(predictions_timesteps, 2)
    return predictions, predictions_timesteps



def __average_predictions_on_timesteps(timesteps_with_predictions:List[Tuple[np.ndarray, np.ndarray]])->Tuple[np.ndarray, np.ndarray]:
    """ Averages the predictions on the same timesteps. timestamps_with_predictions is a list of tuples
    (timesteps, predictions) where timesteps is an array with timesteps and predictions is an array with model predictions

    :param timesteps_with_predictions: List[Tuple[np.ndarray, np.ndarray]]
        List of tuples with timesteps and predictions.
    :return: Tuple[np.ndarray, np.ndarray]
        Tuple with timesteps and averaged predictions
    """
    # get all timesteps
    num_labels = timesteps_with_predictions[0][1].shape[-1]
    all_timesteps = np.concatenate([timesteps for timesteps, _ in timesteps_with_predictions]).reshape((-1,))
    all_predictions = np.concatenate([predictions for _, predictions in timesteps_with_predictions],
                                     axis=0).reshape((-1,num_labels))
    # get unique timesteps
    unique_timesteps = np.unique(all_timesteps)
    # sort
    unique_timesteps.sort()
    # create array to store the averaged predictions
    averaged_predictions = np.zeros((len(unique_timesteps), num_labels))
    # go over unique timesteps and average the predictions
    for i, timestep in enumerate(unique_timesteps):
        # find indices of the current timestep in the all_timesteps
        indices = np.where(all_timesteps == timestep)
        # take the mean of predictions for the current timestep
        predictions = all_predictions[indices]
        averaged_predictions[i] = predictions.mean(axis=0)
    return unique_timesteps, averaged_predictions



def __synchronize_predictions_with_ground_truth(predictions:np.ndarray, predictions_timesteps:np.ndarray,
                                                ground_truth:np.ndarray, ground_truth_timesteps:np.ndarray)->\
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Synchronizes the predictions with ground truth. It means that we take only the predictions
    that have the same timesteps as the ground truth.


    :param predictions: np.ndarray
        The predictions with the shape (num_timesteps1, num_classes or num_regressions)
    :param predictions_timesteps: np.ndarray
        The timesteps of the predictions. Shape: (num_timesteps1,)
    :param ground_truth: np.ndarray
        The ground truth with the shape (num_timesteps2, num_classes or num_regressions)
    :param ground_truth_timesteps: np.ndarray
        The timesteps of the ground truth. Shape: (num_timesteps2,)
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple with synchronized predictions and ground truth. The first two elements are the synchronized predictions
        and timesteps, the last two elements are the synchronized ground truth and timesteps.
    """
    # delete duplicates in the ground truth timesteps and corresponding ground truth
    ground_truth_timesteps, indices = np.unique(ground_truth_timesteps, return_index=True)
    ground_truth = ground_truth[indices]
    # find common timesteps
    common_timesteps = np.intersect1d(ground_truth_timesteps, predictions_timesteps)
    # find indices of the common timesteps in the predictions and ground truth
    predictions_indices = np.where(np.isin(predictions_timesteps, common_timesteps))[0]
    ground_truth_indices = np.where(np.isin(ground_truth_timesteps, common_timesteps))[0]
    # at this point, the number of common timesteps in the predictions and ground truth should be the same
    new_predictions = predictions[predictions_indices]
    new_predictions_timesteps = predictions_timesteps[predictions_indices]
    new_ground_truth = ground_truth[ground_truth_indices]
    new_ground_truth_timesteps = ground_truth_timesteps[ground_truth_indices]
    # check if the number of timesteps is the same
    assert len(new_predictions_timesteps) == len(new_ground_truth_timesteps)
    return new_predictions, new_predictions_timesteps, new_ground_truth, new_ground_truth_timesteps


def __apply_hamming_smoothing(array:np.ndarray, smoothing_window_size:int)->np.ndarray:
    """ Smooths the data by applying the Hamming window to every channel of the array.

    :param array: np.ndarray
        Array with data. Shape: (num_timesteps, num_classes)
    :param smoothing_window_size: int
        Size of the smoothing window
    :return: np.ndarray
        Smoothed array. Shape: (num_timesteps, num_classes)
    """
    # uses numpy.hamming function. Remember to apply the mean to every element as the window is not normalized
    for i in range(array.shape[-1]):
        array[:, i] = np.convolve(array[:, i], np.hamming(smoothing_window_size)/np.hamming(smoothing_window_size).sum(), mode='same')
    return array


def np_concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient.
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """

    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator / denominator



def evaluate_predictions_on_dev_set_full_fps(predictions:Dict[str, pd.DataFrame], labels:Dict[str, pd.DataFrame],
                                             labels_type:str)->Union[float, Tuple[float, float]]:
    """ Evaluates the predictions on the development set. THe predictions and labels should be passed as the
    dictionaries with video names as keys and dataframes as values.
    The dataframes should have the following columns: ["timestep", "frame_num"] + either ["category_0", ...., "category_8"]
                                                                                      or ["valence", "arousal"].

    The function expects that the predictions are already averaged (there are no intersections in timesteps).
    To average the predictions, you can use the __average_predictions_on_timesteps function. (above)

    :param predictions: Dict[str, pd.DataFrame]
        The predictions. The keys are the video names and the values are the dataframes with the predictions.
    :param labels: Dict[str, pd.DataFrame]
        The labels. The keys are the video names and the values are the dataframes with the labels.
    :param labels_type: str
        The type of the labels. Either "Exp" or "VA".
    :return: float
        Either F1 score or CCC score.
    """
    labels_columns = [f"category_{i}" for i in range(8)] if labels_type == 'Exp' else ["valence", "arousal"]
    full_array_predictions = []
    full_array_labels = []
    # go over video names
    for video_name in predictions.keys():
        predictions_timesteps = predictions[video_name]['timestep'].values
        predictions_values = predictions[video_name][labels_columns].values
        ground_truth_timesteps = labels[video_name]['timestep'].values
        ground_truth_values = labels[video_name][labels_columns].values
        # round timesteps to 2 decimal places
        predictions_timesteps = np.round(predictions_timesteps, 2)
        ground_truth_timesteps = np.round(ground_truth_timesteps, 2)
        # get fps of the ground_truths
        fps = 1 / min(np.diff(ground_truth_timesteps))
        # interpolate predictions to the 100 fps
        predictions_values, predictions_timesteps = __interpolate_to_100_fps(predictions_values, predictions_timesteps)
        # synchronize predictions with ground truth
        predictions_values, predictions_timesteps, ground_truth_values, ground_truth_timesteps = \
            __synchronize_predictions_with_ground_truth(predictions_values, predictions_timesteps, ground_truth_values, ground_truth_timesteps)
        # Apply hamming smoothing
        predictions_values = __apply_hamming_smoothing(predictions_values, smoothing_window_size=fps//2)
        # fpr Exp challenge, the softmax and argmax firstly needed
        if labels_type == 'Exp':
            predictions_values = softmax(predictions_values, axis=-1)
            predictions_values = np.argmax(predictions_values, axis=-1)
            ground_truth_values = np.argmax(ground_truth_values, axis=-1)
        # append to the full big array
        full_array_predictions.append(predictions_values.reshape((predictions_values.shape[0], -1)))
        full_array_labels.append(ground_truth_values.reshape((ground_truth_values.shape[0], -1)))
    # calculate the metric
    full_array_predictions = np.concatenate(full_array_predictions, axis=0)
    full_array_labels = np.concatenate(full_array_labels, axis=0)
    if labels_type == 'Exp':
        full_array_predictions = full_array_predictions.flatten()
        full_array_labels = full_array_labels.flatten()
        return f1_score(full_array_labels, full_array_predictions, average='macro')
    elif labels_type == 'VA':
        valence = np_concordance_correlation_coefficient(full_array_labels[:, 0], full_array_predictions[:, 0])
        arousal = np_concordance_correlation_coefficient(full_array_labels[:, 1], full_array_predictions[:, 1])
        # if valence or arousal is nan, than set it to 0.0
        if np.isnan(valence): valence = 0.0
        if np.isnan(arousal): arousal = 0.0
        return (valence, arousal)




