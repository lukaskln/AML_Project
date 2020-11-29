from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import neurokit2 as nk
import heartpy as hp
import numpy as np
import pandas as pd
import os
import traceback as tb
import pywt
ROOT_PATH = ""

######### DON'T FORGET #########
##### Tweak the smoothwindow and gradthreshweight values in _ecg_findpeaks_neurokit(), this avoids some false positives

X_test = pd.read_csv(ROOT_PATH + 'X_test.csv', dtype=np.float32).drop(labels='id', axis=1)
X_train = pd.read_csv(ROOT_PATH + 'X_train.csv', dtype=np.float32).drop(labels='id', axis=1)
y_train = pd.read_csv(ROOT_PATH + 'y_train.csv').drop(labels='id', axis=1)['y']
assert len(X_train) == len(y_train)
# data_train = X_train.join(y_train)
print(len(y_train))

def ecg_process_fixed(ecg_signal, sampling_rate=1000):
    from neurokit2.signal.signal_rate import signal_rate
    from neurokit2.ecg.ecg_clean import ecg_clean
    from neurokit2.ecg.ecg_delineate import ecg_delineate
    from neurokit2.ecg.ecg_peaks import ecg_peaks
    from neurokit2.ecg.ecg_phase import ecg_phase
    from neurokit2.ecg.ecg_quality import ecg_quality

    ecg_cleaned = hp.filter_signal(ecg_signal, 0.5, sampling_rate, order=7, filtertype='highpass')
    ecg_cleaned = ecg_clean(ecg_cleaned, method="biosppy", sampling_rate=sampling_rate)

    # R-peaks
    instant_peaks, rpeaks, = ecg_peaks(
        ecg_cleaned=ecg_cleaned, sampling_rate=sampling_rate, method="neurokit", correct_artifacts=False
    )
    #rpeaks["ECG_R_Peaks"] = np.unique(rpeaks["ECG_R_Peaks"])

    rate = signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned))

    quality = ecg_quality(ecg_cleaned, rpeaks=None, sampling_rate=sampling_rate)

    signals = pd.DataFrame({"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Rate": rate, "ECG_Quality": quality})

    # Additional info of the ecg signal
    delineate_signal, delineate_info = ecg_delineate(
        ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate, method='peak'
    )

    cardiac_phase = ecg_phase(ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, delineate_info=delineate_info)

    signals = pd.concat([signals, instant_peaks, delineate_signal, cardiac_phase], axis=1)

    info = rpeaks
    return signals, info

# Invert signal if needed
def check_if_signal_inverted(signal):
    from neurokit2.ecg.ecg_findpeaks import ecg_findpeaks
    from neurokit2.ecg.ecg_clean import ecg_clean
    filtered = hp.filter_signal(signal, 0.5, 300, order=7, filtertype='highpass')
    filtered = ecg_clean(filtered, method="biosppy", sampling_rate=300)
    peaks = ecg_findpeaks(filtered, sampling_rate=300, method="neurokit", show=False)
    inv_peaks = ecg_findpeaks(filtered*-1, sampling_rate=300, method="neurokit", show=False)
    dist_peaks = np.median(np.abs(filtered[peaks['ECG_R_Peaks']]))
    dist_inv_peaks = np.median(np.abs(filtered[inv_peaks['ECG_R_Peaks']]))
    if dist_inv_peaks > dist_peaks:
        filtered *= -1
    return filtered

from scipy.stats import mstats
def using_mstats(s):
    return mstats.winsorize(s, limits=[0.15, 0.15])

def extract_morphological_features(signal):
    try:
        df, info= ecg_process_fixed(signal, sampling_rate=300)
        df["time"] = df.index / 300
        df = df[(df.ECG_R_Peaks == 1) |(df.ECG_P_Peaks == 1) |(df.ECG_Q_Peaks == 1) |(df.ECG_S_Peaks == 1) |(df.ECG_T_Peaks == 1) |(df.ECG_P_Onsets == 1) |(df.ECG_T_Offsets== 1)]

        # Stats on the peaks
        num_P_vs_R =  df["ECG_P_Peaks"].sum()/df["ECG_R_Peaks"].sum()
        num_Q_vs_R =  df["ECG_Q_Peaks"].sum()/df["ECG_R_Peaks"].sum()
        num_S_vs_R =  df["ECG_S_Peaks"].sum()/df["ECG_R_Peaks"].sum()
        num_T_vs_R =  df["ECG_T_Peaks"].sum()/df["ECG_R_Peaks"].sum()

        # Calcualte all needed time and voltage values
        df["P_onset_time"] = np.where(df["ECG_P_Onsets"] == 1, df["time"], np.nan)
        df["P_time"] = np.where(df["ECG_P_Onsets"] == 1, df["time"], np.nan)
        df["Q_time"] = np.where(df["ECG_Q_Peaks"] == 1, df["time"], np.nan)
        df["R_time"] = np.where(df["ECG_R_Peaks"] == 1, df["time"], np.nan)
        df["S_time"] = np.where(df["ECG_S_Peaks"] == 1, df["time"], np.nan)
        df["T_time"] = np.where(df["ECG_T_Peaks"] == 1, df["time"], np.nan)
        df["T_offset_time"] = np.where(df["ECG_T_Offsets"] == 1, df["time"], np.nan)
        df[["P_onset_time", "P_time", "Q_time", "R_time", "S_time", "T_time", "T_offset_time"]] = df[["P_onset_time", "P_time", "Q_time", "R_time", "S_time", "T_time", "T_offset_time"]].ffill(axis = 0) 

        # Calculate the different intervals
        df["RR_time"] = np.where(df["ECG_R_Peaks"] == 1, df["R_time"] - df["R_time"].shift(periods=1), np.nan)
        RR_time_mean = df["RR_time"].dropna().apply(using_mstats).mean()
        RR_time_std = df["RR_time"].dropna().apply(using_mstats).std()
        df["PQ_time"] = np.where(df["ECG_Q_Peaks"] == 1, df["Q_time"] - df["P_onset_time"], np.nan)
        df["QRS_time"] = np.where(df["ECG_S_Peaks"] == 1, df["S_time"] - df["Q_time"], np.nan)
        df["ST_time"] = np.where(df["ECG_T_Peaks"] == 1, df["T_time"] - df["S_time"], np.nan)
        df["QR_time"] = np.where(df["ECG_R_Peaks"] == 1, df["R_time"] - df["Q_time"], np.nan)
        df["RS_time"] = np.where(df["ECG_S_Peaks"] == 1, df["S_time"] - df["R_time"], np.nan)
        df["QT_time"] = np.where(df["ECG_T_Peaks"] == 1, df["T_time"] - df["Q_time"], np.nan)
        df["TP_time"] = np.where(df["ECG_P_Peaks"] == 1, df["P_time"] - df["T_time"], np.nan)

        # If value obviously wrong set it to null
        df["PQ_time"] = np.where(df["PQ_time"] > RR_time_mean, np.nan, df["PQ_time"])
        df["QRS_time"] = np.where(df["QRS_time"] > RR_time_mean, np.nan, df["QRS_time"])
        df["ST_time"] = np.where(df["ST_time"] > RR_time_mean, np.nan, df["ST_time"])
        df["QR_time"] = np.where(df["QR_time"] > RR_time_mean, np.nan, df["QR_time"])
        df["RS_time"] = np.where(df["RS_time"] > RR_time_mean, np.nan, df["RS_time"])
        df["QT_time"] = np.where(df["QT_time"] > RR_time_mean, np.nan, df["QT_time"])
        df["TP_time"] = np.where(df["TP_time"] > RR_time_mean, np.nan, df["TP_time"])

        # Get stats on the relative amplitude of the signals
        df["R_voltage"] = np.where(df["ECG_R_Peaks"] == 1, df["ECG_Clean"], np.nan)
        df["R_voltage"] = df["R_voltage"].ffill(axis=0)
        R_voltage_mean = df["R_voltage"].dropna().apply(using_mstats).mean()
        R_voltage_std = df["R_voltage"].dropna().apply(using_mstats).std()
        df["P_voltage"] = np.where(df["ECG_P_Peaks"] == 1, df["ECG_Clean"]/df["R_voltage"], np.nan)
        df["Q_voltage"] = np.where(df["ECG_Q_Peaks"] == 1, df["ECG_Clean"]/df["R_voltage"], np.nan)
        df["S_voltage"] = np.where(df["ECG_S_Peaks"] == 1, df["ECG_Clean"]/df["R_voltage"], np.nan)
        df["T_voltage"] = np.where(df["ECG_T_Peaks"] == 1, df["ECG_Clean"]/df["R_voltage"], np.nan)
        df["ST_voltage"] = np.where(df["ECG_T_Peaks"] == 1, (df["T_voltage"] - df["S_voltage"]), np.nan)
        df[["P_voltage", "Q_voltage", "S_voltage", "T_voltage"]] = df[["P_voltage", "Q_voltage", "S_voltage", "T_voltage"]].ffill(axis = 0) 

        # Slopes of QR, RS and ST intervals
        df["QR_slope"] = np.where(df["ECG_R_Peaks"] == 1, (df["R_voltage"] - df["Q_voltage"])/ (df["R_time"] - df["Q_time"]), np.nan)
        df["RS_slope"] = np.where(df["ECG_S_Peaks"] == 1, (df["S_voltage"] - df["R_voltage"])/ (df["S_time"] - df["R_time"]), np.nan)
        df["ST_slope"] = np.where(df["ECG_T_Peaks"] == 1, (df["T_voltage"] - df["S_voltage"])/ (df["T_time"] - df["S_time"]), np.nan)

        output_df = pd.DataFrame(dtype=np.float64)
        output_df["num_P_vs_R"] = [num_P_vs_R]
        output_df["num_Q_vs_R"] = num_Q_vs_R
        output_df["num_S_vs_R"] = num_S_vs_R
        output_df["num_t_vs_R"] = num_T_vs_R
        features = ["RR_time", "PQ_time", "QRS_time", "ST_time", "R_voltage", "P_voltage", "S_voltage", "T_voltage", "Q_voltage", 
                    "ST_voltage", "QT_time", "TP_time", "QT_time", "TP_time", "QR_slope", "RS_slope", "ST_slope",]
        for feature in features:
            output_df[feature+"_mean"] = df[feature].dropna().apply(using_mstats).mean()
            output_df[feature+"_std"] = df[feature].dropna().apply(using_mstats).std()
            output_df[feature+"_median"] = df[feature].dropna().apply(using_mstats).median()
    except Exception as e:
        print("Couldn't extract morphological features", e)
        tb.print_exc()

        columns = ['num_P_vs_R', 'num_Q_vs_R', 'num_S_vs_R', 'num_t_vs_R', 'RR_time_mean',
                   'RR_time_std', 'RR_time_median', 'PQ_time_mean', 'PQ_time_std',
                   'PQ_time_median', 'QRS_time_mean', 'QRS_time_std', 'QRS_time_median',
                   'ST_time_mean', 'ST_time_std', 'ST_time_median', 'R_voltage_mean',
                   'R_voltage_std', 'R_voltage_median', 'P_voltage_mean', 'P_voltage_std',
                   'P_voltage_median', 'S_voltage_mean', 'S_voltage_std',
                   'S_voltage_median', 'T_voltage_mean', 'T_voltage_std',
                   'T_voltage_median', 'Q_voltage_mean', 'Q_voltage_std',
                   'Q_voltage_median', 'ST_voltage_mean', 'ST_voltage_std',
                   'ST_voltage_median', 'QT_time_mean', 'QT_time_std', 'QT_time_median',
                   'TP_time_mean', 'TP_time_std', 'TP_time_median', 'QR_slope_mean',
                   'QR_slope_std', 'QR_slope_median', 'RS_slope_mean', 'RS_slope_std',
                   'RS_slope_median', 'ST_slope_mean', 'ST_slope_std', 'ST_slope_median']
        empty_data = np.empty((1,len(columns)))
        empty_data[:] = np.nan
        output_df = pd.DataFrame(data=empty_data, columns=columns)   
    return output_df

    
# signal = X_train.iloc[0, :].dropna().to_numpy()
# df = extract_morphological_features(signal)
# print(df.columns)

# Extract HRV features if needed
def extract_hrv_features(signal):
    try:
        df, info = ecg_process_fixed(signal, sampling_rate=300)
        analyze_df = nk.ecg_analyze(df, sampling_rate=300, method="interval-related")
    except Exception as e:
        print("Couldn't extract HRV features", e)
        tb.print_exc()
        columns = ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD',
                   'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',
                   'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF',
                   'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn',
                   'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S',
                   'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS',
                   'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d',
                   'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d',
                   'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_ApEn',
                   'HRV_SampEn']
        empty_data = np.empty((1,len(columns)))
        empty_data[:] = np.nan
        analyze_df = pd.DataFrame(data=empty_data, columns=columns)
    return analyze_df

# signal = X_train.iloc[10, :].dropna().to_numpy()
# df = extract_hrv_features(signal)

# Extract wavelet features
def extract_wavelet_features(signal):
    try:
        # Segment the timeseries by heartbeat
        df, info = ecg_process_fixed(signal, sampling_rate=300)
        epochs = nk.ecg_segment(df["ECG_Clean"].to_numpy(), rpeaks=info["ECG_R_Peaks"], sampling_rate=300, show=False)

        # Combine the segments back into one pandas DF
        df_out = None
        segmented_len = 0
        for epoch, df in epochs.items():
            segmented_len += len(df)
            if df_out is None:
                df_out = df
                df_out["epoch"] = int(epoch)
                df_out["index"] = np.arange(len(df_out))
            else:
                df["epoch"] = np.ones(len(df)) * int(epoch)
                df["index"] = np.arange(len(df))
                df_out = pd.concat([df, df_out], axis=0, ignore_index=False)

        # Calculate the representative heartbeat of this signal --> don't we lose a lot of information here?
        df_out["median_val"] = df_out[["Signal", "index"]].groupby("index").transform(np.median)
        groups = df_out.groupby("epoch") 
        max_correlation = 0
        max_epoch = None
        for name, group in groups: 
            correlation = group[["Signal", "median_val"]].corr(method="spearman").loc["Signal", "median_val"]
            if correlation > max_correlation:
                max_correlation = correlation
                max_epoch = name

        # Keep only the "most representative" heartbeat
        df_out = df_out[df_out.epoch == max_epoch]

        # Resample the signal to ensure the same number of features across samples
        import scipy
        resampled_signal = scipy.signal.resample(df_out["median_val"].to_numpy(), 256, t=None, axis=0, window=None, domain='time')

        coeffs = pywt.wavedec(resampled_signal, wavelet='db4', level=3)
        coeffs_filtered = pywt.threshold(coeffs[0], max(10, np.max(coeffs[0]/20.)), mode='soft', substitute=0)  # set to 0 components with small values
        coeffs = pd.DataFrame(data=coeffs_filtered.reshape(1, -1), columns = ["coeff{}".format(i) for i in range(len(coeffs_filtered))])
        # Return the wavelet coeffs as features
    except Exception as e:
        print("Couldn't extract the wavelet features", e)
        coeff_count = 38 # This depends on the sample size (resampled everythign to 256) and the decomposition level (==3)
        empty_data = np.empty((1,coeff_count))
        empty_data[:] = np.nan
        coeffs = pd.DataFrame(data=empty_data, columns = ["coeff{}".format(i) for i in range(coeff_count)])

    return coeffs

# signal = X_train.iloc[10, :].dropna().to_numpy()
# epochs = extract_wavelet_features(signal)

# for idx in range(100):
#     print(idx)
#     signal = X_train.iloc[idx, :].dropna().to_numpy()
#     epochs = extract_signal_features(signal)

# Extact all features from a sample
def process_sample(data, idx_list, cols, test_cols=False):
    if not test_cols:
        df = pd.DataFrame(columns=cols)
    for idx in idx_list:
        try:
            signal = data.iloc[idx, :].dropna().to_numpy()  # dropna drops all nulls, not just trailing nulls
            signal = check_if_signal_inverted(signal)
            hrv_features = extract_hrv_features(signal)
            morphology_features = extract_morphological_features(signal)
            wavelet_features = extract_wavelet_features(signal)
            output_df = pd.concat([hrv_features, morphology_features, wavelet_features], axis=1)
            output_df["index"] = idx  # To be able to keep track of the sample number in parallel execution
            if test_cols:
                return output_df.columns
            df = df.append(output_df)
        except Exception as e:
           print("Failed to extract features for sample: {}".format(idx), e)
    return df

# Preprocess the raw data and save it as the filtered data
def extract_features_pipeline(data):
    # Create the schema of the output file
    # Split the workload into 16 splits for 16 processors
    kf = KFold(n_splits=60, random_state=None, shuffle=False)
    splits = []
    cols = process_sample(data, [0], None, True)
    for _, (_, test_index) in enumerate(kf.split(data)):
        splits.append(test_index)

    print("Starting run")
    # Parallel call to get the processed data
    all_dfs = Parallel(n_jobs=48, verbose=2, prefer="processes")(delayed(process_sample)(data, splits[idx], cols) for idx in range(len(splits)))
    out_df = pd.DataFrame(columns=cols)
    for df in all_dfs:
        out_df = out_df.append(df)
    print("Finished")
    return out_df
       
df = extract_features_pipeline(X_train)
out_path = ROOT_PATH + "X_train_processed.csv"
df.to_csv(out_path, index=False, mode="w", header=True)

df = extract_features_pipeline(X_test)
out_path = ROOT_PATH + "X_test_processed.csv"
df.to_csv(out_path, index=False, mode="w", header=True)

