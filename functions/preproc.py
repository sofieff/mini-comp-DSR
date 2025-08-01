import pandas as pd 

def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # select features we want
    features = [
        "station_max_temp_c",
        "station_min_temp_c",
        "station_avg_temp_c",
        "station_precip_mm",
        "station_diur_temp_rng_c",
        "precipitation_amt_mm",
        "reanalysis_sat_precip_amt_mm",
        "reanalysis_dew_point_temp_k",
        "reanalysis_air_temp_k",
        "reanalysis_relative_humidity_percent",
        "reanalysis_specific_humidity_g_per_kg",
        "reanalysis_precip_amt_kg_per_m2",
        "reanalysis_max_air_temp_k",
        "reanalysis_min_air_temp_k",
        "reanalysis_avg_temp_k",
        "reanalysis_tdtr_k",
        "ndvi_se",
        "ndvi_sw",
        "ndvi_ne",
        "ndvi_nw",
        "ndvi_mean",
        "ndvi_se_sw_diff",
        "ndvi_ne_nw_diff",
        "station_max_temp_c_prev_week",
        "station_min_temp_c_prev_week",
        "station_avg_temp_c_prev_week",
        "station_precip_mm_prev_week",
        "station_diur_temp_rng_c_prev_week",
        "precipitation_amt_mm_prev_week",
        "reanalysis_sat_precip_amt_mm_prev_week",
        "reanalysis_dew_point_temp_k_prev_week",
        "reanalysis_air_temp_k_prev_week",
        "reanalysis_relative_humidity_percent_prev_week",
        "reanalysis_specific_humidity_g_per_kg_prev_week",
        "reanalysis_precip_amt_kg_per_m2_prev_week",
        "reanalysis_max_air_temp_k_prev_week",
        "reanalysis_min_air_temp_k_prev_week",
        "reanalysis_avg_temp_k_prev_week",
        "reanalysis_tdtr_k_prev_week",
        "ndvi_se_prev_week",
        "ndvi_sw_prev_week",
        "ndvi_ne_prev_week",
        "ndvi_nw_prev_week",
        "ndvi_mean_prev_week",
        "ndvi_se_sw_diff_prev_week",
        "ndvi_ne_nw_diff_prev_week",
        "station_max_temp_c_poly2",
        "station_min_temp_c_poly2",
        "station_avg_temp_c_poly2",
        "station_precip_mm_poly2",
        "station_diur_temp_rng_c_poly2",
        "precipitation_amt_mm_poly2",
        "reanalysis_sat_precip_amt_mm_poly2",
        "reanalysis_dew_point_temp_k_poly2",
        "reanalysis_air_temp_k_poly2",
        "reanalysis_relative_humidity_percent_poly2",
        "reanalysis_specific_humidity_g_per_kg_poly2",
        "reanalysis_precip_amt_kg_per_m2_poly2",
        "reanalysis_max_air_temp_k_poly2",
        "reanalysis_min_air_temp_k_poly2",
        "reanalysis_avg_temp_k_poly2",
        "reanalysis_tdtr_k_poly2",
        "ndvi_se_poly2",
        "ndvi_sw_poly2",
        "ndvi_ne_poly2",
        "ndvi_nw_poly2",
        "ndvi_mean_poly2",
        "ndvi_se_sw_diff_poly2",
        "ndvi_ne_nw_diff_poly2",
        "station_max_temp_c_prev_week_poly2",
        "station_min_temp_c_prev_week_poly2",
        "station_avg_temp_c_prev_week_poly2",
        "station_precip_mm_prev_week_poly2",
        "station_diur_temp_rng_c_prev_week_poly2",
        "precipitation_amt_mm_prev_week_poly2",
        "reanalysis_sat_precip_amt_mm_prev_week_poly2",
        "reanalysis_dew_point_temp_k_prev_week_poly2",
        "reanalysis_air_temp_k_prev_week_poly2",
        "reanalysis_relative_humidity_percent_prev_week_poly2",
        "reanalysis_specific_humidity_g_per_kg_prev_week_poly2",
        "reanalysis_precip_amt_kg_per_m2_prev_week_poly2",
        "reanalysis_max_air_temp_k_prev_week_poly2",
        "reanalysis_min_air_temp_k_prev_week_poly2",
        "reanalysis_avg_temp_k_prev_week_poly2",
        "reanalysis_tdtr_k_prev_week_poly2",
        "ndvi_se_prev_week_poly2",
        "ndvi_sw_prev_week_poly2",
        "ndvi_ne_prev_week_poly2",
        "ndvi_nw_prev_week_poly2",
        "ndvi_mean_prev_week_poly2",
        "ndvi_se_sw_diff_prev_week_poly2",
        "ndvi_ne_nw_diff_prev_week_poly2"
    ]

    df = df[features]

    # fill missing values
    #df.fillna(method="ffill", inplace=True) #already done in the feature engineering step

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc["sj"]
    iq = df.loc["iq"]

    return sj, iq