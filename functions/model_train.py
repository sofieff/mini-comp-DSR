from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf


def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = (
        "total_cases ~ 1 + "
        "reanalysis_specific_humidity_g_per_kg + "
        "reanalysis_dew_point_temp_k + "
        "station_min_temp_c + "
        "station_avg_temp_c"
    )

    grid = 10 ** np.arange(-8, -3, dtype=np.float64)

    best_alpha = []
    best_score = 1000

    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(
            formula=model_formula,
            data=train,
            family=sm.families.NegativeBinomial(alpha=alpha),
        )

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print("best alpha = ", best_alpha)
    print("best score = ", best_score)

    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(
        formula=model_formula,
        data=full_dataset,
        family=sm.families.NegativeBinomial(alpha=best_alpha),
    )

    fitted_model = model.fit()
    return fitted_model


sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)
