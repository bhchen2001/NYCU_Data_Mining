import pickle

path = '1222_ex/'

best_parms = {'clf__colsample_bytree': 0.8, 'clf__gamma': 0.1, 'clf__learning_rate': 0.0007, 'clf__max_depth': 3, 'clf__min_child_weight': 1, 'clf__n_estimators': 12000, 'clf__random_state': 42, 'clf__scale_pos_weight': 3, 'clf__subsample': 0.8, 'sfs__k_features': 15}
best_features = ('d1_spo2_min', 'intubated_apache', 'apache_2_diagnosis', 'apache_4a_hospital_death_prob', 'h1_resprate_max', 'icu_id', 'd1_temp_min', 'gcs_verbal_apache', 'ventilated_apache', 'd1_heartrate_min', 'gcs_motor_apache', 'd1_resprate_min', 'h1_sysbp_noninvasive_min', 'd1_mbp_min', 'd1_sysbp_min')

with open(path + 'parms', 'wb') as f:
    pickle.dump(best_parms, f)

with open(path + 'features', 'wb') as f:
    pickle.dump(best_features, f)