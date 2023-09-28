from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import os
from datetime import datetime

# Load the encoder information
with open('encoder_subg.json', 'r') as file:
    encoder_info = json.load(file)

with open('encoder_grade.json', 'r') as file:
    encoder_info_grade = json.load(file)

app = Flask(__name__)

# Load all the models when the Flask app starts
model_subg = joblib.load('best_models_subg.pkl')
model_loan_status = joblib.load('lgbm_loan_status.pkl')
model_grade = joblib.load('xgb_model_grade.pkl')
model_int_rate = joblib.load('xgbr_int_rate_best.pkl')

@app.route('/predict/subg', methods=['POST'])
def predict_subg():
    start_time = datetime.now()

    data = request.json
    df = pd.DataFrame(data)

    # Check if the 'grade' column exists in the provided data
    if 'grade' not in df.columns:
        return jsonify({"error": "The 'grade' column is missing in the provided data."}), 400

    # Get the grade from the dataframe
    grade = df['grade'].iloc[0]

    # Check if a model exists for the provided grade
    if grade not in model_subg:
        return jsonify({"error": f"No model available for grade {grade}."}), 400

    # Adjusting data types for subg model
    df["fico_range_high"] = df["fico_range_high"].astype(float)
    df["last_fico_range_low"] = df["last_fico_range_low"].astype(float)
    df["percent_bc_gt_75"] = df["percent_bc_gt_75"].astype(float)
    df["loan_amnt"] = df["loan_amnt"].astype(float)
    df["total_rev_hi_lim"] = df["total_rev_hi_lim"].astype(float)
    df["inq_last_6mths"] = df["inq_last_6mths"].astype(float)
    df["acc_open_past_24mths"] = df["acc_open_past_24mths"].astype(float)
    df["dti"] = df["dti"].astype(float)
    df["tot_hi_cred_lim"] = df["tot_hi_cred_lim"].astype(float)
    df['term'] = ' ' + df['term'].str.lstrip()

    # Get the model for the provided grade
    model_for_grade = model_subg[grade]
    # model_for_grade.named_steps['classifier'].set_params(boosting_type='goss')
    # Make predictions
    predictions_encoded = model_for_grade.predict(df)
    predictions = [encoder_info[grade]['original'][int(pred)] for pred in predictions_encoded]

    subg_probabilities = model_for_grade.predict_proba(df).tolist()[0]
    subg_prob_dict = {encoder_info[grade]['original'][idx]: prob for idx, prob in enumerate(subg_probabilities)}


    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return jsonify({
        "sub_grade": predictions,
        "subg_probability": subg_prob_dict,
        "processing_time": processing_time
    })

@app.route('/predict/loan_status', methods=['POST'])
def predict_loan_status():
    start_time = datetime.now()

    data = request.json
    df = pd.DataFrame(data)

    # Adjusting data types
    df["loan_amnt"] = df["loan_amnt"].astype(float)
    df["risk_score"] = df["risk_score"].astype(float)
    df["debt_to_income"] = df["debt_to_income"].astype(float)
    df["year_issue"] = df["year_issue"].astype(int)

    predictions = model_loan_status.predict(df).tolist()

    # If your model supports it, get the prediction probabilities
    loan_status_probabilities = model_loan_status.predict_proba(df).tolist()[0]

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return jsonify({
        "loan_status": predictions,
        "loan_status_probability": loan_status_probabilities,
        "processing_time": processing_time
    })

@app.route('/predict/grade', methods=['POST'])
def predict_grade():
    start_time = datetime.now()

    data = request.json
    df = pd.DataFrame(data)

    # Adjusting data types for grade model
    df["fico_range_high"] = df["fico_range_high"].astype(float)
    df["last_fico_range_low"] = df["last_fico_range_low"].astype(float)
    df["percent_bc_gt_75"] = df["percent_bc_gt_75"].astype(float)
    df["loan_amnt"] = df["loan_amnt"].astype(float)
    df["total_rev_hi_lim"] = df["total_rev_hi_lim"].astype(float)
    df["inq_last_6mths"] = df["inq_last_6mths"].astype(float)
    df["acc_open_past_24mths"] = df["acc_open_past_24mths"].astype(float)
    df["dti"] = df["dti"].astype(float)
    df["tot_hi_cred_lim"] = df["tot_hi_cred_lim"].astype(float)
    df['term'] = ' ' + df['term'].str.lstrip()

    predictions_encoded = model_grade.predict(df)
    grade_predictions = [encoder_info_grade[str(int(pred))] for pred in predictions_encoded]

    grade_probabilities = model_grade.predict_proba(df).tolist()[0]
    grade_prob_dict = {encoder_info_grade[str(idx)]: prob for idx, prob in enumerate(grade_probabilities)}


    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return jsonify({
        "grade": grade_predictions,
        "grade_probability": grade_prob_dict,
        "processing_time": processing_time
    })

@app.route('/predict/int_rate', methods=['POST'])
def predict_int_rate():
    start_time = datetime.now()

    data = request.json
    df = pd.DataFrame(data)

    # Adjusting data types for int_rate model
    df["fico_range_high"] = df["fico_range_high"].astype(float)
    df["last_fico_range_low"] = df["last_fico_range_low"].astype(float)
    df["percent_bc_gt_75"] = df["percent_bc_gt_75"].astype(float)
    df["loan_amnt"] = df["loan_amnt"].astype(float)
    df["total_rev_hi_lim"] = df["total_rev_hi_lim"].astype(float)
    df["inq_last_6mths"] = df["inq_last_6mths"].astype(float)
    df["acc_open_past_24mths"] = df["acc_open_past_24mths"].astype(float)
    df["dti"] = df["dti"].astype(float)
    df["tot_hi_cred_lim"] = df["tot_hi_cred_lim"].astype(float)
    df['term'] = ' ' + df['term'].str.lstrip()

    predictions = model_int_rate.predict(df).tolist()

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return jsonify({
        "int_rate": predictions,
        "processing_time": processing_time
    })

@app.route('/predict/all', methods=['POST'])
def predict_all():
    start_time = datetime.now()

    data = request.json
    df = pd.DataFrame(data)

    # Adjust data types for grade model
    df["fico_range_high"] = df["fico_range_high"].astype(float)
    df["last_fico_range_low"] = df["last_fico_range_low"].astype(float)
    df["percent_bc_gt_75"] = df["percent_bc_gt_75"].astype(float)
    df["loan_amnt"] = df["loan_amnt"].astype(float)
    df["total_rev_hi_lim"] = df["total_rev_hi_lim"].astype(float)
    df["inq_last_6mths"] = df["inq_last_6mths"].astype(float)
    df["acc_open_past_24mths"] = df["acc_open_past_24mths"].astype(float)
    df["dti"] = df["dti"].astype(float)
    df["tot_hi_cred_lim"] = df["tot_hi_cred_lim"].astype(float)
    df['term'] = ' ' + df['term'].str.lstrip()

    # Predict grade
    grade_predictions_encoded = model_grade.predict(df)
    grade_predictions = [encoder_info_grade[str(int(pred))] for pred in grade_predictions_encoded]
    grade_pred = grade_predictions[0]
    df['grade'] = grade_pred

    grade_probabilities = model_grade.predict_proba(df).tolist()[0]
    grade_prob_dict = {encoder_info_grade[str(idx)]: prob for idx, prob in enumerate(grade_probabilities)}

    # Predict subgrade
    grade = df['grade'].iloc[0]
    model_for_grade = model_subg[grade]
    subg_predictions_encoded = model_for_grade.predict(df)
    subg_predictions = [encoder_info[grade]['original'][int(pred)] for pred in
                    subg_predictions_encoded]
    df['sub_grade'] = subg_predictions

    subg_probabilities = model_for_grade.predict_proba(df).tolist()[0]
    subg_prob_dict = {encoder_info[grade]['original'][idx]: prob for idx, prob in enumerate(subg_probabilities)}

    # Predict interest rate
    int_rate_pred = model_int_rate.predict(df).tolist()[0]

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return jsonify({
        "grade": grade_pred,
        "sub_grade": subg_predictions,
        "int_rate": int_rate_pred,
        "grade_probability": grade_prob_dict,
        "subg_probability": subg_prob_dict,
        "processing_time": processing_time
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

