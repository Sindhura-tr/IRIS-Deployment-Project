import streamlit as st
import pandas as pd
import joblib

# Set the web app title
st.set_page_config(page_title="IRIS Project")

st.title("IRIS Deployment Project")
st.subheader("By Sindhura Nadendla")

def predict_results(model,sep_len,sep_wid,pet_len,pet_wid):
    # Get the data in a dictionary format
    d = {
        "SepalLengthCm":[sep_len],
        "SepalWidthCm":[sep_wid],
        "PetalLengthCm":[pet_len],
        "PetalWidthCm":[pet_wid]
    }

    # convert the dictionary into dataframe before feeding it to model
    xnew=  pd.DataFrame(d)

    # Predict the results along with probabilities
    preds = model.predict(xnew)
    probs = model.predict_proba(xnew)

    # Probabilities classes as dictionary
    classes = model.classes_

    # Apply a for loop and save the results in dictionary
    prob_d = {}
    for c,p in zip(classes,probs.flatten()):
        prob_d[c] = float(p)
    
    # Return the predicted results
    return preds[0],prob_d

# Take Inputs from user
sep_len = st.number_input("Please enter a Sepal Length:",min_value=0.00,step=0.01)
sep_wid = st.number_input("Please enter a Sepal Width:",min_value=0.00,step=0.01)
pet_len = st.number_input("Please enter a Petal Length:",min_value=0.00,step=0.01)
pet_wid = st.number_input("Please enter a Petal Width:",min_value=0.00,step=0.01)

# Create a button.
submit = st.button("Predict the Species",type="primary")

# After the button is clicked, load the model joblib file
model = joblib.load("Notebook/model.joblib")
if submit:
    pred,prob = predict_results(model,sep_len,sep_wid,pet_len,pet_wid)
    st.subheader(f"Prediction : {pred}")

    for c,p in prob.items():
        st.subheader(f"{c}:{p:.4f}")
        st.progress(p)