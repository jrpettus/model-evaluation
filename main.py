import streamlit as st
import pandas as pd
import altair as alt
from io import StringIO
from metrics import ModelEvaluator
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

st.set_page_config(layout="wide")
st.markdown(f"""
# Model Evaluation Report
""")
            
with st.sidebar:
    st.markdown("""
    # Evaluation Inputs 
    
    Upload your own data or work with example classification data with various models
    """)
    
    # create a form to control the flow
    #with st.form("input_form"):
    data_use = st.radio('Use example data or my own?', ('example','load my own'))

    if data_use == 'load my own':
        uploaded_file = st.file_uploader('Load your csv file (make sure a column exists for the target and predicted values)', )

        if uploaded_file is not None:
            # To read file as bytes:
            df = pd.read_csv(uploaded_file)
            actual_column = st.selectbox('Actual column', df.columns)
            predicted_column = st.selectbox('Predicted column', df.columns)
            y_test = df[actual_column]
            y_pred = df[predicted_column]

        if uploaded_file is None:
            st.write('Select your file to see results')

    else:
        X, y = make_classification(n_samples = 5000,
            n_features=10, n_redundant=0, n_informative=5, random_state=1, n_clusters_per_class=1, flip_y=.2
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        models = {
        "Logistic Regression": LogisticRegression(),
        "XGBoost": XGBClassifier(),
        "Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
        }

        selected = st.selectbox('Select a model',list(models))
        # Fit a model
        clf = models[selected].fit(X_train, y_train)

        # Predict probability
        y_pred = clf.predict_proba(X_test)[:,1]

    #submitted = st.form_submit_button("Submit")
        
    threshold = st.slider('Threshold', 0.0, 1.0, 0.5)

if data_use == 'load my own' and uploaded_file is None:
    st.write('Please load your data or select the example option')

else:
    # Allow the user to select a threshold
    model_eval = ModelEvaluator(y_test, y_pred)
    #model_eval_log = ModelEvaluator(y_test, clf_log.predict_proba(X_test)[:,1])

    # create the model evaluator
    model_eval = ModelEvaluator(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Accuracy',f"{model_eval.metric_summary(threshold)['accuracy']*100:.2f}%")
        st.metric('AUC',f"{model_eval.metric_summary(threshold)['auc']*100:.2f}%")

    with col2:
        st.metric('Precision',f"{model_eval.metric_summary(threshold)['precision']*100:.2f}%")
        st.metric('F1 Score',f"{model_eval.metric_summary(threshold)['f1_score']*100:.2f}%")

    with col3:
        st.metric('Recall',f"{model_eval.metric_summary(threshold)['recall']*100:.2f}%")
        st.metric('Matthews Coeficient',f"{model_eval.metric_summary(threshold)['matthews_coeficient']*100:.2f}%")

    col4, col5 = st.columns(2)
    with col4:
        st.altair_chart(model_eval.plot_confusion_matrix(threshold))
        st.altair_chart(model_eval.plot_precision_recall())

    with col5:
        # Plot the ROC curve
        st.altair_chart(model_eval.plot_roc_curve(threshold=threshold)) #+ model_eval_log.plot_roc_curve(threshold=threshold))
        st.altair_chart(model_eval.plot_tpr_fpr(threshold=threshold))

    st.altair_chart(model_eval.plot_pred_densities(threshold))
 