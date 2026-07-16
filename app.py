import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc)

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TitanicLens — Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

sns.set_style("whitegrid")

# ----------------------------------------------------------------------------
# DATA LOADING + MODEL TRAINING (cached so it only runs once)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("titanic.csv")
    return df

@st.cache_resource
def prepare_and_train(df):
    df_fe = df.copy()

    # Fill missing Age using median per Pclass
    df_fe['Age'] = df_fe.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

    # Fill missing Embarked with mode
    df_fe['Embarked'] = df_fe['Embarked'].fillna(df_fe['Embarked'].mode()[0])

    # Feature engineering
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)
    df_fe['AgeGroup'] = pd.cut(df_fe['Age'], bins=[0, 12, 18, 35, 60, 80],
                                labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])

    le_sex = LabelEncoder()
    df_fe['Sex_enc'] = le_sex.fit_transform(df_fe['Sex'])

    le_emb = LabelEncoder()
    df_fe['Embarked_enc'] = le_emb.fit_transform(df_fe['Embarked'])

    le_age = LabelEncoder()
    df_fe['AgeGroup_enc'] = le_age.fit_transform(df_fe['AgeGroup'].astype(str))

    features = ['Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked_enc', 'FamilySize', 'IsAlone', 'AgeGroup_enc']
    X = df_fe[features]
    y = df_fe['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
    dt_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                       min_samples_split=10, random_state=42)
    rf_model.fit(X_train, y_train)

    return {
        "df_fe": df_fe,
        "features": features,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "dt_model": dt_model, "rf_model": rf_model,
        "le_sex": le_sex, "le_emb": le_emb, "le_age": le_age
    }

df = load_data()
state = prepare_and_train(df)

dt_model = state["dt_model"]
rf_model = state["rf_model"]
X_test = state["X_test"]
y_test = state["y_test"]
features = state["features"]
df_fe = state["df_fe"]

dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
dt_f1 = f1_score(y_test, dt_pred)
rf_f1 = f1_score(y_test, rf_pred)

best_acc = max(dt_acc, rf_acc)
best_f1 = max(dt_f1, rf_f1)
best_model_name = "Random Forest" if rf_acc >= dt_acc else "Decision Tree"

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <h1 style='margin-bottom:0;'>🚢 TitanicLens</h1>
    <p style='color:gray; margin-top:0;'>AI Survival Predictor · v1.0 — Machine Learning Internship Project</p>
    """,
    unsafe_allow_html=True
)

# Top metric cards (like SpamShield-style dashboard)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training Samples", f"{len(state['X_train'])}")
c2.metric("Best Accuracy", f"{best_acc*100:.2f}%")
c3.metric("Best F1 Score", f"{best_f1*100:.2f}%")
c4.metric("ML Models", "2")

st.divider()

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_predict, tab_eda, tab_perf, tab_about = st.tabs(
    ["🔮 Predict Survival", "📊 Data Explorer", "📈 Model Performance", "📋 About"]
)

# ----------------------------------------------------------------------------
# TAB 1 — LIVE PREDICTION
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter Passenger Details")

    colA, colB, colC = st.columns(3)
    with colA:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.radio("Sex", ["male", "female"], horizontal=True)
    with colB:
        age = st.slider("Age", 0, 80, 28)
        fare = st.slider("Fare Paid ($)", 0, 500, 32)
    with colC:
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
        parch = st.number_input("Parents/Children Aboard", 0, 6, 0)

    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"],
                             help="S = Southampton, C = Cherbourg, Q = Queenstown")

    model_choice = st.selectbox("Choose Model", ["Random Forest", "Decision Tree"])

    if st.button("Predict Survival", type="primary"):
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        if age <= 12:
            age_group = "Child"
        elif age <= 18:
            age_group = "Teen"
        elif age <= 35:
            age_group = "Adult"
        elif age <= 60:
            age_group = "MiddleAge"
        else:
            age_group = "Senior"

        sex_enc = state["le_sex"].transform([sex])[0]
        emb_enc = state["le_emb"].transform([embarked])[0]
        age_enc = state["le_age"].transform([age_group])[0]

        input_row = pd.DataFrame([{
            'Pclass': pclass, 'Sex_enc': sex_enc, 'Age': age,
            'SibSp': sibsp, 'Parch': parch, 'Fare': fare,
            'Embarked_enc': emb_enc, 'FamilySize': family_size,
            'IsAlone': is_alone, 'AgeGroup_enc': age_enc
        }])[features]

        model = rf_model if model_choice == "Random Forest" else dt_model
        pred = model.predict(input_row)[0]
        prob = model.predict_proba(input_row)[0][1]

        if pred == 1:
            st.success(f"✅ Likely SURVIVED — confidence: {prob*100:.1f}%")
        else:
            st.error(f"❌ Likely DID NOT SURVIVE — confidence: {(1-prob)*100:.1f}%")

        st.progress(float(prob))
        st.caption(f"Model used: **{model_choice}**")

# ----------------------------------------------------------------------------
# TAB 2 — DATA EXPLORER / EDA
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Survival Count**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Survived', data=df, palette='Set2', ax=ax)
        ax.set_xticklabels(['Died', 'Survived'])
        st.pyplot(fig)

    with col2:
        st.markdown("**Survival Rate by Sex**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x='Sex', y='Survived', data=df, palette='Set1', ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Survival Rate by Passenger Class**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x='Pclass', y='Survived', data=df, palette='Set3', ax=ax)
        st.pyplot(fig)

    with col4:
        st.markdown("**Age Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(df['Age'].dropna(), bins=25, kde=True, color='teal', ax=ax)
        st.pyplot(fig)

    st.markdown("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(8, 5))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

# ----------------------------------------------------------------------------
# TAB 3 — MODEL PERFORMANCE
# ----------------------------------------------------------------------------
with tab_perf:
    st.subheader("Decision Tree vs Random Forest")

    results = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest'],
        'Accuracy': [dt_acc, rf_acc],
        'Precision': [precision_score(y_test, dt_pred), precision_score(y_test, rf_pred)],
        'Recall': [recall_score(y_test, dt_pred), recall_score(y_test, rf_pred)],
        'F1 Score': [dt_f1, rf_f1]
    }).round(4)

    st.dataframe(results, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Metric Comparison**")
        results_melted = results.melt(id_vars='Model', var_name='Metric', value_name='Score')
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Metric', y='Score', hue='Model', data=results_melted, palette='Set2', ax=ax)
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    with col2:
        st.markdown("**ROC Curve**")
        dt_probs = dt_model.predict_proba(X_test)[:, 1]
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={auc(fpr_dt, tpr_dt):.3f})")
        ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc(fpr_rf, tpr_rf):.3f})")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    st.markdown("**Random Forest — Feature Importance**")
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importances.values, y=importances.index, palette='crest', ax=ax)
    st.pyplot(fig)

    with st.expander("View Decision Tree Structure"):
        fig, ax = plt.subplots(figsize=(18, 8))
        plot_tree(dt_model, feature_names=features, class_names=['Died', 'Survived'],
                   filled=True, rounded=True, fontsize=7, max_depth=3, ax=ax)
        st.pyplot(fig)

# ----------------------------------------------------------------------------
# TAB 4 — ABOUT
# ----------------------------------------------------------------------------
with tab_about:
    colL, colR = st.columns(2)

    with colL:
        st.markdown("### 📘 Project Overview")
        st.write(
            "TitanicLens is an end-to-end survival prediction system built using "
            "supervised machine learning. It classifies Titanic passengers as "
            "**survived** or **did not survive** based on demographic and travel attributes."
        )
        st.write(
            "The pipeline cleans missing data, engineers new features "
            "(family size, age groups), and compares two tree-based classifiers."
        )

        st.markdown("### 🧭 Methodology")
        st.markdown(
            """
            1. **Dataset** — Titanic passenger records, 891 rows, 9 columns
            2. **Preprocessing** — Median/mode imputation for missing Age & Embarked
            3. **Feature Engineering** — FamilySize, IsAlone, AgeGroup, label encoding
            4. **Model Training** — Decision Tree, Random Forest
            5. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC
            6. **Deployment** — Streamlit interactive web app
            """
        )

    with colR:
        st.markdown("### 🚀 How to Run")
        st.code(
            "# 1. Install dependencies\n"
            "pip install -r requirements.txt\n\n"
            "# 2. Launch the app\n"
            "streamlit run app.py",
            language="bash"
        )

        st.markdown("### 📦 Requirements")
        st.code(
            "streamlit>=1.32.0\n"
            "scikit-learn>=1.4.0\n"
            "pandas>=2.0.0\n"
            "numpy>=1.26.0\n"
            "matplotlib>=3.8.0\n"
            "seaborn>=0.13.0",
            language="text"
        )

    st.divider()
    st.caption("Built as part of a Machine Learning Internship Project — Decision Tree & Random Forest Classification.")
