# TitanicLens — How to Run

This is your ML project repackaged as a Streamlit web app (just like a deployed dashboard).

## Files in this folder
- `app.py` — the full dashboard application
- `titanic.csv` — the dataset (must stay in the same folder as app.py)
- `requirements.txt` — list of Python packages needed

## Steps to run (on your laptop, with internet access)

1. **Open a terminal / command prompt** in this folder.

2. **Install the required packages** (only needs to be done once):
   ```
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```
   streamlit run app.py
   ```

4. Your browser will automatically open to something like `http://localhost:8501` showing the dashboard.

5. Take your screenshots here for your report:
   - The top metrics bar (Training Samples, Best Accuracy, Best F1 Score, ML Models)
   - The "Predict Survival" tab after entering some values and clicking Predict
   - The "Data Explorer" tab showing the charts
   - The "Model Performance" tab showing the comparison table and ROC curve
   - The "About" tab showing the methodology

6. To stop the app, go back to the terminal and press `Ctrl + C`.

## If pip install fails or is slow
Try:
```
pip install streamlit pandas numpy matplotlib seaborn scikit-learn --break-system-packages
```
(some systems require the `--break-system-packages` flag)

## Notes
- Both your original notebook (`Titanic_ML_Project.ipynb`) and this app use the *same* dataset, same feature engineering, and same two models (Decision Tree + Random Forest) — so they tell a consistent story if anyone compares them.
- You can keep both as deliverables: notebook = the analysis/workflow, app = the polished demo.
