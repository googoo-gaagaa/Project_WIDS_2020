# Project WIDS 2020

The goal of this project is to build a machine learning model that predicts in-hospital mortality (hospital_death) based on patient clinical records. This is a critical problem in healthcare for early warning systems and improving clinical decision-making.

### Problem Challenges faced
- **Severe class imbalance**: Deaths are rare compared to survival.

- **Data leakage risks**: Multiple rows might share the same encounter or patient ID.

- **Correlated features**: Medical datasets often contain redundant features, which can affect model performance.

- **Group dependencies**: Some samples belong to the same encounter, requiring grouped validation.

  ### Key Approach
1.  Data Splitting:
    - Stratified initial split into training and test sets.
    - GroupKFold based on encounter_id to prevent data leakage across splits.
2. Feature Engineering:
    - Removed highly correlated features (threshold > 0.9) based on the training data.
    - Ensured the same features are dropped from validation and test sets.
3. Class Imbalance Handling:
    - Instead of undersampling, used XGBoost's scale_pos_weight to handle imbalance.
    - This avoids losing valuable data from the majority class.
4. Model:
  - Algorithm: XGBoost Classifier
5. Validation Strategy:
    - 5-Fold GroupKFold cross-validation using encounter_id for grouping.
    - Evaluated on AUC score per fold.
6. Evaluation Metrics:

    - AUC-ROC

### Conclusion
- This project demonstrates a robust, leakage-free pipeline for medical prediction tasks involving structured data with class imbalance and group dependencies. XGBoost with proper validation and preprocessing achieves reliable results for hospital mortality prediction.

