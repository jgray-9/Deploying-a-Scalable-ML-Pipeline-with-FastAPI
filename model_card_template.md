# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was developed by Josh Gray and is a RandomForestClassifier imported from sklearn.ensemble.  The model was trained using default hyperparameters.
## Intended Use
The model is used to classify whether an individual's salary is more than $50K or less than or equal to $50K based on census features.  Intended users are data scientists interested in this census classification.
## Training Data
According to UC Irvine Machine Learning Repository, Barry Becker extracted the data from the 1994 Census database.The census data was provided as a CSV file in the data folder in the Deploying-a-Scalable-ML-Pipeline-with-FastAPI repository.  The training set consisted of 80% of the data.  OneHotEncoder was used on training data features, and LabelBinarizer was used on the target label.
## Evaluation Data
The test set consisted of 20% of the data.  The OneHotEncoder and LabelBinarizer fit on the training set were reused to process the test set.
## Metrics
The metrics that were used were precision, recall, and F1 score.  After running train_model.py, these metrics were produced - Precision: 0.7440 | Recall: 0.6346 | F1: 0.6850.  The slice_output.txt file contains the same 3 metrics for different slices of data, showing that model performace varies across categories.
## Ethical Considerations
Because the census data contains demographic information, the model is at risk for unwanted bias.  The model could over-represent some groups and under-represent others.
## Caveats and Recommendations
Something to consider is that the model predicts based on census data from 1994.  Any inferences that can be made should not be applied to the current socioeconomic environment.  The model's use should be limited to educational purposes only.