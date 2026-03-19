# ML-Project

## ML Problem Statement
This project studies Twitter bot detection as a binary classification task (genuine account vs. bot account) using the MIB datasets.

## Data Source and Usage Terms
- Source: MIB datasets from the Institute of Informatics and Telematics (CNR).
- Access note: the full datasets are available to researchers upon request and are not openly redistributed.
- Usage constraints:
	- Cite the original publication in research outputs.
	- Do not redistribute the datasets.

Reference paper:
S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, M. Tesconi,
"The Paradigm-Shift of Social Spambots: Evidence, Theories, and Tools for the Arms Race,"
WWW '17 Companion, pp. 963-972, 2017.

## Dataset Scope
We use the provided account groups from the MIB release for supervised bot detection, including genuine accounts and multiple social/traditional spambot groups.

## Data Processing Plan
- Include all available records from each selected dataset split.
- Standardize numerical features before model training.
- Use the same scaled input pipeline for Logistic Regression, Random Forest, XGBoost, and Neural Network for consistency in comparison.

## Models
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network

## Evaluation Plan
- Primary metrics: accuracy, precision, recall, F1-score, and AUPRC.
- Precision is emphasized because false positives (flagging human users as bots) are costly.
- Compare feature importance across models.

## Expected Outcomes
- XGBoost is expected to perform strongly on structured behavioral features.
- Features such as reply delay and account age are expected to be highly predictive, since automated bots often react quickly and use newer accounts.
