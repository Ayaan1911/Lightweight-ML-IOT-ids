# Project steps: from dataset to model comparison

### 1. Set up the notebook

Create a Jupyter notebook for the project. Add a short introduction explaining the goal: **detect attacks in IoT network traffic and compare models on detection quality and resource use**.

### 2. Get the dataset

Download **CICIoT2023** from the Canadian Institute for Cybersecurity’s [official dataset page](https://www.unb.ca/cic/datasets/iotdataset-2023.html). It’s the main dataset in the synopsis. Start with a manageable sample; the synopsis proposes about **1 million records** for the main experiment.

### 3. Inspect the data

Load the data and check:

- The number of rows and columns
- The names and types of the columns
- The target label column
- The number of records in each label
- Whether there are missing values or duplicate rows

This helps us understand what the models will learn to predict.

### 4. Clean and prepare it

Remove duplicate or unusable rows, handle missing values, and exclude columns that could let a model memorize record details, such as IP addresses or timestamps, as proposed in the synopsis.

Separate the data into:

- **Features (`X`)**: the traffic information used to make a prediction
- **Label (`y`)**: the answer the model should predict

Keep a record of the cleaning steps so someone else can repeat them.

### 5. Choose the first prediction task

Start with **binary classification**:

- Normal traffic
- Attack traffic

After that works, try **multiclass classification**, where the model predicts the attack category. The synopsis proposes starting with normal vs. attack and then moving to attack categories.

### 6. Split the data fairly

Split the data into **70% training, 15% validation, and 15% testing**, keeping the label proportions similar in each split.

- **Training data** teaches the models.
- **Validation data** helps tune choices.
- **Test data** is held back for the final comparison.

Use the **same split for all four models** so their results are comparable. Don’t use test results to tune the models.

### 7. Train the four baseline models

Train each model using the same training data:

1. Decision Tree
2. Random Forest
3. XGBoost
4. Multilayer Perceptron (MLP)

These are the original, or **baseline**, versions. Save the model settings and training details in the notebook.

### 8. Measure detection performance

Evaluate every model on the same held-back test set. Record:

- Macro F1-score — the main comparison score in the synopsis
- Accuracy, precision, and recall
- False positive rate
- Confusion matrix

Macro F1 matters because it gives each attack category a voice in the score, even if some categories have fewer examples.

### 9. Measure resource use

For each trained model, record:

- Saved model size
- Peak RAM use during prediction
- Average CPU use
- Prediction time for one flow at a time
- Throughput for batches of 1,024 predictions

Use the same measurement conditions for each model. The synopsis plans to run this on a Raspberry Pi 4 with 4 GB RAM, or a limited Docker environment if the Pi is unavailable.

### 10. Put the baseline results in one table

Make one comparison table with a row for each model and columns for its detection scores and resource measurements. This shows which baseline model is accurate and which is lighter or faster.

### 11. Reduce the models

Once the baseline comparison works, create smaller versions:

- **Decision Tree, Random Forest, XGBoost:** select fewer features and limit tree depth or number of trees.
- **MLP:** try pruning, 8-bit quantization, and knowledge distillation into a smaller MLP.

### 12. Compare original and reduced versions

Evaluate and benchmark the reduced models the same way. Compare each reduced version against its original model to see what resource savings it achieved and how much its macro F1 changed.

### 13. Decide what the results show

Use the comparison to answer the project question:

> Which model and reduction method lowers memory, CPU use, or inference time while keeping detection performance strong?

Don’t assume one model will win before running the experiments. The results should decide.

### 14. Use the notebook as the project record

Keep the notebook organized in this order: **goal → dataset → inspection → cleaning → split → baseline training → evaluation → resource measurements → model reduction → final comparison**. For an early progress check, show the parts that are genuinely complete and label unfinished experiments as planned work.

**In one sentence:** use the same IoT dataset and data split to train four baseline models, compare both their attack-detection results and resource use, then repeat the comparison after reducing the models.
