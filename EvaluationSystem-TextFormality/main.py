import csv
import LocalDataset
import torch
import torch.nn.functional as F
import sklearn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# models used in this evaluation system
models = (
    "s-nlp/xlmr_formality_classifier",
    "s-nlp/roberta-base-formality-ranker",
    "s-nlp/deberta-large-formality-ranker",
    "s-nlp/mdeberta-base-formality-ranker",
    "s-nlp/mdistilbert-base-formality-ranker"
)
# init model
tokenizer = AutoTokenizer.from_pretrained("s-nlp/xlmr_formality_classifier")
model = AutoModelForSequenceClassification.from_pretrained("s-nlp/xlmr_formality_classifier")

# dictionary to store the results
resultDict = {
    "Accuracy": -1.,
    "Precision": -1.,
    "Recall": -1.,
    "F1 Score": -1.,
    "Log Loss": -1.,
    "ROC AUC": -1.
}

# cleaning the file
with (open('results.csv', mode='w') as file):
    writer = csv.writer(file)
    writer.writerow("")

data_set = LocalDataset.read_test_data() + LocalDataset.read_train_data()

yTrue = []
yPred = []
yProb = []

# iterate over all the elements (models)
for mod in models:
    tokenizer = AutoTokenizer.from_pretrained(mod)
    model = AutoModelForSequenceClassification.from_pretrained(mod)
    i = 0  # variable to show the progress of the process
    for x, y in data_set:
        inputs = tokenizer(x, return_tensors="pt")

        # running the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        # get the probabilities from the tensor
        probs = F.softmax(logits, dim=1)
        # choose the class (0 - formal ; 1 - informal) with higher probability
        pred_class = torch.argmax(probs, dim=1).item()
        # get the probability of formal class (0)
        prob_formal = probs.tolist()[0][0]

        # update lists that will be used in scikit functions
        # invert the predicted y's to make class "1" equal to formal text
        yTrue.append(y)
        yPred.append(1-pred_class)
        yProb.append(prob_formal)

        # show the progress
        i += 1
        if i % 200 == 0:
            print(mod, i)

    # store the results in the dictionary
    resultDict["Accuracy"] = sklearn.metrics.accuracy_score(yTrue, yPred)
    resultDict["Precision"] = sklearn.metrics.precision_score(yTrue, yPred)
    resultDict["Recall"] = sklearn.metrics.recall_score(yTrue, yPred)
    resultDict["F1 Score"] = sklearn.metrics.f1_score(yTrue, yPred)
    resultDict["Log Loss"] = sklearn.metrics.log_loss(yTrue, yProb)
    resultDict["ROC AUC"] = sklearn.metrics.roc_auc_score(yTrue, yProb)

    # print the results and save them to the file
    with (open('results.csv', mode='a') as file):
        writer = csv.writer(file)
        m = list()
        m.append(mod)
        writer.writerow(m)
        for name, value in resultDict.items():
            l = list()
            l.append(name)
            l.append(value)
            writer.writerow(l)
            print(f"{name}: {value:.3f}")
        print("\n")
        writer.writerow("")
