import csv
from datasets import load_dataset


def download_data():
    ds = load_dataset("osyvokon/pavlick-formality-scores")
    return ds


def save_train_data(train_data):
    save_data(train_data, 'osyvokon_train.csv')


def save_test_data(test_data):
    save_data(test_data, 'osyvokon_test.csv')


def save_data(data, fileName):
    avg_score = data.data[1]  # column from the dataset containing ranks
    sentence = data.data[2]  # column from the dataset containing sentences

    with (open(fileName, mode='w') as file):
        for i in range(len(sentence)):
            writer = csv.writer(file)
            a = list()
            a.append(avg_score[i])
            a.append(sentence[i])
            writer.writerow(a)


def read_train_data():
    return read_data('osyvokon_train.csv')


def read_test_data():
    return read_data('osyvokon_test.csv')


def read_data(fileName):
    a = list()
    with open(fileName, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            b = list()
            b.append(row[1])  # sentence
            if float(row[0]) < 0:  # its rank (value from the range [-3.0 , 3.0] where -3.0 means highly informal and 3.0 means highly formal)
                b.append(0)  # class "0" for informal sentences
            else:
                b.append(1)  # class "1" for formal sentences
            a.append(b)
    return a


# download data from the dataset to the local files
if __name__ == "__main__":
    ds = download_data()
    save_train_data(ds["train"])
    save_test_data(ds["test"])
