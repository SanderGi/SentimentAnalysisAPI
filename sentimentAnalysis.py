# code inspired by https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
# code by Alex M

import pandas as pd
import pickle
import os.path

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
import re, string, random, contractions

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# column names
TEXT = "text"
SENT = 'airline_sentiment'

class ProcessData:
    @staticmethod
    def removePunctuation(text: str) -> str:
        final = "".join(letter for letter in text if letter not in ("?", ".", ";", ":",  "!",'"'))
        return final

    @staticmethod
    def expandContractions(text: str) -> str:
        final = "".join(contractions.fix(word)+" " for word in text.split(" "))
        return final

    @staticmethod
    def removeNoise(tweet_tokens, stop_words = ()):
        cleaned_tokens = []

        for token, tag in pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    @staticmethod
    def cleanTokenize(text: str):
        return ProcessData.removeNoise(word_tokenize(text), stopwords.words('english'))

    @staticmethod
    def getDictionary(tokens) -> dict:
        return dict([token, True] for token in tokens)

    @staticmethod
    def getDictionaries(tokensList):
        for tweet_tokens in tokensList:
            yield dict([token, True] for token in tweet_tokens)

class TextData:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @classmethod
    def load(cls, filePath: str, columns: list[str]):
        df = pd.read_csv(filePath)[columns]
        return cls(df)

    def removePunctuation(self) -> None:
        self.df[TEXT] = self.df[TEXT].apply(ProcessData.removePunctuation)

    def expandContractions(self) -> None:
        self.df[TEXT] = self.df[TEXT].apply(ProcessData.expandContractions)

    def tokenize(self) -> None:
        self.df['tokenized_text'] = self.df[TEXT].apply(word_tokenize)

    def removeNoise(self) -> None:
        self.tokenize()
        self.df['tokenized_text'] = self.df['tokenized_text'].apply(lambda text: ProcessData.removeNoise(text, stopwords.words('english')))

    def getTestTrainSplit(self, trainSize: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        train, test = train_test_split(self.df, test_size=1.0-trainSize)
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)
        return (train, test)

    def getPositiveNegative(self):
        positive = self.df[self.df[SENT] == 'positive']
        negative = self.df[self.df[SENT] == 'negative']
        return positive, negative

class LogisticModel():
    def __init__(self, model: LogisticRegression, vectorizer: CountVectorizer) -> None:
        self.model = model
        self.vectorizer = vectorizer

    @classmethod
    def load(cls, modelFilePath: str = 'logisticClassifier.pickle', vectorizorFilePath: str = 'logisticVectorizor.pickle'):
        f = open(modelFilePath,'rb')
        model = pickle.load(f)
        f.close()
        f = open(vectorizorFilePath,'rb')
        vectorizer = pickle.load(f)
        f.close()
        return cls(model, vectorizer)

    @classmethod
    def train(cls, data: TextData, trainSize: int=0.8):
        data.removePunctuation()
        data.expandContractions()
        train, test = data.getTestTrainSplit(trainSize=trainSize)

        vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
        train_matrix = vectorizer.fit_transform(train[TEXT])
        test_matrix = vectorizer.transform(test[TEXT])

        X_train = train_matrix
        X_test = test_matrix
        y_train = train[SENT]
        y_test = test[SENT]

        model = LogisticRegression()
        model.fit(X_train,y_train)

        predictions = model.predict(X_test)
        return cls(model, vectorizer), classification_report(predictions,y_test)

    def predict(self, text: str) -> str:
        text = ProcessData.removePunctuation(text)
        text = ProcessData.expandContractions(text)
        vector = self.vectorizer.transform([text])
        return self.model.predict(vector)[0].lower()

    def save(self) -> None:
        f = open('logisticClassifier.pickle','wb')
        pickle.dump(self.model, f)
        f.close()
        f = open('logisticVectorizor.pickle','wb')
        pickle.dump(self.vectorizer, f)
        f.close()

class BayesModel():
    def __init__(self, model: NaiveBayesClassifier) -> None:
        self.model = model

    @classmethod
    def load(cls, modelFilePath: str = 'NaiveBayesClassifier.pickle'):
        f = open(modelFilePath,'rb')
        model = pickle.load(f)
        f.close()
        return cls(model)

    @classmethod
    def train(cls, data: TextData, trainSize: int=0.8):
        data.removePunctuation()
        data.expandContractions()
        data.removeNoise()

        positive, negative = data.getPositiveNegative()
        positiveTokens = ProcessData.getDictionaries(list(positive['tokenized_text']))
        negativeTokens = ProcessData.getDictionaries(list(negative['tokenized_text']))

        positiveDataset = [(tweetDict, "positive") for tweetDict in positiveTokens]
        negativeDataset = [(tweetDict, "negative") for tweetDict in negativeTokens]

        dataset = positiveDataset + negativeDataset
        random.shuffle(dataset)

        trainData = dataset[:int(len(dataset)*trainSize)]
        testData = dataset[int(len(dataset)*trainSize):]

        model = NaiveBayesClassifier.train(trainData)

        return cls(model), classification_report([model.classify(data) for data, _ in testData],[label for _, label in testData])

    def showMostImportantFeatures(self) -> None:
        # this can later be used to implement API for analysing which words contributed most to a classification
        self.model.show_most_informative_features(10)

    def predict(self, text: str) -> str:
        text = ProcessData.removePunctuation(text)
        text = ProcessData.expandContractions(text)
        tokens = ProcessData.cleanTokenize(text)
        dict = ProcessData.getDictionary(tokens)
        return self.model.classify(dict).lower()

    def save(self) -> None:
        f = open('NaiveBayesClassifier.pickle','wb')
        pickle.dump(self.model, f)
        f.close()
        
def getModels() -> tuple[LogisticModel, BayesModel]:
    '''Loads pretrained models or trains them again'''
    bayesModelExists = os.path.isfile('NaiveBayesClassifier.pickle')
    logisticModelExists = os.path.isfile('logisticClassifier.pickle')
    vectorizorExists = os.path.isfile('logisticVectorizor.pickle')
    if bayesModelExists:
        bayesModel = BayesModel.load()
    else:
        data = TextData.load('airline_sentiment_analysis.csv',[SENT, TEXT])
        bayesModel, report = BayesModel.train(data)
        bayesModel.save()
        print('Trained NaiveBayesModel:')
        print(report)
    if logisticModelExists and vectorizorExists:
        logisticModel = LogisticModel.load()
    else:
        data = TextData.load('airline_sentiment_analysis.csv',[SENT, TEXT])
        logisticModel, report = LogisticModel.train(data)
        logisticModel.save()
        print('Trained LogisticModel:')
        print(report)
    return (logisticModel, bayesModel)

# only train models if file not called as module
if __name__ == "__main__":
    getModels()
