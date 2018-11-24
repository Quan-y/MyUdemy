import os
import fnmatch
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class file_classifier:

    def __init__(self, path):

        self.path = path


    def get_files(self):

        the_files = pd.DataFrame()

        for root, dir, files in os.walk(self.path):
            for items in fnmatch.filter(files, "*"):
                path = root
                label = root.split('\\')[-1]
                if 'Workshop' not in label:
                    the_files = the_files.append({"Label": label, "File Name": items, "File Path": path+"\\"+items}, ignore_index=True)

        return the_files


    def clean_text(self, df, stopwords):

        result = pd.DataFrame()

        for label, file_name, path in zip(df['Label'], df['File Name'], df['File Path']):
            f = open(path, "r", encoding='utf-8')

            text_raw = f.readlines()
            text_split = " ".join(text_raw).split()
            text_low = [word.lower() for word in text_split]
            text_re = [re.sub(r"[^a-zA-Z]+", ' ', token) for token in text_low]
            while ' ' in text_re:
                text_re.remove(' ')
            text = " ".join([word for word in text_re if word not in stopwords])

            result = result.append({'Body': text, 'Label': label}, ignore_index=True)

            f.close()

        return result


    def train_model(self, model, df):

        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        data = pd.DataFrame(tfidf.fit_transform(df.Body).toarray())

        label_enc = LabelEncoder()
        label_enc.fit(df.Label)
        data["Label"] = label_enc.transform(df.Label)

        model_trained = model.fit(data.iloc[:, :-1], data.iloc[:, -1])

        return model_trained


if __name__ == "__main__":

    the_path = "C:\\Users\\sheng\\OneDrive\\First Semester in CU\\Programming Language （Python）\\Workshop"
    stopwords = set(stopwords.words('english'))

    fc = file_classifier(the_path)
    file_names = fc.get_files()

    text_label_df = fc.clean_text(file_names, stopwords)
    RfClf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

    trained = fc.train_model(RfClf_model, text_label_df)

    print(1)










"""
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,3))
    data = pd.DataFrame(tfidf.fit_transform(text_label_df.Body).toarray())
    data.columns = tfidf.get_feature_names()

    label_enc = LabelEncoder()
    label_enc.fit(text_label_df.Label)
    data["Label"]=label_enc.transform(text_label_df.Label)

    RfClf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    RfClf_trained = RfClf_model.fit(data.iloc[:, :-1], data.iloc[:, -1])
"""
    # scores = cross_val_score(RfClf_model, data.iloc[:, :-1], data.iloc[:, -1], cv=10)

    # print(scores)









