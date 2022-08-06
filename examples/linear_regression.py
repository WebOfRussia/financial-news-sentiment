from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("../data/data.tsv", sep='\t')

train, test = train_test_split(df, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
vectorizer.fit(df['title'])

model = LinearRegression()
model.fit(vectorizer.transform(train['title']), train['score'])

print(mean_squared_error(test['score'], model.predict(vectorizer.transform(test['title']))))

print(model.predict(vectorizer.transform(["Яндекс увеличил прибыль на $1 млрд"])))