import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#  read  the dataset
resume_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\UpdatedResumeDataSet.csv")
print(resume_data.head())
print(resume_data.keys())
print(resume_data['Resume'][0])
print(resume_data['Category'].unique())
# show the Categorical data in suitable tabular form

# plt.figure(figsize=(15,5))
# sns.countplot(resume_data['Category'])
# plt.xticks(rotation=90)
# plt.show()

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText
# Demo  of cleanResume function
print(cleanResume(" my @resume is a screening is https:\\\loacalhist  are a languagww #hub "))
# Apply the function on the resume column

resume_data['Resume']=resume_data['Resume'].apply(lambda x : cleanResume(x))
print(resume_data['Resume'][0])

# Make the categorical data numeric
le=LabelEncoder()
le.fit(resume_data['Category'])
resume_data['Category']=le.transform(resume_data['Category'])
print(resume_data['Category'].unique())

# Clean and processed data
print(resume_data.head())
# spllit the dataset in train test division
X=resume_data['Resume'].values
Y=resume_data['Category'].values
# Corrected train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print(X_train.shape)
print(Y_train.shape)

 # Vectorization
tfd=TfidfVectorizer()
X_train_vc=tfd.fit_transform(X_train)
X_train_np=X_train_vc.toarray()
X_test_vc=tfd.transform(X_test)
model= KNeighborsClassifier()
model.fit(X_train_np,Y_train)

# accuracy on training data
X_train_prediction=model.predict(X_train_np)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)

myresume='''Education Details 
January 2016 B.Sc. Information Technology Mumbai, Maharashtra University of Mumbai
January 2012 HSC  Allahabad, Uttar Pradesh Allahabad university
January 2010 SSC dot Net Allahabad, Uttar Pradesh Allahabad university
Web designer and Developer Trainer 

Web designer and Developer
Skill Details 
Web design- Exprience - 12 months
Php- Exprience - 12 monthsCompany Details 
company - NetTech India
description - Working. ( salary - 12k)
PERSONAL INTEREST

Listening to Music, Surfing net, Watching Movie, Playing Cricket.
company - EPI Center Academy
description - Working.  ( Salary Contract based)
company - Aptech Charni Road
description - Salary Contract based) '''
cleaned_resume = cleanResume(myresume)

# Transform the cleaned resume using the trained TfidfVectorizer
input_features = tfd.transform([cleaned_resume])

# Make the prediction using the loaded classifier
prediction_id = model.predict(input_features)[0]

# Map category ID to category name
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}
category_name = category_mapping.get(prediction_id, "Unknown")

print("Predicted Category:", category_name)
print(prediction_id)

import pickle
pickle.dump(tfd,open('tfdfv.pkl','wb'))
pickle.dump(model, open('model.pkl', 'wb'))