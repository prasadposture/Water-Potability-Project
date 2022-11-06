#Importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt
#setting page configuration
st.set_page_config(page_title='Water Potability', page_icon=':droplet:')

#loading the preprocessed data
df=pd.read_csv('water_potability_preprocessed.csv')

st.write("""# Water Potability""")

st.write("___")

#seperating the dependent anad independant variables
x=df.drop('Potability',axis=1)
Y=df['Potability']

#fitting the model
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state=3)
clf=clf.fit(x, Y)

#creating sidebar
st.sidebar.header('Water Quality Metrics')

#Taking user inputs form the side
def user_input_features():
    PH = st.sidebar.slider('PH', df.PH.min(), df.PH.max(), df.PH.median()),
    Hardness = st.sidebar.slider('Hardness', df.Hardness.min(), df.Hardness.max(), df.Hardness.median()),
    Solids = st.sidebar.slider('Solids', df.Solids.min(), df.Solids.max(), df.Solids.median()),
    Chloramines = st.sidebar.slider('Chloramines', df.Chloramines.min(), df.Chloramines.max(), df.Chloramines.median()),
    Sulfate = st.sidebar.slider('Sulfate', df.Sulfate.min(), df.Sulfate.max(), df.Sulfate.median()),
    Conductivity = st.sidebar.slider('Conductivity', df.Conductivity.min(), df.Conductivity.max(), df.Conductivity.median()),
    Organic_carbon = st.sidebar.slider('Organic_carbon', df.Organic_carbon.min(), df.Organic_carbon.max(), df.Organic_carbon.median()),
    Trihalomethanes = st.sidebar.slider('Trihalomethanes', df.Trihalomethanes.min(), df.Trihalomethanes.max(), df.Trihalomethanes.median()),
    Turbidity = st.sidebar.slider('Turbidity', df.Turbidity.min(), df.Turbidity.max(), df.Turbidity.median()),
    data = {'PH': PH,
            'Hardness': Hardness,
            'Solids': Solids,
            'Chloramines': Chloramines,
            'Sulfate': Sulfate,
            'Conductivity': Conductivity,
            'Organic_carbon': Organic_carbon,
            'Trihalomethanes': Trihalomethanes,
            'Turbidity': Turbidity}
    features = pd.DataFrame(data, index=[0])
    return features

df1 = user_input_features()

st.write('### Water Quality Metrics :')
st.write(df1)

#Makinf predictions
prediction = clf.predict(df1)
prediction_proba = clf.predict_proba(df1)
st.write("___")

st.write('### Prediction :')
if prediction==0:
    st.write("##### Water is not potabale :warning:")
else:
    st.write("##### Water is potable :innocent:")
st.write("___")
st.write('### Prediction Probability :')
st.write(prediction_proba)
st.write("0 : Not Potable")
st.write("1 : Potable")

st.write('___')

#Data Visualization Inputs
st.write('### Data Visualization :')
with st.container():
    l,m,r=st.columns(3)
    with l:
        option1=st.selectbox('On X Axis',('PH','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'))
    with m:
        option2=st.selectbox('On Y Axis',('PH','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'))
    with r:
        option4=st.selectbox('Type of Plot', ('KDE Plot (X Axis)','Histplot (X Axis)','Scatterplot', 'Barplot (Y Axis)','Boxplot (Y Axis)' ))
#Plots
if option4=='Boxplot (Y Axis)':
    sns.boxplot(data=df, y=option2, x='Potability')
    plt.title('Boxplot of '+option2)
    st.pyplot(plt.gcf())
elif option4=='Scatterplot':
    sns.scatterplot(data=df, x=option1, y=option2, hue='Potability',alpha=0.5)
    plt.title(option1+' vs '+option2)
    st.pyplot(plt.gcf())
elif option4=='KDE Plot (X Axis)':
    sns.kdeplot(data=df, x=option1, hue='Potability')
    plt.title('Distribution of '+option1)
    st.pyplot(plt.gcf())
elif option4=='Histplot (X Axis)':
    sns.histplot(data=df, x=option1, hue='Potability')
    plt.title('Histplot of '+option1)
    st.pyplot(plt.gcf())
elif option4=='Barplot (Y Axis)':
    sns.barplot(data=df, y=option2, x='Potability')
    plt.title('Barplot of '+option2)
    st.pyplot(plt.gcf())
else:
    pass
st.write('___')
st.write('### Statistical Information :')
if st.button('Potable Water'):
    st.write(df[df.Potability==1].describe())
else:
    pass
if st.button('Not Potable Water'):
    st.write(df[df.Potability==0].describe())
else:
    pass
st.write("___")
st.write('My Profiles :')
with st.container():
    left, middle, right = st.columns(3)
    with left:
        st.write('[Kaggle](https://www.kaggle.com/prasadposture121)')
    with middle:
        st.write('[GitHub](https://github.com/prasadposture)')
    with right:
        st.write('[LinkedIn](https://www.linkedin.com/in/prasad-posture-6a3a77215/)')
