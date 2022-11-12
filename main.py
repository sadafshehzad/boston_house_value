import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score




header=st.beta_container()
dataset=st.beta_container()
features=st.beta_container()
model_training=st.beta_container()





@st.cache
def get_data(filname):
    boston_data=pd.read_csv(filname)

    return boston_data

with header:
    st.title(" Excited to start a new Project in Streamlit ")
    st.write('Welcome to my awesome Data Science project in  Streamlit App:')
    st.text('In This model I would like to look into the features that could affect Husing prices in Boston:')

with dataset:
    st.header('Boston House Pricing ')
    st.text('I found this data set on Kaggle...')

    # calling function once to get data... because its cached
    boston_data=get_data('Boston.csv')

    st.write(boston_data.head())
    st.subheader('Age of House Distribution')
    #houseAge_data=pd.DataFrame(boston_data['AGE'].value_counts())
    fig, ax = plt.subplots()
    ax.hist(boston_data['AGE'], bins=20)

    st.pyplot(fig)
    







with features:
    st.header('The Features I Created')
    st.markdown('* ** First feature:**I created this feature because...')
    st.markdown('* ** Second feature:**I created this feature because...')






with model_training:
    st.header('Its Time to Train The Model:......')
    st.text('Here you get to choose te hyper parameters of the model and see how the performance change .....')
    sel_column,display_column=st.beta_columns(2)

    max_depth=sel_column.slider('What should be the max_depth of the Model?',min_value=10,max_value=100,value=20,step=10)

    n_estimators=sel_column.selectbox('How many tress should be there in.....?',options=[100,200,300,'No limit'],index=0)
    
    sel_column.text('Here is the list of Features in my data... You can choose and see the difference in the r2 results:')
    sel_column.write(boston_data.columns)
    
    
    input_features=sel_column.text_input('Which feature should be used?....','AGE')

    

    if n_estimators=='No limit':
        regr=RandomForestRegressor(max_depth=max_depth)
    else:

        regr=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

    X=boston_data[[input_features]]
    y=boston_data[['MEDV']]

    regr.fit(X,y)
    prediction=regr.predict(y)

    display_column.subheader('Mean Absolute Error of the model is :  ')
    display_column.write(mean_absolute_error(y,prediction))

    display_column.subheader('Mean Squared Error of the model is :  ')
    display_column.write(mean_squared_error(y,prediction))



    display_column.subheader('r2 score of the model is :  ')
    display_column.write(r2_score(y,prediction))




     
