import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Salarize Yourself - Predict Your Own Salary", page_icon=":moneybag:")

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_values = X.values
y_values = y.values

st.title("Salarize Yourself")
st.header("Predict Your Own Salary")

col1, col2, col3 =  st.columns(3)

with col1:
    st.markdown("### Salary Dataset")
    st.dataframe(dataset)

with col2:
    st.markdown("### Training Dataset")
    st.dataframe(X)

with col3:
    st.markdown("### Test Dataset")
    st.dataframe(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

X_test_df = pd.DataFrame(X_test)

def training(X_values, y_values):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

reg = training(X_values, y_values)

def predicted_values(X_test):
    st.write("# Predicted Salaries")
    y_pred = reg.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    return st.dataframe(y_pred_df)

predicted_values(X_test)

def train_results_visualize(X_train, y_train):
    st.write("# Visualizing the Training Set Results")

    # Create the scatter plot
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, reg.predict(X_train), color='blue', label='Regression line')
    plt.title('Salary vs Experience (Training set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()  # Show legend

    # Display the plot in Streamlit
    st.pyplot(plt)

train_results_visualize(X_train, y_train)

def test_results_visualize(X_test, y_test):
    st.write("# Visualizing the Test Set Results")

    # Create the scatter plot
    plt.scatter(X_test, y_test, color='red', label='Test data')
    plt.plot(X_test, reg.predict(X_test), color='blue', label='Regression line')
    plt.title('Salary vs Experience (Test set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()  # Show legend

    # Display the plot in Streamlit
    st.pyplot(plt)

test_results_visualize(X_test, y_test)

st.sidebar.title("Years of Experience")
testing = st.sidebar.slider(label = "What's your years of experience?", min_value=0, max_value=100)

def predict(testing):
    st.sidebar.title("Your Predicted Salary")
    y_pred = training(X_values, y_values).predict([[testing]])
    return st.sidebar.markdown(f"#### _Rs. {round(y_pred[0], 2)} per month_")

predict(testing)

st.sidebar.write("---")

def coefficients():
    st.sidebar.markdown("## Coefficient")
    st.sidebar.markdown(reg.coef_[0])

coefficients()

def intercept():
    st.sidebar.markdown("## Intercept")
    st.sidebar.markdown(reg.intercept_)

intercept()

st.write("---")

st.markdown("##### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Created by [Gandharv Kulkarni](https://share.streamlit.io/user/gandharvk422)")

st.markdown("&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[![GitHub](https://img.shields.io/badge/GitHub-100000?style=the-badge&logo=github&logoColor=white&logoBackground=white)](https://github.com/gandharvk422) &emsp; [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/gandharvk422) &emsp; [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/gandharvk422)")
