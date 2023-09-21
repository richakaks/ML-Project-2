import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#import mlflow

from sklearn.model_selection import KFold

#################################################################################################################################################

class LinearRegression(object):
    
    #Adding Crossvalidation for each experiment
    kfold = KFold(n_splits=3)
            
    def __init__(self, regularization, lr=0.001, method='batch', grad_init='xavier', polynomial=True, degree=3,
                 use_momentum=True, momentum=0.9, num_epochs=500, batch_size=50, cv=kfold):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self.use_momentum = use_momentum        # will store:'true' or 'false' depending on the exp
        self.momentum = momentum                # if defined will use the value 0.9
        self.grad_init = grad_init             
        self.polynomial = polynomial            # will store true or false depending on the exp
        self.degree = degree                    # if defined will use the ploynomial of degree 2
        self.prev_grad = 0

    # function to calculate mse for one fold in cv
    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    # function to calculate r2 for one fold in cv
    def r2(self, ytrue, ypred):
        return 1 - ((ytrue - ypred) ** 2).sum() / ((ytrue - ytrue.mean()) ** 2).sum()

    # function to calculate average mse for all kfold_scores
    def avgMse(self):
        return np.sum(np.array(self.kfold_scores))/len(self.kfold_scores)
    
    # function to calculate average r2 for all kfold_scores
    def avgr2(self):
        return np.sum(np.array(self.kfold_r2))/len(self.kfold_r2)
    
    # function where we will do training!!
    def fit(self, X_train, y_train):
        
        # Store column names for later use
        #self.columns = X_train.columns

        #Check if our experiment is polynomial or normal linear regression
        # if self.polynomial is True will use polynomial, By default self.polynomial is True
        if self.polynomial == True:
            X_train = self._transform_features(X_train)
            print("Using Polynomial")
        else:
            print("Using Linear")
            #X_train = X_train.to_numpy()

        #Convert the array to numpy array    
        y_train = y_train.to_numpy()

        #create a list of kfold scores
        self.kfold_scores = list()
        self.kfold_r2 = list()
        
        #reset val loss ----> This will be used in our first epoch after first epoch we will replace this value 
        # with the loss value in validation set in first epoch for 2nd epoch and so on..
        self.val_loss_old = np.infty

        #kfold.split in the sklearn.....
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            #initialize weights using Xavier method
            if self.grad_init == 'xavier':
                #calculate the range for the weights with number of samples
                lower, upper = -(1 / np.sqrt(X_cross_train.shape[0])), 1 / np.sqrt(X_cross_train.shape[0])
                #randomize weights then scale them using lower and upper bounds
                self.theta = np.random.rand(X_cross_train.shape[1])
                self.theta = lower + self.theta * (upper - lower)

            #initialize weights with zero
            elif self.grad_init == 'zero':
                self.theta = np.zeros(X_cross_train.shape[1])
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):                    
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323

                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx].reshape(1, )  #(1, )
                            train_loss = self._train(X_method_train, y_method_train)                   
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]    
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)                   ###
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)                        ###

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    val_r2_new = self.r2(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    mlflow.log_metric(key="val_r2", value=val_r2_new, step=epoch)
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):              
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                self.kfold_r2.append(val_r2_new)
                print(f"Fold {fold} mse: {val_loss_new}")
                print(f"Fold {fold} r2: {val_r2_new}")

    # def _transform_features(self, X):
    #     X = X.to_numpy()
    #     for i in range(2, self.degree+1):
    #         X = np.concatenate((X, X[:, 1:] ** i), axis=1)
    #     return X
    
    def _transform_features(self, X):
        # Transform input features to include polynomial degree --> highest degree is taken
        X_poly = np.column_stack([X ** (self.degree)])        
        return X_poly

    def _train(self, X, y): #X is (m, n) and y is (m, )
        yhat = self.predict(X) #===>(m, )
        m    = X.shape[0] #number of samples
        if self.regularization:
            grad = (1/m) * X.T @ (yhat - y) + self.regularization.derivation(self.theta) #===>(n, m) @ (m, ) = (n, )
        else:
            grad = (1/m) * X.T @ (yhat - y)
        if self.use_momentum == True:
            self.step = self.lr * grad
            self.theta = self.theta - self.step  +  self.momentum * self.prev_grad
            self.prev_grad = self.step
        else:
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)
 
    def predict(self, X):
        if self.polynomial == True:
            X = self._transform_features(X)
        return X @ self.theta   #===>(m, n) @ (n, )

    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]


class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)

class Lasso(LinearRegression):
    def __init__(self, l, lr, method, grad_init, polynomial, degree, use_momentum, momentum):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method, grad_init, polynomial, degree, use_momentum, momentum)
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

class Ridge(LinearRegression):
    def __init__(self, l, lr, method, grad_init, polynomial, degree, use_momentum, momentum):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, grad_init, polynomial, degree, use_momentum, momentum)
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

class ElasticNet(LinearRegression):
    def __init__(self, l, lr, method, grad_init, polynomial, degree, use_momentum, momentum, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method, grad_init, polynomial, degree, use_momentum, momentum)
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

class Normal(LinearRegression):  
    def __init__(self, l, lr, method, grad_init, polynomial, degree, use_momentum, momentum):
        self.regularization = None  # No regularization
        super().__init__(self.regularization, lr, method, grad_init, polynomial, degree, use_momentum, momentum)
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

#################################################################################################################################################

# Load the first model
with open('car_price_prediction.model', 'rb') as model_file:
    model1 = pickle.load(model_file)

# with open('car-a2-prediction.model', 'rb') as model_file2:
#     model2 = pickle.load(model_file2)

model2 = pickle.load(open('car-a2-prediction.pkl', 'rb'))

X_train = pd.read_csv("X_train.csv")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

scaler2 = pickle.load(open('a2r-scalar.model', 'rb'))

def predict_output1(year, transmission, engine, max_power):
    input_data = [[float(year), 0 if transmission == 'Automatic' else 1, float(engine), float(max_power)]]
    input_data = scaler.transform(input_data)
    output = model1.predict(input_data)
    output = np.exp(output)
    return output
def predict_output2(model, index, mileage, engine, max_power):
    input_data = [[float(mileage), float(engine), float(max_power) ]]
    input_data = scaler2.transform(input_data)
    input_data = np.insert(input_data, 0, index, axis=1)
    output = model.predict(input_data)
    output = np.exp(output)
    return output

def main():
    st.title("Car Price Predictor")
    page = st.sidebar.selectbox("Select a Page", ["Home", "Old Model", "New Model"])

    if page == "Home":
        st.header("Welcome to Car Model Predictor")
        st.write("Please select a model from the sidebar to make predictions.")

    elif page == "Old Model":
        st.header("Old Model")
        year = st.text_input("Car model year", key="year", value='2014')
        transmission = st.selectbox('Transmission type', ['Manual', 'Automatic'])
        engine = st.slider('Engine power', min_value=700.0, max_value=2000.0, value=1000.0)
        max_power = st.slider('Maximum power', min_value=50.0, max_value=200.0, value=125.0)

        # Create a button with a label and a callback function
        button_clicked = st.button("Predict")

        # Check if the button has been clicked
        if button_clicked:
            # Use default values if the user didn't input anything
            if not year:
                year = '2014'
            if not transmission:
                transmission = 'Manual'
            if not engine:
                engine = 1463.565853 # mean value of df.engine
            if not max_power:
                max_power = 92.057087 # mean value of df.max_power

            # Call the prediction function
            output = predict_output1(year, transmission, engine, max_power)
            output = round(float(output), 2)
            # Display the prediction result
            st.markdown(f"<h1 style='text-align: center;'>{'Predicted Car Selling Price'}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center;'>{output}</h1>", unsafe_allow_html=True)

    elif page == "New Model":
        st.header("New Model")
        # year = st.text_input("Car model year", key="year1", value='2014')
        # transmission = st.selectbox('Transmission type', ['Manual', 'Automatic'])
        engine = st.slider('Engine power', min_value=700.0, max_value=3000.0, value=1463.565853)
        max_power = st.slider('Maximum power', min_value=30.0, max_value=200.0, value=92.057087)
        mileage = st.slider('Mileage', min_value=8.0, max_value=30.0, value=15.0)

        # Create a button with a label and a callback function
        button_clicked = st.button("Predict")

        # Check if the button has been clicked
        if button_clicked:
            # Use default values if the user didn't input anything
            # if not year:
            #     year = '2014'
            # if not transmission:
            #     transmission = 'Manual'
            if not engine:
                engine = 1463.565853  # mean value of df.engine
            if not max_power:
                max_power = 92.057087  # mean value of df.max_power
            if not mileage:
                mileage = 15.0

            # Call the prediction function for Model 2
            output = predict_output2(model2, 1,  mileage, engine, max_power)
            output = round(float(output), 2)
            # Display the prediction result
            st.markdown(f"<h1 style='text-align: center;'>{'Predicted Car Selling Price'}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center;'>{output}</h1>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
