from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from matplotlib import pyplot as plt
import numpy as np
import datetime

class CHFitter(object):
    '''
    
    Description
    -----------------------------------------
    
    This class provides methods to estimate the CH function as presented in the following paper:
    
    Title: "Forecasting life expectancy: Evidence from a new survival function."
    By: Wong, Chi Heem, and Albert K. Tsui. 
    Journal: Insurance: Mathematics and Economics 65 (2015): 208-226.
    
    Please cite the paper if you used this code in your work!
    
    @version: 2021-Jan-08
    @author: Chi Heem, Wong
    
    ...
    
    Methods
    -----------------------------------------
    - set_params(alpha1, beta1, gamma1, alpha2, beta2, gamma2)
        Manually set parameters to the CH function
    
    - fit(df, sx_col, age_col, x0=None, plot_fit=False)
         Estimate the parameters of the CH function from data
    
    - predict(x)
        Predicts the S(x) given by the CH function
        
    - print_summary
        Prints a summary of the estimation (if model was fitted) and 
        the parameters of CH function
    
    Example usage
    --------
    .. code:: python
        from CHFitter import CHFitter
        chf = CHFitter()
        chf.fit(df, 'sx', 'age', plot_fit=True)
        chf.print_summary()
    
    
    '''
    def __init__(self, ):
        '''
        Constructor
        '''
        # Function parameters
        self.params = None
        
        # Diagnostics info
        self._params_estimated = False
        self._params_set = False
        self._time_fit_was_called = None
        self._time_taken = None
        self._Sx_col = None
        self._age_col = None
        self._n = None
        
        self._r_squared = None # R-squred
        self._mse = None # mean squared error
        self._rss = None # sum of squared residuals
        self._mape = None # mean absolute percentage error

    def fit(self, df, sx_col, age_col, x0=None, plot_fit=False):
        '''
        Estimate the parameters of the CH function from data
        
        Parameters
        -------------------
        df: DataFrame
            a Pandas DataFrame with necessary columns `sx_col` and
            `age_col` (see below). `sx_col` refers to the survival probabilities (y-axis).
            `age_col` refers to the age axis (x-axis).
        
        sx_col: string
            the name of the column in DataFrame that contains the survival probabilities.
            
        age_col: string
            the name of the column in DataFrame that contains the ages.
            
        x0 (optional): list
            the initial conditions of minimization. must be of the format "[alpha1, beta1, gamma1, alpha2, beta2, gamma2]"
            
        plot_fit (optional): Bool
            option to plot the estimated CH function against data points (default = False)
            
            
        Returns
        -------
        params: Numpy array
            estimated parameters of the CH function
            
        '''
        if x0 is None:
            self.params = [0.4, 65, 4, 2, 80, 4]
        else:
            self.params = x0
        
        self._time_fit_was_called = datetime.datetime.utcnow()
        self._Sx_col = sx_col
        self._age_col = age_col
        
        sx = df[sx_col]; ages=df[age_col]
        self._n = len(sx)
        x0 = self.params
        
        # Set constraints
        LB = np.ones(7)*1e-6
        UB = np.ones(7)*100
        X = np.eye(7,6)
        X[-1,:] = [-1, 0, 0, 1, 0, 0]
        linear_constraint = LinearConstraint(X, LB, UB)
        
        # Minimize
        def loss(params, ages, sx):
            sx_het = params[0]*np.exp(-np.exp(np.power(ages/params[1],params[2]))) + params[3]*np.exp(-np.cosh(np.power(ages/params[4],params[5])))
            diff = sx_het-sx
            return np.matmul(np.transpose(diff), diff)/len(sx)
        
        res = minimize(loss,
                     x0,
                     method='trust-constr',
                     args=(ages, sx),
                     #jac=self.jac,
                     constraints=linear_constraint,
                     )
        self._params_estimated = True
        self._time_taken = datetime.datetime.utcnow() - self._time_fit_was_called
        
        # Store estimated parameters
        self.params= res.x
        
        # Obtain and store statistics
        sx_het = self.predict(ages)
        err = sx - sx_het
        self._rss = np.sum(np.power(err, 2))
        ape = np.abs(err/sx)
        n = np.sum(np.where(ape==np.inf, 0, 1))
        ape[ape == np.inf] = 0 # Handle cases where sx=0
        self._mape = np.nansum(np.abs(ape))/n
        self._r_squared = 1- self._rss/np.sum(np.power(sx-np.mean(sx),2))
        self._mse = res.fun
        
        if plot_fit==True:
            plt.scatter(ages, sx)
            plt.plot(ages, sx_het)
            plt.legend(['Data points', 'Fitted CH function'])
            plt.show()
        return self.params
    
    def predict(self, x):
        '''
        Predicts the S(x) given by the CH function given a Numpy array, x
        
        Parameters
        -------------------
        x: Numpy array
            points that require predictions
            
        Returns
        -------
        S_x: Numpy array
            values from the CH function
        '''
        if (self._params_estimated or self._params_estimated):
            return self.params[0]*np.exp(-np.exp(np.power(x/self.params[1],self.params[2]))) \
                    + self.params[3]*np.exp(-np.cosh(np.power(x/self.params[4],self.params[5])))
        else:
            raise Exception('Parameters for the model are not set. Please set parameters or fit model to data.')
    
    def set_params(self, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
        '''
        Manually set parameters to the CH function. Useful if you know the
        parameters and want to do predictions.
        
        Parameters
        -------------------
        alpha1: float
            alpha1 in the CH function
        beta1: float
            beta1 in the CH function
        gamma1: float
            gamma1 in the CH function
        alpha2: float
            alpha2 in the CH function
        beta2: float
            beta2 in the CH function
        gamma2: float
            gamma2 in the CH function
        '''
        self.params = [alpha1, beta1, gamma1, alpha2, beta2, gamma2]
        self._params_estimated = True
        
    def print_summary(self,):
        '''
        Prints a summary of the estimation (if model was fitted) and 
        the parameters of CH function
        '''
        if self._params_estimated:
            print(''.join([''.join('-') for _ in range(50)]))
            print('Fitted on %s in %.3f seconds' %((self._time_fit_was_called.strftime("%Y-%m-%d %H:%M:%S") + " UTC"), self._time_taken.total_seconds()))
            print('Age variable : "%s"'%self._age_col)
            print('Sx variable  : "%s"' %self._Sx_col)
            print('N            : %d' %self._n)
            print(''.join([''.join('-') for _ in range(50)]))
            print('R-squared                             : %f' %self._r_squared)
            print('Mean squared error (MSE)              : %f' %self._mse)
            print('Residual sum of squares (RSS)         : %f' %self._rss)
            print('Mean absolute percentage error (MAPE) : %f' %self._mape)
            print(''.join([''.join('-') for _ in range(50)]))
        print('Parameters')
        print('alpha1 : %.5f' %self.params[0])
        print('beta1  : %.5f' %self.params[1])
        print('gamma1 : %.5f' %self.params[2])
        print('alpha1 : %.5f' %self.params[3])
        print('beta2  : %.5f' %self.params[4])
        print('gamma2 : %.5f' %self.params[5])
        print(''.join([''.join('-') for _ in range(50)]))
            
            