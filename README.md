# CH-function
This code provides a class to estimate the CH function as presented in the following paper:

Wong, Chi Heem, and Albert K. Tsui. "Forecasting life expectancy: Evidence from a new survival function." Insurance: Mathematics and Economics 65 (2015): 208-226.

Please cite the paper if you used this code in your work!

# Example usage
sx = [1, 0.9, 0.889, 0.858, 0.813, 0.739, 0.617, 0.421, 0.181, 0.0278, 0.011, 0.000001]
ages = [x*10 for x in range(11)]
df = pd.DataFrame(list(zip(sx, ages)), columns =['Sx', 'Age'])

chf = CHFitter()
chf.fit(df, 'Sx', 'Age', plot_fit=True)
chf.print_summary()

