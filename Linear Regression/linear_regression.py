import numpy as np
from sklearn.linear_model import LinearRegression

# Step 1: Generate a synthetic dataset
np.random.seed(0)

# House sizes betwwen 1000 and 3500 sq ft
x = 2500 * np.random.rand(100,1) + 1000

# Prices as a linear function of size (with some random noise)
y = 200 * x + 50000 + (np.random.randn(100,1) * 50000)

# Step 2: Train the model on the dataset
model = LinearRegression()
model.fit(x,y)

# Step 3: Allow user input for house size and predict the price
def predict_house_price():
    try:
        house_size = float(input("Enter the size of the house in square feet: "))
        house_size = np.array([[house_size]])  # Reshape the input to match the model's expected input shape
        predicted_price = model.predict(house_size)
        print(f"The predicted price for a house of size {house_size[0][0]} sq ft is ${predicted_price[0][0]:,.2f}")
    except ValueError:
        print("Invalid input. Please enter a numerical value.")

# Call the function to input house size and predict the price
predict_house_price()