# TODO

## Check out how and why do we need to sample 100 times in diffusion model and get the median

## Read and understand evaluation

# 📘 Mean Absolute Error (MAE) in Machine Learning

## 🔍 Definition
**Mean Absolute Error (MAE)** is a common metric used in machine learning to measure the accuracy of a model's predictions for regression tasks.

MAE is the average of the absolute differences between predicted values and actual values.

## 📏 Formula

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

Where:
- $n$ = number of data points  
- $y_i$ = actual (true) value  
- $\hat{y}_i$ = predicted value  
- $| \cdot |$ = absolute value

## 💡 Intuition
It tells you how wrong, on average, your predictions are.  
For example, an MAE of **5** means your predictions are, on average, **5 units** away from the actual values.

## ✅ Pros
- Easy to understand and interpret
- Less sensitive to outliers than other metrics like **Mean Squared Error (MSE)**

## ⚠️ Cons
- Doesn’t penalize large errors as harshly as MSE, which may or may not be desirable depending on the problem


# 📘 Root Mean Squared Error (RMSE) in Machine Learning

## 🔍 Definition

**Root Mean Squared Error (RMSE)** is a metric that measures the **average magnitude of the error** between predicted values and actual values.  

It penalizes larger errors more heavily by squaring the differences before averaging, and then takes the square root of that average.

## 📏 Formula

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 }
$$

Where:
- $n$ = number of data points  
- $y_i$ = actual (true) value  
- $\hat{y}_i$ = predicted value  

## 💡 Intuition

- RMSE tells you how far off, on average, your predictions are from the actual values — just like MAE.
- But it **squares the errors**, so larger mistakes have a **bigger impact**.
- It's especially useful when **large errors are more serious** than small ones.

## ✅ Pros

- Emphasizes **larger errors**, which is helpful if big mistakes are costly.
- **Differentiable**, making it useful during model training and optimization.

## ⚠️ Cons

- **Sensitive to outliers** — even one large error can skew the result.
- **Harder to interpret** than MAE, especially since it uses squared units.

### CRPS
### CRPS_sum?


## Add different evaluation on the code

## Add scenario map to evaluation

## Doing data augmentation 