install.packages("readr", "dplyr", "ggplot", "lubridate")
install.packages("caret", "randomForest", "xgboost")
install.packages("keras", "stats")
install.packages("stringr")
install.packages("tidyverse")
install.packages("gridExtra")
library(gridExtra)
library(tidyverse)
library(readr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(caret)
library(randomForest)
library(xgboost)
library(keras)
library(stats)
library(stringr)

train_data <- read.csv("train.csv")
head(train_data)

str(train_data)
summary(train_data$sales)

# Plot sales
ggplot(train_data, aes(x = date, y = sales)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Sales Over Time", x = "Date", y = "Sales")
monthlyORyears_sales <- function(data, time = c("monthly", "years")) {
  
  # Check if the Date column exists and is named correctly
  if (!"date" %in% colnames(data)) {
    stop("The 'date' column does not exist in the dataset.")
  }
  
  # Convert Date column to character
  data <- data %>%
    mutate(date = as.character(pull(data, date)))  # Explicitly pull the Date column
  
  if (time == "monthly") {
    # Drop the day indicator, keep year and month
    data <- data %>%
      mutate(date = str_sub(date, 1, 7))  # Keep YYYY-MM
  } else if (time == "years") {
    # Keep only the year
    data <- data %>%
      mutate(date = str_sub(date, 1, 4))  # Keep YYYY
  }
  
  # Sum sales by the new date grouping
  result <- data %>%
    group_by(date) %>%
    summarise(sales = sum(sales, na.rm = TRUE)) %>%
    ungroup()
  
  # Convert the date column back to Date class
  result$date <- if (time == "monthly") {
    as.Date(paste0(result$date, "-01"))  # Convert to YYYY-MM-DD
  } else {
    as.Date(paste0(result$date, "-01-01"))  # Convert to YYYY-01-01
  }
  
  return(result)
}

# Example use case
m_df <- monthlyORyears_sales(train_data, "monthly")

# Save to CSV
write.csv(m_df, './monthly_data.csv', row.names = FALSE)

# View first 10 rows
head(m_df, 10)
y_df <- monthlyORyears_sales(train_data, "years")

# Create yearly sales bar plot
years_plot <- ggplot(y_df, aes(x = date, y = sales)) +
  geom_bar(stat = "identity", fill = "mediumblue") +
  labs(title = "Distribution of Sales Per Year", x = "Years", y = "Sales") +
  theme_minimal()

# Assuming you have already calculated 'm_df' for monthly sales
# Create monthly sales line plot
months_plot <- ggplot(m_df, aes(x = date, y = sales)) +
  geom_line(color = "darkorange") +
  geom_point(color = "darkorange") +
  labs(title = "Distribution of Sales Per Month", x = "Months", y = "Sales") +
  theme_minimal()

# Arrange both plots in a 1x2 grid layout
grid.arrange(years_plot, months_plot, ncol = 2)

sales_time <- function(data) {
  # Convert the date column to Date format
  data$date <- as.Date(data$date)
  
  # Calculate the number of days and years between the max and min date
  n_of_days <- max(data$date) - min(data$date)
  n_of_years <- as.integer(n_of_days / 365)
  
  # Print the results
  cat("Days:", as.numeric(n_of_days), "\n")
  cat("Years:", n_of_years, "\n")
  cat("Months:", 12 * n_of_years, "\n")
}

# Example use
sales_time(train_data)
sales_per_store <- function(data) {
  # Group by 'store' and calculate the sum of sales per store
  sales_by_store <- data %>%
    group_by(store) %>%
    summarise(sales = sum(sales, na.rm = TRUE)) %>%
    ungroup()
  
  # Create the bar plot using ggplot2
  ggplot(sales_by_store, aes(x = as.factor(store), y = sales)) +
    geom_bar(stat = "identity", fill = "darkred") +
    labs(x = "Store Id", y = "Sum of Sales", title = "Total Sales Per Store") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability
  
  return(sales_by_store)
}

# Example use
sales_per_store(train_data)

average_m_sales <- mean(train_data$sales, na.rm = TRUE)
cat("Overall Average Monthly Sales: $", average_m_sales, "\n")

average_12months <- function() {
  average_m_sales_1y <- mean(tail(m_df$sales, 12), na.rm = TRUE)
  cat("Last 12 months average monthly sales: $", average_m_sales_1y, "\n")
}

# Call the function
average_12months()

time_plot <- function(data, x_col, y_col, title) {
  
  # Create the base plot for total sales
  p <- ggplot(data, aes_string(x = x_col, y = y_col)) +
    geom_line(color = 'darkblue', size = 1) +
    labs(x = "Years", y = "Sales", title = title) +
    theme_minimal()
  
  # Calculate the yearly mean of sales
  s_mean <- data %>%
    mutate(year = year(!!sym(x_col))) %>%
    group_by(year) %>%
    summarise(mean_sales = mean(!!sym(y_col), na.rm = TRUE)) %>%
    ungroup()
  
  # Plot the mean sales line (adjusting the date to reflect mid-year for visibility)
  s_mean <- s_mean %>%
    mutate(mid_year = as.Date(paste0(year, "-07-01")))  # Mid-year
  
  # Add the mean sales line to the plot
  p <- p + geom_line(data = s_mean, aes(x = mid_year, y = mean_sales), color = "red", linewidth = 1) +
    labs(color = "Legend")  # Optional legend adjustment
  
  print(p)
}

# Call the function to plot
time_plot(m_df, "date", "sales", "Monthly Sales Before Diff Transformation")

get_diff <- function(data) {
  # Calculate the difference in sales month over month
  data <- data %>%
    mutate(sales_diff = sales - lag(sales)) %>%
    na.omit()  # Drop rows with NA values generated by lag
  
  # Save the result to a CSV file
  write.csv(data, './stationary_df.csv', row.names = FALSE)
  
  return(data)
}

# Call the function
stationary_df <- get_diff(m_df)

# Plot the transformed sales difference data
time_plot(stationary_df, 'date', 'sales_diff', 'Monthly Sales After Diff Transformation')

install.packages("tidyverse", "gridExtra")
install.packages("forecast")
library(tidyverse)
library(forecast)
library(gridExtra)

build_arima_data <- function(data) {
  # Generates a csv-file with a datetime index and a dependent sales column for ARIMA modeling.
  
  da_data <- data %>%
    select(-sales) %>%
    arrange(date) %>%
    na.omit()
  
  write_csv(da_data, './arima_df.csv')
  
  return(da_data)
}

datatime_df <- build_arima_data(stationary_df)
print(datatime_df)  # ARIMA Dataframe

plots_lag <- function(data, lags = NULL) {
  # Convert dataframe to datetime index and plot ACF and PACF
  dt_data <- data %>%
    select(-sales) %>%
    arrange(date) %>%
    na.omit()
  
  # Create time series object
  ts_data <- ts(dt_data$sales_diff, frequency = 12)
  
  # Main plot
  p1 <- ggplot(dt_data, aes(x = date, y = sales_diff)) +
    geom_line(color = 'orange') +
    theme_minimal() +
    labs(title = "Time Series Plot", x = "Date", y = "Sales Difference")
  
  # ACF plot
  acf_data <- acf(ts_data, lag.max = lags, plot = FALSE)
  p2 <- ggplot(data.frame(lag = acf_data$lag, acf = acf_data$acf), aes(x = lag, y = acf)) +
    geom_bar(stat = "identity", fill = "mediumblue") +
    theme_minimal() +
    labs(title = "ACF Plot", x = "Lag", y = "ACF")
  
  # PACF plot
  pacf_data <- pacf(ts_data, lag.max = lags, plot = FALSE)
  p3 <- ggplot(data.frame(lag = pacf_data$lag, pacf = pacf_data$acf), aes(x = lag, y = pacf)) +
    geom_bar(stat = "identity", fill = "mediumblue") +
    theme_minimal() +
    labs(title = "PACF Plot", x = "Lag", y = "PACF")
  
  # Combine plots
  grid.arrange(p1, p2, p3, ncol = 2, nrow = 2)
}

# Assuming stationary_df is already defined
plots_lag(stationary_df, lags = 24)


#Regressive Modelling

built_supervised <- function(data) {
  supervised_df <- data
  
  # Create column for each lag
  for (i in 1:12) {
    col_name <- paste0('lag_', i)
    supervised_df <- supervised_df %>%
      mutate(!!col_name := lag(sales_diff, i))
  }
  
  # Drop null values and reset index
  supervised_df <- supervised_df %>%
    drop_na() %>%
    rowid_to_column("index") %>%
    select(-index)
  
  # Write to CSV
  write_csv(supervised_df, './model_df.csv')
  
  return(supervised_df)
}

# Assuming stationary_df is already defined
model_df <- built_supervised(stationary_df)
print(model_df)

# Display information about the dataframe
glimpse(model_df)


train_test_split <- function(data) {
  data <- data %>% select(-sales, -date)
  n <- nrow(data)
  train <- data[1:(n-12), ]
  test <- data[(n-11):n, ]
  
  return(list(train = as.matrix(train), test = as.matrix(test)))
}

# Assuming model_df is already defined
split_data <- train_test_split(model_df)
train <- split_data$train
test <- split_data$test

cat(sprintf("Shape of Train: %s\nShape of Test: %s\n", 
            paste(dim(train), collapse = ", "),
            paste(dim(test), collapse = ", ")))

#Scale the data
scale_data <- function(train_set, test_set) {
  # Apply Min Max Scaler
  scaler <- preProcess(train_set, method = c("range"), range = c(-1, 1))
  
  # Scale training set
  train_set_scaled <- predict(scaler, train_set)
  
  # Scale test set
  test_set_scaled <- predict(scaler, test_set)
  
  # Separate features and target
  X_train <- train_set_scaled[, -1]
  y_train <- train_set_scaled[, 1]
  X_test <- test_set_scaled[, -1]
  y_test <- test_set_scaled[, 1]
  
  return(list(X_train = X_train, y_train = y_train, 
              X_test = X_test, y_test = y_test, 
              scaler = scaler))
}

scaled_data <- scale_data(train, test)
X_train <- scaled_data$X_train
y_train <- scaled_data$y_train
X_test <- scaled_data$X_test
y_test <- scaled_data$y_test
scaler_object <- scaled_data$scaler

cat(sprintf("Shape of X Train: %s\nShape of y Train: %s\nShape of X Test: %s\nShape of y Test: %s\n",
            paste(dim(X_train), collapse = ", "),
            paste(length(y_train), collapse = ", "),
            paste(dim(X_test), collapse = ", "),
            paste(length(y_test), collapse = ", ")))


#Reverse Scaling
re_scaling <- function(y_pred, x_test, scaler_obj, lstm = FALSE) {
  # For visualizing and comparing results, undoes the scaling effect on predictions.
  # y_pred: model predictions
  # x_test: features from the test set used for predictions
  # scaler_obj: the scaler objects used for min-max scaling
  # lstm: indicate if the model run is the lstm. If TRUE, additional transformation occurs
  
  # Reshape y_pred:
  y_pred <- array(y_pred, dim = c(length(y_pred), 1, 1))
  
  if (!lstm) {
    x_test <- array(x_test, dim = c(nrow(x_test), 1, ncol(x_test)))
  }
  
  # Rebuild test set for inverse transform:
  pred_test_set <- lapply(seq_len(length(y_pred)), function(index) {
    cbind(y_pred[index,,], x_test[index,,])
  })
  
  # Convert list to array:
  pred_test_set <- do.call(rbind, pred_test_set)
  
  # Inverse transform:
  pred_test_set_inverted <- predict(scaler_obj, pred_test_set)
  
  return(pred_test_set_inverted)
}

#Predictions Dataframe
prediction_df <- function(unscale_predictions, origin_df) {
  # Generates a dataframe that shows the predicted sales for each month
  # for plotting results.
  
  # unscale_predictions: the model predictions that do not have min-max or other scaling applied
  # origin_df: the original monthly sales dataframe
  
  # Create dataframe that shows the predicted sales:
  sales_dates <- tail(origin_df$date, 13)
  act_sales <- tail(origin_df$sales, 13)
  
  result_list <- lapply(seq_along(unscale_predictions), function(index) {
    list(
      pred_value = as.integer(unscale_predictions[index, 1] + act_sales[index]),
      date = sales_dates[index + 1]
    )
  })
  
  df_result <- do.call(rbind.data.frame, result_list)
  
  return(df_result)
}

#Scoring the models
install.packages("Metrics")
library(Metrics)
library(ggplot2)

model_scores <- list()

get_scores <- function(unscale_df, origin_df, model_name) {
  # Prints the root mean squared error, mean absolute error, and r2 scores
  # for each model. Saves all results in a model_scores list for comparison.
  
  rmse <- sqrt(mse(tail(origin_df$sales, 12), tail(unscale_df$pred_value, 12)))
  
  mae <- mae(tail(origin_df$sales, 12), tail(unscale_df$pred_value, 12))
  
  r2 <- cor(tail(origin_df$sales, 12), tail(unscale_df$pred_value, 12))^2
  
  model_scores[[model_name]] <<- c(rmse = rmse, mae = mae, r2 = r2)
  
  cat(sprintf("RMSE: %f\nMAE: %f\nR2 Score: %f\n", rmse, mae, r2))
}

#Graph of Results
regressive_model <- function(train_data, test_data, model, model_name) {
  # Runs regressive models in R framework. First calls scale_data
  # to split into X and y and scale the data. Then fits and predicts. Finally,
  # predictions are unscaled, scores are printed, and results are plotted and saved.
  
  # Split into X & y and scale data:
  scaled_data <- scale_data(train_data, test_data)
  X_train <- scaled_data$X_train
  y_train <- scaled_data$y_train
  X_test <- scaled_data$X_test
  y_test <- scaled_data$y_test
  scaler_object <- scaled_data$scaler_object
  
  # Run R models:
  mod <- model
  mod_fit <- mod(X_train, y_train)
  predictions <- predict(mod_fit, X_test)
  
  # Undo scaling to compare predictions against original data:
  origin_df <- m_df  # Assuming m_df is defined in the global environment
  unscaled <- re_scaling(predictions, X_test, scaler_object)
  unscaled_df <- prediction_df(unscaled, origin_df)
  
  # Print scores and plot results:
  get_scores(unscaled_df, origin_df, model_name)
  plot_results(unscaled_df, origin_df, model_name)
}

# Placeholder for plot_results function
plot_results <- function(unscaled_df, origin_df, model_name) {
  # Implement your plotting logic here
  # For example:
  ggplot() +
    geom_line(data = origin_df, aes(x = date, y = sales), color = "blue") +
    geom_line(data = unscaled_df, aes(x = date, y = pred_value), color = "red") +
    ggtitle(paste("Sales Prediction -", model_name)) +
    xlab("Date") +
    ylab("Sales")
  
  ggsave(paste0(model_name, "_plot.png"))
}

#Modelling
#Linear Regression
regressive_model <- function(train_data, test_data, model, model_name) {
  # Split into X & y and scale data:
  scaled_data <- scale_data(train_data, test_data)
  X_train <- scaled_data$X_train
  y_train <- scaled_data$y_train
  X_test <- scaled_data$X_test
  y_test <- scaled_data$y_test
  scaler_object <- scaled_data$scaler_object
  
  # Run R models:
  if (model_name == "LinearRegression") {
    # For linear regression, we need to create a data frame and use formula
    train_df <- as.data.frame(cbind(y_train, X_train))
    colnames(train_df) <- c("y", paste0("X", 1:(ncol(train_df)-1)))
    formula <- as.formula(paste("y ~", paste(colnames(train_df)[-1], collapse = " + ")))
    mod_fit <- lm(formula, data = train_df)
    predictions <- predict(mod_fit, newdata = as.data.frame(X_test))
  } else {
    # For other models (RandomForest, XGBoost)
    mod_fit <- model(X_train, y_train)
    predictions <- predict(mod_fit, X_test)
  }
  
  # Undo scaling to compare predictions against original data:
  origin_df <- m_df  # Assuming m_df is defined in the global environment
  unscaled <- re_scaling(predictions, X_test, scaler_object)
  unscaled_df <- prediction_df(unscaled, origin_df)
  
  # Print scores and plot results:
  get_scores(unscaled_df, origin_df, model_name)
  plot_results(unscaled_df, origin_df, model_name)
}

# Linear Regression
regressive_model <- function(train_data, test_data, model, model_name) {
  # Split into X & y and scale data:
  scaled_data <- scale_data(train_data, test_data)
  X_train <- scaled_data$X_train
  y_train <- scaled_data$y_train
  X_test <- scaled_data$X_test
  y_test <- scaled_data$y_test
  scaler_object <- scaled_data$scaler_object
  
  # Run R models:
  if (model_name == "LinearRegression") {
    # For linear regression, we need to create a data frame and use formula
    train_df <- as.data.frame(cbind(y = y_train, X_train))
    formula <- as.formula(paste("y ~", paste(colnames(train_df)[-1], collapse = " + ")))
    mod_fit <- lm(formula, data = train_df)
    test_df <- as.data.frame(X_test)
    colnames(test_df) <- colnames(X_train)
    predictions <- predict(mod_fit, newdata = test_df)
  } else {
    # For other models (RandomForest, XGBoost)
    mod_fit <- model(X_train, y_train)
    predictions <- predict(mod_fit, X_test)
  }
  
  # Undo scaling to compare predictions against original data:
  origin_df <- m_df  # Assuming m_df is defined in the global environment
  unscaled <- re_scaling(predictions, X_test, scaler_object)
  unscaled_df <- prediction_df(unscaled, origin_df)
  
  # Print scores and plot results:
  get_scores(unscaled_df, origin_df, model_name)
  plot_results(unscaled_df, origin_df, model_name)
}



yearly_sales <- train_data %>%
  mutate(year = year(as.Date(date))) %>%
  group_by(year) %>%
  summarise(total_sales = sum(sales)) %>%
  filter(year >= 2013 & year <= 2018)

# Print the yearly_sales to check available years
print(yearly_sales)

# Prepare data for Random Forest
rf_data <- data.frame(
  year = yearly_sales$year,
  total_sales = yearly_sales$total_sales
)

# Create and train the Random Forest model
rf_model <- randomForest(total_sales ~ year, data = rf_data, ntree = 500)

# Create a sequence of years for projection
years_seq <- data.frame(year = seq(min(yearly_sales$year), max(yearly_sales$year), 1))

# Make predictions
predictions <- predict(rf_model, newdata = years_seq)

# Combine actual data and predictions
plot_data <- years_seq %>%
  left_join(yearly_sales, by = "year") %>%
  rename(actual_sales = total_sales) %>%
  mutate(predicted_sales = predictions)

# Create the plot
ggplot(plot_data, aes(x = year)) +
  geom_line(aes(y = actual_sales, color = "Actual Sales"), size = 1) +
  geom_line(aes(y = predicted_sales, color = "Projected Sales"), size = 1) +
  geom_point(aes(y = actual_sales, color = "Actual Sales"), size = 3) +
  geom_point(aes(y = predicted_sales, color = "Projected Sales"), size = 3) +
  scale_color_manual(values = c("Actual Sales" = "blue", "Projected Sales" = "red")) +
  labs(title = "Yearly Sales Projection using Random Forest (2013-2018)",
       x = "Year",
       y = "Total Sales",
       color = "Legend") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Print model summary
print(rf_model)

# Save the plot
ggsave("yearly_sales_rf_projection_2013_2018.png", width = 10, height = 6, dpi = 300)

# Print actual vs. predicted values
print(plot_data)

# Calculate performance metrics
mse <- mean((plot_data$actual_sales - plot_data$predicted_sales)^2, na.rm = TRUE)
rmse <- sqrt(mse)
mae <- mean(abs(plot_data$actual_sales - plot_data$predicted_sales), na.rm = TRUE)
r_squared <- 1 - sum((plot_data$actual_sales - plot_data$predicted_sales)^2, na.rm = TRUE) / 
  sum((plot_data$actual_sales - mean(plot_data$actual_sales, na.rm = TRUE))^2, na.rm = TRUE)

# Print performance metrics
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("R-squared (R²):", r_squared, "\n")


yearly_sales <- train_data %>%
  mutate(year = year(as.Date(date))) %>%
  group_by(year) %>%
  summarise(total_sales = sum(sales)) %>%
  filter(year >= 2013 & year <= 2018)

# Print the yearly_sales to check available years
print(yearly_sales)

# Prepare data for XGBoost
xgb_data <- data.frame(
  year = yearly_sales$year,
  total_sales = yearly_sales$total_sales
)

# Create DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(xgb_data["year"]), label = xgb_data$total_sales)

# Set XGBoost parameters
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 3,
  nrounds = 100
)

# Create and train the XGBoost model
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)

# Create a sequence of years for projection
years_seq <- data.frame(year = seq(min(yearly_sales$year), max(yearly_sales$year), 1))

# Make predictions
predictions <- predict(xgb_model, as.matrix(years_seq))

# Combine actual data and predictions
plot_data <- years_seq %>%
  left_join(yearly_sales, by = "year") %>%
  rename(actual_sales = total_sales) %>%
  mutate(predicted_sales = predictions)

# Create the plot
ggplot(plot_data, aes(x = year)) +
  geom_line(aes(y = actual_sales, color = "Actual Sales"), size = 1) +
  geom_line(aes(y = predicted_sales, color = "Projected Sales"), size = 1) +
  geom_point(aes(y = actual_sales, color = "Actual Sales"), size = 3) +
  geom_point(aes(y = predicted_sales, color = "Projected Sales"), size = 3) +
  scale_color_manual(values = c("Actual Sales" = "blue", "Projected Sales" = "red")) +
  labs(title = "Yearly Sales Projection using XGBoost (2013-2018)",
       x = "Year",
       y = "Total Sales",
       color = "Legend") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Print model summary (feature importance)
importance <- xgb.importance(feature_names = "year", model = xgb_model)
print(importance)

# Save the plot
ggsave("yearly_sales_xgboost_projection_2013_2018.png", width = 10, height = 6, dpi = 300)

# Print actual vs. predicted values
print(plot_data)

# Calculate performance metrics
mse <- mean((plot_data$actual_sales - plot_data$predicted_sales)^2, na.rm = TRUE)
rmse <- sqrt(mse)
mae <- mean(abs(plot_data$actual_sales - plot_data$predicted_sales), na.rm = TRUE)
r_squared <- 1 - sum((plot_data$actual_sales - plot_data$predicted_sales)^2, na.rm = TRUE) / 
  sum((plot_data$actual_sales - mean(plot_data$actual_sales, na.rm = TRUE))^2, na.rm = TRUE)

# Print performance metrics
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("R-squared (R²):", r_squared, "\n")

yearly_sales <- train_data %>%
  mutate(year = year(as.Date(date))) %>%
  group_by(year) %>%
  summarise(total_sales = sum(sales)) %>%
  filter(year >= 2013 & year <= 2018)

# Print the yearly_sales to check available years
print(yearly_sales)

# Perform linear regression
lm_model <- lm(total_sales ~ year, data = yearly_sales)

# Create a sequence of years for projection
years_seq <- data.frame(year = seq(min(yearly_sales$year), max(yearly_sales$year), 1))

# Make predictions
predictions <- predict(lm_model, newdata = years_seq)

# Combine actual data and predictions
plot_data <- years_seq %>%
  left_join(yearly_sales, by = "year") %>%
  rename(actual_sales = total_sales) %>%
  mutate(predicted_sales = predictions)

# Create the plot
ggplot(plot_data, aes(x = year)) +
  geom_line(aes(y = actual_sales, color = "Actual Sales"), size = 1) +
  geom_line(aes(y = predicted_sales, color = "Projected Sales"), size = 1) +
  geom_point(aes(y = actual_sales, color = "Actual Sales"), size = 3) +
  geom_point(aes(y = predicted_sales, color = "Projected Sales"), size = 3) +
  scale_color_manual(values = c("Actual Sales" = "blue", "Projected Sales" = "red")) +
  labs(title = "Yearly Sales Projection using Linear Regression (2013-2018)",
       x = "Year",
       y = "Total Sales",
       color = "Legend") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Print model summary
summary(lm_model)

# Save the plot
ggsave("yearly_sales_linear_regression_projection_2013_2018.png", width = 10, height = 6, dpi = 300)

# Print actual vs. predicted values
print(plot_data)

# Calculate performance metrics
mse <- mean((plot_data$actual_sales - plot_data$predicted_sales)^2, na.rm = TRUE)
rmse <- sqrt(mse)
mae <- mean(abs(plot_data$actual_sales - plot_data$predicted_sales), na.rm = TRUE)
r_squared <- summary(lm_model)$r.squared

# Print performance metrics
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("R-squared (R²):", r_squared, "\n")

monthly_sales <- train_data %>%
  mutate(date = as.Date(date)) %>%
  filter(year(date) >= 2013 & year(date) <= 2018) %>%
  group_by(date = floor_date(date, "month")) %>%
  summarise(total_sales = sum(sales))

# Convert to time series object
sales_ts <- ts(monthly_sales$total_sales, start = c(2013, 1), frequency = 12)

# Function to plot actual vs predicted values
plot_forecast <- function(forecast_obj, title) {
  autoplot(forecast_obj) +
    autolayer(sales_ts, series = "Actual") +
    labs(title = title, y = "Sales", x = "Year") +
    theme_minimal()
}

# ARIMA Model
arima_model <- auto.arima(sales_ts, seasonal = FALSE)
arima_forecast <- forecast(arima_model, h = 12)

# Plot ARIMA results
arima_plot <- plot_forecast(arima_forecast, "ARIMA Sales Forecast (2013-2018)")
print(arima_plot)
ggsave("arima_sales_forecast_2013_2018.png", arima_plot, width = 10, height = 6, dpi = 300)

# SARIMA Model
sarima_model <- auto.arima(sales_ts, seasonal = TRUE)
sarima_forecast <- forecast(sarima_model, h = 12)

# Plot SARIMA results
sarima_plot <- plot_forecast(sarima_forecast, "SARIMA Sales Forecast (2013-2018)")
print(sarima_plot)
ggsave("sarima_sales_forecast_2013_2018.png", sarima_plot, width = 10, height = 6, dpi = 300)

# Print model summaries
cat("ARIMA Model Summary:\n")
summary(arima_model)

cat("\nSARIMA Model Summary:\n")
summary(sarima_model)

# Calculate performance metrics
calculate_metrics <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predicted), na.rm = TRUE)
  
  cat("Mean Squared Error (MSE):", mse, "\n")
  cat("Root Mean Squared Error (RMSE):", rmse, "\n")
  cat("Mean Absolute Error (MAE):", mae, "\n")
}

cat("\nARIMA Model Performance Metrics:\n")
calculate_metrics(sales_ts, arima_model$fitted)

cat("\nSARIMA Model Performance Metrics:\n")
calculate_metrics(sales_ts, sarima_model$fitted)

# Compare AIC and BIC
cat("\nModel Comparison:\n")
cat("ARIMA AIC:", arima_model$aic, "BIC:", arima_model$bic, "\n")
cat("SARIMA AIC:", sarima_model$aic, "BIC:", sarima_model$bic, "\n")

