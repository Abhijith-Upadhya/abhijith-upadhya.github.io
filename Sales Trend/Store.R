install.packages("tidyverse", "dplyr", "ggplot", "lubridate", "tidyr")
install.packages("GGally", "corrplot")
install.packages("caret")
install.packages("randomForest")
install.packages("xgboost")
install.packages("tseries", "gridExtra", "forecast")
install.packages("zoo")
install.packages("viridus", "reshape2")
install.packages("Metrics")
library(Metrics)
library(zoo)
library(viridis)
library(reshape2)
library(tseries)
library(gridExtra)
library(forecast)
library(xgboost)
library(caret)
library(randomForest)
library(GGally)
library(corrplot)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyr)

store <- read.csv('store.csv')
train <- read.csv('train.csv') %>%
  mutate(Date = as.Date(Date)) %>%
  arrange(Date)

# Display the first few rows of each dataset
print("Store data:")
print(head(store))
print("\nTrain data:")
print(head(train))

# Merge train and store data
train_store_joined <- inner_join(train, store, by = "Store")

# Extract month and year from the date
train_store_joined <- train_store_joined %>%
  mutate(Year = year(Date),
         Month = month(Date))

sales_summary <- train_store_joined %>%
  group_by(Year, Month, Promo, Promo2) %>%
  summarise(
    mean_sales = mean(Sales, na.rm = TRUE),
    se_sales = sd(Sales, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )

# Create the line plot with error bars
sales_trend_plot <- ggplot(sales_summary, aes(x = factor(Month), y = mean_sales, group = interaction(Year, Promo2), color = factor(Promo2))) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = mean_sales - se_sales, ymax = mean_sales + se_sales), width = 0.2) +
  facet_grid(Year ~ Promo) +
  labs(title = "Average Sales Trend by Month, Year, and Promotions",
       x = "Month", y = "Average Sales", color = "Promo2") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_discrete(name = "Promo2", labels = c("No", "Yes"))

# Print the plot
print(sales_trend_plot)


# Calculate mean sales for each day of the week and promo status
sales_summary <- train_store_joined %>%
  group_by(DayOfWeek, Promo) %>%
  summarise(
    mean_sales = mean(Sales, na.rm = TRUE),
    se_sales = sd(Sales, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )

# Create the plot
sales_trend_plot <- ggplot(sales_summary, aes(x = factor(DayOfWeek), y = mean_sales, group = factor(Promo), color = factor(Promo))) +
  geom_line(size = 1.2) +
  geom_point(size = 3, shape = 21, fill = "white") +
  geom_errorbar(aes(ymin = mean_sales - se_sales, ymax = mean_sales + se_sales), width = 0.2, alpha = 0.7) +
  labs(
    title = "Sales Trend Over Days of Week",
    subtitle = "Comparing sales with and without promotions",
    x = "Day of Week", 
    y = "Average Sales", 
    color = "Promotion Status"
  ) +
  scale_x_discrete(labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")) +
  scale_y_continuous(labels = scales::comma_format(scale = 1e-3, suffix = "K")) +
  scale_color_viridis(discrete = TRUE, begin = 0.3, end = 0.7,
                      labels = c("No Promotion", "Promotion Running")) +
  theme_minimal(base_size = 14) +
  theme(
    plot.subtitle = element_text(size = 12, color = "gray50"),
    legend.position = "bottom",
    legend.background = element_rect(fill = "white", color = NA),
    legend.key = element_rect(fill = "white", color = NA),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.title = element_text(face = "bold")
  ) +
  annotate("text", x = 7, y = max(sales_summary$mean_sales), 
           label = "Weekend sales\nare highest", 
           hjust = 1, vjust = 1, color = "gray30", size = 4, fontface = "italic")

# Print the plot
print(sales_trend_plot)


correlation_matrix <- train_store_joined %>%
  select(Sales, Customers, Promo, SchoolHoliday, CompetitionDistance) %>%
  cor()

corr_melted <- melt(correlation_matrix)

# Create the improved correlation plot
correlation_plot <- ggplot(corr_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.2f", value)), 
            color = ifelse(abs(corr_melted$value) > 0.5, "white", "black"), 
            size = 3.5) +
  scale_fill_viridis(option = "C", begin = 0.3, end = 0.9, 
                     name = "Correlation", limit = c(-1,1)) +
  coord_equal() +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(hjust = 1),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = NA, color = "grey80", size = 0.5),
    legend.position = "right",
    legend.title = element_text(angle = 90),
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "grey50", size = 10),
    plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
  ) +
  labs(
    title = "Correlation Heatmap of Key Features",
    subtitle = "Strength and direction of relationships between variables",
    x = "", 
    y = ""
  )

# Print the plot
print(correlation_plot)




train$Sales <- as.numeric(train$Sales)  # Ensure Sales is numeric

# Assigning one store from each category (Store 2, type 'a')
sales_a <- train %>% 
  filter(Store == 2) %>%
  select(Date, Sales) %>%
  arrange(Date)

# Resample weekly sales and calculate sum for each week
weekly_sales_a <- sales_a %>%
  mutate(week = as.Date(cut(Date, breaks = "week"))) %>%
  group_by(week) %>%
  summarise(Sales = sum(Sales))

# Function to test the stationarity of time series
test_stationarity <- function(timeseries) {
  
  # Calculating rolling mean and standard deviation
  roll_mean <- rollmean(timeseries$Sales, 7, fill = NA)
  roll_std <- rollapply(timeseries$Sales, 7, sd, fill = NA)
  
  # Plotting original, rolling mean, and rolling std
  plot(timeseries$week, timeseries$Sales, type = "l", col = "blue", lwd = 2, ylab = "Sales", xlab = "Date", main = "Stationarity Check")
  lines(timeseries$week, roll_mean, col = "red", lwd = 2)
  lines(timeseries$week, roll_std, col = "green", lwd = 2)
  legend("topright", legend = c("Original", "Rolling Mean", "Rolling Std"), col = c("blue", "red", "green"), lwd = 2)
  
  # Perform Dickey-Fuller test
  adf_test <- adf.test(timeseries$Sales, alternative = "stationary", k = 0)
  
  # Output the results of the Dickey-Fuller test
  print("Results of Dickey-Fuller Test:")
  print(paste("ADF Statistic: ", adf_test$statistic))
  print(paste("p-value: ", adf_test$p.value))
  print("Critical Values:")
  print(adf_test$critical.values)
  
  if(adf_test$p.value < 0.05) {
    print("Data is stationary (Reject the null hypothesis)")
  } else {
    print("Data is not stationary (Fail to reject the null hypothesis)")
  }
}

# Testing the stationarity of store type 'a'
test_stationarity(weekly_sales_a)


plot_timeseries <- function(sales, StoreType) {
  # Convert to ts object
  ts_data <- ts(sales$Sales, frequency = 365)
  
  # Perform decomposition
  decomp <- stl(ts_data, s.window = "periodic")
  
  # Plot decomposition
  plot(decomp, main = paste("Decomposition Plots for Store Type", StoreType))
}

# Plot decomposition for store type 'a'
plot_timeseries(sales_a, "a")


# Ensure the Date column in `weekly_sales_a` is properly formatted as Date type.
weekly_sales_a$week <- as.Date(weekly_sales_a$week)

# Convert data to a zoo object for better date handling
weekly_sales_zoo <- zoo(weekly_sales_a$Sales, order.by = weekly_sales_a$week)

# Convert zoo object to time series object with weekly frequency (52 weeks/year)
weekly_sales_ts <- ts(weekly_sales_a$Sales, frequency = 52, start = c(as.numeric(format(min(weekly_sales_a$week), "%Y")),
                                                                      as.numeric(format(min(weekly_sales_a$week), "%U"))))

# Function to plot seasonality and trend using STL decomposition
plot_timeseries <- function(sales_ts) {
  
  # Decompose the time series using STL (seasonal-trend decomposition)
  decomposition <- stl(sales_ts, s.window = "periodic")
  decomposition_trend <- decompose(sales_ts, type = "additive")
  # Extract components
  estimated_trend <- decomposition_trend$trend
  #estimated_trend <- decomposition$time.series[, "trend"]
  estimated_seasonal <- decomposition$time.series[, "seasonal"]
  
  # Create a 2-row plot layout
  par(mfrow = c(2, 1), mar = c(2, 5, 5, 4))
  
  # Plot the trend with dates on the x-axis
  plot(weekly_sales_a$week, estimated_trend, type = "l", col = "blue", 
       main = "Estimated Trend", ylab = "Trend", xlab = "Date")
  
  # Plot the seasonality with dates on the x-axis
  plot(weekly_sales_a$week, estimated_seasonal, type = "l", col = "green", 
       main = "Estimated Seasonality", ylab = "Seasonality", xlab = "Date")
  
  # Reset the layout
  par(mfrow = c(1, 1))
  
  title("Decomposition Plots", outer = TRUE, line = -1)
}

# Testing the decomposition plot function with the weekly time series
plot_timeseries(weekly_sales_ts)

weekly_sales_ts <- ts(weekly_sales_a$Sales, frequency = 52, start = c(2013, 1))
fit_arima <- auto.arima(weekly_sales_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)

# Print ARIMA model summary
summary(fit_arima)
tuned_arima <- auto.arima(weekly_sales_ts, 
                          seasonal = TRUE, 
                          stepwise = FALSE, 
                          approximation = FALSE, 
                          trace = TRUE)

# Print tuned ARIMA model summary
summary(tuned_arima)
forecasted_sales <- forecast(tuned_arima, h = 12)

# Plot the forecasted sales
par(mfrow = c(2, 1), mar = c(4, 4, 4, 4), cex = 1)  # Adjust margins and size scaling
plot(forecasted_sales, main = "Forecasted Sales", xlab = "Weeks", ylab = "Sales")

tsdiag(tuned_arima)

# Plot residuals
plot(residuals(tuned_arima), main = "Residuals of ARIMA Model")
acf(residuals(tuned_arima), main = "ACF of Residuals")
decomp <- stl(weekly_sales_ts, s.window = "periodic")
plot(decomp)


create_lagged_features <- function(data, lags = 1:4) {
  lagged_data <- data.frame(Sales = data$Sales)
  
  for (lag in lags) {
    lagged_data[[paste0("Lag_", lag)]] <- dplyr::lag(data$Sales, n = lag)
  }
  
  return(na.omit(lagged_data))
}

# Apply the function to create lagged features
weekly_sales_lagged <- create_lagged_features(weekly_sales_a, lags = 1:4)

# Split data into training and testing sets (e.g., last 12 weeks as test)
train_size <- nrow(weekly_sales_lagged) - 12
train_data <- weekly_sales_lagged[1:train_size, ]
test_data <- weekly_sales_lagged[(train_size + 1):nrow(weekly_sales_lagged), ]

train_matrix <- xgb.DMatrix(as.matrix(train_data[, -1]), label = train_data$Sales)
test_matrix <- xgb.DMatrix(as.matrix(test_data[, -1]), label = test_data$Sales)

# Set parameters for XGBoost
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

# Train XGBoost model
xgboost_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix, test = test_matrix),
  early_stopping_rounds = 10,
  print_every_n = 10
)

# Predict on test data
preds <- predict(xgboost_model, test_matrix)

# Calculate Mean Absolute Error (MAE)
mae_value <- mae(test_data$Sales, preds)
print(paste("Mean Absolute Error (MAE):", mae_value))

# Calculate Root Mean Square Error (RMSE)
rmse_value <- rmse(test_data$Sales, preds)
print(paste("Root Mean Square Error (RMSE):", rmse_value))

# Set up the hyperparameter grid
grid <- expand.grid(
  nrounds = c(50, 100, 150),
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 5, 7),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# Set up training control
train_control <- trainControl(
  method = "cv", 
  number = 3, 
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Train the XGBoost model using caret
xgb_tuned <- train(
  Sales ~ ., 
  data = train_data, 
  method = "xgbTree", 
  tuneGrid = grid, 
  trControl = train_control,
  metric = "RMSE"
)

# Best model and results
print(xgb_tuned$bestTune)
print(xgb_tuned$results)
