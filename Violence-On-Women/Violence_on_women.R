# Install and load required packages
install.packages("tidyverse")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("viridis")
install.packages("RColorBrewer")
install.packages("dplyr")
install.packages("tidyr")
install.packages("gridExtra")
install.packages("caret")
install.packages("ROSE")
install.packages("e1071")
install.packages("rpart")
install.packages("randomForest")
install.packages("class")
install.packages("DMwR")
library(e1071)
library(rpart)
library(randomForest)
library(class)
library(DMwR)
library(ROSE)
library(caret)
library(gridExtra)
library(tidyr)
library(RColorBrewer)
library(tidyverse)
library(corrplot)
library(ggplot2)
library(viridis)
library(dplyr)

# Read the dataset
df <- read.csv("Domestic_violence.csv")

# Data preparation
df <- df %>% 
  select(-'Sl.No.') %>%
  mutate(across(c('Education', 'Employment', 'Marital_status'), as.character)) %>%
  mutate(Employment = trimws(Employment))

# Check dimensions
dim(df)

# Check data types
str(df)

# Check for null values
colSums(is.na(df))

# Count distinct values for each column
sapply(df, function(x) length(unique(x)))

# Binarize the dataset
df <- df %>%
  mutate(
    marital_binary = case_when(
      Marital_status == 'married' ~ 1,
      Marital_status == 'unmarred' ~ 0
    ),
    Violence = case_when(
      Violence == 'yes' ~ 1,
      Violence == 'no' ~ 0
    ),
    education_binary = case_when(
      Education == 'none' ~ 0,
      Education == 'primary' ~ 1,
      Education == 'secondary' ~ 2,
      Education == 'tertiary' ~ 3
    ),
    employment_binary = case_when(
      Employment == 'unemployed' ~ 0,
      Employment == 'semi employed' ~ 1,
      Employment == 'employed' ~ 2
    )
  )

# Check zero income count
sum(df$Income == 0)

# Check outliers - Employed/Semi-employed with zero income
mask <- df %>%
  filter(Income == 0 & (Employment == "employed" | Employment == "semi employed"))
print(mask)

# Check duplicates
duplicate_rows <- df[duplicated(df), ]
cat("Number of duplicate rows:", nrow(duplicate_rows))

# Summary statistics
summary(df)

# Create histograms
par(mfrow = c(1, 2))
hist_age <- ggplot(df, aes(x = Age)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30) +
  geom_density() +
  theme_minimal() +
  labs(title = "Age Distribution")

hist_income <- ggplot(df, aes(x = Income)) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  labs(title = "Income Distribution")

grid.arrange(hist_age, hist_income, ncol = 2)

# Create boxplot
ggplot(df, aes(y = Income)) +
  geom_boxplot(width = 0.2) +
  theme_minimal() +
  labs(title = "Income Distribution Boxplot")

# Filter zero income
zero_income <- df %>% filter(Income == 0)
print(zero_income)

# Correlation matrix
df_binary <- df %>%
  select(Age, education_binary, employment_binary, Income, marital_binary, Violence)

correlation_matrix <- cor(df_binary)
corrplot(correlation_matrix, 
         method = "color", 
         type = "upper", 
         addCoef.col = "black",
         number.cex = 0.7,
         col = viridis(100),
         title = "Correlation between Features")

# Define color palettes
palette <- c("#0096c7", "#00b4d8", "#48cae4", "#90e0ef")
palette_pie <- c("#0096c7", "#e3f2fd", "#bbdefb", "#90caf9")

# Display first 10 rows
head(df, 10)

# Distribution of features - Create 4 pie charts
# Violence View
p1 <- df %>%
  count(Violence) %>%
  mutate(
    prop = n/sum(n),
    label = c("No", "Yes"),
    pct = paste0(round(prop * 100, 1), "%")
  ) %>%
  ggplot(aes(x = "", y = prop, fill = factor(label))) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = pct), position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = palette_pie) +
  labs(title = "Violence View") +
  theme_void() +
  theme(legend.title = element_blank())

# Marital View
p2 <- df %>%
  count(Marital_status) %>%
  mutate(
    prop = n/sum(n),
    pct = paste0(round(prop * 100, 1), "%")
  ) %>%
  ggplot(aes(x = "", y = prop, fill = Marital_status)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = pct), position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = palette_pie) +
  labs(title = "Marital View") +
  theme_void() +
  theme(legend.title = element_blank())

# Employment View
p3 <- df %>%
  count(Employment) %>%
  mutate(
    prop = n/sum(n),
    pct = paste0(round(prop * 100, 1), "%")
  ) %>%
  ggplot(aes(x = "", y = prop, fill = Employment)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = pct), position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = palette_pie) +
  labs(title = "Employment View") +
  theme_void() +
  theme(legend.title = element_blank())

# Education View
p4 <- df %>%
  count(Education) %>%
  mutate(
    prop = n/sum(n),
    pct = paste0(round(prop * 100, 1), "%")
  ) %>%
  ggplot(aes(x = "", y = prop, fill = Education)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = pct), position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = palette_pie) +
  labs(title = "Education View") +
  theme_void() +
  theme(legend.title = element_blank())

# Arrange all pie charts in a grid
grid.arrange(p1, p2, p3, p4, ncol = 2)

# Age & Income x Violence
violence_data <- df %>% filter(Violence == 1)

# Create scatter plots for Age and Income
p5 <- ggplot(violence_data, aes(x = Age, y = Age)) +
  geom_point() +
  theme_minimal()

p6 <- ggplot(violence_data, aes(x = Income, y = Age)) +
  geom_point() +
  theme_minimal()

grid.arrange(p5, p6, ncol = 2)

# Education x Violence
ggplot(df, aes(y = Education, fill = factor(Violence))) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = palette, labels = c("No", "Yes")) +
  labs(title = "Violence x Education", x = "Violence Qty", fill = "Violence") +
  theme_minimal()

# Marital Status and Employment for Violence cases
violence_plots <- df %>%
  filter(Violence == 1) %>%
  {
    list(
      ggplot(., aes(x = Marital_status)) +
        geom_bar(fill = palette[1]) +
        theme_minimal() +
        labs(y = ""),
      
      ggplot(., aes(x = Employment)) +
        geom_bar(fill = palette[1]) +
        theme_minimal() +
        labs(y = "")
    )
  }

grid.arrange(grobs = violence_plots, ncol = 2)

# Education x Age, Marital Status x Age, Education x Income
p7 <- ggplot(df, aes(x = Education, y = Age)) +
  geom_jitter(width = 0.2) +
  theme_minimal()

p8 <- ggplot(df, aes(x = Marital_status, y = Age)) +
  geom_jitter(width = 0.2) +
  theme_minimal()

p9 <- ggplot(df, aes(x = Education, y = Income)) +
  geom_point() +
  theme_minimal()

grid.arrange(p7, p8, p9, ncol = 3)

# Average income per Education Level
average_income_per_education <- df %>%
  group_by(Education) %>%
  summarise(Income = round(mean(Income), 2)) %>%
  arrange(Income)

print(average_income_per_education)

ggplot(average_income_per_education, aes(x = Education, y = Income, fill = Education)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = palette) +
  theme_minimal() +
  theme(legend.position = "none")


# Binarizing the dataset
binary_features <- df %>% select(Age, Income, Violence)

# Creating dummy variables for the categorical features
# Excluding the already binarized columns
dummie_features <- df %>% 
  select(-Age, -Income, -Violence, -marital_binary, -education_binary, -employment_binary) %>%
  mutate(across(everything(), ~ ifelse(. == TRUE, 1, 0)))  # Transform TRUE/FALSE into 1/0

# Combining binary features with dummy features
df_ml <- cbind(binary_features, dummie_features)

# Plot original dataset Violence distribution
p1 <- ggplot(df_ml, aes(x = as.factor(Violence))) +
  geom_bar(fill = palette[1]) +
  labs(title = "Violence Distribution in Original Data", x = "Violence") +
  theme_minimal()

# Re-sampling the dataset using SMOTE
set.seed(123)  # Set seed for reproducibility
table(df_ml$Violence)

# Set N to be larger than the total number of rows in the dataset
total_samples <- nrow(df_ml)
N_target <- 2 * total_samples  # Set to twice the total number of rows to ensure oversampling

# Apply SMOTE using the larger N value
smote_data <- ovun.sample(Violence ~ ., data = df_ml, method = "over", N = N_target)$data

# Plot resampled dataset Violence distribution
p2 <- ggplot(smote_data, aes(x = as.factor(Violence))) +
  geom_bar(fill = palette[1]) +
  labs(title = "Violence Distribution after SMOTE", x = "Violence") +
  theme_minimal()

# Arrange both plots side by side
grid.arrange(p1, p2, ncol = 2)

# Randomly shuffle the resampled dataset
df_ml_sample <- smote_data[sample(nrow(smote_data)), ]

SEED <- 158020
set.seed(SEED)

# Prepare the dataset: Exclude the 'Violence' column from features (X) and define the target (y)
x <- df_ml_sample %>% select(-Violence)
y <- df_ml_sample$Violence

# Normalize the features using the preProcess function from the caret package
norm <- preProcess(x, method = c("center", "scale"))
x_norm <- predict(norm, x)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
train_x <- x_norm[trainIndex, ]
test_x <- x_norm[-trainIndex, ]
train_y <- y[trainIndex]
test_y <- y[-trainIndex]

# Function to run model scores
model <- function(model, test_x, test_y) {
  predictions <- predict(model, newdata = test_x)
  accuracy <- mean(predictions == test_y) * 100
  conf_matrix <- confusionMatrix(predictions, as.factor(test_y))
  
  print(conf_matrix)
  print(sprintf("Accuracy: %.2f%%", accuracy))
}

# Dummy Classifier - Baseline
dummy_model <- train(train_x, as.factor(train_y), method = "rf", trControl = trainControl(method = "cv", number = 5))
model(dummy_model, test_x, test_y)

# Naive Bayes Classifier
nb_model <- naiveBayes(as.factor(train_y) ~ ., data = train_x)
model(nb_model, test_x, test_y)


# Decision Tree Classifier
tree_model <- rpart(as.factor(train_y) ~ ., data = train_x, method = "class", control = rpart.control(maxdepth = 2))

# Updated model function to ensure class predictions for decision trees
model <- function(model, test_x, test_y) {
  if ("rpart" %in% class(model)) {
    # For rpart models, get class predictions directly
    predictions <- predict(model, newdata = test_x, type = "class")
  } else {
    # For other models, assume normal predict works fine
    predictions <- predict(model, newdata = test_x)
  }
  
  accuracy <- mean(predictions == test_y) * 100
  conf_matrix <- confusionMatrix(predictions, as.factor(test_y))
  
  print(conf_matrix)
  print(sprintf("Accuracy: %.2f%%", accuracy))
}

# Run the model evaluation again for Decision Tree
model(tree_model, test_x, test_y)

# K-Nearest Neighbors (KNN)
# With Euclidean distance
knn_model_euclidean <- knn(train = train_x, test = test_x, cl = train_y, k = 5)
confusionMatrix(knn_model_euclidean, as.factor(test_y))

# With Hamming distance (more suitable for binary data)
knn_model_hamming <- knn(train = train_x, test = test_x, cl = train_y, k = 4)
confusionMatrix(knn_model_hamming, as.factor(test_y))

# Random Forest Classifier
rf_model <- randomForest(as.factor(train_y) ~ ., data = train_x, ntree = 100)
model(rf_model, test_x, test_y)


# Define a function to print results
print_results <- function(results) {
  mean_acc <- mean(results$results$Accuracy)
  std_acc <- sd(results$results$Accuracy)
  cat(sprintf("Average Accuracy: %.2f%%\n", mean_acc * 100))
  cat(sprintf("Accuracy interval: [%.2f, %.2f]%%\n", (mean_acc - 2 * std_acc) * 100, (mean_acc + 2 * std_acc) * 100))
}

# Set random seed
set.seed(158020)

# Define cross-validation method
train_control <- trainControl(method = "cv", number = 10)

# Prepare the dataset (x and y are from previous steps, assuming they are ready)
# x_norm and y are normalized features and target (Violence) from the previous steps

# Ensure y is named properly and combine with x_norm
df_combined <- cbind(x_norm, Violence = y)  # Rename your target variable appropriately

# Check column names of df_combined to ensure there's no mismatch
colnames(df_combined)

# Decision Tree Classifier with cross-validation
tree_model <- train(as.factor(Violence) ~ ., data = df_combined, 
                    method = "rpart", 
                    trControl = train_control, 
                    tuneGrid = expand.grid(cp = 0.01),  # Cross-validation on complexity parameter
                    control = rpart.control(maxdepth = 2))

# Check the results
tree_model

# Print cross-validation results for Decision Tree
print_results(tree_model)

# K-Nearest Neighbors with cross-validation
knn_model <- train(as.factor(Violence) ~ ., data = df_combined, 
                   method = "knn", 
                   metric = "Accuracy", 
                   tuneGrid = expand.grid(k = 3),  # 3 neighbors
                   trControl = train_control)

# Print cross-validation results for KNN
print_results(knn_model)

# Random Forest with cross-validation
rf_model <- train(as.factor(Violence) ~ ., data = df_combined, 
                  method = "rf", 
                  trControl = train_control, 
                  tuneGrid = expand.grid(mtry = 3))  # mtry can be adjusted or optimized

# Print cross-validation results for Random Forest
print_results(rf_model)

# Grid Search for KNN (search for optimal 'k')
knn_grid <- expand.grid(k = 1:31)  # Searching k from 1 to 31

knn_grid_search <- train(as.factor(Violence) ~ ., data = df_combined, 
                         method = "knn", 
                         trControl = trainControl(method = "cv", number = 10), 
                         tuneGrid = knn_grid)

# Make predictions using the best model from Grid Search
knn_predictions <- predict(knn_grid_search, newdata = test_x)

# Evaluate the KNN model
confusionMatrix(knn_predictions, as.factor(test_y))
cat("Best number of neighbors:", knn_grid_search$bestTune$k, "\n")

# Define the tuning grid for mtry (number of predictors randomly sampled at each split)
rf_grid <- expand.grid(mtry = c(1, 2, 3, 4, 5))  # Adjust values as needed

# Perform grid search with cross-validation for Random Forest
rf_grid_search <- train(as.factor(Violence) ~ ., data = df_combined, 
                        method = "rf", 
                        trControl = trainControl(method = "cv", number = 10), 
                        tuneGrid = rf_grid)  # Use the proper tuneGrid with mtry

# Check the results
rf_grid_search
# Make predictions using the best Random Forest model from Grid Search
rf_predictions <- predict(rf_grid_search, newdata = test_x)

# Evaluate the Random Forest model
confusionMatrix(rf_predictions, as.factor(test_y))
