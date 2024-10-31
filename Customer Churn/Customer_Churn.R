install.packages(c("tidyverse", "visdat", "caret", "rpart", "randomForest", 
                   "e1071", "class", "kernlab", "nnet", "ada", "gbm", 
                   "xgboost", "catboost", "pROC"))
install.packages("CatBoost")
install.packages("plotly", "ggplot2", "gridExtra", "scales")
install.packages("ggridges", "corrplot")
install.packages("tidyr", "reshape2")

library(tidyverse)  # for data manipulation and visualization
library(visdat)     # for visualizing missing data
library(caret)      # for machine learning workflows
library(rpart)      # for decision trees
library(randomForest)
library(e1071)      # for naive bayes and svm
library(class)      # for knn
library(kernlab)    # for svm
library(nnet)       # for neural networks
library(ada)        # for adaboost
library(gbm)        # for gradient boosting
library(xgboost)
library(CatBoost)
library(pROC)       # for ROC curves
library(plotly)
library(gridExtra)
library(ggplot2)
library(scales)
library(corrplot)
library(ggridges)
library(tidyr)
library(reshape2)

# Load and examine data
df <- read.csv('Telecom_Customer_Churn.csv')
head(df)
dim(df)
str(df)
colnames(df)
sapply(df, class)

# Visualize missing values
vis_miss(df)

# Drop customerID column
df <- df %>% select(-customerID)
head(df)

# Convert TotalCharges to numeric
df$TotalCharges <- as.numeric(as.character(df$TotalCharges))

# Check missing values
colSums(is.na(df))

# View rows where TotalCharges is NA
df[is.na(df$TotalCharges), ]

# Find and remove rows where tenure is 0
zero_tenure_indices <- which(df$tenure == 0)
df <- df[-zero_tenure_indices, ]

# Verify removal
sum(df$tenure == 0)

# Fill NA values with mean
df$TotalCharges[is.na(df$TotalCharges)] <- mean(df$TotalCharges, na.rm = TRUE)

# Check missing values again
colSums(is.na(df))

# Map SeniorCitizen values
df$SeniorCitizen <- ifelse(df$SeniorCitizen == 0, "No", "Yes")
head(df)

# Describe InternetService column
summary(df$InternetService)
table(df$InternetService)

# Define numerical columns and get summary statistics
numerical_cols <- c('tenure', 'MonthlyCharges', 'TotalCharges')
summary(df[numerical_cols])       # For missing value handling alternatives

# Gender and Churn Distribution (Donut charts)
# Plot 1: Gender Distribution
gender_plot <- plot_ly() %>%
  add_pie(data = as.data.frame(table(df$gender)),
          labels = ~Var1,
          values = ~Freq,
          hole = 0.4,
          name = "Gender") %>%
  layout(title = "Gender Distribution",
         annotations = list(text = "Gender", 
                            x = 0.5, 
                            y = 0.5,
                            showarrow = FALSE))

# Plot 2: Churn Distribution
churn_plot <- plot_ly() %>%
  add_pie(data = as.data.frame(table(df$Churn)),
          labels = ~Var1,
          values = ~Freq,
          hole = 0.4,
          name = "Churn") %>%
  layout(title = "Churn Distribution",
         annotations = list(text = "Churn", 
                            x = 0.5, 
                            y = 0.5,
                            showarrow = FALSE))

# Arrange plots side by side
subplot(gender_plot, churn_plot)

# Churn counts by gender
churn_no <- table(df$gender[df$Churn == "No"])
churn_yes <- table(df$gender[df$Churn == "Yes"])

# Nested donut chart for Churn and Gender
# Using ggplot2 for more complex visualization
churn_gender_data <- data.frame(
  category = c("Churn: Yes", "Churn: No"),
  value = c(1869, 5163),
  gender = c("F", "M", "F", "M"),
  gender_sizes = c(939, 930, 2544, 2619)
)

ggplot() +
  # Outer donut (Churn)
  geom_bar(data = churn_gender_data[1:2,],
           aes(x = 2, y = value, fill = category),
           stat = "identity", width = 2) +
  # Inner donut (Gender)
  geom_bar(data = data.frame(gender = c("F", "M", "F", "M"),
                             value = c(939, 930, 2544, 2619)),
           aes(x = 1, y = value, fill = gender),
           stat = "identity", width = 2) +
  coord_polar(theta = "y") +
  theme_void() +
  scale_fill_manual(values = c("#ff6666", "#66b3ff", "#c2c2f0", "#ffb3e6")) +
  ggtitle("Churn Distribution w.r.t Gender: Male(M), Female(F)") +
  theme(plot.title = element_text(hjust = 0.5, size = 15))

# Method 1: Using basic ggplot2
ggplot(df, aes(x = Churn, fill = Contract)) +
  geom_bar(position = "dodge") +
  labs(title = "Customer Contract Distribution") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")

# Payment Method Distribution
plot_ly(data = as.data.frame(table(df$PaymentMethod)),
        labels = ~Var1,
        values = ~Freq,
        type = 'pie',
        hole = 0.3) %>%
  layout(title = "Payment Method Distribution")

# Payment Method vs Churn
ggplot(df, aes(x = Churn, fill = PaymentMethod)) +
  geom_bar(position = "dodge") +
  ggtitle("Customer Payment Method Distribution w.r.t. Churn") +
  theme_minimal()+
  scale_fill_brewer(palette = "Set1")

# Internet Service Analysis
# Unique values
unique(df$InternetService)

# Count by gender and churn
male_counts <- table(df$InternetService[df$gender == "Male"], 
                     df$Churn[df$gender == "Male"])
female_counts <- table(df$InternetService[df$gender == "Female"], 
                       df$Churn[df$gender == "Female"])

# Internet Service by Gender and Churn
internet_data <- data.frame(
  Churn = rep(c("No", "No", "Yes", "Yes"), 3),
  Gender = rep(c("Female", "Male", "Female", "Male"), 3),
  Service = rep(c("DSL", "Fiber optic", "No Internet"), each = 4),
  Count = c(965, 992, 219, 240,  # DSL
            889, 910, 664, 633,   # Fiber optic
            690, 717, 56, 57)     # No Internet
)


ggplot(internet_data, aes(x = interaction(Churn, Gender), y = Count, fill = Service)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Churn Distribution w.r.t. Internet Service and Gender") +
  theme_minimal()+
  scale_fill_brewer(palette = "Set4")

# Dependents Distribution
ggplot(df, aes(x = Churn, fill = Dependents)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#AB63FA", "#FF97FF")) +
  ggtitle("Dependents Distribution") +
  theme_minimal()

# Partner Distribution

ggplot(df, aes(x = Churn, fill = Partner)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#00CC96", "#FFA15A")) +
  ggtitle("Churn Distribution w.r.t. Partners") +
  theme_minimal()


# Senior Citizen Distribution
ggplot(df, aes(x = Churn, fill = SeniorCitizen)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#B6E880", "#00CC96")) +
  labs(title = "Churn Distribution w.r.t. Senior Citizen") +
  theme_minimal()

# Alternative with more customization
ggplot(df, aes(x = Churn, fill = OnlineSecurity)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("No" = "#AB63FA", 
                               "Yes" = "#FF97FF", 
                               "No internet service" = "#BFBFBB")) +
  labs(title = "Churn Distribution w.r.t. Online Security",
       x = "Churn",
       y = "Count",
       fill = "Online Security") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    legend.position = "right"
  )

# Paperless Billing Distribution

ggplot(df, aes(x = Churn, fill = PaperlessBilling)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#00CC96", "#FFA15A")) +
  labs(title = "Churn Distribution w.r.t. Paperless Billing") +
  theme_minimal()

# Tech Support Distribution
ggplot(df, aes(x = Churn, fill = TechSupport)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn Distribution w.r.t. TechSupport") +
  theme_minimal()

# Phone Service Distribution

ggplot(df, aes(x = Churn, fill = PhoneService)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#B6E880", "#00CC96")) +
  labs(title = "Churn Distribution w.r.t. Phone Service") +
  theme_minimal()


# Monthly Charges Density Plot
ggplot(df, aes(x = MonthlyCharges, fill = Churn)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("Red", "Blue")) +
  labs(title = "Distribution of Monthly Charges by Churn",
       x = "Monthly Charges",
       y = "Density") +
  theme_minimal()

# Total Charges Density Plot
ggplot(df, aes(x = TotalCharges, fill = Churn)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("Gold", "Green")) +
  labs(title = "Distribution of Total Charges by Churn",
       x = "Total Charges",
       y = "Density") +
  theme_minimal()

# Tenure Box Plot
plot_ly(df, x = ~Churn, y = ~tenure, type = "box") %>%
  layout(title = "Tenure vs Churn",
         xaxis = list(title = "Churn"),
         yaxis = list(title = "Tenure (Months)"),
         width = 750, height = 600)

# Correlation Matrix
# First convert categorical variables to numeric
df_numeric <- df %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)

# Calculate correlation matrix
corr_matrix <- cor(df_numeric, use = "complete.obs")

# Correlation heatmap using ggplot2
ggplot(data = reshape2::melt(corr_matrix), aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Matrix",
       x = "",
       y = "") +
  coord_fixed()

# Function to convert categorical variables to numeric (Label Encoding)
object_to_int <- function(x) {
  if(is.character(x) || is.factor(x)) {
    return(as.numeric(factor(x)) - 1)  # Subtract 1 to match Python's 0-based encoding
  }
  return(x)
}

# Apply label encoding to all columns
df_encoded <- df %>%
  mutate_all(object_to_int)

# Show first few rows
head(df_encoded)

# Correlation with Churn (sorted)
correlations <- cor(df_encoded)[,'Churn']
correlations_sorted <- sort(correlations, decreasing = TRUE)
print(correlations_sorted)

# Visualize correlations
ggplot(data = data.frame(
  variable = names(correlations_sorted),
  correlation = correlations_sorted
), aes(x = reorder(variable, correlation), y = correlation)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Correlations with Churn",
       x = "Variables",
       y = "Correlation")

# Split features and target
X <- df_encoded %>% select(-Churn)
y <- df_encoded$Churn

# Split data into training and testing sets
set.seed(40)  # for reproducibility
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Function to create distribution plots
plot_distribution <- function(feature, data, color = "red") {
  ggplot(data, aes_string(x = feature)) +
    geom_density(fill = color, alpha = 0.5) +
    theme_minimal() +
    ggtitle(paste("Distribution for", feature))
}

# Plot distributions for numeric columns
num_cols <- c("tenure", "MonthlyCharges", "TotalCharges")
plots <- lapply(num_cols, function(col) {
  plot_distribution(col, df_encoded)
})

# Display plots in a grid
gridExtra::grid.arrange(grobs = plots, ncol = 2)

# Standardize numeric columns
# Create preprocessing object
preproc <- preProcess(df_encoded[num_cols], method = c("center", "scale"))

# Apply standardization
df_std <- predict(preproc, df_encoded[num_cols])

# Plot standardized distributions
std_plots <- lapply(num_cols, function(col) {
  plot_distribution(col, df_std, color = "cyan")
})

# Display standardized plots in a grid
gridExtra::grid.arrange(grobs = std_plots, ncol = 2)

# Define columns for different encoding methods
cat_cols_ohe <- c('PaymentMethod', 'Contract', 'InternetService')
cat_cols_le <- setdiff(
  setdiff(names(X_train), num_cols),
  cat_cols_ohe
)

# Standardize numeric columns in training and test sets
X_train[num_cols] <- predict(preproc, X_train[num_cols])
X_test[num_cols] <- predict(preproc, X_test[num_cols])

# Function to perform one-hot encoding
perform_ohe <- function(data, columns) {
  # Create dummy variables
  dummies <- dummyVars(~ ., data = data[columns])
  encoded <- predict(dummies, newdata = data[columns])
  
  # Convert to data frame
  encoded_df <- as.data.frame(encoded)
  
  # Remove original columns and add encoded ones
  data <- data %>%
    select(-all_of(columns)) %>%
    bind_cols(encoded_df)
  
  return(data)
}

# Apply one-hot encoding
X_train <- perform_ohe(X_train, cat_cols_ohe)
X_test <- perform_ohe(X_test, cat_cols_ohe)

# Create final preprocessed datasets
train_data <- data.frame(X_train, Churn = y_train)
test_data <- data.frame(X_test, Churn = y_test)


# KNN
knn_pred <- knn(train_data[, -ncol(train_data)], test_data[, -ncol(test_data)], train_data$Churn, k = 5)
knn_conf_matrix <- confusionMatrix(as.factor(knn_pred), as.factor(test_data$Churn))
print(knn_conf_matrix)

# SVC
svc_model <- svm(Churn ~ ., data = train_data, kernel = "radial")
svc_pred <- predict(svc_model, test_data)
svc_conf_matrix <- confusionMatrix(factor(svc_pred, levels = unique(train_data$Churn)), factor(test_data$Churn, levels = unique(train_data$Churn)))
print(svc_conf_matrix)

# Random Forest
rf_model <- randomForest(Churn ~ ., data = train_data)
rf_pred <- predict(rf_model, test_data)
rf_conf_matrix <- confusionMatrix(factor(rf_pred, levels = unique(train_data$Churn)), factor(test_data$Churn, levels = unique(train_data$Churn)))
print(rf_conf_matrix)

#Plot ROC Curve
train_data$Churn <- as.factor(train_data$Churn)
test_data$Churn <- as.factor(test_data$Churn)

# Retrain Random Forest model with Churn as a factor
rf_model <- randomForest(Churn ~ ., data = train_data)

# Get predicted probabilities for the positive class
rf_prob <- predict(rf_model, test_data, type = "prob")[, 2]

# Plot ROC curve
roc_curve <- roc(test_data$Churn, rf_prob)
plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for RandomForest Model")
abline(a = 0, b = 1, col = "gray", lty = 2)  # Add diagonal line for reference

# Add AUC to the plot
auc <- auc(roc_curve)
text(0.6, 0.4, paste("AUC =", round(auc, 2)), col = "blue", cex = 1.2)

# Perform logistic regression
# Fit logistic regression model
#logit_model <- glm(Churn ~ ., data = train_data, family = "binomial")

# Predict probabilities for the test set
#logit_prob <- predict(logit_model, test_data, type = "response")

# Convert probabilities to binary predictions (using 0.5 as threshold)
#logit_pred <- ifelse(logit_prob > 0.5, "Yes", "No")

# Convert predictions to factors to match the actual values
#logit_pred <- factor(logit_pred, levels = levels(test_data$Churn))

# Print confusion matrix
#logit_conf_matrix <- confusionMatrix(logit_pred, test_data$Churn)
#print(logit_conf_matrix)

#roc_curve <- roc(test_data$Churn, logit_prob)
#plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Logistic Regression Model")
#abline(a = 0, b = 1, col = "gray", lty = 2)  # Add diagonal line for reference

# Add AUC to the plot
#auc <- auc(roc_curve)
#text(0.6, 0.4, paste("AUC =", round(auc, 2)), col = "blue", cex = 1.2)

# Fit a Decision Tree model
dt_model <- rpart(Churn ~ ., data = train_data, method = "class")

# Predict on the test set
dt_pred <- predict(dt_model, test_data, type = "class")

# Print the confusion matrix
dt_conf_matrix <- confusionMatrix(dt_pred, test_data$Churn)
print(dt_conf_matrix)
