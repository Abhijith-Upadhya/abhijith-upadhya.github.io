Abhijith
2024-11-06

# **Customer Churn Analysis**

<figure>
<img
src="https://www.voxco.com/wp-content/uploads/2021/09/Everything-you-need-to-know-about-Customer-Churn2.jpg"
height="600" alt="Source: Voxco.com" />
<figcaption aria-hidden="true">Source: Voxco.com</figcaption>
</figure>

## Introduction

The goal of this analysis is to understand the factors influencing
customer churn in a telecom dataset. We aim to explore and visualize
various features, inspect patterns associated with churn, and assess the
relationships between churn and various demographic, service-related,
and financial characteristics. By doing so, we aim to identify key
factors contributing to churn and leverage these insights in building
predictive models that can help in customer retention efforts.

The analysis starts by examining the dataset structure and handling
missing values, followed by visualizing data distributions across
several categorical and numerical variables. Subsequently, machine
learning models are employed to predict churn, with performance
comparisons among various classifiers, including K-Nearest Neighbors
(KNN), Support Vector Classifier (SVC), Random Forest, Logistic
Regression, and Decision Tree models.

## Libraries

``` r
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
library(pROC)       # for ROC curves
library(plotly)
library(gridExtra)
library(ggplot2)
library(scales)
library(corrplot)
library(ggridges)
library(tidyr)
library(reshape2)
```

## Loading the dataset

``` r
# Load and examine data
df <- read.csv('Telecom_Customer_Churn.csv')
head(df)
```

    ##   customerID gender SeniorCitizen Partner Dependents tenure PhoneService
    ## 1 7590-VHVEG Female             0     Yes         No      1           No
    ## 2 5575-GNVDE   Male             0      No         No     34          Yes
    ## 3 3668-QPYBK   Male             0      No         No      2          Yes
    ## 4 7795-CFOCW   Male             0      No         No     45           No
    ## 5 9237-HQITU Female             0      No         No      2          Yes
    ## 6 9305-CDSKC Female             0      No         No      8          Yes
    ##      MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection
    ## 1 No phone service             DSL             No          Yes               No
    ## 2               No             DSL            Yes           No              Yes
    ## 3               No             DSL            Yes          Yes               No
    ## 4 No phone service             DSL            Yes           No              Yes
    ## 5               No     Fiber optic             No           No               No
    ## 6              Yes     Fiber optic             No           No              Yes
    ##   TechSupport StreamingTV StreamingMovies       Contract PaperlessBilling
    ## 1          No          No              No Month-to-month              Yes
    ## 2          No          No              No       One year               No
    ## 3          No          No              No Month-to-month              Yes
    ## 4         Yes          No              No       One year               No
    ## 5          No          No              No Month-to-month              Yes
    ## 6          No         Yes             Yes Month-to-month              Yes
    ##               PaymentMethod MonthlyCharges TotalCharges Churn
    ## 1          Electronic check          29.85        29.85    No
    ## 2              Mailed check          56.95      1889.50    No
    ## 3              Mailed check          53.85       108.15   Yes
    ## 4 Bank transfer (automatic)          42.30      1840.75    No
    ## 5          Electronic check          70.70       151.65   Yes
    ## 6          Electronic check          99.65       820.50   Yes

``` r
dim(df)
```

    ## [1] 7043   21

``` r
str(df)
```

    ## 'data.frame':    7043 obs. of  21 variables:
    ##  $ customerID      : chr  "7590-VHVEG" "5575-GNVDE" "3668-QPYBK" "7795-CFOCW" ...
    ##  $ gender          : chr  "Female" "Male" "Male" "Male" ...
    ##  $ SeniorCitizen   : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ Partner         : chr  "Yes" "No" "No" "No" ...
    ##  $ Dependents      : chr  "No" "No" "No" "No" ...
    ##  $ tenure          : int  1 34 2 45 2 8 22 10 28 62 ...
    ##  $ PhoneService    : chr  "No" "Yes" "Yes" "No" ...
    ##  $ MultipleLines   : chr  "No phone service" "No" "No" "No phone service" ...
    ##  $ InternetService : chr  "DSL" "DSL" "DSL" "DSL" ...
    ##  $ OnlineSecurity  : chr  "No" "Yes" "Yes" "Yes" ...
    ##  $ OnlineBackup    : chr  "Yes" "No" "Yes" "No" ...
    ##  $ DeviceProtection: chr  "No" "Yes" "No" "Yes" ...
    ##  $ TechSupport     : chr  "No" "No" "No" "Yes" ...
    ##  $ StreamingTV     : chr  "No" "No" "No" "No" ...
    ##  $ StreamingMovies : chr  "No" "No" "No" "No" ...
    ##  $ Contract        : chr  "Month-to-month" "One year" "Month-to-month" "One year" ...
    ##  $ PaperlessBilling: chr  "Yes" "No" "Yes" "No" ...
    ##  $ PaymentMethod   : chr  "Electronic check" "Mailed check" "Mailed check" "Bank transfer (automatic)" ...
    ##  $ MonthlyCharges  : num  29.9 57 53.9 42.3 70.7 ...
    ##  $ TotalCharges    : num  29.9 1889.5 108.2 1840.8 151.7 ...
    ##  $ Churn           : chr  "No" "No" "Yes" "No" ...

``` r
colnames(df)
```

    ##  [1] "customerID"       "gender"           "SeniorCitizen"    "Partner"         
    ##  [5] "Dependents"       "tenure"           "PhoneService"     "MultipleLines"   
    ##  [9] "InternetService"  "OnlineSecurity"   "OnlineBackup"     "DeviceProtection"
    ## [13] "TechSupport"      "StreamingTV"      "StreamingMovies"  "Contract"        
    ## [17] "PaperlessBilling" "PaymentMethod"    "MonthlyCharges"   "TotalCharges"    
    ## [21] "Churn"

``` r
sapply(df, class)
```

    ##       customerID           gender    SeniorCitizen          Partner 
    ##      "character"      "character"        "integer"      "character" 
    ##       Dependents           tenure     PhoneService    MultipleLines 
    ##      "character"        "integer"      "character"      "character" 
    ##  InternetService   OnlineSecurity     OnlineBackup DeviceProtection 
    ##      "character"      "character"      "character"      "character" 
    ##      TechSupport      StreamingTV  StreamingMovies         Contract 
    ##      "character"      "character"      "character"      "character" 
    ## PaperlessBilling    PaymentMethod   MonthlyCharges     TotalCharges 
    ##      "character"      "character"        "numeric"        "numeric" 
    ##            Churn 
    ##      "character"

## Data Transformation

### Data Preparation and Cleaning

``` r
# Drop customerID column
df <- df %>% select(-customerID)
head(df)
```

    ##   gender SeniorCitizen Partner Dependents tenure PhoneService    MultipleLines
    ## 1 Female             0     Yes         No      1           No No phone service
    ## 2   Male             0      No         No     34          Yes               No
    ## 3   Male             0      No         No      2          Yes               No
    ## 4   Male             0      No         No     45           No No phone service
    ## 5 Female             0      No         No      2          Yes               No
    ## 6 Female             0      No         No      8          Yes              Yes
    ##   InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport
    ## 1             DSL             No          Yes               No          No
    ## 2             DSL            Yes           No              Yes          No
    ## 3             DSL            Yes          Yes               No          No
    ## 4             DSL            Yes           No              Yes         Yes
    ## 5     Fiber optic             No           No               No          No
    ## 6     Fiber optic             No           No              Yes          No
    ##   StreamingTV StreamingMovies       Contract PaperlessBilling
    ## 1          No              No Month-to-month              Yes
    ## 2          No              No       One year               No
    ## 3          No              No Month-to-month              Yes
    ## 4          No              No       One year               No
    ## 5          No              No Month-to-month              Yes
    ## 6         Yes             Yes Month-to-month              Yes
    ##               PaymentMethod MonthlyCharges TotalCharges Churn
    ## 1          Electronic check          29.85        29.85    No
    ## 2              Mailed check          56.95      1889.50    No
    ## 3              Mailed check          53.85       108.15   Yes
    ## 4 Bank transfer (automatic)          42.30      1840.75    No
    ## 5          Electronic check          70.70       151.65   Yes
    ## 6          Electronic check          99.65       820.50   Yes

``` r
# Convert TotalCharges to numeric
df$TotalCharges <- as.numeric(as.character(df$TotalCharges))

# Check missing values
colSums(is.na(df))
```

    ##           gender    SeniorCitizen          Partner       Dependents 
    ##                0                0                0                0 
    ##           tenure     PhoneService    MultipleLines  InternetService 
    ##                0                0                0                0 
    ##   OnlineSecurity     OnlineBackup DeviceProtection      TechSupport 
    ##                0                0                0                0 
    ##      StreamingTV  StreamingMovies         Contract PaperlessBilling 
    ##                0                0                0                0 
    ##    PaymentMethod   MonthlyCharges     TotalCharges            Churn 
    ##                0                0               11                0

``` r
# View rows where TotalCharges is NA
df[is.na(df$TotalCharges), ]
```

    ##      gender SeniorCitizen Partner Dependents tenure PhoneService
    ## 489  Female             0     Yes        Yes      0           No
    ## 754    Male             0      No        Yes      0          Yes
    ## 937  Female             0     Yes        Yes      0          Yes
    ## 1083   Male             0     Yes        Yes      0          Yes
    ## 1341 Female             0     Yes        Yes      0           No
    ## 3332   Male             0     Yes        Yes      0          Yes
    ## 3827   Male             0     Yes        Yes      0          Yes
    ## 4381 Female             0     Yes        Yes      0          Yes
    ## 5219   Male             0     Yes        Yes      0          Yes
    ## 6671 Female             0     Yes        Yes      0          Yes
    ## 6755   Male             0      No        Yes      0          Yes
    ##         MultipleLines InternetService      OnlineSecurity        OnlineBackup
    ## 489  No phone service             DSL                 Yes                  No
    ## 754                No              No No internet service No internet service
    ## 937                No             DSL                 Yes                 Yes
    ## 1083              Yes              No No internet service No internet service
    ## 1341 No phone service             DSL                 Yes                 Yes
    ## 3332               No              No No internet service No internet service
    ## 3827              Yes              No No internet service No internet service
    ## 4381               No              No No internet service No internet service
    ## 5219               No              No No internet service No internet service
    ## 6671              Yes             DSL                  No                 Yes
    ## 6755              Yes             DSL                 Yes                 Yes
    ##         DeviceProtection         TechSupport         StreamingTV
    ## 489                  Yes                 Yes                 Yes
    ## 754  No internet service No internet service No internet service
    ## 937                  Yes                  No                 Yes
    ## 1083 No internet service No internet service No internet service
    ## 1341                 Yes                 Yes                 Yes
    ## 3332 No internet service No internet service No internet service
    ## 3827 No internet service No internet service No internet service
    ## 4381 No internet service No internet service No internet service
    ## 5219 No internet service No internet service No internet service
    ## 6671                 Yes                 Yes                 Yes
    ## 6755                  No                 Yes                  No
    ##          StreamingMovies Contract PaperlessBilling             PaymentMethod
    ## 489                   No Two year              Yes Bank transfer (automatic)
    ## 754  No internet service Two year               No              Mailed check
    ## 937                  Yes Two year               No              Mailed check
    ## 1083 No internet service Two year               No              Mailed check
    ## 1341                  No Two year               No   Credit card (automatic)
    ## 3332 No internet service Two year               No              Mailed check
    ## 3827 No internet service Two year               No              Mailed check
    ## 4381 No internet service Two year               No              Mailed check
    ## 5219 No internet service One year              Yes              Mailed check
    ## 6671                  No Two year               No              Mailed check
    ## 6755                  No Two year              Yes Bank transfer (automatic)
    ##      MonthlyCharges TotalCharges Churn
    ## 489           52.55           NA    No
    ## 754           20.25           NA    No
    ## 937           80.85           NA    No
    ## 1083          25.75           NA    No
    ## 1341          56.05           NA    No
    ## 3332          19.85           NA    No
    ## 3827          25.35           NA    No
    ## 4381          20.00           NA    No
    ## 5219          19.70           NA    No
    ## 6671          73.35           NA    No
    ## 6755          61.90           NA    No

``` r
# Find and remove rows where tenure is 0
zero_tenure_indices <- which(df$tenure == 0)
df <- df[-zero_tenure_indices, ]

# Verify removal
sum(df$tenure == 0)
```

    ## [1] 0

``` r
# Fill NA values with mean
df$TotalCharges[is.na(df$TotalCharges)] <- mean(df$TotalCharges, na.rm = TRUE)

# Check missing values again
colSums(is.na(df))
```

    ##           gender    SeniorCitizen          Partner       Dependents 
    ##                0                0                0                0 
    ##           tenure     PhoneService    MultipleLines  InternetService 
    ##                0                0                0                0 
    ##   OnlineSecurity     OnlineBackup DeviceProtection      TechSupport 
    ##                0                0                0                0 
    ##      StreamingTV  StreamingMovies         Contract PaperlessBilling 
    ##                0                0                0                0 
    ##    PaymentMethod   MonthlyCharges     TotalCharges            Churn 
    ##                0                0                0                0

``` r
# Map SeniorCitizen values
df$SeniorCitizen <- ifelse(df$SeniorCitizen == 0, "No", "Yes")
head(df)
```

    ##   gender SeniorCitizen Partner Dependents tenure PhoneService    MultipleLines
    ## 1 Female            No     Yes         No      1           No No phone service
    ## 2   Male            No      No         No     34          Yes               No
    ## 3   Male            No      No         No      2          Yes               No
    ## 4   Male            No      No         No     45           No No phone service
    ## 5 Female            No      No         No      2          Yes               No
    ## 6 Female            No      No         No      8          Yes              Yes
    ##   InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport
    ## 1             DSL             No          Yes               No          No
    ## 2             DSL            Yes           No              Yes          No
    ## 3             DSL            Yes          Yes               No          No
    ## 4             DSL            Yes           No              Yes         Yes
    ## 5     Fiber optic             No           No               No          No
    ## 6     Fiber optic             No           No              Yes          No
    ##   StreamingTV StreamingMovies       Contract PaperlessBilling
    ## 1          No              No Month-to-month              Yes
    ## 2          No              No       One year               No
    ## 3          No              No Month-to-month              Yes
    ## 4          No              No       One year               No
    ## 5          No              No Month-to-month              Yes
    ## 6         Yes             Yes Month-to-month              Yes
    ##               PaymentMethod MonthlyCharges TotalCharges Churn
    ## 1          Electronic check          29.85        29.85    No
    ## 2              Mailed check          56.95      1889.50    No
    ## 3              Mailed check          53.85       108.15   Yes
    ## 4 Bank transfer (automatic)          42.30      1840.75    No
    ## 5          Electronic check          70.70       151.65   Yes
    ## 6          Electronic check          99.65       820.50   Yes

``` r
# Describe InternetService column
summary(df$InternetService)
```

    ##    Length     Class      Mode 
    ##      7032 character character

``` r
table(df$InternetService)
```

    ## 
    ##         DSL Fiber optic          No 
    ##        2416        3096        1520

``` r
# Define numerical columns and get summary statistics
numerical_cols <- c('tenure', 'MonthlyCharges', 'TotalCharges')
summary(df[numerical_cols])       # For missing value handling alternatives
```

    ##      tenure      MonthlyCharges    TotalCharges   
    ##  Min.   : 1.00   Min.   : 18.25   Min.   :  18.8  
    ##  1st Qu.: 9.00   1st Qu.: 35.59   1st Qu.: 401.4  
    ##  Median :29.00   Median : 70.35   Median :1397.5  
    ##  Mean   :32.42   Mean   : 64.80   Mean   :2283.3  
    ##  3rd Qu.:55.00   3rd Qu.: 89.86   3rd Qu.:3794.7  
    ##  Max.   :72.00   Max.   :118.75   Max.   :8684.8

## Data Visualisation and Analysis

### Gender and Churn Distribution (Donut Charts):

Gender Distribution: Displays the proportions of male and female
customers, providing a demographic breakdown. Churn Distribution: Shows
the proportion of customers who churned versus those who stayed.
Visualizing these together gives an understanding of the churn rate
within each gender group.

``` r
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
```

<div class="plotly html-widget html-fill-item-overflow-hidden html-fill-item" id="htmlwidget-b31ff10fa935ac13e767" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-b31ff10fa935ac13e767">{"x":{"data":[{"values":[3483,3549],"labels":["Female","Male"],"type":"pie","hole":0.40000000000000002,"name":"Gender","marker":{"color":"rgba(31,119,180,1)","line":{"color":"rgba(255,255,255,1)"}},"frame":null},{"values":[5163,1869],"labels":["No","Yes"],"type":"pie","hole":0.40000000000000002,"name":"Churn","marker":{"color":"rgba(255,127,14,1)","line":{"color":"rgba(255,255,255,1)"}},"frame":null}],"layout":{"NA":{"anchor":[],"domain":[0,1]},"NA2":{"anchor":[],"domain":[0,1]},"annotations":[{"text":"Gender","x":0.5,"y":0.5,"showarrow":false},{"text":"Churn","x":0.5,"y":0.5,"showarrow":false}],"margin":{"b":40,"l":60,"t":25,"r":10},"title":"Churn Distribution","hovermode":"closest","showlegend":true},"attrs":{"33642e5d1097":{"alpha_stroke":1,"sizes":[10,100],"spans":[1,20],"values":{},"labels":{},"type":"pie","hole":0.40000000000000002,"name":"Gender","inherit":true},"336448b137f9":{"alpha_stroke":1,"sizes":[10,100],"spans":[1,20],"values":{},"labels":{},"type":"pie","hole":0.40000000000000002,"name":"Churn","inherit":true}},"source":"A","config":{"modeBarButtonsToAdd":["hoverclosest","hovercompare"],"showSendToCloud":false},"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.20000000000000001,"selected":{"opacity":1},"debounce":0},"subplot":true,"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script>

``` r
# Churn counts by gender
churn_no <- table(df$gender[df$Churn == "No"])
churn_yes <- table(df$gender[df$Churn == "Yes"])
```

### Customer Contract Distribution

This bar plot categorizes customers by their contract type and shows the
churn rate within each category. It illustrates how contract length
might impact customer retention.

``` r
# Method 1: Using basic ggplot2
ggplot(df, aes(x = Churn, fill = Contract)) +
  geom_bar(position = "dodge") +
  labs(title = "Customer Contract Distribution") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")
```

![](README_figs/README-unnamed-chunk-2-1.png)<!-- -->

### Payment Method Distribution:

Visualized as a pie chart, this plot shows the distribution of different
payment methods and their association with churn. Understanding payment
preferences can help in customizing retention strategies.

``` r
# Payment Method Distribution
plot_ly(data = as.data.frame(table(df$PaymentMethod)),
        labels = ~Var1,
        values = ~Freq,
        type = 'pie',
        hole = 0.3) %>%
  layout(title = "Payment Method Distribution")
```

<div class="plotly html-widget html-fill-item-overflow-hidden html-fill-item" id="htmlwidget-bc19c64175500a5f7ac8" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-bc19c64175500a5f7ac8">{"x":{"visdat":{"33643f307411":["function () ","plotlyVisDat"]},"cur_data":"33643f307411","attrs":{"33643f307411":{"labels":{},"values":{},"hole":0.29999999999999999,"alpha_stroke":1,"sizes":[10,100],"spans":[1,20],"type":"pie"}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"title":"Payment Method Distribution","hovermode":"closest","showlegend":true},"source":"A","config":{"modeBarButtonsToAdd":["hoverclosest","hovercompare"],"showSendToCloud":false},"data":[{"labels":["Bank transfer (automatic)","Credit card (automatic)","Electronic check","Mailed check"],"values":[1542,1521,2365,1604],"hole":0.29999999999999999,"type":"pie","marker":{"color":"rgba(31,119,180,1)","line":{"color":"rgba(255,255,255,1)"}},"frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.20000000000000001,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script>

### Customer Payment Method Distribution w.r.t. Churn

``` r
# Payment Method vs Churn
ggplot(df, aes(x = Churn, fill = PaymentMethod)) +
  geom_bar(position = "dodge") +
  ggtitle("Customer Payment Method Distribution w.r.t. Churn") +
  theme_minimal()+
  scale_fill_brewer(palette = "Set1")
```

![](README_figs/README-unnamed-chunk-3-1.png)<!-- -->

### Internet Service by Gender and Churn:

A grouped bar plot displays churn distribution based on internet service
type across genders. This plot helps assess whether certain internet
services are associated with higher churn rates.

``` r
# Internet Service Analysis
# Unique values
unique(df$InternetService)
```

    ## [1] "DSL"         "Fiber optic" "No"

``` r
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
```

![](README_figs/README-internetservice_analysis-1.png)<!-- -->

### Dependents, Partner, and Senior Citizen Distributions

- Dependents Distribution: Shows the breakdown of churn among customers
  with and without dependents.
- Partner Distribution: This plot examines the influence of having a
  partner on churn likelihood.
- Senior Citizen Distribution: Displays churn rates among senior versus
  non-senior customers.

``` r
# Dependents Distribution
ggplot(df, aes(x = Churn, fill = Dependents)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#AB63FA", "#FF97FF")) +
  ggtitle("Dependents Distribution") +
  theme_minimal()
```

![](README_figs/README-unnamed-chunk-4-1.png)<!-- -->

``` r
# Partner Distribution

ggplot(df, aes(x = Churn, fill = Partner)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#00CC96", "#FFA15A")) +
  ggtitle("Churn Distribution w.r.t. Partners") +
  theme_minimal()
```

![](README_figs/README-unnamed-chunk-4-2.png)<!-- -->

``` r
# Senior Citizen Distribution
ggplot(df, aes(x = Churn, fill = SeniorCitizen)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#B6E880", "#00CC96")) +
  labs(title = "Churn Distribution w.r.t. Senior Citizen") +
  theme_minimal()
```

![](README_figs/README-unnamed-chunk-4-3.png)<!-- -->

### Churn Distribution with Online Security

Illustrates the role of online security in customer retention. Churn
rates for customers with online security services are compared to those
without.

``` r
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
```

![](README_figs/README-churn_onlinesecurity-1.png)<!-- -->

### Paperless Billing, Tech Support, and Phone Service

Each plot visualizes churn rates for customers who opted for these
services, showing if certain service features correlate with churn
tendencies.

``` r
# Paperless Billing Distribution

ggplot(df, aes(x = Churn, fill = PaperlessBilling)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#00CC96", "#FFA15A")) +
  labs(title = "Churn Distribution w.r.t. Paperless Billing") +
  theme_minimal()
```

![](README_figs/README-paperless_tech_phone-1.png)<!-- -->

``` r
# Tech Support Distribution
ggplot(df, aes(x = Churn, fill = TechSupport)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn Distribution w.r.t. TechSupport") +
  theme_minimal()
```

![](README_figs/README-paperless_tech_phone-2.png)<!-- -->

``` r
# Phone Service Distribution

ggplot(df, aes(x = Churn, fill = PhoneService)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#B6E880", "#00CC96")) +
  labs(title = "Churn Distribution w.r.t. Phone Service") +
  theme_minimal()
```

![](README_figs/README-paperless_tech_phone-3.png)<!-- -->

### Monthly Charges Density Plot

This density plot shows the distribution of monthly charges among
churned and retained customers, highlighting any differences in spending
patterns that correlate with churn.

``` r
# Monthly Charges Density Plot
ggplot(df, aes(x = MonthlyCharges, fill = Churn)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("Red", "Blue")) +
  labs(title = "Distribution of Monthly Charges by Churn",
       x = "Monthly Charges",
       y = "Density") +
  theme_minimal()
```

![](README_figs/README-monthlydensity-1.png)<!-- -->

### Total Charges Density Plot

Similar to monthly charges, this plot shows the distribution of total
charges among churned and non-churned customers, allowing for insights
into long-term customer spending.

``` r
# Total Charges Density Plot
ggplot(df, aes(x = TotalCharges, fill = Churn)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("Gold", "Green")) +
  labs(title = "Distribution of Total Charges by Churn",
       x = "Total Charges",
       y = "Density") +
  theme_minimal()
```

![](README_figs/README-totaldensity-1.png)<!-- -->

### Tenure vs. Churn

A box plot compares tenure between churned and retained customers,
providing a visual summary of how customer longevity might influence
churn.

``` r
# Tenure Box Plot
plot_ly(df, x = ~Churn, y = ~tenure, type = "box") %>%
  layout(title = "Tenure vs Churn",
         xaxis = list(title = "Churn"),
         yaxis = list(title = "Tenure (Months)"),
         width = 750, height = 600)
```

<div class="plotly html-widget html-fill-item-overflow-hidden html-fill-item" id="htmlwidget-7de04487eedb9eee5902" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-7de04487eedb9eee5902">{"x":{"visdat":{"33645fdb3785":["function () ","plotlyVisDat"]},"cur_data":"33645fdb3785","attrs":{"33645fdb3785":{"x":{},"y":{},"alpha_stroke":1,"sizes":[10,100],"spans":[1,20],"type":"box"}},"layout":{"width":750,"height":600,"margin":{"b":40,"l":60,"t":25,"r":10},"title":"Tenure vs Churn","xaxis":{"domain":[0,1],"automargin":true,"title":"Churn","type":"category","categoryorder":"array","categoryarray":["No","Yes"]},"yaxis":{"domain":[0,1],"automargin":true,"title":"Tenure (Months)"},"hovermode":"closest","showlegend":false},"source":"A","config":{"modeBarButtonsToAdd":["hoverclosest","hovercompare"],"showSendToCloud":false},"data":[{"fillcolor":"rgba(31,119,180,0.5)","x":["No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No"],"y":[1,34,2,45,2,8,22,10,28,62,13,16,58,49,25,69,52,71,10,21,1,12,1,58,49,30,47,1,72,17,71,2,27,1,1,72,5,46,34,11,10,70,17,63,13,49,2,2,52,69,43,15,25,8,60,18,63,66,34,72,47,60,72,18,9,3,47,31,50,10,1,52,64,62,3,56,46,8,30,45,1,11,7,42,49,9,35,48,46,29,30,1,66,65,72,12,71,5,52,25,1,1,38,66,68,5,72,32,43,72,55,52,43,37,64,3,36,10,41,27,56,6,3,7,4,33,27,72,1,71,13,25,67,1,2,43,23,64,57,1,72,8,61,64,71,65,3,1,30,15,8,7,70,62,6,14,22,22,16,10,13,20,2,53,11,69,4,72,58,16,43,2,14,53,32,34,15,7,15,61,1,1,8,33,13,1,20,3,13,40,43,6,69,72,59,20,24,59,72,1,27,14,71,13,44,33,72,1,19,64,2,1,61,29,23,57,72,66,65,8,4,71,1,4,12,24,31,1,30,47,54,50,1,72,29,2,10,18,11,16,72,72,41,65,13,4,41,15,1,42,51,2,1,32,10,67,61,50,2,29,3,13,57,31,45,61,50,19,59,71,16,57,1,20,1,5,52,21,14,5,6,10,1,68,18,22,20,1,8,10,24,35,23,6,12,1,71,35,40,1,23,4,4,68,38,52,32,29,38,48,1,22,43,5,5,51,71,38,24,35,54,72,1,9,69,52,11,2,28,17,35,8,46,7,2,68,43,68,36,63,32,71,66,63,41,1,2,70,23,64,37,17,7,4,21,10,16,64,27,42,5,41,58,47,18,5,23,1,71,72,33,2,24,56,37,43,1,25,61,17,41,1,72,1,48,11,55,42,44,1,27,27,2,19,42,66,33,34,33,23,32,11,69,68,20,72,60,32,1,1,3,46,29,51,48,16,70,40,22,1,5,7,29,44,10,55,52,10,18,68,61,72,2,12,41,26,36,72,35,1,16,49,54,18,36,60,1,52,8,72,64,22,60,28,61,24,28,30,2,1,6,24,4,7,72,70,64,72,44,13,17,1,9,24,1,24,35,7,5,15,11,48,20,72,8,72,15,72,1,63,2,2,61,1,22,28,70,5,12,34,71,70,52,69,20,11,2,6,1,20,61,5,56,30,40,28,5,27,12,67,29,55,23,34,52,72,58,35,56,24,70,2,68,1,12,63,33,69,60,72,11,1,10,13,34,39,65,50,15,72,72,55,23,32,56,1,38,11,1,56,3,7,59,7,71,15,71,35,11,60,47,11,56,28,61,31,9,35,2,12,1,4,1,3,1,52,5,72,71,72,46,63,30,1,12,16,4,51,65,16,2,66,46,32,72,38,51,72,65,9,9,66,44,50,15,8,66,57,7,10,62,40,20,7,25,23,66,72,49,43,46,72,10,40,65,31,68,56,10,68,43,1,49,15,20,1,50,2,24,3,1,35,17,8,10,68,45,2,37,4,10,1,65,57,3,2,49,4,70,53,53,1,22,52,65,48,2,3,45,1,61,3,40,1,1,51,2,52,51,1,31,47,3,22,1,72,3,47,72,66,35,29,2,4,25,65,27,29,29,1,20,58,14,72,46,71,32,26,68,2,61,4,3,33,9,22,5,30,65,45,5,25,72,27,32,30,70,42,72,47,2,10,61,5,72,72,3,48,63,27,70,7,2,20,66,3,15,72,1,22,3,72,65,11,22,14,41,17,11,15,1,5,33,72,3,2,59,2,71,5,27,1,63,46,72,34,24,72,60,68,8,34,6,2,31,20,1,62,70,10,39,46,6,72,18,71,40,1,58,70,42,34,5,25,2,55,21,70,61,43,47,5,62,16,7,14,60,34,50,38,70,37,4,60,62,1,36,44,55,72,12,13,1,15,65,12,72,72,72,52,2,5,68,62,72,1,66,72,26,64,20,3,22,4,62,5,59,3,72,57,66,60,45,3,15,51,60,33,10,26,6,67,49,1,7,27,37,63,31,50,32,1,63,30,71,53,12,50,2,9,17,56,67,9,4,19,8,71,10,15,72,12,72,1,23,72,26,21,60,12,16,63,22,32,3,13,68,30,16,33,72,4,12,4,6,65,15,24,13,24,72,54,3,4,32,35,35,2,8,22,15,22,1,71,4,25,32,7,17,8,56,1,8,7,3,71,2,1,49,58,44,59,71,1,11,62,35,20,40,39,1,72,33,12,1,27,34,56,58,22,10,13,35,34,4,72,2,7,27,4,37,21,53,18,2,32,23,3,71,9,1,18,12,71,64,4,23,39,28,5,45,37,60,8,47,26,3,50,27,8,62,71,66,68,13,56,38,14,16,14,32,8,43,52,3,29,1,12,16,40,5,40,36,5,10,2,23,26,72,34,10,14,23,47,24,49,20,2,2,22,7,1,59,58,41,59,3,32,46,2,52,13,11,32,17,16,51,29,70,71,41,1,7,25,67,5,15,20,3,54,42,9,63,69,69,40,60,4,71,37,32,39,38,52,48,70,20,50,19,25,12,39,7,23,27,47,26,14,11,2,26,72,63,71,11,14,13,6,11,18,1,32,29,3,2,13,41,1,7,52,45,70,53,62,60,3,23,1,67,12,71,25,5,26,1,70,72,60,32,1,14,13,6,46,15,43,39,21,57,53,18,1,58,71,35,3,38,35,7,47,14,20,66,15,42,17,37,12,53,60,18,1,3,9,1,56,17,11,7,69,19,3,54,62,24,62,17,9,64,2,1,16,72,30,49,61,47,20,34,70,54,61,3,13,16,3,25,30,21,1,15,23,45,24,11,1,56,1,1,7,55,2,72,45,47,46,2,2,12,68,69,56,4,64,59,62,63,53,5,49,62,55,71,72,36,25,72,36,1,72,59,7,1,30,64,63,72,8,62,67,6,70,20,5,24,11,72,66,45,69,15,28,70,36,16,18,34,42,48,47,39,11,7,3,8,1,32,60,10,71,4,1,43,59,23,72,22,1,69,50,1,2,15,31,1,66,3,8,64,28,57,14,19,10,51,67,11,72,66,18,9,9,48,10,9,13,4,4,72,51,59,10,61,54,33,27,1,23,1,45,39,5,72,58,70,61,2,46,1,22,48,64,72,12,34,72,29,33,1,62,41,64,4,24,14,3,4,18,8,35,1,66,8,71,43,2,29,15,65,35,64,58,18,67,63,60,9,70,15,48,12,71,44,1,45,23,43,35,9,12,65,2,27,40,5,8,58,52,3,41,20,1,4,23,6,8,18,52,31,29,36,16,42,1,60,5,22,36,4,9,1,12,23,62,37,8,31,13,24,45,69,2,61,41,44,39,72,13,51,71,22,2,56,1,23,66,1,19,11,8,52,3,51,15,64,37,13,49,45,18,1,68,54,23,17,71,67,14,1,63,41,17,56,5,2,3,37,29,8,63,7,3,72,19,59,2,35,14,14,69,7,69,72,8,4,63,72,46,5,30,63,60,63,25,1,6,22,31,39,26,53,1,12,16,2,39,1,7,4,10,55,72,10,11,15,23,1,3,47,15,66,68,17,7,12,21,21,56,6,65,42,68,48,50,7,63,17,42,4,62,2,2,48,27,70,1,46,30,15,69,65,72,13,17,51,51,72,67,34,67,49,53,27,23,69,2,35,46,54,56,9,20,11,30,68,38,17,48,1,63,3,48,66,68,17,7,72,29,37,34,42,59,11,60,27,1,1,17,58,1,3,53,35,50,68,47,65,5,51,46,9,8,14,45,8,1,66,72,41,23,29,4,6,67,7,56,72,72,23,35,27,26,12,40,7,70,60,39,72,1,54,3,63,71,42,47,66,21,11,1,55,69,3,4,30,5,71,29,52,68,46,8,72,17,3,2,9,51,6,3,17,30,31,45,64,1,1,61,1,9,72,1,7,66,1,40,16,2,67,41,56,72,3,54,52,50,14,27,72,62,12,44,54,68,20,50,58,35,2,63,58,27,71,63,71,41,13,2,68,1,65,72,28,72,2,18,60,26,1,4,68,38,42,57,54,12,44,42,72,71,19,23,30,35,10,1,22,7,36,34,72,36,1,23,32,71,23,17,1,12,72,1,72,60,61,6,32,31,19,72,32,65,45,42,8,32,22,57,1,1,1,24,1,54,4,65,56,45,71,59,69,19,55,38,10,47,2,1,1,1,46,38,65,19,52,71,1,52,6,26,48,64,3,1,72,1,51,41,72,43,72,47,72,3,1,2,26,29,35,27,24,67,16,23,14,1,1,4,16,46,68,38,30,5,17,4,12,72,3,56,41,40,7,69,7,5,72,44,65,3,24,44,72,24,1,22,70,25,37,22,59,49,47,31,1,3,53,1,20,3,51,51,13,1,1,63,3,46,1,8,71,55,70,2,67,65,14,20,1,1,49,72,46,24,5,33,42,23,8,66,24,24,69,53,60,7,20,23,72,11,21,1,31,57,45,10,58,14,27,14,12,69,25,58,35,16,45,17,1,22,1,67,67,2,23,9,5,54,57,24,49,5,2,4,70,5,53,47,31,13,28,10,38,1,67,52,62,16,5,12,72,71,24,15,67,2,5,15,1,41,43,1,1,26,22,71,7,28,16,7,69,1,3,21,69,71,69,48,47,2,45,51,22,72,37,71,7,66,51,30,34,64,65,47,1,49,67,39,14,43,56,14,1,16,70,72,23,21,1,1,32,17,4,36,50,48,50,72,10,18,1,1,9,2,40,69,37,18,11,8,3,55,33,46,34,3,30,33,45,40,71,1,72,22,46,55,1,12,31,5,67,1,40,41,1,51,42,23,1,1,56,15,12,54,7,33,16,21,30,3,11,62,18,6,46,21,68,1,25,24,30,2,51,57,15,72,2,28,29,70,13,59,13,7,62,21,2,1,4,19,30,67,72,53,5,71,50,56,2,2,24,46,71,29,69,71,1,56,56,1,28,19,66,17,52,19,36,7,72,67,34,57,7,1,8,69,50,10,12,14,70,64,66,71,20,72,71,38,28,17,33,23,58,70,4,45,10,36,54,23,41,5,27,1,67,72,56,44,66,34,69,1,40,30,11,15,11,64,72,72,1,15,60,56,8,3,49,2,6,70,12,52,72,40,1,3,40,1,30,23,1,44,65,7,72,8,16,66,1,3,53,8,69,5,72,13,4,54,72,12,1,1,54,69,48,48,8,71,2,67,34,3,9,71,57,72,48,18,43,72,35,4,49,71,11,63,65,49,29,15,4,72,26,35,57,28,25,47,57,16,5,17,56,72,21,48,68,30,3,14,4,71,8,61,72,5,49,8,3,9,67,46,67,55,33,62,1,49,1,14,18,1,1,72,64,69,1,71,66,2,71,11,47,35,32,60,11,29,21,48,3,43,5,1,71,8,8,20,33,71,31,38,1,2,12,9,11,6,71,42,8,5,2,45,28,43,60,42,7,25,40,27,10,27,11,4,68,1,18,57,26,17,1,38,59,30,2,50,9,3,14,31,7,8,17,32,2,7,72,31,27,18,7,14,11,72,28,15,4,71,5,47,57,50,8,48,70,1,8,1,1,60,49,4,29,67,53,67,6,47,53,69,3,4,56,59,61,2,46,12,14,28,24,31,68,39,42,13,6,35,38,18,4,27,41,50,72,70,44,2,34,72,71,64,72,1,29,23,52,25,64,16,1,24,2,34,36,53,47,72,72,1,9,8,45,7,71,41,67,69,70,25,72,34,65,70,72,35,13,12,62,25,52,8,2,56,12,47,2,18,8,45,3,38,72,46,71,66,25,18,13,65,60,15,72,30,42,71,1,39,35,53,1,31,48,30,10,12,57,58,37,44,27,8,3,25,57,12,62,65,71,21,71,7,72,1,72,64,72,29,13,31,1,7,61,39,10,14,1,67,72,6,1,25,33,18,71,28,2,17,56,60,33,1,2,63,7,55,65,1,63,70,36,52,22,22,5,47,33,18,1,56,2,35,64,15,24,1,70,1,4,39,29,14,61,13,66,2,59,62,33,66,72,1,19,51,63,27,22,4,42,29,4,30,4,71,46,4,7,69,72,19,28,5,72,8,7,22,72,8,52,68,71,2,34,35,61,1,1,53,72,2,3,13,41,24,28,8,1,54,41,19,72,62,56,15,10,32,21,62,2,27,5,25,2,49,63,4,1,11,52,60,64,43,61,1,5,66,67,42,1,31,7,4,34,3,19,31,1,3,46,1,69,5,1,26,10,25,64,30,13,64,46,12,15,17,13,67,24,6,53,16,10,13,9,25,7,38,43,4,25,27,72,71,24,50,57,15,4,28,9,55,3,10,55,20,62,32,43,9,60,58,7,2,37,65,39,66,68,62,3,72,41,29,4,53,1,41,39,63,15,13,1,1,8,60,12,40,66,42,66,49,1,41,41,23,3,4,52,4,11,2,26,24,12,60,64,66,60,17,42,1,47,10,70,67,1,7,1,4,66,12,24,26,6,57,14,42,25,64,22,19,61,22,70,12,31,11,68,72,67,60,1,1,58,47,1,1,22,48,37,13,43,6,71,1,72,6,12,25,21,6,20,18,43,35,1,32,52,32,72,51,68,8,49,72,9,28,54,11,50,69,1,68,40,31,33,55,68,12,71,40,64,53,12,53,72,46,40,12,9,51,49,41,56,4,20,26,20,7,7,51,4,1,27,22,12,3,34,24,51,14,59,3,65,5,59,72,62,28,3,19,1,24,57,72,67,52,71,26,35,55,33,72,1,10,37,12,1,62,1,18,69,2,19,12,9,27,27,1,24,14,32,11,1,38,9,54,29,44,59,3,18,67,22,33,5,2,72,9,67,16,8,5,23,1,50,17,68,1,25,67,32,67,72,71,1,46,2,1,48,61,32,2,3,5,71,37,65,67,49,50,25,17,64,25,23,24,37,21,1,10,6,51,10,6,47,61,52,35,71,6,45,2,4,2,4,51,60,9,3,17,8,46,68,1,4,1,28,39,11,71,2,30,17,55,58,5,1,9,26,50,72,43,56,1,72,72,36,5,13,44,70,44,32,69,16,68,16,68,4,26,29,5,70,24,72,1,70,36,38,17,41,1,2,14,2,1,13,6,4,5,15,47,8,17,15,26,23,4,29,25,9,18,3,69,14,19,39,31,24,14,64,50,52,28,21,25,17,58,17,51,72,52,27,3,64,45,3,71,1,58,34,8,15,66,12,58,3,43,9,3,22,40,68,54,50,1,72,40,72,6,5,48,1,64,17,40,41,51,41,1,2,68,24,70,3,2,3,7,13,7,12,53,12,63,15,36,4,24,61,16,65,26,16,54,1,5,19,10,23,3,72,10,10,11,37,17,36,17,66,61,22,1,6,31,68,34,52,10,29,72,47,24,65,4,12,1,33,34,14,4,13,65,23,55,49,60,69,40,67,35,19,13,41,4,24,5,5,1,72,24,42,4,68,33,1,31,4,69,38,3,48,15,25,1,48,1,1,37,66,26,63,10,2,18,64,9,28,1,4,38,66,1,18,51,1,12,41,12,55,7,12,68,5,49,40,16,10,72,2,23,71,11,1,16,1,12,54,68,4,1,27,21,13,64,1,57,21,19,31,52,46,11,53,11,57,2,2,71,1,68,72,2,1,41,72,6,4,12,58,7,65,1,56,4,58,62,26,62,58,68,61,42,18,56,4,4,35,64,31,67,4,70,3,53,2,29,47,68,12,8,54,69,26,72,70,1,10,28,1,21,51,53,53,24,70,61,11,2,25,41,18,72,71,34,29,40,36,46,58,39,4,52,70,65,1,70,29,1,67,1,26,30,48,55,7,37,31,4,72,5,1,15,8,35,56,42,65,2,65,18,23,4,70,4,19,18,38,2,47,52,9,26,8,44,3,2,9,1,25,2,43,1,58,59,44,66,68,9,19,4,70,1,8,53,51,11,60,17,3,70,1,43,16,57,37,72,11,50,5,1,16,2,17,16,15,10,46,64,1,25,71,8,72,49,29,72,31,50,71,70,71,61,32,1,68,62,7,20,6,33,28,27,7,26,5,30,63,1,53,14,21,17,16,35,32,28,1,59,72,36,40,40,9,63,3,40,8,34,5,9,9,31,50,2,1,8,9,2,3,25,1,45,51,55,38,2,38,34,70,13,39,61,12,41,21,55,69,26,69,18,47,72,33,2,72,37,62,71,23,16,9,17,4,1,24,1,72,72,11,9,2,60,29,49,30,53,39,9,39,8,51,71,71,70,1,38,28,32,49,37,10,67,7,51,9,9,4,71,50,24,22,44,33,1,54,42,1,1,30,1,16,1,9,46,1,71,43,50,13,19,41,1,24,40,3,37,67,32,6,32,59,30,20,27,20,9,68,69,26,69,11,1,10,55,44,46,69,11,11,29,57,28,42,2,23,18,62,1,16,3,67,62,57,2,23,25,72,2,8,5,35,24,2,72,41,4,26,7,1,4,48,2,12,60,55,1,1,4,1,42,1,7,3,72,15,4,11,5,1,72,55,40,57,1,1,1,52,41,43,47,3,66,55,29,12,66,35,10,27,58,54,9,2,6,26,9,8,12,15,43,42,31,66,18,1,61,10,1,18,24,3,50,1,2,17,69,72,3,50,53,58,46,72,1,6,72,4,52,2,65,43,4,25,51,12,57,24,64,4,26,15,64,36,27,1,35,4,8,10,2,58,51,46,1,46,50,53,61,5,47,54,19,26,70,17,30,1,19,26,21,50,68,3,9,51,9,41,22,21,71,1,26,71,4,12,18,3,72,11,1,13,72,42,17,7,68,56,38,72,48,52,35,67,1,53,34,3,1,19,60,11,47,18,60,72,39,59,2,1,20,6,71,24,67,1,48,37,11,3,18,50,67,25,2,9,10,70,9,4,2,1,19,7,1,1,9,3,9,5,56,18,49,70,72,6,17,29,6,63,16,59,3,8,7,68,68,52,72,32,72,1,42,25,45,43,37,20,4,63,3,66,28,8,71,1,72,16,66,11,51,8,14,4,70,70,54,28,24,69,42,2,39,45,72,38,72,1,72,55,51,63,1,23,1,2,52,36,1,28,7,14,72,1,10,42,7,4,72,20,63,56,5,72,68,67,8,52,18,59,60,7,59,46,5,59,70,14,44,64,58,46,58,72,30,11,34,54,3,72,40,2,54,14,1,10,1,1,56,68,14,68,55,16,9,14,58,53,70,14,22,10,29,1,49,68,1,30,72,10,7,9,1,20,1,29,1,3,20,64,1,6,50,6,7,72,8,67,24,72,33,2,70,22,59,36,51,53,20,63,40,35,26,27,53,34,19,43,6,56,57,34,10,1,13,56,55,36,47,12,1,24,63,35,67,25,21,13,35,71,29,71,7,57,65,27,6,72,1,11,39,59,26,2,72,65,72,6,32,50,61,15,72,9,1,12,37,61,18,21,68,12,2,62,29,1,5,1,62,36,28,69,11,63,23,10,71,45,70,22,52,55,65,72,10,7,5,24,72,21,69,44,61,24,1,6,4,72,72,14,7,48,55,1,45,3,71,8,3,69,1,72,11,71,1,33,16,56,1,5,57,56,8,22,1,40,46,63,68,69,56,10,63,24,19,22,29,13,70,49,43,3,42,57,2,72,46,66,62,72,35,17,72,28,56,31,45,1,2,6,48,25,64,50,52,4,32,45,9,66,3,54,1,64,31,14,12,67,35,45,10,29,24,66,51,45,49,29,40,37,25,22,72,7,33,23,24,1,69,3,56,65,71,14,2,32,40,1,1,7,15,17,19,71,54,31,11,18,72,71,5,38,5,2,52,8,68,69,42,50,1,1,33,7,64,1,59,6,3,15,13,23,31,29,49,56,63,63,24,36,9,3,21,13,1,25,71,66,45,22,67,68,49,4,63,2,21,55,1,17,30,22,9,1,21,19,69,1,72,70,66,7,46,39,32,24,6,37,8,72,71,16,57,66,17,21,66,17,1,58,8,27,34,30,33,1,14,16,49,19,70,32,18,37,4,16,17,19,60,51,28,43,42,3,1,3,63,3,68,30,60,15,45,70,10,4,1,68,22,38,1,18,29,16,1,12,31,4,48,15,50,7,41,68,26,57,3,1,19,3,59,1,42,7,67,1,66,61,4,42,64,54,1,54,18,3,1,72,60,11,12,61,39,55,17,37,72,72,8,22,1,38,17,70,72,28,15,72,11,8,57,1,46,30,10,23,32,13,39,44,9,67,9,15,71,1,30,1,17,3,67,1,1,32,41,1,1,12,62,22,17,72,56,9,72,20,19,2,53,27,6,9,8,71,10,1,71,68,34,26,22,7,20,60,72,72,4,16,62,10,31,71,58,70,71,69,1,72,26,33,10,57,10,39,11,21,68,18,6,18,52,56,45,67,3,65,63,11,1,55,25,72,72,65,54,7,72,21,2,4,3,72,6,52,69,8,8,63,60,12,13,22,5,1,72,2,40,44,71,2,26,1,1,65,3,13,33,1,4,2,72,37,15,23,30,42,32,22,42,8,65,2,70,22,4,2,67,25,20,2,51,46,25,13,25,26,43,19,10,2,72,18,9,27,24,69,46,72,22,70,2,31,56,16,52,13,35,59,72,66,49,2,21,54,24,1,6,1,49,56,56,6,32,50,58,65,64,66,38,20,36,64,1,60,1,50,1,72,60,46,69,31,19,71,12,39,44,56,72,5,11,24,15,72,56,64,34,2,35,22,5,9,11,23,4,68,33,31,1,56,1,66,72,34,58,2,37,71,1,71,35,6,3,69,44,53,24,5,2,62,19,9,53,5,71,1,18,72,4,59,1,31,3,65,49,2,53,55,72,36,10,1,72,28,38,61,52,67,34,54,1,15,4,9,46,22,38,55,1,64,53,58,56,72,1,72,22,8,16,39,12,54,18,32,41,67,65,25,1,67,7,43,24,9,69,37,20,7,37,5,41,54,3,69,53,18,64,31,20,57,63,13,48,2,57,71,7,16,34,37,16,48,58,72,7,38,48,10,30,31,46,50,28,66,8,41,72,7,38,44,47,53,4,20,2,57,44,24,15,3,4,37,1,24,5,33,58,72,71,28,51,30,72,36,14,72,22,2,15,51,70,71,39,61,52,1,64,62,30,4,63,1,15,27,4,72,45,45,36,17,1,16,3,4,71,10,20,4,26,4,5,4,29,2,29,1,1,8,13,59,1,50,18,17,47,26,6,19,3,68,2,7,18,71,13,3,72,66,24,1,56,22,14,61,40,42,72,12,71,26,7,6,58,51,72,18,7,47,2,62,16,6,19,69,11,64,39,15,25,6,66,61,43,12,23,71,34,5,41,72,14,41,23,71,1,72,6,23,10,72,7,6,9,12,1,48,20,16,2,10,2,20,20,19,19,22,35,1,39,54,1,66,56,18,16,68,53,72,9,30,36,18,55,39,21,2,33,44,30,71,4,35,1,23,22,49,42,33,7,67,15,67,53,21,40,22,39,45,2,57,8,7,6,7,49,65,55,71,35,3,11,1,17,72,28,18,40,52,47,23,66,8,47,7,71,50,46,1,66,42,5,7,29,27,15,25,11,57,67,47,13,8,44,71,24,15,1,2,55,71,50,1,5,66,49,3,66,11,28,65,62,2,2,55,41,17,30,17,16,72,9,1,23,8,19,7,1,61,57,9,15,1,12,54,4,7,20,26,36,53,3,68,72,12,34,68,50,1,41,30,1,29,23,60,72,22,72,66,72,47,51,70,9,59,3,38,37,37,24,14,72,53,8,72,17,2,8,48,10,1,29,65,8,61,45,72,12,7,9,43,58,16,2,8,40,9,41,26,33,68,65,55,20,19,45,70,2,27,12,72,12,5,71,35,70,31,52,37,69,30,33,54,59,55,69,66,37,9,69,10,40,13,6,69,66,11,46,6,56,70,33,72,3,19,5,71,8,1,1,61,71,68,46,33,53,50,57,54,60,28,1,29,10,43,13,43,19,1,69,61,43,6,1,56,70,1,49,6,32,72,37,69,26,58,24,5,15,30,55,25,10,44,47,13,49,64,1,20,37,30,38,1,37,52,71,26,66,72,25,69,53,12,26,21,1,48,26,60,18,10,5,4,65,70,18,62,66,65,3,34,16,54,50,71,10,1,18,4,58,56,2,32,56,36,4,53,10,4,1,51,12,6,63,1,48,5,35,6,2,50,33,31,9,54,46,34,71,63,51,26,64,1,61,15,64,18,57,14,18,72,70,38,68,13,65,30,51,31,9,72,10,37,2,55,33,46,1,20,9,32,19,70,61,26,45,62,1,3,41,67,1,71,37,60,1,6,13,11,7,10,34,62,64,1,25,26,10,53,7,33,71,29,24,20,1,54,5,72,52,9,1,1,33,55,69,1,54,33,45,11,6,21,65,6,8,11,43,49,1,15,60,17,16,35,44,12,1,28,70,5,18,70,9,67,1,18,4,71,30,1,55,59,1,7,45,54,51,72,44,2,66,68,31,21,21,55,9,71,1,22,1,61,67,14,59,21,4,3,70,3,21,20,22,1,63,70,13,5,72,13,61,1,56,4,35,18,72,49,44,3,37,61,70,1,41,70,1,51,42,70,48,68,48,26,11,1,27,46,1,46,25,4,13,31,23,2,65,22,55,9,7,35,6,1,17,10,15,40,13,29,3,58,45,72,68,1,38,2,11,20,72,3,23,40,62,22,11,7,13,1,39,3,58,6,1,22,14,64,1,6,1,39,20,1,1,64,1,46,28,33,39,42,1,7,70,65,1,18,24,63,44,4,1,37,10,34,35,4,39,43,17,61,49,4,64,3,1,40,1,8,1,34,1,39,58,45,6,43,41,5,72,4,9,72,33,72,22,70,21,15,29,15,71,72,19,1,1,2,11,12,70,20,23,49,4,32,2,69,6,24,32,27,27,58,1,18,47,70,13,36,67,10,19,71,72,48,1,18,1,67,69,19,72,38,40,61,10,32,21,59,13,47,69,2,22,15,53,28,22,1,16,48,30,3,57,68,23,65,44,71,37,12,69,35,5,58,72,72,1,39,53,27,1,1,18,46,72,36,4,25,40,63,15,14,72,39,47,19,5,13,17,34,42,5,71,19,2,57,72,6,17,61,1,48,16,9,3,1,65,70,60,69,35,22,66,1,1,34,72,31,30,9,20,19,65,30,6,2,53,7,61,70,13,35,2,3,3,62,72,63,20,35,21,62,15,55,11,17,61,71,2,35,17,21,47,3,3,44,1,44,5,24,1,18,10,65,1,53,3,33,3,34,14,13,46,23,47,17,49,59,69,11,10,12,45,39,71,71,33,67,37,49,9,52,70,1,14,1,1,52,6,7,47,26,25,69,72,4,59,67,26,27,72,6,62,20,6,51,61,62,72,13,5,3,26,13,38,8,34,18,56,36,9,1,12,57,42,33,70,68,1,37,4,1,20,72,31,18,11,33,62,1,16,22,49,36,42,4,12,31,5,66,15,64,10,7,29,57,46,53,17,38,15,22,14,57,11,1,12,3,36,16,65,2,42,72,62,6,48,35,52,1,6,71,67,60,23,39,15,53,24,37,5,50,54,3,68,5,33,41,34,13,20,51,3,41,13,35,12,4,43,12,68,25,7,66,53,63,70,27,1,5,37,3,12,38,9,13,29,47,61,16,41,43,36,6,58,19,11,39,8,26,53,70,1,59,2,7,12,59,61,72,13,64,1,10,65,62,55,25,1,1,59,64,36,3,61,26,1,1,68,2,72,71,57,4,1,72,21,71,29,69,64,16,4,52,2,1,18,2,19,40,66,21,8,72,48,69,72,14,6,8,17,65,57,13,19,56,14,52,58,47,67,2,6,71,46,5,67,3,3,52,42,50,23,67,25,39,69,1,32,9,16,60,72,5,26,3,2,2,36,7,60,19,45,4,31,47,1,1,1,59,10,35,4,32,43,4,54,11,66,61,72,44,41,50,47,8,18,72,1,42,18,13,68,4,69,17,25,43,59,5,21,69,13,42,52,46,61,29,25,5,15,19,44,6,58,62,70,1,10,26,66,7,51,72,65,2,70,72,1,1,5,3,58,22,33,1,54,72,1,3,72,72,54,59,54,60,60,3,69,1,50,56,60,69,1,1,3,60,13,62,45,25,44,2,33,1,22,35,29,27,54,2,57,62,15,2,70,21,23,6,4,3,23,26,8,26,2,67,71,59,39,21,1,48,31,64,46,52,67,67,5,71,9,26,71,32,2,71,60,55,54,2,6,48,63,1,12,54,30,30,4,40,9,17,62,28,70,46,23,47,68,60,67,14,57,55,1,1,23,13,47,38,38,2,1,15,26,35,3,50,42,10,61,68,10,65,72,55,1,7,2,9,27,7,64,70,2,67,45,24,4,44,72,1,66,1,13,10,65,1,38,23,10,4,72,35,1,58,70,38,60,26,8,41,36,54,71,55,72,3,54,72,52,60,39,15,69,43,63,2,72,32,40,58,67,51,31,69,32,21,52,72,72,52,41,41,6,67,16,17,35,58,1,52,70,19,1,35,32,17,67,9,31,4,58,60,58,1,27,66,15,47,41,59,50,17,6,51,44,49,2,59,50,59,18,10,14,35,8,18,60,1,6,19,53,72,60,1,13,5,1,13,37,64,5,61,1,1,26,1,24,17,26,1,40,52,1,1,21,67,44,70,3,56,13,58,42,1,46,63,11,15,72,29,1,6,1,63,2,18,43,15,10,55,49,6,70,2,63,25,18,28,53,35,1,70,2,26,34,19,15,62,42,9,24,68,31,1,21,63,2,61,1,18,6,33,16,56,23,9,14,15,5,61,70,15,8,8,4,34,68,45,9,22,2,70,10,72,49,54,71,22,50,43,45,64,23,68,1,2,26,55,14,71,64,7,57,13,3,72,40,14,2,66,38,1,22,1,5,29,1,3,71,9,43,48,26,9,1,46,2,1,64,12,6,59,7,72,16,25,34,1,10,24,10,69,57,50,28,16,25,3,61,2,51,71,20,6,6,29,36,28,7,63,48,49,27,72,1,72,47,1,36,43,27,9,38,35,59,27,2,7,36,41,13,19,60,48,3,69,43,11,45,72,2,12,67,37,39,41,25,8,71,5,30,40,54,72,28,18,2,59,22,1,72,14,50,48,49,28,68,13,11,3,57,3,72,70,49,67,46,64,37,2,13,72,68,15,24,24,27,12,71,67,63,1,4,40,12,52,10,68,54,4,52,1,70,43,52,12,56,42,22,51,27,51,4,1,35,71,1,69,14,57,72,48,4,31,38,37,1,57,62,3,72,29,13,3,11,21,19,61,11,35,25,1,67,19,56,72,43,55,2,27,13,70,14,19,20,43,5,70,40,6,39,4,15,1,45,64,57,72,1,72,3,55,59,18,32,4,66,27,4,60,8,8,35,7,53,18,15,67,6,6,13,11,1,5,13,9,29,1,1,18,2,30,66,38,44,54,2,42,58,58,25,71,37,14,4,48,3,8,1,67,13,45,49,52,63,68,31,64,62,1,6,21,72,32,71,34,3,12,8,35,3,3,53,4,48,6,3,54,1,62,22,1,51,30,56,35,64,30,25,41,9,1,70,57,9,69,43,72,44,72,33,54,27,54,3,53,1,15,56,5,48,25,3,58,10,1,71,65,5,28,67,35,72,61,68,1,3,70,48,68,47,32,5,49,48,13,15,12,67,9,13,38,42,24,27,9,49,61,50,25,22,1,4,18,56,53,51,24,62,24,70,1,16,8,72,23,31,37,30,35,23,20,36,8,71,50,43,57,41,27,13,3,67,3,64,26,38,23,40,72,3,23,1,4,62,40,41,34,1,51,1,39,12,12,72,63,44,18,9,13,68,6,2,55,1,38,67,19,12,72,24,72,11,4,66],"type":"box","marker":{"color":"rgba(31,119,180,1)","line":{"color":"rgba(31,119,180,1)"}},"line":{"color":"rgba(31,119,180,1)"},"xaxis":"x","yaxis":"y","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.20000000000000001,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script>

### Correlation Matrix

A heatmap displays the correlations between numerical variables,
including the encoded categorical ones. This plot provides an overview
of how different features correlate with each other and with churn.

``` r
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
```

![](README_figs/README-heatmap-1.png)<!-- -->

## Label Encoding

``` r
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
```

    ##   gender SeniorCitizen Partner Dependents tenure PhoneService MultipleLines
    ## 1      0             0       1          0      1            0             1
    ## 2      1             0       0          0     34            1             0
    ## 3      1             0       0          0      2            1             0
    ## 4      1             0       0          0     45            0             1
    ## 5      0             0       0          0      2            1             0
    ## 6      0             0       0          0      8            1             2
    ##   InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport
    ## 1               0              0            2                0           0
    ## 2               0              2            0                2           0
    ## 3               0              2            2                0           0
    ## 4               0              2            0                2           2
    ## 5               1              0            0                0           0
    ## 6               1              0            0                2           0
    ##   StreamingTV StreamingMovies Contract PaperlessBilling PaymentMethod
    ## 1           0               0        0                1             2
    ## 2           0               0        1                0             3
    ## 3           0               0        0                1             3
    ## 4           0               0        1                0             0
    ## 5           0               0        0                1             2
    ## 6           2               2        0                1             2
    ##   MonthlyCharges TotalCharges Churn
    ## 1          29.85        29.85     0
    ## 2          56.95      1889.50     0
    ## 3          53.85       108.15     1
    ## 4          42.30      1840.75     0
    ## 5          70.70       151.65     1
    ## 6          99.65       820.50     1

### Correlations with Churn

A bar plot highlights features with the highest correlations to churn,
aiding in identifying the most significant factors for predictive
modeling.

``` r
# Correlation with Churn (sorted)
correlations <- cor(df_encoded)[,'Churn']
correlations_sorted <- sort(correlations, decreasing = TRUE)
print(correlations_sorted)
```

    ##            Churn   MonthlyCharges PaperlessBilling    SeniorCitizen 
    ##      1.000000000      0.192858218      0.191454321      0.150541053 
    ##    PaymentMethod    MultipleLines     PhoneService           gender 
    ##      0.107852015      0.038043274      0.011691399     -0.008544643 
    ##      StreamingTV  StreamingMovies  InternetService          Partner 
    ##     -0.036302722     -0.038801748     -0.047097165     -0.149981926 
    ##       Dependents DeviceProtection     OnlineBackup     TotalCharges 
    ##     -0.163128439     -0.177883195     -0.195290209     -0.199484084 
    ##      TechSupport   OnlineSecurity           tenure         Contract 
    ##     -0.282232487     -0.289050176     -0.354049359     -0.396149533

``` r
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
```

![](README_figs/README-churn_barplot-1.png)<!-- -->

## Data Resampling

``` r
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
```

![](README_figs/README-data_resampling-1.png)<!-- -->

``` r
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
```

![](README_figs/README-data_resampling-2.png)<!-- -->

``` r
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
```

## Model Training and Evaluation

``` r
# KNN
knn_pred <- knn(train_data[, -ncol(train_data)], test_data[, -ncol(test_data)], train_data$Churn, k = 5)
knn_conf_matrix <- confusionMatrix(as.factor(knn_pred), as.factor(test_data$Churn))
print(knn_conf_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 1309  281
    ##          1  227  292
    ##                                           
    ##                Accuracy : 0.7591          
    ##                  95% CI : (0.7403, 0.7772)
    ##     No Information Rate : 0.7283          
    ##     P-Value [Acc > NIR] : 0.0007038       
    ##                                           
    ##                   Kappa : 0.3728          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.0186982       
    ##                                           
    ##             Sensitivity : 0.8522          
    ##             Specificity : 0.5096          
    ##          Pos Pred Value : 0.8233          
    ##          Neg Pred Value : 0.5626          
    ##              Prevalence : 0.7283          
    ##          Detection Rate : 0.6207          
    ##    Detection Prevalence : 0.7539          
    ##       Balanced Accuracy : 0.6809          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
# Random Forest
rf_model <- randomForest(Churn ~ ., data = train_data)
rf_pred <- predict(rf_model, test_data)
rf_conf_matrix <- confusionMatrix(factor(rf_pred, levels = unique(train_data$Churn)), factor(test_data$Churn, levels = unique(train_data$Churn)))
print(rf_conf_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction 0 1
    ##          0 0 0
    ##          1 0 0
    ##                                   
    ##                Accuracy : NaN     
    ##                  95% CI : (NA, NA)
    ##     No Information Rate : NA      
    ##     P-Value [Acc > NIR] : NA      
    ##                                   
    ##                   Kappa : NaN     
    ##                                   
    ##  Mcnemar's Test P-Value : NA      
    ##                                   
    ##             Sensitivity :  NA     
    ##             Specificity :  NA     
    ##          Pos Pred Value :  NA     
    ##          Neg Pred Value :  NA     
    ##              Prevalence : NaN     
    ##          Detection Rate : NaN     
    ##    Detection Prevalence : NaN     
    ##       Balanced Accuracy :  NA     
    ##                                   
    ##        'Positive' Class : 0       
    ## 

``` r
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
```

![](README_figs/README-train_model-1.png)<!-- -->

``` r
# Fit a Decision Tree model
dt_model <- rpart(Churn ~ ., data = train_data, method = "class")

# Predict on the test set
dt_pred <- predict(dt_model, test_data, type = "class")

# Print the confusion matrix
dt_conf_matrix <- confusionMatrix(dt_pred, test_data$Churn)
print(dt_conf_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 1422  357
    ##          1  114  216
    ##                                           
    ##                Accuracy : 0.7767          
    ##                  95% CI : (0.7583, 0.7943)
    ##     No Information Rate : 0.7283          
    ##     P-Value [Acc > NIR] : 2.025e-07       
    ##                                           
    ##                   Kappa : 0.3492          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 0.9258          
    ##             Specificity : 0.3770          
    ##          Pos Pred Value : 0.7993          
    ##          Neg Pred Value : 0.6545          
    ##              Prevalence : 0.7283          
    ##          Detection Rate : 0.6743          
    ##    Detection Prevalence : 0.8435          
    ##       Balanced Accuracy : 0.6514          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

## Conclusion

Through this analysis, we identified several key drivers of churn,
including monthly charges, tenure, contract type, and additional
services like online security. Customers with shorter tenures, higher
monthly charges, and flexible contracts were found to have higher churn
rates. Moreover, the absence of online security and tech support
services was also associated with increased churn.

In predictive modeling, the Random Forest classifier outperformed other
models, providing a strong balance between accuracy and
interpretability. This suggests that a Random Forest model could be
effectively used in customer retention strategies to proactively
identify at-risk customers based on these key factors.

Moving forward, the findings from this analysis could be integrated into
strategic initiatives, such as offering targeted discounts or
personalized service upgrades to at-risk customers. By leveraging these
insights, telecom companies can reduce churn rates and enhance customer
satisfaction.
