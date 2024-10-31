install.packages("dplyr", "ggplot", "tidyr", "readr", "forcats")
install.packages("RColorBrewer")
library(RColorBrewer)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(forcats)

# Load the dataset
data <- read_csv("SuperStore_Sales_DataSet.csv")

data <- data %>%
  mutate(Order_Date = as.Date(Order_Date, format="%m/%d/%Y"),
         Ship_Date = as.Date(Ship_Date, format="%m/%d/%Y"),
         Sales = as.numeric(Sales),
         Quantity = as.numeric(Quantity),
         Profit = as.numeric(Profit)) %>%
  filter(!is.na(Sales), !is.na(Quantity), !is.na(Profit))

# A. Calculate metrics
total_orders <- nrow(data)
total_sales_inr <- sum(data$Sales) * 84.10  # Assuming 1 USD = 75 INR
average_product_quantity <- mean(data$Quantity)
average_delivery_days <- mean(as.numeric(data$Ship_Date - data$Order_Date))

# Print results
cat("Total Orders:", total_orders, "\n")
cat("Total Sales in Indian Rupees:", total_sales_inr, "\n")
cat("Average Product Quantity:", average_product_quantity, "\n")
cat("Average Delivery Days:", average_delivery_days, "\n")

# B. Create plots for Sales and Profit for 2019 and 2020
data_years <- data %>%
  filter(format(Order_Date, "%Y") %in% c("2019", "2020")) %>%
  group_by(year = format(Order_Date, "%Y")) %>%
  summarise(Total_Sales = sum(Sales), Total_Profit = sum(Profit))


ggplot(data_years, aes(x = year)) +
  geom_bar(aes(y = Total_Sales), stat = "identity", fill = brewer.pal(3, "Blues")[2], width = 0.6) +
  geom_bar(aes(y = Total_Profit), stat = "identity", fill = brewer.pal(3, "Reds")[2], width = 0.4) +
  labs(title = "Sales and Profit for 2019 and 2020", 
       y = "Amount", x = "Year") +
  theme_minimal(base_size = 15) + 
  theme(legend.position = "right",
        legend.title = element_blank(),
        text = element_text(family = "serif"),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))
  scale_y_continuous(labels = scales::comma)

# C. Create donut charts for various categories
sales_by_region <- data %>%
  group_by(Region) %>%
  summarise(Sales = sum(Sales))

sales_by_payment_mode <- data %>%
  group_by(Payment_Mode) %>%
  summarise(Sales = sum(Sales))

sales_by_segment <- data %>%
  group_by(Segment) %>%
  summarise(Sales = sum(Sales))

sales_by_shipment_mode <- data %>%
  group_by(Ship_Mode) %>%
  summarise(Sales = sum(Sales))

sales_by_category <- data %>%
  group_by(Category) %>%
  summarise(Sales = sum(Sales))

sales_by_sub_category <- data %>%
  group_by(Sub_Category) %>%
  summarise(Sales = sum(Sales))


# Function to create donut chart
create_donut_chart <- function(data, category, title) {
  ggplot(data, aes(x = 2, y = Sales, fill = !!sym(category))) + 
    geom_col(width = 1, color = "white") + 
    coord_polar(theta = "y") +
    xlim(0.5, 2.5) + # This makes it a donut chart instead of a pie chart
    theme_void() + 
    theme(legend.position = "right",
          legend.title = element_blank(),
          text = element_text(family = "serif"),
          plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
    labs(title = title, fill = category)
}

# Plotting donut charts for each category
create_donut_chart(sales_by_region, "Region", "Sales by Region")
create_donut_chart(sales_by_payment_mode, "Payment_Mode", "Sales by Payment Mode")
create_donut_chart(sales_by_segment, "Segment", "Sales by Segment")
create_donut_chart(sales_by_shipment_mode, "Ship_Mode", "Sales by Shipment Mode")
create_donut_chart(sales_by_category, "Category", "Sales by Category")
create_donut_chart(sales_by_sub_category, "Sub_Category", "Sales by Sub-Category")

# D. Create stacked charts for various categories
popular_payment_modes <- data %>%
  group_by(Payment_Mode) %>%
  summarise(Total_Sales = sum(Sales), Total_Profit = sum(Profit))

product_segments <- data %>%
  group_by(Segment) %>%
  summarise(Total_Sales = sum(Sales))

categories_data <- data %>%
  group_by(Category) %>%
  summarise(Total_Sales = sum(Sales))

delivery_type_region <- data %>%
  group_by(Ship_Mode, Region) %>%
  summarise(Total_Sales = sum(Sales))

# Enhanced Stacked Bar Chart
create_stacked_bar_chart <- function(data, x_var, fill_var, title) {
  ggplot(data, aes_string(x = x_var, y = "Total_Sales", fill = fill_var)) + 
    geom_bar(stat = "identity", color = "black", width = 0.6) + 
    labs(title = title, x = x_var, fill = fill_var) + 
    theme_minimal(base_size = 14) +
    theme(text = element_text(family = "serif"),
          plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.title.y = element_text(face = "bold")) +
    scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.1))) +
    scale_fill_brewer(palette = "Dark2")
}

# Create stacked bar charts
create_stacked_bar_chart(popular_payment_modes, "Payment_Mode", "Payment_Mode", "Most Popular Payment Modes")
create_stacked_bar_chart(product_segments, "Segment", "Segment", "Product Segments")
create_stacked_bar_chart(categories_data, "Category", "Category", "Categories")
create_stacked_bar_chart(delivery_type_region, "Ship_Mode", "Region", "Delivery Type by Region of Order")

# E. Create stacked chart for product features by payment methods
product_features_payment_modes <- data %>%
  group_by(Payment_Mode, Product_Name) %>%
  summarise(Total_Sales = sum(Sales))

create_stacked_bar_chart(product_features_payment_modes, "Product_Name", "Payment_Mode", "Product Features by Payment Methods")


# F. Create stacked chart for sales and profit analysis
top_sales_profit_payment_modes <- data %>%
  group_by(Payment_Mode) %>%
  summarise(Total_Sales = sum(Sales), Total_Profit = sum(Profit))

customer_segments_data <- data %>%
  group_by(Segment) %>%
  summarise(Total_Sales = sum(Sales), Total_Profit = sum(Profit))

product_categories_data <- data %>%
  group_by(Category) %>%
  summarise(Total_Sales = sum(Sales), Total_Profit = sum(Profit))

shipping_mode_region_data <- data %>%
  group_by(Ship_Mode, Region) %>%
  summarise(Total_Sales = sum(Sales), Total_Profit = sum(Profit))

create_stacked_bar_chart1 <- function(data, x_var, fill_var, title) {
  ggplot(data, aes_string(x = x_var, y = "Total_Sales", fill = fill_var)) + 
    geom_bar(stat = "identity", color = "black", width = 0.6) + 
    labs(title = title, x = x_var, fill = fill_var) + 
    theme_minimal(base_size = 14) +
    theme(text = element_text(family = "serif"),
          plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.title.y = element_text(face = "bold")) +
    scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.1))) +
    scale_fill_gradient(low = "lightblue", high = "darkblue")  # Use gradient for continuous values
}

create_stacked_bar_chart1(top_sales_profit_payment_modes, "Payment_Mode", "Total_Sales", "Top Sales and Profit by Payment Modes")
create_stacked_bar_chart1(customer_segments_data, "Segment", "Total_Sales", "Customer_Segments")
create_stacked_bar_chart1(product_categories_data, "Category", "Total_Sales", "Product_Categories")
create_stacked_bar_chart(shipping_mode_region_data, "Ship_Mode", "Region", "Shipping Mode by Regions of Order")


install.packages("maps", "sf")
library(maps)
library(sf)

profit_by_state <- data %>%
  group_by(State) %>%
  summarise(Total_Profit = sum(Profit, na.rm = TRUE))

# Get US state map data
us_states <- map_data("state")

# Convert state names in both datasets to lowercase for matching
profit_by_state$State <- tolower(profit_by_state$State)
us_states$region <- tolower(us_states$region)

# Merge map data with profit data
map_data <- merge(us_states, profit_by_state, by.x = "region", by.y = "State", all.x = TRUE)

# Create a map with profit levels
ggplot(map_data, aes(x = long, y = lat, group = group, fill = Total_Profit)) +
  geom_polygon(color = "white", linewidth = 0.2) +  # Thinner borders for better clarity
  scale_fill_gradient(low = "lightyellow", high = "darkred", na.value = "gray90", name = "Profit") + 
  coord_fixed(1.3) + 
  labs(title = "Profit by States") +
  theme_void() +
  theme(text = element_text(family = "serif"),
        plot.title = element_text(hjust = 0.5, size = 18, face = "bold")) +
  geom_text(data = map_data %>%
              group_by(region) %>%
              summarise(long = mean(long), lat = mean(lat)),
            aes(label = region, x = long, y = lat), size = 2, color = "black", inherit.aes = FALSE)  # Add state names

# Top 10 Customers by Sales
top_customers_by_sales <- data %>%
  group_by(Customer_Name) %>%
  summarise(Total_Sales = sum(Sales, na.rm = TRUE)) %>%
  arrange(desc(Total_Sales)) %>%
  top_n(10)

# Plotting Top 10 Customers by Sales
ggplot(top_customers_by_sales, aes(x = reorder(Customer_Name, Total_Sales), y = Total_Sales)) +
  geom_col(fill = "steelblue", width = 0.6) +
  coord_flip() +
  labs(title = "Top 10 Customers by Sales", x = "Customer Name", y = "Total Sales") +
  theme_minimal(base_size = 14) +
  theme(text = element_text(family = "serif"),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.title.y = element_text(face = "bold"))

# Top 10 Cities by Profit and Sales
top_cities_by_profit_sales <- data %>%
  group_by(City) %>%
  summarise(Total_Sales = sum(Sales, na.rm = TRUE), Total_Profit = sum(Profit, na.rm = TRUE)) %>%
  arrange(desc(Total_Sales)) %>%
  top_n(10)

# Plotting Top 10 Cities by Profit and Sales
ggplot(top_cities_by_profit_sales, aes(x = reorder(City, Total_Sales), y = Total_Sales, fill = Total_Profit)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top 10 Cities by Profit and Sales", x = "City", y = "Total Sales") +
  scale_fill_gradient(low = "yellow", high = "red", name = "Profit") +
  theme_minimal()

# Top 10 Profit-Making Categories and Sub-Categories
top_profit_categories_subcategories <- data %>%
  group_by(Category, Sub_Category) %>%
  summarise(Total_Profit = sum(Profit, na.rm = TRUE)) %>%
  arrange(desc(Total_Profit)) %>%
  top_n(10)

# Plotting Top 10 Profit-Making Categories and Sub-Categories
ggplot(top_profit_categories_subcategories, aes(x = reorder(Sub_Category, Total_Profit), y = Total_Profit, fill = Category)) +
  geom_col(width = 0.6, color = "black") +
  coord_flip() +
  labs(title = "Top 10 Profit-Making Categories and Sub-Categories", x = "Sub-Category", y = "Total Profit") +
  theme_minimal(base_size = 14) +
  theme(text = element_text(family = "serif"),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.title.y = element_text(face = "bold")) +
  scale_fill_brewer(palette = "Set2")

# Number of Category-Wise Returns
category_wise_returns <- data %>%
  filter(Returns == "Yes") %>%
  group_by(Category) %>%
  summarise(Num_Returns = n())

# Plotting Number of Category-Wise Returns
ggplot(category_wise_returns, aes(x = reorder(Category, Num_Returns), y = Num_Returns, fill = Category)) +
  geom_col(color = "black", width = 0.8) +
  labs(title = "Number of Category-Wise Returns", x = "Category", y = "Number of Returns") +
  theme_minimal(base_size = 10) +
  theme(text = element_text(family = "serif"),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.title.y = element_text(face = "bold")) +
  scale_fill_brewer(palette = "Set1")
