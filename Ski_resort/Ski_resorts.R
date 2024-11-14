install.packages("dplyr", "ggplot2", "leaflet", "cluster")
install.packages("tidyverse", "readr", "lubridate")
install.packages("ggplot2")
install.packages("lubridate")
install.packages("knitr")
install.packages("tinytex")
tinytex::install_tinytex()
# Load libraries
library(knitr)
library(tinytex)
library(dplyr)
library(ggplot2)
library(readr)
library(lubridate)
library(leaflet)
library(cluster)
library(tidyverse)

# Load the data
resorts <- read_csv("resorts.csv")
snow <- read_csv("snow.csv")

# Clean `resorts` data
resorts <- resorts %>%
  mutate(
    Price = as.numeric(Price),
    Season = as.factor(Season),
    Child_friendly = as.logical(Child_friendly),
    Snowparks = as.logical(Snowparks),
    Nightskiing = as.logical(Nightskiing),
    Summer_skiing = as.logical(Summer_skiing)
  ) %>%
  na.omit()

# Check for non-numeric values in Latitude and Longitude
summary(resorts$Latitude)
summary(resorts$Longitude)

# Check the number of rows with NA values in Latitude or Longitude
missing_data <- resorts %>%
  filter(is.na(Latitude) | is.na(Longitude))
nrow(missing_data)  # Display the count of rows with missing Latitude or Longitude

# Re-import data ensuring Latitude and Longitude are read as numeric
resorts <- read_csv("resorts.csv", col_types = cols(
  Latitude = col_double(),
  Longitude = col_double()
)) %>%
  filter(!is.na(Latitude) & !is.na(Longitude))  # Filter out rows with NA in Latitude or Longitude

# Check the number of rows after filtering
print(nrow(resorts))

# Adjust the number of clusters to the available data
num_clusters <- min(5, nrow(resorts) - 1)  # Ensure we don't exceed available data

# Perform clustering if we have at least two data points
if (num_clusters > 1) {
  set.seed(123)
  geographical_clusters <- kmeans(resorts[, c("Latitude", "Longitude")], centers = num_clusters)
  resorts$Cluster <- as.factor(geographical_clusters$cluster)
} else {
  print("Insufficient data for clustering even after cleaning.")
}

# Ensure that all character columns are in UTF-8 encoding
resorts <- resorts %>%
  mutate(across(where(is.character), ~ iconv(., from = "", to = "UTF-8")))

# Join resorts and snow data by matching Latitude and Longitude
resort_snow <- resorts %>%
  inner_join(snow, by = c("Latitude", "Longitude"))

snow_by_season <- resort_snow %>%
  mutate(Month = month(Month, label = TRUE)) %>%
  group_by(Season, Month) %>%
  summarize(Average_Snow = mean(Snow, na.rm = TRUE), .groups = "drop")

# Calculate elevation change and find resorts with the highest peaks and elevation changes
resorts <- resorts %>%
  mutate(Elevation_Change = Highest_point - Lowest_point)

resorts$Resort <- gsub("[0-9?/~`^\\\\]", "", resorts$Resort)
resorts$Resort <- gsub("\\(.*?\\)", "", resorts$Resort)

# Highest mountain peaks
highest_peaks <- resorts %>%
  arrange(desc(Highest_point)) %>%
  select(Resort, Country, Highest_point) %>%
  head(15)

# Largest elevation changes
largest_elevation_changes <- resorts %>%
  arrange(desc(Elevation_Change)) %>%
  select(Resort, Country, Elevation_Change) %>%
  head(15)

# Define criteria for beginner-friendly resorts (e.g., high percentage of beginner slopes)
beginner_resorts <- resorts %>%
  arrange(desc(Beginner_slopes)) %>%
  select(Resort, Country, Beginner_slopes) %>%
  head(15)

# Define criteria for expert-friendly resorts (e.g., high percentage of difficult slopes)
expert_resorts <- resorts %>%
  filter(!is.na(resorts$Resort)) %>%
  arrange(desc(Difficult_slopes)) %>%
  filter(!is.na(Difficult_slopes)) %>%  # Remove rows with NA in Difficult_slopes
  select(Resort, Country, Difficult_slopes) %>%
  head(15)

# Plot Highest Mountain Peaks
ggplot(highest_peaks, aes(x = reorder(Resort, Highest_point), y = Highest_point)) +
  geom_bar(stat = "identity", fill = "darkorange") +
  labs(title = "Top 10 Highest Mountain Peaks", x = "Resort", y = "Highest Point (m)") +
  theme_minimal() +
  coord_flip()  # Flip for better readability

# Plot Largest Elevation Changes
ggplot(largest_elevation_changes, aes(x = reorder(Resort, Elevation_Change), y = Elevation_Change)) +
  geom_bar(stat = "identity", fill = "forestgreen") +
  labs(title = "Top 10 Largest Elevation Changes", x = "Resort", y = "Elevation Change (m)") +
  theme_minimal() +
  coord_flip()  # Flip for better readability

# Plot Beginner-Friendly Resorts
ggplot(beginner_resorts, aes(x = reorder(Resort, Beginner_slopes), y = Beginner_slopes)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  labs(title = "Top 10 Beginner-Friendly Resorts", x = "Resort", y = "Beginner Slopes (%)") +
  theme_minimal() +
  coord_flip()  # Flip for better readability

# Plot Expert-Friendly Resorts
ggplot(expert_resorts, aes(x = reorder(Resort, Difficult_slopes), y = Difficult_slopes)) +
  geom_bar(stat = "identity", fill = "darkred") +
  labs(title = "Top 10 Expert-Friendly Resorts", x = "Resort", y = "Difficult Slopes (%)") +
  theme_minimal() +
  coord_flip()  # Flip for better readability

snow_summary <- snow %>%
  group_by(Latitude, Longitude) %>%
  summarise(MeanSnow = mean(Snow))

# Calculate the center of the map based on the data
center_lat <- mean(snow_summary$Latitude, na.rm = TRUE)
center_long <- mean(snow_summary$Longitude, na.rm = TRUE)

pal <- colorFactor(palette = "Set1", domain = resorts$Continent)

leaflet(data = resorts) %>%
  addTiles() %>%
  setView(lng = center_long, lat = center_lat, zoom = 2) %>%
  addCircleMarkers(
    ~Longitude, ~Latitude,
    label = ~Resort,
    color = ~pal(Continent),  # Color by Continent
    fillOpacity = 0.5,
    radius = 2
  ) %>%
  addLegend("topright", pal = pal, values = ~Continent, title = "Continents")

# Choose clustering features: Elevation change, Total lifts, Lift capacity, Total slopes
resort_features <- resorts %>%
  select(Elevation_Change, Total_lifts, Lift_capacity, Total_slopes) %>%
  na.omit()

# Normalize features for clustering
resort_features_scaled <- scale(resort_features)

# Perform k-means clustering (tuning the number of clusters based on needs)
set.seed(123)
traffic_clusters <- kmeans(resort_features_scaled, centers = 4)
resorts$Traffic_Cluster <- as.factor(traffic_clusters$cluster)

# Visualize traffic clusters
ggplot(resorts, aes(x = Elevation_Change, y = Total_lifts, color = Traffic_Cluster)) +
  geom_point() +
  labs(title = "Traffic Control Clustering of Ski Resorts", x = "Elevation Change", y = "Total Lifts")

# Count ski resorts by country
country_resort_count <- resorts %>%
  group_by(Country) %>%
  summarise(ResortCount = n())

# Plotting noticeable clusters using a bar chart
ggplot(country_resort_count, aes(x = reorder(Country, -ResortCount), y = ResortCount)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Countries with Most Ski Resorts", x = "Country", y = "Number of Resorts")

# Calculate the center of the map based on the data
center_lat <- mean(snow_summary$Latitude, na.rm = TRUE)
center_long <- mean(snow_summary$Longitude, na.rm = TRUE)

pal <- colorNumeric(palette = c("blue", "red"), domain = snow_summary$MeanSnow)

# Plot interactive map with initial zoom level
leaflet(data = snow_summary) %>%
  addTiles() %>%
  setView(lng = center_long, lat = center_lat, zoom = 2) %>%  # Set initial center and zoom level
  addCircleMarkers(
    ~Longitude, ~Latitude,
    color = ~pal(MeanSnow),
    fillOpacity = 0.2,
    radius = 1,
    popup = ~paste("Mean Snow Cover:", MeanSnow)
  ) %>%
  addLegend("topright", pal = pal, values = ~MeanSnow, title = "Mean Snow Cover")

# Continent-wise resort counts
continent_counts <- resorts %>%
  group_by(Continent) %>%
  summarise(ResortCount = n()) %>%
  arrange(desc(ResortCount))

# Top 10 child-friendly countries (assuming 'Child_friendly' is a logical variable or 1/0)
child_friendly_countries <- resorts %>%
  filter(Child_friendly == "Yes") %>%
  group_by(Country) %>%
  summarise(ChildFriendlyResorts = n()) %>%
  arrange(desc(ChildFriendlyResorts)) %>%
  head(15)

# Plot continent-wise resort counts
ggplot(continent_counts, aes(x = reorder(Continent, ResortCount), y = ResortCount)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Continent-wise Resort Counts", x = "Continent", y = "Number of Resorts") +
  theme_minimal() +
  coord_flip()  # Flip coordinates for better readability

# Plot top 10 child-friendly countries
ggplot(child_friendly_countries, aes(x = reorder(Country, ChildFriendlyResorts), y = ChildFriendlyResorts)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  labs(title = "Top 10 Child-Friendly Countries", x = "Country", y = "Number of Child-Friendly Resorts") +
  theme_minimal() +
  coord_flip()  # Flip coordinates for better readability

# Select the top 15 resorts based on Longest Run
top_longest_runs <- resorts %>%
  filter(!is.na(resorts$Resort)) %>%
  arrange(desc(Longest_run)) %>%
  filter(!is.na(Longest_run)) %>%
  head(15)
print(top_longest_runs$Resort)
# Plot
ggplot(top_longest_runs, aes(x = reorder(Resort, Longest_run), y = Longest_run)) +
  geom_bar(stat = "identity", fill = "maroon") +
  coord_flip() +
  labs(title = "Top 15 Resorts by Longest Run", x = "Resort", y = "Longest Run (km)")

# Select the top 15 resorts based on Lift Capacity
top_lift_capacity <- resorts %>%
  filter(!is.na(resorts$Resort)) %>%
  arrange(desc(Lift_capacity)) %>%
  filter(!is.na(Lift_capacity)) %>%
  head(15)

# Plot
ggplot(top_lift_capacity, aes(x = reorder(Resort, Lift_capacity), y = Lift_capacity)) +
  geom_bar(stat = "identity", fill = "orange") +
  coord_flip() +
  labs(title = "Top 15 Resorts by Lift Capacity", x = "Resort", y = "Lift Capacity (people/hour)")

# Filter resorts located in Europe with Nightskiing facilities
nightskiing_europe <- resorts %>%
  filter(!is.na(resorts$Resort)) %>% 
  filter(Continent == "Europe" & Nightskiing == "Yes") 
print(nightskiing_europe$Resort)
nightskiing_europe <- nightskiing_europe %>% filter(!is.na(Resort))

leaflet(data = nightskiing_europe) %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  setView(lng = center_long, lat = center_lat, zoom = 4) %>%
  addCircleMarkers(
    ~Longitude, ~Latitude,
    label = ~as.character(nightskiing_europe),
    color = "purple",
    radius = 2,
    popup = ~paste("Resort:", Resort, "<br>", "Nightskiing: Yes"),
    fillOpacity = 0.5,
    labelOptions = labelOptions(noHide = FALSE)
  ) %>%
  addLegend("topright", colors = "purple", labels = "Nightskiing Available", title = "Nightskiing Facilities")

# Select resorts with the highest number of Gondola Lifts and arrange in descending order
top_gondola_lifts <- resorts %>%
  arrange(desc(Gondola_lifts)) %>%
  filter(!is.na(Gondola_lifts)) %>%
  head(15)

# Plot
ggplot(top_gondola_lifts, aes(x = reorder(Resort, Gondola_lifts), y = Gondola_lifts)) +
  geom_bar(stat = "identity", fill = "forestgreen") +
  coord_flip() +
  labs(title = "Top Resorts by Number of Gondola Lifts", x = "Resort", y = "Number of Gondola Lifts")

# Filter resorts that offer Summer Skiing and have Snow Cannons
summer_skiing_with_snow_cannons <- resorts %>%
  filter(Summer_skiing == "Yes" & Snow_cannons > 0 & Continent == "Europe")
center_lat <- mean(snow_summary$Latitude, na.rm = TRUE)
center_long <- mean(snow_summary$Longitude, na.rm = TRUE)

# Create map with Leaflet
leaflet(data = summer_skiing_with_snow_cannons) %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  setView(lng = center_long, lat = center_lat, zoom = 5) %>%
  addCircleMarkers(
    ~Longitude, ~Latitude,
    color = "green",
    radius = 4,
    label = ~Resort,
    popup = ~paste("Resort:", Resort, "<br>", "Summer Skiing: Yes", "<br>", "Snow Cannons: Yes"),
    fillOpacity = 0.5
  ) %>%
  addLegend("topright", colors = "green", labels = "Summer Skiing & Snow Cannons", title = "Summer Skiing with Snow Cannons")
