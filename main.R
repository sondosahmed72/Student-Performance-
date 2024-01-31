# Load necessary libraries for data manipulation, machine learning and visualization
library(tidyverse)
library(caret) # linear Regression
library(rpart) # Decision Tree algorithm
library(nnet)  # Logistic Regression algorithm
library(e1071) # Support Vector Machine (SVM)
library(randomForest) # Random Forest algorithm
library(ggplot2) #boxplots

# Read dataset 'students' 
students <- read.csv("StudentsPerformance.csv", header = TRUE)

# Replace empty strings
students$internet[students$internet == ""] <- NA

# Calculate the mode (most common value) for 'internet' column
mode_internet <- names(sort(table(students$internet), decreasing = TRUE))[1]

# Impute missing values in 'internet' column with the mode
students$internet[is.na(students$internet)] <- mode_internet

# Convert 'internet' column to a factor type if it is character type
students$internet <- as.factor(students$internet)
# Remove the first column from the dataset
students <- students[,-1]

# Display the modified dataframe``
students

# Count and print the total missing values
total_missing_values <- sum(is.na(students))
print(paste("Total missing values in the dataset:", total_missing_values))

# Clean the data by removing rows with missing values
students <- na.omit(students)


# Identify and count duplicate rows in the dataset
duplicates <- duplicated(students) | duplicated(students, fromLast = TRUE)
print(paste("Number of duplicate rows:", sum(duplicates)))

# Convert 'sex' column values from 'M'/'F' to 'Male'/'Female'
students$sex <- ifelse(students$sex == "Male", "M", ifelse(students$sex == "Female", "F", students$sex))

# Display the modified dataframe
students

# Create scatter plots to visualize the relationship between study time and grades (G1, G2, G3)
ggplot(students, aes(x = studytime, y = G3)) + geom_point() + labs(title = "Study Time vs Final Grade")
ggplot(students, aes(x = studytime, y = G1)) + geom_point() + labs(title = "Study Time vs First Period Grade")
ggplot(students, aes(x = studytime, y = G2)) + geom_point() + labs(title = "Study Time vs Second Period Grade")

# Plot sorted grades for G1, G2, and G3
plot(sort(students$G1), type = 'l', col = 'blue', xlab = 'Student ID', ylab = 'Grades', main = 'Sorted G1 Grades')
plot(sort(students$G2), type = 'l', col = 'red', xlab = 'Student ID', ylab = 'Grades', main = 'Sorted G2 Grades')
plot(sort(students$G3), type = 'l', col = 'green', xlab = 'Student ID', ylab = 'Grades', main = 'Sorted G3 Grades')

# Feature Engineering: Create a new feature 'Sum_Grades' as the sum of G1, G2, and G3
students$Sum_Grades <- students$G1 + students$G2 + students$G3
# Remove the original grade columns (G1, G2, G3) and plot a scatter plot for the sum of grades
students <- subset(students, select = -c(G1, G2, G3))
ggplot(students, aes(x = studytime, y = Sum_Grades)) + geom_point() + labs(title = "Study Time vs Sum of 3 Grades")
plot(sort(students$Sum_Grades), type = 'l', col = 'orange', xlab = 'Student ID', ylab = 'Grades', main = 'Sum of 3 Grades')

# Print a summary of the entire dataset
print(summary(students))

# Calculate IQR for each numerical column in the dataset
numerical_columns <- sapply(students, is.numeric)
iqr_values <- apply(students[, numerical_columns], 2, IQR)
print("IQR for each numerical column:")
print(iqr_values)

# Optional: Use IQR for outlier detection in a specific column (e.g., 'AVG_grades')
Q1 <- quantile(students$AVG_grades, 0.25)
Q3 <- quantile(students$AVG_grades, 0.75)
IQR <- Q3 - Q1

# Define outlier thresholds
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Identify outliers
outliers <- students$AVG_grades < lower_bound | students$AVG_grades > upper_bound
print(paste("Number of outliers in AVG_grades:", sum(outliers)))
# Identify numerical columns
numerical_columns <- sapply(students, is.numeric)

# Create boxplots for each numerical column
for(col_name in names(students)[numerical_columns]) {
  p <- ggplot(students, aes_string(x = "factor(1)", y = col_name)) +
    geom_boxplot() +
    labs(title = paste("Boxplot of", col_name), x = "", y = col_name) +
    theme_minimal()
  
  print(p)
}

# Encode categorical variables 'romantic', 'internet', 'school', and 'sex' into numeric formats
students$romantic <- as.integer(students$romantic == "yes")
students$internet <- as.integer(students$internet == "yes")
students$school <- as.integer(students$school == "GP")
students$sex <- as.integer(students$sex == "Male")

# Prepare the model by calculating the average grades and categorizing them
students$AVG_grades <- students$Sum_Grades / 3
# Categorize AVG_grades into Low, Average, High based on quantile thresholds
low_threshold <- quantile(students$AVG_grades, 1/3)
high_threshold <- quantile(students$AVG_grades, 2/3)
students$Grade_Category <- cut(students$AVG_grades, breaks = c(-Inf, low_threshold, high_threshold, Inf), labels = c("Low", "Average", "High"), include.lowest = TRUE)
# Convert Grade_Category to a numeric factor
students$Grade_Category_Num <- as.numeric(factor(students$Grade_Category))
# Drop the 'Sum_Grades' column
students <- students[, -which(names(students) == "Sum_Grades")]

# Split the data into training and testing sets with a fixed random seed for reproducibility
set.seed(123)
splitIndex <- createDataPartition(students$Grade_Category_Num, p = .8, list = FALSE)
train <- students[splitIndex,]
test <- students[-splitIndex,]

# Define various models for training
models <- list(
  "Linear Regression" = lm(AVG_grades ~ ., data = train),
  "Decision Tree" = rpart(AVG_grades ~ ., data = train),
  "Logistic Regression" = glm(AVG_grades ~ ., data = train),
  "SVM" = svm(AVG_grades ~ ., data = train),
  "Random Forest" = randomForest(AVG_grades ~ ., data = train)
)

# Train models and evaluate their performance using R-squared
results <- lapply(names(models), function(model_name) {
  model <- models[[model_name]]
  prediction <- predict(model, newdata = test)
  
  # Calculate R-squared for performance evaluation
  rss <- sum((prediction - test$AVG_grades)^2)
  tss <- sum((test$AVG_grades - mean(test$AVG_grades))^2)
  r_squared <- 1 - rss/tss
  r_squared_percent <- r_squared * 100  # Convert R-squared to percentage
  
  # Print R-squared percentage with model name
  cat(paste("R-squared for", model_name, ":", round(r_squared_percent, 2), "%\n"))
  
  return(r_squared_percent)
})

# Create a dataframe to display model accuracy and plot a comparison
accuracy_df <- data.frame(Model = names(models), R_Squared_Percentage = unlist(results))
ggplot(accuracy_df, aes(x = Model, y = R_Squared_Percentage, fill = Model)) +
  geom_col() +
  scale_fill_brewer(palette = "Set3")
  theme_minimal() + 
  labs(title = "Model Performance Comparison (Higher R-Squared % is better)", x = "Model", y = "R-Squared %")

