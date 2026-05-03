# 1. SET WORKING DIRECTORY
setwd("~/HK252/Probability and Statistics/Project/archive")

# kiểm tra file
list.files()

# 2. INSTALL + LOAD PACKAGES

#install.packages("tidyverse")
#install.packages("ggcorrplot")
#install.packages("patchwork")
#install.packages("pROC")
#install.packages("viridis")

library(tidyverse)
library(ggcorrplot)
library(patchwork)
library(pROC)
library(viridis)
library(caret)
library(randomForest)

# 3. LOAD DATA
data <- read.csv("data.csv")

# convert categorical → numeric
data$infill_pattern <- ifelse(data$infill_pattern == "grid", 0, 1)
data$material <- ifelse(data$material == "abs", 0, 1) 

head(data)
str(data)

# 4. PREPROCESSING
# checking NA

apply(is.na(data), 2, which)
apply(is.na(data), 2, sum)
apply(is.na(data), 2, mean)

# 5. Descriptive Statistics
# Histogram

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, ggcorrplot, patchwork, caret, randomForest, viridis)

summary(data)  

distribution_data <- data %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "measurement")

ggplot(distribution_data, aes(x = measurement, group = feature)) +
  geom_density(alpha = 0.3, fill = "gray50") +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  theme_minimal() +
  labs(title = NULL, x = "Measured Range", y = "Density") +
  theme(legend.position = "none")

boxplot_data <- distribution_data %>%
  filter(!feature %in% c("infill_pattern", "material"))

ggplot(boxplot_data, aes(x = feature, y = measurement)) +
  geom_violin(alpha = 0.1, color = "gray80", fill = NA) +
  geom_boxplot(width = 0.1, outlier.shape = 21, fill = "white") +
  facet_wrap(~ feature, scales = "free", nrow = 2) +
  theme_light() +
  labs(title = NULL, x = NULL, y = "Value") +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(),
        strip.background = element_rect(fill = "gray20"))

correlation_matrix <- round(cor(data), 2)

ggcorrplot(correlation_matrix,
           hc.order = TRUE,
           type = "lower",
           lab = TRUE,
           outline.color = "white",
           colors = c("#d73027", "#f7f7f7", "#4575b4"),
           title = "", 
           legend.title = "Pearson R")

create_comparison <- function(x_var, y_var, x_lab, y_lab) {
  ggplot(data, aes(x = .data[[x_var]], y = .data[[y_var]], shape = factor(material))) +
    geom_point(size = 2, alpha = 0.6, color = "black") +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.5, color = "black") +
    theme_classic() +
    scale_shape_discrete(labels = c("ABS", "PLA"), name = "Filament") +
    labs(x = x_lab, y = y_lab, title = NULL)
}

fig_a <- create_comparison("layer_height", "roughness", "Resolution (mm)", "Surface Roughness")
fig_b <- create_comparison("fan_speed", "tension_strenght", "Fan Speed (%)", "Tensile Strength (MPa)")
fig_c <- create_comparison("infill_pattern", "elongation", "Pattern Type (0:Grid, 1:Other)", "Ductility (%)")

(fig_a + fig_b) / fig_c 

# 6. Multiple Linear Regression

model_roughness_1 <- lm(formula = roughness ~ layer_height + wall_thickness + 
                          infill_density + infill_pattern + nozzle_temperature + 
                          bed_temperature + print_speed + material, data = data)

summary(model_roughness_1)
model_roughness_2 <- lm(formula = roughness ~ layer_height + nozzle_temperature + 
                          bed_temperature + print_speed + material, data = data)

summary(model_roughness_2)

mae_MLR <- MAE(data$roughness, predict(model_roughness_2))
nmae_MLR <- (mae_MLR / (max(data$roughness) - min(data$roughness)))
print(nmae_MLR)

rmse_MLR <- sqrt(mean((data$roughness - predict(model_roughness_2, data))^2))
print(rmse_MLR)


model_tensile_1 <- lm(formula = tension_strenght ~ layer_height + wall_thickness + 
                          infill_density + infill_pattern + nozzle_temperature + 
                          bed_temperature + print_speed + material, data = data)

summary(model_tensile_1)

model_tensile_2 <- lm(formula = tension_strenght ~ layer_height + wall_thickness
                      + infill_density + nozzle_temperature + bed_temperature + 
                        material, data = data)

summary(model_tensile_2)

mae_MLR <- MAE(data$tension_strenght, predict(model_tensile_2))
nmae_MLR <- (mae_MLR / (max(data$tension_strenght) - min(data$tension_strenght)))
print(nmae_MLR)

rmse_MLR <- sqrt(mean((data$tension_strenght - predict(model_tensile_2, data))^2))
print(rmse_MLR)

model_elongation_1 <- lm(formula = elongation ~ layer_height + wall_thickness + 
                        infill_density + infill_pattern + nozzle_temperature + 
                        bed_temperature + print_speed + material, data = data)

summary(model_elongation_1)

model_elongation_2 <- lm(formula = elongation ~ layer_height + print_speed +
                        infill_density + nozzle_temperature + bed_temperature + 
                        material, data = data)

summary(model_elongation_2)

mae_MLR <- MAE(data$elongation, predict(model_elongation_2))
nmae_MLR <- (mae_MLR / (max(data$elongation) - min(data$elongation)))
print(nmae_MLR)

rmse_MLR <- sqrt(mean((data$elongation - predict(model_elongation_2, data))^2))
print(rmse_MLR)

# 7. Random Forest Model
set.seed(75) # seed for reproducibility
train.rows <- sample(rownames(data), dim(data)[1] * 0.8)
test.rows <- setdiff(rownames(data), train.rows)
train.data <- data[train.rows,]
test.data <- data[test.rows,]

# Predict roughness while excluding tension strength and elongation
cols_to_remove <- c("tension_strenght", "elongation", "quality_binary", "quality")
train.data <- train.data[, !colnames(train.data) %in% cols_to_remove]
test.data <- test.data[, !colnames(test.data) %in% cols_to_remove]

set.seed(75) # seed for randomness in model
# Define cross-validation parameters
control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

# Train random forest model
rf_model <- train(roughness ~ ., data = train.data, method = "rf",
                  trControl = control)

# Make predictions
rf_model.predict1 <- predict(rf_model, newdata = train.data)
rf_model.predict2 <- predict(rf_model, newdata = test.data)

# Calculate NMAE for training data
nmae_train <- mean(abs(rf_model.predict1 - train.data$roughness)) / 
  (max(train.data$roughness) - min(train.data$roughness))

# Calculate NMAE for test data
nmae_test <- mean(abs(rf_model.predict2 - test.data$roughness)) / 
  (max(test.data$roughness) - min(test.data$roughness))

print(paste("NMAE_Train__RFM_(Roughness):", round(nmae_train, 7)))
print(paste("NMAE_Test_RFM_(Roughness):", round(nmae_test, 7)))

#Calculate RMSE

rmse_train <- sqrt(mean((rf_model.predict1 - train.data$roughness)^2))
rmse_test <- sqrt(mean((rf_model.predict2 - test.data$roughness)^2))
print(paste("RMSE_Train__RFM_(Roughness):", round(rmse_train, 7)))
print(paste("RMSE_Test_RFM_(Roughness):", round(rmse_test, 7)))

Sactual <- test.data$roughness
predict_rf <- rf_model.predict2
sse <- sum((actual - predict_rf) ^ 2)
sst <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - sse/sst
print(rsq)

train.data.ts <- data[train.rows,]
test.data.ts <- data[test.rows,]

cols_to_remove_ts <- c("roughness", "elongation", "quality_binary", "quality")
train.data.ts <- train.data.ts[, !colnames(train.data.ts) %in% cols_to_remove_ts]
test.data.ts <- test.data.ts[, !colnames(test.data.ts) %in% cols_to_remove_ts]

set.seed(75) 

rf_model_ts <- train(tension_strenght ~ ., 
                     data = train.data.ts, 
                     method = "rf",
                     trControl = control)

predict_ts_train <- predict(rf_model_ts, newdata = train.data.ts)
predict_ts_test <- predict(rf_model_ts, newdata = test.data.ts)

nmae_train_ts <- mean(abs(predict_ts_train - train.data.ts$tension_strenght)) / 
  (max(train.data.ts$tension_strenght) - min(train.data.ts$tension_strenght))

nmae_test_ts <- mean(abs(predict_ts_test - test.data.ts$tension_strenght)) / 
  (max(test.data.ts$tension_strenght) - min(test.data.ts$tension_strenght))

print(paste("NMAE_Train_(Tensile Strength):", round(nmae_train_ts, 7)))
print(paste("NMAE_Test_(Tensile Strength):", round(nmae_test_ts, 7)))

rmse_train_ts <- sqrt(mean((predict_ts_train - train.data.ts$tension_strenght)^2))
rmse_test_ts <- sqrt(mean((predict_ts_test - test.data.ts$tension_strenght)^2))
print(paste("RMSE_Train__RFM_(Tensile Strength):", round(rmse_train_ts, 7)))
print(paste("RMSE_Test_RFM_(Tensile Strength):", round(rmse_test_ts, 7)))

actual_ts <- test.data.ts$tension_strenght
sse_ts <- sum((actual_ts - predict_ts_test) ^ 2)
sst_ts <- sum((actual_ts - mean(actual_ts)) ^ 2)
rsq_ts <- 1 - sse_ts/sst_ts
print(paste("R-squared (Tensile Strength):", round(rsq_ts, 4)))

train.data.el <- data[train.rows,]
test.data.el <- data[test.rows,]

cols_to_remove_el <- c("roughness", "tension_strenght", "quality_binary", "quality")
train.data.el <- train.data.el[, !colnames(train.data.el) %in% cols_to_remove_el]
test.data.el <- test.data.el[, !colnames(test.data.el) %in% cols_to_remove_el]

set.seed(75) 

rf_model_el <- train(elongation ~ ., 
                     data = train.data.el, 
                     method = "rf",
                     trControl = control)

predict_el_train <- predict(rf_model_el, newdata = train.data.el)
predict_el_test <- predict(rf_model_el, newdata = test.data.el)

nmae_train_el <- mean(abs(predict_el_train - train.data.el$elongation)) / 
  (max(train.data.el$elongation) - min(train.data.el$elongation))

nmae_test_el <- mean(abs(predict_el_test - test.data.el$elongation)) / 
  (max(test.data.el$elongation) - min(test.data.el$elongation))

print(paste("NMAE_Train_(Elongation):", round(nmae_train_el, 7)))
print(paste("NMAE_Test_(Elongation):", round(nmae_test_el, 7)))

rmse_train_el <- sqrt(mean((predict_el_train - train.data.el$elongation)^2))
rmse_test_el <- sqrt(mean((predict_el_test - test.data.el$elongation)^2))
print(paste("RMSE_Train__RFM_(Elongation):", round(rmse_train_el, 7)))
print(paste("RMSE_Test_RFM_(Elongation):", round(rmse_test_el, 7)))

actual_el <- test.data.el$elongation
sse_el <- sum((actual_el - predict_el_test) ^ 2)
sst_el <- sum((actual_el - mean(actual_el)) ^ 2)
rsq_el <- 1 - sse_el/sst_el
print(paste("R-squared (Elongation):", round(rsq_el, 4)))
