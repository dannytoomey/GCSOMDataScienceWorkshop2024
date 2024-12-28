x_linear <- read.csv("workshop_2/data/x_create_linear_data.csv", header = FALSE)
y_linear <- read.csv("workshop_2/data/y_create_linear_data.csv", header = FALSE)

colnames(y_linear)[1] <- "Dependent_Var"
colnames(x_linear)[1] <- "Covariate_A"
colnames(x_linear)[2] <- "Covariate_B"
colnames(x_linear)[3] <- "Covariate_C"

df <- data.frame(y_linear, x_linear)
model <- lm(Dependent_Var ~ Covariate_A + Covariate_B + Covariate_C, data = df)

if (!dir.exists(file.path("outputs"))) {
  dir.create(file.path("outputs"))
}
if (!dir.exists(file.path("outputs/workshop_2"))) {
  dir.create(file.path("outputs/workshop_2"))
}

capture.output(
  summary(model),
  file = "outputs/workshop_2/Linear Regression.txt"
)

new <- data.frame(Covariate_A = 30.0, Covariate_B = 40.0, Covariate_C = 50.0)
pred <- predict(model, new, type = "response")
print(paste("Prediction: ", pred))
print(paste("Correct: ", sum(new)))
