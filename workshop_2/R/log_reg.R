required_packages <- c("nnet")
# check if packages are installed and install if needed
new_packages <- required_packages[!
    (required_packages %in% installed.packages()[, "Package"])
]
if (length(new_packages)) {
  install.packages(new_packages, repos = "http://cran.us.r-project.org")
}
# load required packages
lapply(required_packages, require, character.only = TRUE)

x_cat <- read.csv("workshop_2/data/x_create_vote_data.csv", header = FALSE)
y_cat <- read.csv("workshop_2/data/y_create_vote_data.csv", header = FALSE)

reg_y_max <- max(y_cat)

colnames(y_cat)[1] <- "Dependent_Var"
colnames(x_cat)[1] <- "Covariate_A"
colnames(x_cat)[2] <- "Covariate_B"
colnames(x_cat)[3] <- "Covariate_C"

df <- data.frame(y_cat, x_cat)

model <- multinom(Dependent_Var ~ Covariate_A + Covariate_B + Covariate_C,
  data = df
)

if (!dir.exists(file.path("outputs"))) {
  dir.create(file.path("outputs"))
}
if (!dir.exists(file.path("outputs/workshop_2"))) {
  dir.create(file.path("outputs/workshop_2"))
}

capture.output(
  summary(model),
  file = "outputs/workshop_2/Logistic Regression.txt"
)

new <- data.frame(Covariate_A = 30.0, Covariate_B = 40.0, Covariate_C = 35.0)
pred <- predict(model, new)
print(paste("Prediction: ", pred))
print(paste("Correct: ", which.max(new) - 1))
