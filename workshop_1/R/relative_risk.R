# --- 1. Set up dependencies for script ---
# define required packages
required_packages <- c("readr", "epitools")
# check if packages are installed and install if needed
new_packages <- required_packages[!
    (required_packages %in% installed.packages()[, "Package"])
]
if (length(new_packages)) {
  install.packages(new_packages, repos = "http://cran.us.r-project.org")
}
# load required packages
lapply(required_packages, require, character.only = TRUE)

# --- 2. Load and preprocess data ---
# load data
data <- read_csv("workshop_1/data/data.csv")
# create 2x2 contigency table
group <- c("Experimental", "Control")
outcome <- c("Affected", "Not affected")
data <- matrix(c(
    sum(data$Number_affected_experimental),
    sum(data$Number_not_affected_experimental),
    sum(data$Number_affected_control),
    sum(data$Number_not_affected_control)), nrow = 2, ncol = 2, byrow = TRUE
)
# label the dimensions
dimnames(data) <- list("Group" = group, "Outcome" = outcome)

# --- 3. Run analysis and show results ---
rr <- riskratio.wald(data, rev = c("both"))
print(rr)
