required_packages <- c("readr", "meta")
# check if packages are installed and install if needed
new_packages <- required_packages[!
    (required_packages %in% installed.packages()[, "Package"])
]
if (length(new_packages)) {
  install.packages(new_packages, repos = "http://cran.us.r-project.org")
}
# load required packages
lapply(required_packages, require, character.only = TRUE)

data <- read_csv("workshop_1/data/data.csv")

if (!dir.exists(file.path("outputs"))) {
  dir.create(file.path("outputs"))
}
if (!dir.exists(file.path("outputs/workshop_1"))) {
  dir.create(file.path("outputs/workshop_1"))
}

or_meta_bin <- metabin(
  event.e = Number_affected_experimental,
  n.e = Number_affected_experimental + Number_not_affected_experimental,
  event.c = Number_affected_control,
  n.c = Number_affected_control + Number_not_affected_control,
  studlab = Study_name,
  data = data,
  sm = "OR",
  method = "MH",
  MH.exact = TRUE,
  fixed = FALSE,
  random = TRUE,
  method.tau = "PM",
  hakn = TRUE,
)

pdf(file = "outputs/workshop_1/OR Meta Analysis Forest Plot.pdf",
  width = 11, height = 3
)
metafor::forest(
  or_meta_bin,
  sortvar = TE,
  print.tau2 = TRUE,
  leftlabs = c("Study", "Affected","Not affected","Affected","Not affected"),
  lab.e = "Experimental",
  lab.c = "Control",
)
dev.off()

capture.output(
  summary(or_meta_bin),
  file = "outputs/workshop_1/OR Meta Analysis Summary.txt"
)

rr_meta_bin <- metabin(
  event.e = Number_affected_experimental,
  n.e = Number_affected_experimental + Number_not_affected_experimental,
  event.c = Number_affected_control,
  n.c = Number_affected_control + Number_not_affected_control,
  studlab = Study_name,
  data = data,
  sm = "RR",
  method = "MH",
  MH.exact = TRUE,
  fixed = FALSE,
  random = TRUE,
  method.tau = "PM",
  hakn = TRUE,
)

pdf(file = "outputs/workshop_1/RR Meta Analysis Forest Plot.pdf",
  width = 11, height = 3
)
metafor::forest(
  rr_meta_bin,
  sortvar = TE,
  print.tau2 = TRUE,
  leftlabs = c("Study", "Affected","Not affected","Affected","Not affected"),
  lab.e = "Experimental",
  lab.c = "Control",
)
dev.off()

capture.output(
  summary(rr_meta_bin),
  file = "outputs/workshop_1/RR Meta Analysis Summary.txt"
)
