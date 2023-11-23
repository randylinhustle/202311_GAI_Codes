################################################################################
################################### Libraries ##################################
################################################################################

# Load necessary libraries
library(stm)                  # For Structural Topic Models
library(jiebaR)               # For text segmentation
library(dplyr)                # For data manipulation
library(quanteda)             # For quantitative text analysis
library(quanteda.textstats)   # For text statistics in quanteda
library(quanteda.textmodels)  # For text models in quanteda
library(sysfonts)             # For font management
library(showtext)             # For displaying text and fonts
library(furrr)                # For parallel processing
library(ggplot2)              # For plotting
library(tidyverse)            # For tidy data manipulation
library(tidytext)             # For text manipulation in tidyverse
library(stminsights)          # For insights in STM
library(igraph)               # For network analysis
library(ggraph)               # For graph plotting
library(lmtest)               # For testing linear models
library(MASS)                 # For statistical functions

################################################################################
################################ Data Preparation ##############################
################################################################################

# Read input data
input_data <- read.csv(file = 'cleaned-dataset.csv')

# Create corpus from input data
corpus <- corpus(input_data, docid_field = "thread_id", text_field = "tokenz") %>%
  tokenizers::tokenize_regex(pattern = " ") %>%
  tokens()

# Add metadata to the corpus
docvars(corpus, "thread_id") <- as.character(input_data$thread_id)
docvars(corpus, "thread_permalink") <- as.character(input_data$thread_permalink)
docvars(corpus, "thread_title") <- as.character(input_data$thread_title)

# Create document-feature matrix
dfm <- dfm(corpus)

# Trim the data to remove rare and common words
dfm_trimmed <- dfm_trim(dfm, min_docfreq = 0.0005, max_docfreq = 0.1, docfreq_type = "prop")

# Convert the dfm to STM format
stm_dfm <- convert(dfm_trimmed, to = "stm", docvars = docvars(corpus))

################################################################################
############################### Topic Modeling #################################
################################################################################

# Create stm list
stm <- list(
  documents = stm_dfm$documents,
  vocab = stm_dfm$vocab,
  meta = stm_dfm$meta
)

# Try several different models and compare performance
models <- tibble(K = ((1:10)*5)) %>%
  mutate(
    topic_model = future_map(K, ~stm(
      stm$documents, stm$vocab, K = .,
      data = stm$meta, init.type = "Spectral",
      seed = 2023
    ))
  )

# Evaluate model quality
heldout <- make.heldout(dfm_trimmed)

# Evaluate model quality for each value of K
# Note: This section includes calculations for model diagnostics
k_result <- models %>%
  mutate(
    exclusivity = map(topic_model, exclusivity),
    semantic_coherence = map(topic_model, semanticCoherence, dfm_trimmed),
    eval_heldout = map(topic_model, eval.heldout, heldout$missing),
    residual = map(topic_model, checkResiduals, dfm_trimmed),
    bound = map_dbl(topic_model, function(x) max(x$convergence$bound)),
    lfact = map_dbl(topic_model, function(x) lfactorial(x$settings$dim$K)),
    lbound = bound + lfact,
    iterations = map_dbl(topic_model, function(x) length(x$convergence$bound))
  )

# Create plots to compare model quality for different values of K
# Note: This section involves plotting diagnostics for different models
k_result %>%
  transmute(
    K,
    `Lower bound` = lbound,
    Residuals = map_dbl(residual, "dispersion"),
    `Semantic coherence` = map_dbl(semantic_coherence, mean),
    `Held-out likelihood` = map_dbl(eval_heldout, "expected.heldout")
  ) %>%
  gather(Metric, Value, -K) %>%
  ggplot(aes(K, Value, color = Metric)) +
  geom_line(size = 1.5, alpha = 0.7, show.legend = FALSE) +
  facet_wrap(~Metric, scales = "free_y") +
  labs(
    x = "K (number of topics)",
    y = NULL,
    title = "Model diagnostics by number of topics",
    subtitle = ""
  )

################################################################################
############################# Model Analysis ###################################
################################################################################

# Extract and analyze top-performing models
# Note: This section includes extraction and labeling of top models
model_15 <- models %>% filter(K == 15) %>% pull(topic_model) %>% .[[1]]
model_20 <- models %>% filter(K == 20) %>% pull(topic_model) %>% .[[1]]
model_25 <- models %>% filter(K == 25) %>% pull(topic_model) %>% .[[1]]
model_30 <- models %>% filter(K == 30) %>% pull(topic_model) %>% .[[1]]
model_35 <- models %>% filter(K == 35) %>% pull(topic_model) %>% .[[1]]

# Print the top 10 terms for each topic in each of the best models
T_15 <- labelTopics(model_15, n = 10)
capture.output(T_15, file = "T_15.txt")
T_20 <- labelTopics(model_20, n = 10)
capture.output(T_20, file = "T_20.txt") 
T_25 <- labelTopics(model_25, n = 10)
capture.output(T_25, file = "T_25.txt") 
T_30 <- labelTopics(model_30, n = 10)
capture.output(T_30, file = "T_30.txt") 
T_35 <- labelTopics(model_35, n = 10)
capture.output(T_35, file = "T_35.txt") 

# Find most relevant documents for each topic in each model
findThoughts(model_15, texts = stm[["meta"]][["thread_title"]], n = 4)
findThoughts(model_20, texts = stm[["meta"]][["thread_title"]], n = 4)
findThoughts(model_35, texts = stm[["meta"]][["thread_title"]], n = 4)
findThoughts(model_35, texts = stm[["meta"]][["thread_permalink"]], n = 4)

# Visualize topic quality for each topic in each model
# Note: This section includes plotting for topic quality in different models
par(mfrow = c(2,2), mar = c(2, 2, 2, 2))
topicQuality(model_15, documents = stm$documents, main = "model_10")
topicQuality(model_20, documents = stm$documents, main = "model_15")
topicQuality(model_25, documents = stm$documents, main = "model_25")
topicQuality(model_30, documents = stm$documents, main = "model_30")
topicQuality(model_35, documents = stm$documents, main = "model_35")

################################################################################
################################ NB Regression #################################
################################################################################

# Load the required libraries for NB Regression
library(dplyr)                # For data manipulation
library(lubridate)            # For date-time functions
library(MASS)                 # For statistical functions including NB regression

# Read the CSV into a dataframe
data_frame <- read.csv(file = 'theta.csv')

# Create a new column for word length based on the "merged" column
data_frame$word_length <- nchar(as.character(data_frame$merged))

# Create a new column for days based on the "thread_created" column
start_date <- min(data_frame$thread_created, na.rm = TRUE)
data_frame$days_from_start <- as.integer(difftime(data_frame$thread_created, start_date, units = "days")) + 1

# Generate topic names based on the provided topic numbers
topic_numbers <- c(2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 18, 19, 20, 22, 25, 28, 30, 31, 33, 34)
topic_names <- paste0("Topic", topic_numbers)

# Scale the topic columns
data_frame[topic_names] <- scale(data_frame[topic_names])

# Create formulas for both dependent variables
thread_formula_string <- paste("thread_score ~", paste(topic_names, collapse = " + "))
comment_formula_string <- paste("comment_score ~", paste(topic_names, collapse = " + "))

thread_formula <- as.formula(thread_formula_string)
comment_formula <- as.formula(comment_formula_string)

# Fit negative binomial regression for thread_score
thread_model <- glm.nb(thread_formula, data = data_frame)
print(summary(thread_model))

# Fit negative binomial regression for comment_score
comment_model <- glm.nb(comment_formula, data = data_frame)
print(summary(comment_model))

# Function to convert the summary of a glm.nb model to a dataframe
summary_to_df <- function(model) {
  coef_df <- as.data.frame(summary(model)$coefficients)
  return(coef_df)
}

# Convert the summaries of both models to dataframes
thread_summary_df <- summary_to_df(thread_model)
comment_summary_df <- summary_to_df(comment_model)

# Save the dataframes to CSV files
write.csv(thread_summary_df, file = "thread_model_summary.csv", row.names = TRUE)
write.csv(comment_summary_df, file = "comment_model_summary.csv", row.names = TRUE)

# Calculate the mean and standard deviation for thread_score
thread_score_mean <- mean(data_frame$thread_score, na.rm = TRUE)
thread_score_sd <- sd(data_frame$thread_score, na.rm = TRUE)

# Calculate the mean and standard deviation for comment_score
comment_score_mean <- mean(data_frame$comment_score, na.rm = TRUE)
comment_score_sd <- sd(data_frame$comment_score, na.rm = TRUE)

# Calculate the median for thread_score
thread_score_median <- median(data_frame$thread_score, na.rm = TRUE)

# Calculate the median for comment_score
comment_score_median <- median(data_frame$comment_score, na.rm = TRUE)

# Print the results
cat("Mean of thread_score: ", thread_score_mean, "\n")
cat("Median of thread_score: ", thread_score_median, "\n")
cat("Standard Deviation of thread_score: ", thread_score_sd, "\n")
cat("Mean of comment_score: ", comment_score_mean, "\n")
cat("Median of comment_score: ", comment_score_median, "\n")
cat("Standard Deviation of comment_score: ", comment_score_sd, "\n")
