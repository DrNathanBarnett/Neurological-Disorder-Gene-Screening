PrecisionNeuromuscularAI <- function(genomic_file, clinical_file, output_path, task_type = "classification", model_type = "deep_learning") {

  # Function to install and load required packages
  install_and_load_packages <- function(package_list) {
    missing_packages <- package_list[!(package_list %in% installed.packages()[, "Package"])]
    if (length(missing_packages) > 0) install.packages(missing_packages, dependencies = TRUE)
    invisible(lapply(package_list, library, character.only = TRUE))
  }

  required_packages <- c(
    "data.table", "dplyr", "tidyr", "keras", "tensorflow",
    "caret", "ggplot2", "org.Hs.eg.db", "clusterProfiler"
  )
  install_and_load_packages(required_packages)

  # Define neuromuscular disorder genes and associated conditions
  neuromuscular_genes <- list(
    MuscularDystrophies = c("DMD", "SMN1", "SMN2", "COL6A1", "COL6A2", "COL6A3", "CAPN3", "FKRP"),
    SpinalMuscularAtrophy = c("SMN1", "SMN2"),
    Myopathies = c("RYR1", "TTN", "ACTA1", "SEPN1", "TPM3", "MYH7", "TNNT1", "DES", "PLEC")
  )

  disorder_weights <- list(
    MuscularDystrophies = c("Duchenne Muscular Dystrophy" = 0.7, "Becker Muscular Dystrophy" = 0.3),
    SpinalMuscularAtrophy = c("Type I SMA" = 0.5, "Type II SMA" = 0.3, "Type III SMA" = 0.2),
    Myopathies = c("Nemaline Myopathy" = 0.4, "Central Core Disease" = 0.4, "Congenital Myopathy" = 0.2)
  )

  # Preprocess Genomic Data
  preprocess_genomic_data <- function(genomic_file) {
    genomic_data <- fread(genomic_file)
    genomic_data <- genomic_data %>%
      mutate(impact_score = case_when(
        Variant_Consequence %in% c("missense_variant", "stop_gained") ~ 3,
        Variant_Consequence %in% c("splice_region_variant", "inframe_insertion") ~ 2,
        TRUE ~ 1
      ))
    return(genomic_data)
  }

  # Identify Neuromuscular Disorder Susceptibilities
  identify_neuromuscular_susceptibilities <- function(genomic_data) {
    susceptibility <- genomic_data %>%
      filter(Gene %in% unlist(neuromuscular_genes)) %>%
      group_by(SampleID) %>%
      summarize(
        Disorder_Groups = paste(unique(names(neuromuscular_genes)[sapply(neuromuscular_genes, function(x) any(Gene %in% x))]), collapse = ", "),
        Disorder_Types = paste(unique(unlist(names(disorder_weights[sapply(neuromuscular_genes, function(x) any(Gene %in% x))]))), collapse = ", "),
        Disorder_Likelihoods = list(
          unlist(lapply(names(neuromuscular_genes), function(group) {
            if (any(Gene %in% neuromuscular_genes[[group]])) {
              sapply(names(disorder_weights[[group]]), function(disorder) {
                disorder_weights[[group]][[disorder]]
              })
            } else {
              NULL
            }
          }))
        )
      ) %>%
      unnest(cols = c(Disorder_Likelihoods))
    return(susceptibility)
  }

  # Feature Engineering
  feature_engineering <- function(genomic_data, clinical_data, susceptibility) {
    features <- genomic_data %>%
      group_by(SampleID) %>%
      summarize(
        rare_variant_count = n(),
        pathogenic_score = sum(impact_score)
      ) %>%
      left_join(clinical_data, by = "SampleID") %>%
      left_join(susceptibility, by = "SampleID")
    return(features)
  }

  # Train Model
  train_model <- function(features, model_type) {
    set.seed(123)
    train_index <- createDataPartition(features$Diagnosis, p = 0.8, list = FALSE)
    train_data <- features[train_index, ]
    test_data <- features[-train_index, ]

    x_train <- as.matrix(train_data %>% select(-Diagnosis))
    x_test <- as.matrix(test_data %>% select(-Diagnosis))
    y_train <- as.matrix(train_data$Diagnosis)
    y_test <- as.matrix(test_data$Diagnosis)

    if (model_type == "deep_learning") {
      model <- keras_model_sequential() %>%
        layer_dense(units = 64, activation = "relu", input_shape = ncol(x_train)) %>%
        layer_dropout(rate = 0.3) %>%
        layer_dense(units = 32, activation = "relu") %>%
        layer_dropout(rate = 0.3) %>%
        layer_dense(units = 1, activation = "sigmoid")

      model %>% compile(
        optimizer = optimizer_adam(learning_rate = 0.001),
        loss = "binary_crossentropy",
        metrics = c("accuracy")
      )

      history <- model %>% fit(
        x = x_train,
        y = y_train,
        epochs = 50,
        batch_size = 16,
        validation_split = 0.2,
        verbose = 2
      )

      evaluation <- model %>% evaluate(x_test, y_test)
      return(list(model = model, history = history, evaluation = evaluation))
    }
  }

  # Generate Neuromuscular Disorder Report
  generate_neuromuscular_report <- function(features, output_path, patient_id) {
    patient_data <- features %>% filter(SampleID == patient_id)
    if (nrow(patient_data) == 0) stop("Patient ID not found in the features data.")

    rare_variant_count <- patient_data$rare_variant_count[1]
    pathogenic_score <- patient_data$pathogenic_score[1]
    disorder_groups <- patient_data$Disorder_Groups[1]
    disorder_types <- patient_data$Disorder_Types[1]
    disorder_likelihoods <- unlist(patient_data$Disorder_Likelihoods[1])
    age <- patient_data$Age[1]
    gender <- patient_data$Gender[1]

    report <- paste0(
      "Neuromuscular Disorder Report for Patient: ", patient_id, "\n",
      "========================================\n",
      "\n--- Patient Overview ---\n",
      "Age: ", age, "\n",
      "Gender: ", gender, "\n",
      "\n--- Genomic Findings ---\n",
      "Rare Variant Count: ", rare_variant_count, "\n",
      "Pathogenic Score: ", pathogenic_score, "\n",
      "\n--- Disorder Groups Identified ---\n",
      disorder_groups, "\n",
      "\n--- Associated Disorders and Likelihoods ---\n"
    )

    for (disorder in names(disorder_likelihoods)) {
      report <- paste0(report, "- ", disorder, ": ", round(disorder_likelihoods[[disorder]] * 100, 2), "% likelihood\n")
    }

    report <- paste0(
      report,
      "\n--- Recommendations ---\n",
      "1. Schedule a follow-up appointment with a neuromuscular specialist to discuss these findings.\n",
      "2. Consider targeted therapies or clinical trials based on identified conditions.\n",
      "3. Share this report with family members if conditions are hereditary.\n",
      "\nThis report is generated as part of an AI-driven WES analysis pipeline for neuromuscular disorders.\n"
    )

    report_path <- file.path(output_path, paste0("neuromuscular_report_", patient_id, ".txt"))
    writeLines(report, report_path)
    return(report_path)
  }

  # Pipeline Execution
  genomic_data <- preprocess_genomic_data(genomic_file)
  susceptibility <- identify_neuromuscular_susceptibilities(genomic_data)
  clinical_data <- fread(clinical_file)
  features <- feature_engineering(genomic_data, clinical_data, susceptibility)
  model_results <- train_model(features, model_type)
  export_results(features, model_results$model, output_path, model_type)

  # Generate Reports for All Patients
  lapply(features$SampleID, function(patient_id) {
    generate_neuromuscular_report(features, output_path, patient_id)
  })

  return(list(features = features, model = model_results$model, history = model_results$history, evaluation = model_results$evaluation))
}
