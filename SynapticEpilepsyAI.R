#The SynapticEpilepsyAI function defines a deep learning model using pre-trained modules and convolutional layers to integrate genomic sequences, SNP sequences, and rare variant annotations,
#along with epigenomic and conservation features, for predicting epilepsy risk and classifying its subtypes. The model utilizes residual blocks
#for feature refinement and outputs a binary risk prediction and multi-class subtype classification.


library(keras)

# Define the 'sa' and 'ss' models (reused as-is from earlier definitions)

# Define the 'rare_disease_epilepsyAI_model'
SynapticEpilepsyAI <- function(weight_ss_file, weight_sa_file) {
  seq_length <- 51
  L <- 40  # Number of filters
  N <- c(2, 2, 2)  # Depth of the model
  W <- c(5, 5, 5)  # Filter length
  AR <- c(1, 1, 1)  # Dilation rates

  # Install and load required packages
  required_packages <- c("keras", "tensorflow")

  # Function to check and install missing packages
  install_and_load <- function(packages) {
    for (pkg in packages) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg)
      }
      library(pkg, character.only = TRUE)
    }
  }

  # Install and load all required packages
  install_and_load(required_packages)

  # Set up Keras and TensorFlow if not already done
  if (!keras::is_keras_available()) {
    message("Installing Keras...")
    keras::install_keras()
  }

  if (!tensorflow::tf_config()$available) {
    message("Installing TensorFlow...")
    tensorflow::install_tensorflow()
  }

  # Verify TensorFlow and Keras setup
  print(keras::is_keras_available())  # Should return TRUE if Keras is available
  print(tensorflow::tf_config())      # Should display TensorFlow configuration

  # Define inputs
  input0 <- layer_input(shape = c(seq_length, 20), name = "genomic_seq")
  input1 <- layer_input(shape = c(seq_length, 20), name = "epilepsy_snp_seq")
  input2 <- layer_input(shape = c(seq_length, 20), name = "human_conservation")
  input3 <- layer_input(shape = c(seq_length, 20), name = "rare_variant_annotations")
  input4 <- layer_input(shape = c(seq_length, 20), name = "epigenomic_features")

  # Combine conservation and rare variant data
  input_combined <- layer_add(name = "combined_rare_disease_features")([input2, input3, input4])

  # Pre-trained models
  ss_model <- get_ss_model(weight_ss_file)
  struc <- ss_model(input_combined)

  sa_model <- get_sa_model(weight_sa_file)
  solv <- sa_model(input_combined)

  # Process genomic sequence and SNP sequence
  conv_genomic <- layer_conv_1d(filters = L, kernel_size = 1, activation = "relu", name = "conv_genomic_seq")(input0)
  conv_snp <- layer_conv_1d(filters = L, kernel_size = 1, activation = "relu", name = "conv_snp_seq")(input1)

  # Combine all processed inputs
  combined_features <- layer_add(name = "combined_features")(
    list(conv_genomic, conv_snp, struc, solv)
  )

  # Residual block to refine features
  refined_features <- combined_features
  for (i in seq_along(N)) {
    for (j in seq_len(N[i])) {
      refined_features <- layer_conv_1d(
        filters = L, kernel_size = W[i], dilation_rate = AR[i], padding = "same",
        activation = "relu", name = paste0("residual_block_", i, "_", j)
      )(refined_features)
    }
  }

  # Dense layer for subtype classification and risk prediction
  dense_subtype <- layer_dense(units = 128, activation = "relu", name = "dense_subtype")(refined_features)
  dropout_subtype <- layer_dropout(rate = 0.3, name = "dropout_subtype")(dense_subtype)

  # Final outputs
  output_risk <- layer_dense(units = 1, activation = "sigmoid", name = "risk_prediction")(dropout_subtype)
  output_subtype <- layer_dense(units = 5, activation = "softmax", name = "subtype_classification")(dropout_subtype)

  # Create the model
  model <- keras_model(
    inputs = list(input0, input1, input2, input3, input4),
    outputs = list(output_risk, output_subtype)
  )

  # Compile the model
  model %>% compile(
    optimizer = "adam",
    loss = list("binary_crossentropy", "categorical_crossentropy"),
    loss_weights = c(0.7, 0.3),
    metrics = list("accuracy", "accuracy")
  )

  return(model)
}
