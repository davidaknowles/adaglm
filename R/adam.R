
adam_update = function(g, adam_state) {

  adam_state$iteration <- adam_state$iteration + 1

  for (param_name in names(adam_state$parameters)) {
    # Update biased first moment estimate
    adam_state$m[[param_name]] <- adam_state$beta1 * adam_state$m[[param_name]] + (1 - adam_state$beta1) * g[[param_name]]

    # Update biased second moment estimate
    adam_state$v[[param_name]] <- adam_state$beta2 * adam_state$v[[param_name]] + (1 - adam_state$beta2) * g[[param_name]]^2

    # Bias correction
    m_hat <- adam_state$m[[param_name]] / (1 - adam_state$beta1^adam_state$iteration)
    v_hat <- adam_state$v[[param_name]] / (1 - adam_state$beta2^adam_state$iteration)

    # Update parameters
    scaled_grad = m_hat / (sqrt(v_hat) + adam_state$epsilon)
    adam_state$parameters[[param_name]] <- adam_state$parameters[[param_name]] - adam_state$learning_rate * scaled_grad
    mean(scaled_grad^2)
  }

  adam_state
}

init_adam = function(
    parameters,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,
    learning_rate = 0.01
) {
  list(
    m = lapply(parameters, function(param) param*0),
    v = lapply(parameters, function(param) param*0),
    iteration = 0,
    parameters = parameters,
    beta1 = beta1,
    beta2 = beta2,
    epsilon = epsilon,
    learning_rate = learning_rate
  )
}
