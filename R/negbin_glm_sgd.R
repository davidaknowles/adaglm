#' Negative binomial log likelihood function
#'
#' Gives Poisson if conc is infinite.
#'
#' @param conc The concentration parameter = 1/dispersion.
#' @param mu The mean, a N-vector
#' @param y The observed counts, an N-vector.
#' @param w Observation weights.
#'
#' @returns Log likelihood (scalar)
loglik <- function(conc, mu, y, w)
  if (is.finite(conc)) {
    sum(w*(lgamma(conc + y) - lgamma(conc) - lgamma(y + 1) + conc * log(conc) +
           y * log(mu + (y == 0)) - (conc + y) * log(conc + mu)))
  } else { # Poisson
    sum(w*( y * log(mu + (y == 0)) - mu - lgamma(y + 1)))
  }

#' Gradient of the NB log likelihood wrt to the concentration
grad_conc <- function(conc, mu, y, w) # derivative wrt conceta (concentration)
    sum(w*(digamma(conc + y) - digamma(conc) + log(conc) +
             1 - log(conc + mu) - (y + conc)/(mu + conc)))

#' Gradient of the NB log likelihood wrt to the mean
grad_mu <- function(conc, mu, y, w)
  w * (y / (mu + (y == 0)) - if (is.finite(conc)) {(conc + y) / (conc + mu)} else 1)

#' Hessian of the NB log likelihood wrt to the concentration
neg_hess_conc <- function(conc, mu, y, w) # just for getting SE
  sum(w*( - trigamma(conc + y) + trigamma(conc) - 1/conc +
            2/(mu + conc) - (y + conc)/(mu + conc)^2))

#' Hessian of the NB log likelihood wrt to the mean
neg_hess_mu <- function(conc, mu, y, w)
  w * (y / (mu + (y == 0))^2 + if (is.finite(conc)) {(conc + y) / (conc + mu)^2} else 0)

#' Off-diagonal block of the Hessian of the NB log likelihood wrt to the mu and conc jointly
hess_conc_mu <- function(conc, mu, y, w)
  w * (y - mu) / (mu + conc)^2

one_epoch = function(adam_state, X, y, w, o, batches, b = NULL, conc = NULL, learn_beta = T, learn_conc = T) {
  for (batch in batches) {
    y_batch = y[batch]
    w_batch = w[batch]
    X_batch = X[batch,,drop=F]
    if (learn_beta) b = adam_state$parameters$b
    mu = exp(X_batch %*% b + o[batch])

    if (learn_conc) conc = exp(adam_state$parameters$logconc)

    g = list()
    # could divide by conc here ala Andrew's paper (https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf)
    if (learn_beta) g$b = -t(X_batch) %*%  (mu * grad_mu(conc, mu, y_batch, w_batch)) # minus because we do sgD
    if (learn_conc) g$logconc = -conc * grad_conc(conc, mu, y_batch, w_batch)

    adam_state <- adam_update(g, adam_state)

  }

  adam_state
}

get_standard_errors = function(X, y, w, o, b, conc, learn_conc = T) {

  mu = exp(X %*% b + o)

  se_logconc = NA

  dia = mu * grad_mu(conc, mu, y, w) - mu^2 * neg_hess_mu(conc, mu, y, w)
  hess_b = t(X) %*% sweep(X, 1, dia, "*")

  if (is.infinite(conc) || !learn_conc) {

    cov_b = chol2inv(chol(-hess_b))
    se_beta = sqrt(diag(cov_b))

  } else {

    hess_c = -neg_hess_conc(conc, mu, y, w) * conc^2 + grad_conc(conc, mu, y, w) * conc
    # se_logconc = sqrt(-1/hess_c)

    hess_b_conc = t(X) %*% (mu * hess_conc_mu(conc, mu, y, w) * conc) # off diagonal of hessian

    joint_hess = rbind( cbind(hess_b, hess_b_conc), c(hess_b_conc, hess_c) )
    joint_cov = chol2inv(chol(-joint_hess))

    joint_se = sqrt(diag(joint_cov))
    P = ncol(X)
    se_beta = joint_se[1:P]
    se_logconc = joint_se[P+1] # slightly wider SE than when just using hess_c

  }

  list(se_beta = se_beta, se_logconc = se_logconc)
}

#' Fit negative binomial (or Poisson) regression using Adam
#'
#' @param X NxP matrix of samples x covariates
#' @param y N-vector of observed counts
#' @param w Optional N-vector of observation weights
#' @param b Optional P-vector of coefficients. This is the initialization if b is being learned,
#' @param conc Optional concentration parameter. This is the initialization if conc is being learned.
#' @param learn_beta Whether to optimize the coefficients.
#' @param learn_conc Whether to optimize the concentration parameter.
#' @param batch_size For Adam
#' @param epochs Number of sweeps through dataset
#' @param loglik_tol Once the loglikelihood changes by less than this, stop.
#' @param verbosity Integer between 0 (only print warnings) and 3 (print at every iteration of optimization)
#' @param ... passed to init_adam so can be used to specify e.g. learning_rate.
#'
#' @returns Vector of log likelihoods through optimization (one per epoch).
#' @return adam_state list.
#'
#' @export
sgd_glm = function(
    X,
    y,
    w = numeric(nrow(X))+1,
    o = numeric(nrow(X)),
    b = numeric(ncol(X)), # initial coefficients
    conc = 10.,
    learn_beta = T,
    learn_conc = T,
    batch_size = 10000,
    epochs = 1000,
    loglik_tol = 0.1,
    verbosity = 0,
    ... # to pass to init_adam
) {

  if (is.infinite(conc)) stopifnot(!learn_conc)

  N = nrow(X)
  P = ncol(X)
  parameters = list()
  if (learn_conc) parameters$logconc = log(conc)
  if (learn_beta) parameters$b = b

  adam_state = init_adam(parameters, ...)

  batches = split(
    sample.int(N),
    ceiling(seq(1,N)/batch_size))

  old_loglik = loglik(conc, exp(X %*% b + o), y, w)

  logliks = old_loglik

  if (verbosity >= 3) {
    cat("epoch", "loglik", "conc", "\n")
    cat(0, old_loglik, conc, "\n")
  }

  ll_got_worse = F

  for (epoch in 1:epochs){

    adam_state = one_epoch(adam_state, X, y, w, o, batches, b = b, conc = conc, learn_beta = learn_beta, learn_conc = learn_conc)

    if (learn_beta) b = adam_state$parameters$b
    if (learn_conc) conc = exp(adam_state$parameters$logconc)

    ll = loglik(conc, exp(X %*% b + o), y, w)
    if (verbosity >= 3) cat(epoch, ll, conc, "\n")

    logliks = c(logliks, ll)
    if (ll+1. < old_loglik) ll_got_worse = T

    if ((epoch == epochs) || abs(ll - old_loglik) < loglik_tol) break
    old_loglik = ll

  }

  if (verbosity  >= 2) cat("Adam finished after",epoch,"epochs, log likelihood=",ll,".\n")

  if (epoch == epochs) {
    if (ll_got_worse)
      warning("Log likelihood decreased at least once during optimization. You might want to decrease learning_rate (currently ",adam_state$learning_rate,") or increase batch_size (currently ",batch_size,").") else warning("Reached maximum number of epochs before convergence. The last two log likelihoods were ",format(old_loglik, digits=5),", ",format(ll, digits=5),". You might consider increasing the max number of epochs (currently ",epochs,") or learning_rate (currently ",adam_state$learning_rate,").")
  }

  se = get_standard_errors(X, y, w, o, b, conc, learn_conc = learn_conc)

  names(b) = colnames(X)

  saturated_loglik = loglik(conc, y, y, w) # MASS:ng.glm uses this "trick"

  list(
    b = b,
    conc = conc,
    loglik = ll,
    deviance = 2 * (saturated_loglik - ll),
    logliks = logliks,
    se_beta = se$se_beta,
    se_logconc = se$se_logconc,
    epochs = epoch,
    converged = epoch < epochs
  )
}

#' Fit negative binomial (or Poisson) regression using Adam
#'
#" Jointly optimizing the coefficients and concentration parameter from arbitrary points has poor learning dynamics. The initially poor coefficients result in an artificially small concentration. An effective solution is: 1) Fit a Poisson GLM. 2) Starting with coefficients initialized from the Poisson GLM, fit a NB GLM jointly optimizing the coefficients and concentration.
#'
#' @param X NxP matrix of samples x covariates
#' @param y N-vector of observed counts
#' @param w Optional N-vector of observation weights
#' @param o Offsets
#' @param conc Concentration parameter. NULL to learn it, or can be fixed to some value. Setting conc=Inf gives Poisson GLM.
#' @param consider_poisson Whether to compare the final NB GLM fit to the Poisson GLM and choose the later if it has higher loglikelihood. .
#' @param verbosity Integer between 0 (only print warnings) and 3 (print at every iteration of optimization)
#' @param ... passed to sgd_glm
#'
#' @export
smart_fit_nb_glm = function(X, y, w, o, conc = NULL, consider_poisson = T, verbosity = 0, ...) {

  P = ncol(X)
  if (verbosity >= 1) cat("1. Linear model based initialization\n")
  b = if (P > 0) solve(t(X) %*% X, t(X) %*% log(y + 0.1)) else numeric(0)

  if (verbosity >= 1)
    cat(
      "Poisson GLM log likelihood=",
      loglik(Inf, exp(X %*% b + o), y, w),
      "\n2. Fit Poisson GLM\n")

  pois_fit = sgd_glm(X, y, w, o, b = b, conc = Inf, learn_beta = T, learn_conc = F, verbosity = verbosity, ...)

  if (verbosity >= 1) cat("3. Fit concentration parameter under NB GLM\n")
  # it doesn't seem to hurt to also update beta at this stage.
  # should we regularize (log)conc a bit? if the data is Poisson (not overdispersed) then i believe
  # the likelihood is flat for conc -> inf, which is a bit weird for optimization
  if (is.null(conc))
    conc = 1
  nb_fit = sgd_glm(X, y, w, o, b = pois_fit$b, conc = conc, learn_beta = T, learn_conc = T, verbosity = verbosity, ...)

  if (consider_poisson && (pois_fit$loglik > nb_fit$loglik)) pois_fit else nb_fit
}


#' Main function for fitting negative binomial or Poisson GLM using Adam.
#'
#' This aims to have roughly the same functionality as `base::glm`. Offsets are not currently supported however, and currently only log link can be used.
#'
#' @param formula an object of class "formula" (or one that can be coerced to that class): a symbolic description of the model to be fitted.
#' @param data an optional data frame, list or environment (or object coercible by as.data.frame to a data frame) containing the variables in the model. If not found in data, the variables are taken from environment(formula), typically the environment from which adaglm is called.
#' @param weights an optional vector of ‘prior weights’ to be used in the fitting process. Should be NULL or a numeric vector.
#' @param conc Concentration parameter. NULL to learn it, or can be fixed to some value. Setting conc=Inf gives Poisson GLM.
#' @param return_x Whether to return the model matrix
#' @param return_y Whether to return the response vector
#' @param return_model Whether to return the model frame.
#' @param contrasts an optional list. See the contrasts.arg of model.matrix.default.
#' @param verbosity Integer between 0 (only print warnings) and 3 (print at every iteration of optimization)
#' @param ... Passed to smart_fit_nb_glm, then sgd_glm, and finally init_adam. Can control the optimization through `batch_size`, `epochs`, `loglik_tol`, `beta1`, `beta2`, `epsilon` and `learning_rate`.
#'
#' @export
adaglm = function(
    formula,
    data,
    weights,
    conc = NULL,
    return_model = FALSE,
    return_x = FALSE,
    return_y = FALSE,
    contrasts = NULL,
    verbosity = 0,
    ...) {

  mf <- Call <- match.call() # mf = model frame
  m <- match(c("formula", "data", "subset", "weights", "offset"), names(mf), 0)
  mf <- mf[c(1, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval.parent(mf)
  Terms <- attr(mf, "terms")
  y <- model.response(mf, "numeric")
  ## null model support
  X <- if (!is.empty.model(Terms)) model.matrix(Terms, mf, contrasts) else matrix(NA,NROW(y),0)
  w <- model.weights(mf)
  if(!length(w)) w <- rep(1, nrow(mf))
  else if(any(w < 0)) stop("negative weights not allowed")
  o <- model.offset(mf)
  if (is.null(o)) o = y*0
  if (verbosity >= 1) cat("Fitting NB GLM.\n")
  myfit = smart_fit_nb_glm(X, y, w, o, conc = conc, verbosity = verbosity, ...)

  X_null = if ("(Intercept)" %in% colnames(X)) X[, "(Intercept)", drop = FALSE] else X[,0,drop=F]

  if (verbosity >= 1) cat("Fitting null model\n") # Note conc is fixed here.
  null_fit = sgd_glm(X_null, y, w, o, conc = myfit$conc, learn_beta = T, learn_conc = F, verbosity = verbosity, ...)

  eta = X %*% myfit$b + o
  fitted.values = exp(eta)

  stds = sqrt(fitted.values + fitted.values*fitted.values / myfit$conc)
  resid = (y - fitted.values) / stds

  fit = list(
    call = Call,
    rank = ncol(X), # assumes no colinearity
    df.residual = ncol(X) - ncol(X_null),
    resid = resid,
    null.deviance = null_fit$deviance,
    df.null = ncol(X_null),
    contrasts = attr(X, "contrasts"),
    xlevels = .getXlevels(Terms, mf),
    coefficients = myfit$b,
    fitted.values = fitted.values,
    linear.predictors	= eta,
    twologlik = 2. * myfit$loglik,
    deviance = myfit$deviance,
    aic = - 2. * myfit$loglik + 2*ncol(X) + 2,
    terms = Terms,
    formula = as.vector(attr(Terms, "formula")),
    converged = myfit$converged,
    theta = myfit$conc,
    iter = myfit$epochs,
    se_beta = myfit$se_beta,
    se_logconc = myfit$se_logconc
  )

  if (return_x) fit$x = x
  if (return_y) fit$y = y
  if (return_model) fit$model <- mf

  class(fit) <- c("adanb", "glm", "lm")

  fit
}

#' @export
anova.adanb <- function(..., test = "Chisq")
{
  if(test != "Chisq")
      warning("only Chi-squared LR tests are implemented")
  mlist <- list(...)
  nt <- length(mlist)
  dflis <- sapply(mlist, function(x) x$df.residual)
  s <- sort.list(dflis)
  mlist <- mlist[s]
  if(any(!sapply(mlist, inherits, "adanb")))
    stop("not all objects are of class \"adanb\"")
  rsp <- unique(sapply(mlist, function(x) paste(formula(x)[2L])))
  mds <- sapply(mlist, function(x) paste(formula(x)[3L]))
  ths <- sapply(mlist, function(x) x$theta)
  dfs <- dflis[s]
  lls <- sapply(mlist, function(x) x$twologlik)
  tss <- c("", paste(1L:(nt - 1L), 2:nt, sep = " vs "))
  df <- c(NA,  diff(dfs))
  x2 <- c(NA, diff(lls))
  pr <- c(NA, 1 - pchisq(x2[-1L], df[-1L]))
  out <- data.frame(Model = mds, theta = ths, Resid.df = dfs,
                    "2 x log-lik." = lls, Test = tss, df = df, LRtest = x2,
                    Prob = pr)
  names(out) <- c("Model", "theta", "Resid. df",
                  "   2 x log-lik.", "Test", "   df", "LR stat.", "Pr(Chi)")
  class(out) <- c("Anova", "data.frame")
  attr(out, "heading") <-
    c("Likelihood ratio tests of Negative Binomial Models\n",
      paste("Response:", rsp))
  out
}

#' @export
summary.adanb = function (object,  ...) {
  est.disp <- FALSE
  df.r <- object$df.residual

  aliased <- is.na(coef(object))
  if (object$rank > 0) {
    tvalue <- object$coefficients/object$se_beta
    pvalue <- 2 * pnorm(-abs(tvalue))
    coef.table <- cbind(object$coefficients, object$se_beta, tvalue, pvalue)
    dimnames(coef.table) <- list(
      names(object$coefficients),
      c("Estimate", "Std. Error", "z value", "Pr(>|z|)"))
  } else {
    coef.table <- matrix(NA, 0L, 4L)
    dimnames(coef.table) <- list(NULL, c("Estimate", "Std. Error",
                                         "t value", "Pr(>|t|)"))
  }
  keep <- match(c("call", "terms", "family", "deviance", "aic",
                  "contrasts", "df.residual", "null.deviance", "df.null",
                  "iter", "na.action"), names(object), 0L)
  ans <- c(
    object[keep],
    list(coefficients = coef.table, aliased = aliased, dispersion = 1/object$theta, df = c(
      object$rank, object$df.residual, object$rank)))
  class(ans) <- "summary.glm"
  return(ans)
}

#' Pearson residuals
#'
#' These are calculated as in scTrasform, i.e. (y-mu)/s where mu is the expected value from the GLM and s is the standard deviation s=sqrt(mu+mu^2/theta) where theta is the "concentration" parameter, i.e. 1/dispersion.
#'
#' @param object A negative binomial GLM fit (adanb).
#'
#' @returns Pearson residuals
#'
#' @export
residuals.adanb = function(object, ...) {
  object$resid
}

#' Per datapoint p-values
#'
#' Calculates both one-sided and two-sided p-values against the null hypothesis that the counts are NB distributed with mean and overdispersion from the GLM fit.
#'
#' @param fit A negative binomial GLM fit (adanb).
#' @param y Per datapoint counts.
#'
#' @returns List of
#' \item{p_two_sided}{Tests whether y is lower OR higher than expected.}
#' \item{p_lower}{p-value for y being lower than expected}
#' \item{p_higher}{p for y being higher than expected}
#'
#' @export
pointwise_pvalues = function(fit, y) {
  mu = fitted(fit)
  p_lower = pnbinom(y, mu=fitted(fit), size=fit$theta)
  p_higher = pnbinom(y, mu=mu, size=fit$theta, lower=F) + dnbinom(y, mu=mu, size=fit$theta)
  p_two_sided = pmin(p_lower, p_higher)
  list(p_two_sided=p_two_sided, p_lower=p_lower, p_higher=p_higher)
}



