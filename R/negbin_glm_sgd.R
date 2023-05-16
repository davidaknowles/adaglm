loglik_nb <- function(conc, mu, y, w)
  sum(w*(lgamma(conc + y) - lgamma(conc) - lgamma(y + 1) + conc * log(conc) +
           y * log(mu + (y == 0)) - (conc + y) * log(conc + mu)))

loglik_poisson <- function(conc, mu, y, w)
  sum(w*( y * log(mu + (y == 0)) - mu - lgamma(y + 1)))

grad_conc <- function(conc, mu, y, w) # derivative wrt conceta (concentration)
  sum(w*(digamma(conc + y) - digamma(conc) + log(conc) +
           1 - log(conc + mu) - (y + conc)/(mu + conc)))

neg_hess_conc <- function(conc, mu, y, w) # just for getting SE
  sum(w*( - trigamma(conc + y) + trigamma(conc) - 1/conc +
            2/(mu + conc) - (y + conc)/(mu + conc)^2))

grad_nb_mu <- function(conc, mu, y, w)
  w * (y / (mu + (y == 0)) - (conc + y) / (conc + mu))

grad_poisson_mu <- function(conc, mu, y, w)
  w * (y / (mu + (y == 0)) - 1)

neg_hess_mu <- function(conc, mu, y, w)
  w * (y / (mu + (y == 0))^2 + (conc + y) / (conc + mu)^2)

neg_hess_poisson_mu <- function(conc, mu, y, w)
  w * (y / (mu + (y == 0))^2 )

hess_conc_mu <- function(conc, mu, y, w)
  w * (y - mu) / (mu + conc)^2


one_epoch = function(adam_state, X, y, w, batches, grad_mu = grad_nb_mu, b = NULL, conc = NULL, learn_beta = T, learn_conc = T) {
  for (batch in batches) {
    y_batch = y[batch]
    w_batch = w[batch]
    X_batch = X[batch,,drop=F]
    if (learn_beta) b = adam_state$parameters$b
    mu = exp(X_batch %*% b)

    if (learn_conc) conc = exp(adam_state$parameters$logconc)

    g = list()
    # could divide by conc here ala Andrew's paper (https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf)
    if (learn_beta) g$b = -t(X_batch) %*%  (mu * grad_mu(conc, mu, y_batch, w_batch)) # minus because we do sgD
    if (learn_conc) g$logconc = -conc * grad_conc(conc, mu, y_batch, w_batch)

    adam_state <- adam_update(g, adam_state)

  }

  adam_state
}

sgd_glm = function(
    X,
    y,
    w = numeric(nrow(X))+1,
    b = numeric(ncol(X)), # initial coefficients
    conc = 10.,
    learn_beta = T,
    learn_conc = T,
    true_beta = NULL,
    batch_size = 100,
    epochs = 100,
    loglik_tol = 0.01,
    verbose = F,
    ... # to pass to init_adam
) {

  if (is.infinite(conc)) {
    grad_mu = grad_poisson_mu
    loglik = loglik_poisson
    stopifnot(!learn_conc)
  } else {
    grad_mu = grad_nb_mu
    loglik = loglik_nb
  }

  N = nrow(X)
  P = ncol(X)
  parameters = list()
  if (learn_conc) parameters$logconc = log(conc)
  if (learn_beta) parameters$b = b

  adam_state = init_adam(parameters, ...)

  batches = split(
    sample.int(N),
    ceiling(seq(1,N)/batch_size))

  true_cor = NA

  old_loglik = -Inf
  logliks = c()

  if (verbose) cat("epoch", "loglik", "conc", "mean(abs(b))", "true_cor", "\n")

  for (epoch in 1:epochs){

    adam_state = one_epoch(adam_state, X, y, w, batches, grad_mu = grad_mu, b = b, conc = conc, learn_beta = learn_beta, learn_conc = learn_conc)

    if (learn_beta) b = adam_state$parameters$b
    if (learn_conc) conc = exp(adam_state$parameters$logconc)

    if (!is.null(true_beta)) true_cor = cor(true_beta, b)

    ll = loglik(conc, exp(X %*% b), y, w)
    if (verbose) cat(epoch, ll, conc, mean(abs(b)), true_cor, "\n")
    logliks = c(logliks, ll)
    if (abs(ll - old_loglik) < loglik_tol) break
    old_loglik = ll

  }

  list(logliks = logliks, adam_state = adam_state)
}

# ... is passed to sgd_glm
smart_fit_nb_glm = function(X, y, w, consider_poisson = T, true_b = NULL, ...) {

  # 1. lm based initialization
  P = ncol(X)
  b = if (P > 0) solve(t(X) %*% X, t(X) %*% log(y + 0.1)) else numeric(0)
  # loglik_poisson(1, exp(X %*% b), y, w) # this is surprisingly bad

  # 2. fit beta under Poisson GLM
  res = sgd_glm(X, y, b = b, true_beta = true_b, conc = Inf, learn_beta = T, learn_conc = F, ...)

  b_pois = res$adam_state$parameters$b
  mu_pois = exp(X %*% b_pois)
  ll_poi = loglik_poisson(1, mu_pois, y, w)

  #  3. fit concentration. it doesn't seem to hurt to also update beta at this stage.
  # should we regularize (log)conc a bit? if the data is Poisson (not overdispersed) then i believe
  # the likelihood is flat for conc -> inf, which is a bit weird for optimization
  conc = 1
  res = sgd_glm(X, y, b = b_pois, conc = conc, true_beta = true_b, learn_beta = T, learn_conc = T, ...)
  b = res$adam_state$parameters$b
  conc = exp(res$adam_state$parameters$logconc)

  mu = exp(X %*% b)
  ll = loglik_nb(conc, mu, y, w)

  if (consider_poisson && (ll_poi > ll)) {
    b = b_pois
    ll = ll_poi
    conc = Inf
    mu = mu_pois
    dia = mu * grad_poisson_mu(conc, mu, y, w) - mu^2 * neg_hess_poisson_mu(conc, mu, y, w)
    hess_b = t(X) %*% sweep(X, 1, dia, "*")
    cov_b = chol2inv(chol(-hess_b))
    se_b = sqrt(diag(cov_b))

    se_logconc = NA
  } else {

    hess_c = -neg_hess_conc(conc, mu, y, w) * conc^2 + grad_conc(conc, mu, y, w) * conc
    # se_logconc = sqrt(-1/hess_c)

    dia = mu * grad_nb_mu(conc, mu, y, w) - mu^2 * neg_hess_mu(conc, mu, y, w)
    hess_b = t(X) %*% sweep(X, 1, dia, "*")

    # cov_b = chol2inv(chol(-hess_b))
    # se_b = sqrt(diag(cov_b))

    hess_b_conc = t(X) %*% (mu * hess_conc_mu(conc, mu, y, w) * conc) # off diagonal of hessian

    joint_hess = rbind( cbind(hess_b, hess_b_conc), c(hess_b_conc, hess_c) )
    joint_cov = chol2inv(chol(-joint_hess))

    joint_se = sqrt(diag(joint_cov))
    se_beta = joint_se[1:P]
    se_logconc = joint_se[P+1] # slightly wider
  }

  names(b) = colnames(X)

  list(b = b, conc = conc, loglik = ll, se_beta = se_beta, se_logconc = se_logconc)
}

nb_glm_wrapper = function(
    formula,
    data,
    weights,
    conc = NULL,
    return_model = FALSE,
    return_x = FALSE,
    return_y = FALSE,
    contrasts = NULL,
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
  offset <- model.offset(mf) # can't currently handle this

  myfit = if (is.null(conc)) smart_fit_nb_glm(X, y, w, ...) else sgd_glm(X, y, conc = conc, learn_beta = T, learn_conc = F, ...)

  # consider_poisson = T, true_b = NULL

  X_null = if ("(Intercept)" %in% colnames(X)) X[, "(Intercept)", drop = FALSE] else X[,0,drop=F]

  null_fit = sgd_glm(X_null, y, conc = myfit$conc, learn_beta = T, learn_conc = F, ...)
  null.deviance = -2. * null_fit$logliks[length(null_fit$logliks)] # might be better to fix conc here?
  eta = X %*% myfit$b

  fit = list(
    call = Call,
    rank = ncol(X), # assumes no collinearity
    df.residual = ncol(X) - ncol(X_null),
    null.deviance = null.deviance,
    df.null = ncol(X_null),
    contrasts = attr(X, "contrasts"),
    xlevels = .getXlevels(Terms, mf),
    coefficients = myfit$b,
    fitted.values = exp(eta),
    linear.predictors	= eta,
    twologlik = 2. * myfit$loglik,
    deviance = -2. * myfit$loglik,
    aic = - 2. * myfit$loglik + 2*ncol(X) + 2,
    terms = Terms,
    formula = as.vector(attr(Terms, "formula")),
    theta = myfit$conc,
    se_beta = myfit$se_beta,
    se_logconc = myfit$se_logconc
  )

  if (return_x) fit$x = x
  if (return_y) fit$y = y
  if (return_model) fit$model <- mf

  class(fit) <- c("negbin", "glm", "lm")

  fit
}

anova.negbin <- function(..., test = "Chisq")
{
  if(test != "Chisq")
      warning("only Chi-squared LR tests are implemented")
  mlist <- list(...)
  nt <- length(mlist)
  dflis <- sapply(mlist, function(x) x$df.residual)
  s <- sort.list(dflis)
  mlist <- mlist[s]
  if(any(!sapply(mlist, inherits, "negbin")))
    stop("not all objects are of class \"negbin\"")
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

summary.negbin = function (object,  ...) {
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



