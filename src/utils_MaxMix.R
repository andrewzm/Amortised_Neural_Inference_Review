V <- function(x,y,rho){
  temp <- sqrt(x^2-2*x*y*rho+y^2)
  res <- (1/x+1/y)*(1/2)*(1+temp/(x+y))
  return(res)
}

V1 <- function(x,y,rho){
  temp <- sqrt(x^2-2*x*y*rho+y^2)
  res <- (1/x^2)*(-1/2) + (1/2)*(rho/x-y/x^2)*temp^(-1)
  return(res)
}

V2 <- function(x,y,rho){
  temp <- sqrt(x^2-2*x*y*rho+y^2)
  res <- (1/y^2)*(-1/2) + (1/2)*(rho/y-x/y^2)*temp^(-1)
  return(res)
}

V12 <- function(x,y,rho){
  temp <- sqrt(x^2-2*x*y*rho+y^2)
  res <- -(1/2)*(1-rho^2)*temp^(-3)
}

fitMSP <- function(data,coord,distmax,init){
  nstat <- ncol(data)
  nrep <- nrow(data)
  allpairs <- t(combn(1:nstat,2))
  distmat <- as.matrix(dist(coord))
  distpairs <- sapply(1:nrow(allpairs),FUN=function(i){return(distmat[allpairs[i,1],allpairs[i,2]])})
  subpairs <- allpairs[which(distpairs<distmax),]
  
  datapairsall <- c()
  for(k in 1:nrep){
    datapairsall <- rbind(datapairsall,t(sapply(1:nrow(subpairs),FUN=function(i){return(c(data[k,subpairs[i,1]],data[k,subpairs[i,2]]))})))
  }
  distpairsall <- rep(distpairs[which(distpairs<distmax)],nrep)
  
  nloglikMSP <- function(param,datapairsall,distpairsall){
    if(param[1]<0 | param[2]<0 | param[2]>2){
      return(10^10)
    } else{
      #rhopairsall <- exp(-(distpairsall/param[1])^param[2])
      rhopairsall <- matern(distpairsall, param[1], param[2])
      v <- V(datapairsall[,1],datapairsall[,2],rhopairsall)
      v1 <- V1(datapairsall[,1],datapairsall[,2],rhopairsall)
      v2 <- V2(datapairsall[,1],datapairsall[,2],rhopairsall)
      v12 <- V12(datapairsall[,1],datapairsall[,2],rhopairsall)
      res <- sum(-log(v1*v2-v12)+v)
      return(res)
    }
  }
  fit <- optim(par=init,fn=nloglikMSP,datapairsall=datapairsall,distpairsall=distpairsall,method="BFGS",control=list(maxit=1000))
  return(fit)
}


fitIMSP <- function(data,coord,distmax,init){
  nstat <- ncol(data)
  nrep <- nrow(data)
  allpairs <- t(combn(1:nstat,2))
  distmat <- as.matrix(dist(coord))
  distpairs <- sapply(1:nrow(allpairs),FUN=function(i){return(distmat[allpairs[i,1],allpairs[i,2]])})
  subpairs <- allpairs[which(distpairs<distmax),]
  
  datapairsall <- c()
  for(k in 1:nrep){
    datapairsall <- rbind(datapairsall,t(sapply(1:nrow(subpairs),FUN=function(i){return(c(data[k,subpairs[i,1]],data[k,subpairs[i,2]]))})))
  }
  distpairsall <- rep(distpairs[which(distpairs<distmax)],nrep)
  
  g <- function(x){
    return( -1/log(1-exp(-1/x)) )
  } 
  dg <- function(x){
    return( 1/(log(1-exp(-1/x)))^2 * 1/(1-exp(-1/x)) * (-exp(-1/x)) * (1/x^2) )
  }
  logabsdg <- function(x){
    return( -2*log(-log(1-exp(-1/x))) -log(1-exp(-1/x)) -1/x -2*log(x) )
  }
  
  nloglikIMSP <- function(param,datapairsall,distpairsall){
    if(param[1]<0 | param[1] > 0.6 | param[2]<0 | param[2]>3){
      return(10^10)
    } else{
      #rhopairsall <- exp(-(distpairsall/param[1])^param[2])
      rhopairsall <- matern(distpairsall, param[1], param[2])
      v <- V(g(datapairsall[,1]),g(datapairsall[,2]),rhopairsall)
      v1 <- V1(g(datapairsall[,1]),g(datapairsall[,2]),rhopairsall)
      v2 <- V2(g(datapairsall[,1]),g(datapairsall[,2]),rhopairsall)
      v12 <- V12(g(datapairsall[,1]),g(datapairsall[,2]),rhopairsall)
      res <- sum(-log(v1*v2-v12)+v)-sum(logabsdg(datapairsall))
      return(res)
    }
  }
  fit <- optim(par=init,fn=nloglikIMSP,datapairsall=datapairsall,distpairsall=distpairsall,method="BFGS",control=list(maxit=1000))
  return(fit)
}


fitMaxMix <- function(data,coord,distmax,init){
  nstat <- ncol(data)
  nrep <- nrow(data)
  allpairs <- t(combn(1:nstat,2))
  distmat <- as.matrix(dist(coord))
  distpairs <- sapply(1:nrow(allpairs),FUN=function(i){return(distmat[allpairs[i,1],allpairs[i,2]])})
  subpairs <- allpairs[which(distpairs<distmax),]
  
  datapairsall <- c()
  for(k in 1:nrep){
    datapairsall <- rbind(datapairsall,t(sapply(1:nrow(subpairs),FUN=function(i){return(c(data[k,subpairs[i,1]],data[k,subpairs[i,2]]))})))
  }
  distpairsall <- rep(distpairs[which(distpairs<distmax)],nrep)
  
  g <- function(x){
    return( -1/log(1-exp(-1/x)) )
  } 
  dg <- function(x){
    return( 1/(log(1-exp(-1/x)))^2 * 1/(1-exp(-1/x)) * (-exp(-1/x)) * (1/x^2) )
  }
  logabsdg <- function(x){
    return( -2*log(-log(1-exp(-1/x))) -log(1-exp(-1/x)) -1/x -2*log(x) )
  }
  
  nloglikMaxMix <- function(param,datapairsall,distpairsall){
    if(param[1]<0 | param[1] > 0.6 | param[2] < 0 | param[2] > 3 | param[3]<0 | param[4]<0 | param[4]>2 | param[5]<0 | param[5]>1){
      return(10^10)
    } else{
      #rhopairsall1 <- exp(-(distpairsall/param[1])^param[2])
      rhopairsall1 <- matern(distpairsall, param[1], param[2])
      v <- V(datapairsall[,1]/param[5],datapairsall[,2]/param[5],rhopairsall1)
      v1 <- V1(datapairsall[,1]/param[5],datapairsall[,2]/param[5],rhopairsall1)
      v2 <- V2(datapairsall[,1]/param[5],datapairsall[,2]/param[5],rhopairsall1)
      v12 <- V12(datapairsall[,1]/param[5],datapairsall[,2]/param[5],rhopairsall1)
      
      logG <- -v
      logG1 <- log(-v1)-v
      logG2 <- log(-v2)-v
      logG12 <- log(v1*v2-v12)-v
      
      rhopairsall2 <- exp(-(distpairsall/param[3])^param[4])
      w <- V(g(datapairsall[,1]/(1-param[5])),g(datapairsall[,2]/(1-param[5])),rhopairsall2)
      w1 <- V1(g(datapairsall[,1]/(1-param[5])),g(datapairsall[,2]/(1-param[5])),rhopairsall2)
      w2 <- V2(g(datapairsall[,1]/(1-param[5])),g(datapairsall[,2]/(1-param[5])),rhopairsall2)
      w12 <- V12(g(datapairsall[,1]/(1-param[5])),g(datapairsall[,2]/(1-param[5])),rhopairsall2)
      
      logH <- -w
      logH1 <- log(-w1)-w+logabsdg(datapairsall[,1]/(1-param[5]))
      logH2 <- log(-w2)-w+logabsdg(datapairsall[,2]/(1-param[5]))
      logH12 <- log(w1*w2-w12)-w+logabsdg(datapairsall[,1]/(1-param[5]))+logabsdg(datapairsall[,2]/(1-param[5]))
      
      res <- sum( -log( (1/param[5]^2)*exp(logG12)*exp(logH)+(1/(param[5]*(1-param[5])))*exp(logG1)*exp(logH2)+(1/(param[5]*(1-param[5])))*exp(logG2)*exp(logH1)+(1/(1-param[5]))^2*exp(logG)*exp(logH12) ) )
      return(res)
    }
  }
  fit <- optim(par=init,fn=nloglikMaxMix,datapairsall=datapairsall,distpairsall=distpairsall,method="BFGS",control=list(maxit=1000))
  return(fit)
}


fitMaxMix2 <- function(data,coord,distmax,init){
  nstat <- ncol(data)
  nrep <- nrow(data)
  allpairs <- t(combn(1:nstat,2))
  distmat <- as.matrix(dist(coord))
  distpairs <- sapply(1:nrow(allpairs),FUN=function(i){return(distmat[allpairs[i,1],allpairs[i,2]])})
  subpairs <- allpairs[which(distpairs<distmax),]
  
  datapairsall <- c()
  for(k in 1:nrep){
    datapairsall <- rbind(datapairsall,t(sapply(1:nrow(subpairs),FUN=function(i){return(c(data[k,subpairs[i,1]],data[k,subpairs[i,2]]))})))
  }
  distpairsall <- rep(distpairs[which(distpairs<distmax)],nrep)
  
  g <- function(x){
    return( -1/log(1-exp(-1/x)) )
  } 
  dg <- function(x){
    return( 1/(log(1-exp(-1/x)))^2 * 1/(1-exp(-1/x)) * (-exp(-1/x)) * (1/x^2) )
  }
  logabsdg <- function(x){
    return( -2*log(-log(1-exp(-1/x))) -log(1-exp(-1/x)) -1/x -2*log(x) )
  }
  
  nloglikMaxMix2 <- function(param,datapairsall,distpairsall){
    if(param[1]<0 | param[2]<0 | param[3]<0 | param[3]>1){
      return(10^10)
    } else{
      rhopairsall1 <- exp(-(distpairsall/param[1]))
      v <- V(datapairsall[,1]/param[3],datapairsall[,2]/param[3],rhopairsall1)
      v1 <- V1(datapairsall[,1]/param[3],datapairsall[,2]/param[3],rhopairsall1)
      v2 <- V2(datapairsall[,1]/param[3],datapairsall[,2]/param[3],rhopairsall1)
      v12 <- V12(datapairsall[,1]/param[3],datapairsall[,2]/param[3],rhopairsall1)
      
      logG <- -v
      logG1 <- log(-v1)-v
      logG2 <- log(-v2)-v
      logG12 <- log(v1*v2-v12)-v
      
      rhopairsall2 <- exp(-(distpairsall/param[2]))
      w <- V(g(datapairsall[,1]/(1-param[3])),g(datapairsall[,2]/(1-param[3])),rhopairsall2)
      w1 <- V1(g(datapairsall[,1]/(1-param[3])),g(datapairsall[,2]/(1-param[3])),rhopairsall2)
      w2 <- V2(g(datapairsall[,1]/(1-param[3])),g(datapairsall[,2]/(1-param[3])),rhopairsall2)
      w12 <- V12(g(datapairsall[,1]/(1-param[3])),g(datapairsall[,2]/(1-param[3])),rhopairsall2)
      
      logH <- -w
      logH1 <- log(-w1)-w+logabsdg(datapairsall[,1]/(1-param[3]))
      logH2 <- log(-w2)-w+logabsdg(datapairsall[,2]/(1-param[3]))
      logH12 <- log(w1*w2-w12)-w+logabsdg(datapairsall[,1]/(1-param[3]))+logabsdg(datapairsall[,2]/(1-param[3]))
      
      res <- sum( -log( (1/param[3]^2)*exp(logG12)*exp(logH)+(1/(param[3]*(1-param[3])))*exp(logG1)*exp(logH2)+(1/(param[3]*(1-param[3])))*exp(logG2)*exp(logH1)+(1/(1-param[3]))^2*exp(logG)*exp(logH12) ) )
      return(res)
    }
  }
  fit <- optim(par=init,fn=nloglikMaxMix2,datapairsall=datapairsall,distpairsall=distpairsall,method="BFGS",control=list(maxit=1000))
  return(fit)
}
