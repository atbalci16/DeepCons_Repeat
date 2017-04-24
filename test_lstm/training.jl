using Knet

function predict(w,x,cell,hidden)
  for input in x
    cell, hidden = lstm(cell,input,hidden,w[1:2])
  end

  y = hidden * w[3] .+ w[4]
  return y
end

function loss(w,x,ygold,cell,hidden,predict)
  ypred = predict(w,x,cell,hidden)
  ynorm = logp(ypred,2)
  lost = -sum(ygold .* ynorm) / size(ygold, 1)
  return lost
end

lossgradient =  grad(loss)

function train(w,dtrn,params,cell,hidden,predict)
    for (x,y) in dtrn
        w_grad = lossgradient(w, x, y, cell, hidden,predict)
        for i=1:length(w)
          update!(w[i],w_grad[i],params[i])
        end
    end
    return w
end

function accuracy(w,dtst,cell,hidden,pred)
    ncorrect = 0
    ninstance = 0
    nloss = 0
    for (x,y) in dtst
        pred_y = pred(w,x,cell,hidden)
        pred_y = pred_y .== maximum(pred_y,2)
        ncorrect += sum(pred_y .* y)
        ninstance += size(pred_y,2)
    end
    nloss = ninstance - ncorrect
    return (ncorrect, nloss,ninstance)
end

function avgloss(w,data,cell,hidden,loss,pred)
    sum = cnt = 0
    for (x,y) in data
        sum += loss(w,x,y,cell,hidden,pred)
        cnt += 1
    end
    return sum, cnt
end