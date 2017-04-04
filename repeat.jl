# for p in ("Knet")
#     Pkg.installed(p) == nothing && Pkg.add(p)
# end

using Knet, JLD

function writeData(itxt,otxt,n)
  ofile = open(otxt, "w")
  index = 1
  open(itxt, "r") do file
    for line in readlines(file)
      if length(line)-2 < 200
        write(ofile,line)
        index += 1
      end
      if index > n
        break
      end
    end
  end
  close(ofile)
end

function prepData()
  isfile("data_te.seq") || download("https://cbcl.ics.uci.edu/public_data/DeepCons/data_te.seq", "data_te.seq")
  isfile("data_tr.seq") || download("https://cbcl.ics.uci.edu/public_data/DeepCons/data_tr.seq", "data_tr.seq")
  isfile("data_va.seq") || download("https://cbcl.ics.uci.edu/public_data/DeepCons/data_va.seq", "data_va.seq")
  writeData("data_tr.seq","data_train.seq",600000)
  writeData("data_te.seq","data_test.seq",150000)
  writeData("data_va.seq","data_valid.seq",150000)
end

function read_data()
  dtrn = open("data_train.seq") do f
    readlines(f)
  end
  dtst = open("data_test.seq") do f
    readlines(f)
  end
  dva = open("data_valid.seq") do f
    readlines(f)
  end
  return dtrn, dtst, dva
end

function getChunk(data, sz, i)
  if i+sz-1 > length(data)
    return map(x->split(x), data[i:length(data)])
  else
    return map(x->split(x), data[i:(i+sz-1)])
  end
end

function getIter(ltr,lts,lva,sz)
  return convert(Int32,ceil(ltr/sz)), convert(Int32,ceil(lva/sz)), convert(Int32,ceil(lts/sz))
end

function seqMap(na)
  seqMap = Dict{Char,UInt8}('A' => 1, 'G' => 2, 'C' => 3, 'T' => 4)
  return seqMap[na]
end

function seqToVector!(matrix, seq, index)
  index_m = map(x-> sub2ind(size(matrix),x[1],x[2],1,index) , 
              map(x -> (seqMap(x[2]),x[1]), 
                filter(x -> x[2] != 'N', enumerate(seq))))
  matrix[index_m] = 1
end

function preprocess(data)
  xdata = [elm[1] for elm in data]
  ydata = [elm[2] for elm in data]
  preprocessed = zeros(UInt8,4,200,1,length(xdata))
  y_out = zeros(UInt8,2, length(ydata))
  map( x -> seqToVector!(preprocessed, x[2], x[1]),enumerate(xdata))
  y = [parse(elm)+1 for elm in ydata]
  index = map(x -> sub2ind((2,length(y)),x[2],x[1]),enumerate(y))
  y_out[index] = 1
  return preprocessed, y_out
end

function minibatch(x,y,sz)
    data = Any[]
    for i=1:sz:size(x,4)
      if i+sz-1 > size(x,4)
        push!(data,(convert(KnetArray{Float32},x[:,:,:,i:end]),convert(KnetArray{Float32},y[:,i:end])))
      else
        push!(data,(convert(KnetArray{Float32},x[:,:,:,i:i+sz-1]),convert(KnetArray{Float32},y[:,i:i+sz-1])))
      end
    end
    return data
end

function weights(;winit=0.1)
    return Any[convert(KnetArray{Float32},randn(4,10,1,500)*winit), 
               convert(KnetArray{Float32},zeros(1,1,500,1)),
               convert(KnetArray{Float32},randn(4,20,1,250)*winit), 
               convert(KnetArray{Float32},zeros(1,1,250,1)),
               convert(KnetArray{Float32},randn(2,750)*winit),
               convert(KnetArray{Float32},zeros(2,1))]
end

function initparams(weights;learning_rate=0.005)
  return map(x -> Adam(;lr=learning_rate), weights)
end

function predict(w,x)
    pool_1 = pool(dropout(relu(conv4(w[1],x) .+ w[2]), 0.25);stride=191,window=191)
    pool_2 = pool(relu(conv4(w[3],x) .+ w[4]);stride=181,window=181)
    pool_1 = reshape(pool_1, (size(pool_1,1)*size(pool_1,2)*size(pool_1,3),size(pool_1,4)))
    pool_2 = reshape(pool_2, (size(pool_2,1)*size(pool_2,2)*size(pool_2,3),size(pool_2,4)))
    pool_out = [pool_1 ; pool_2]
    y = dropout(relu(w[5]*pool_out .+ w[6]), 0.5)
    y = y .- maximum(y,1)
    expsum = sum(exp(y),1)
    pred = exp(y)./expsum
    return pred
end

function loss(w,x,ygold)
    pred_y = predict(w,x)
    lost = -sum(ygold .*pred_y) / size(x, 4)
    return lost
end

lossgradient =  grad(loss)

function train(w,dtrn,params)
    for (x,y) in dtrn
        w_grad = lossgradient(w, x, y)
        for i=1:length(w)
          update!(w[i],w_grad[i],params[i])
        end
    end
    return w
end

function avgloss(w,data)
    sum = cnt = 0
    for (x,y) in data
        sum += loss(w,x,y)
        cnt += 1
    end
    return sum, cnt
end

function accuracy(w,dtst,pred=predict)
    ncorrect = 0
    ninstance = 0
    nloss = 0
    for (x,y) in dtst
        pred_y = pred(w,x)
        pred_y = pred_y .== maximum(pred_y,1)
        ncorrect += sum(pred_y .* y)
        ninstance += size(pred_y,2)
    end
    nloss = ninstance - ncorrect
    return (ncorrect, nloss,ninstance)
end

function main()
  prepData()
  chunk_size = 12800
  w = weights()
  params = initparams(w)
  dtrain,dtest,dvalid = read_data()
  itrain, ival, itest = getIter(length(dtrain),length(dtest),length(dvalid),chunk_size)
  patience = 0
  bests = 0
  bestw = Any[]
  @time for epoch=1:100
    if patience > 10
      print("Early stopping no progess for 10 epoch... \n")
      break
    end
    print("epoch $epoch... \n")
    average_loss = 0
    @time for i=1:itrain
      data = getChunk(dtrain, chunk_size, i)
      xtrn, ytrn = preprocess(data)
      dtrn = minibatch(xtrn,ytrn,128)
      train(w,dtrn,params)
      gc()
    end
    nloss = 0;ninstance = 0
    tloss = 0;tinstance = 0
    for i=1:ival
      data = getChunk(dvalid, chunk_size, i)
      xva, yva = preprocess(data)
      dva = minibatch(xva,yva,128)
      nloss, ninstance= avgloss(w,dva)
      tloss += nloss; tinstance += ninstance
      xva = 0;yva = 0;data = 0;dva = 0;
      gc()
    end
    average_loss = tloss/tinstance
    print("$epoch: validation lost is $(average_loss) \n")
    if average_loss < bests
      bestw = w
      bests = average_loss
      patience = 0
    else
      patience += 1
    end
  end
  corr=0;wrong=0;instance = 0
  for i=1:itest
    data = getChunk(dtest, chunk_size, i)
    xtst, ytst = preprocess(data)
    dtst = minibatch(xtst,ytst,128)
    ncorr, nwrong, ninstance = accuracy(bestw,dtst)
    corr += ncorr; wrong += nwrong; instance += ninstance
    xtst = 0;ytst = 0;data = 0;dtst = 0;
    gc()
  end
  print("Test accuracy is $(corr/instance) $(wrong/instance) \n")
  print("Writing the weights of the best model to weights.jld \n")
  save("weights.jld","w",map(x->convert(Array{Float32},x),bestw))
end

main()
