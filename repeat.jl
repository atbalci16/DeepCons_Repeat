# for p in ("Knet")
#     Pkg.installed(p) == nothing && Pkg.add(p)
# end

using Knet, JLD, ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=100; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=128; help="Number of sequences to train on in parallel.")
        ("--lr"; arg_type=Float64; default=0.005; help="Initial learning rate.")
        ("--weight"; arg_type=String; default=""; help="Initial weights instead of randomly initialized ones.")
    end
    return parse_args(s;as_symbols = true)        
end

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
  writeData("data_tr.seq","data_train.seq",1200000) #1290000
  writeData("data_te.seq","data_test.seq",100000) #165000
  writeData("data_va.seq","data_valid.seq",100000) #165000
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
  if ((i-1)*sz+sz) > length(data)
    return map(x->split(x), data[((i-1)*sz+1):length(data)])
  else
    return map(x->split(x), data[((i-1)*sz+1):((i-1)*sz+sz)])
  end
end

function getIter(ltr,lts,lva,sz)
  return convert(Int32,ceil(ltr/sz)), convert(Int32,ceil(lva/sz)), convert(Int32,ceil(lts/sz))
end

function seqMap(na)
  seqMap = Dict{Char,UInt8}('A' => 1, 'G' => 2, 'C' => 3, 'T' => 4)
  return seqMap[na]
end

function seqToVector(seq)
  m = convert(Array{Float32}, zeros(4,200,1,1))
  index = map(x-> sub2ind(size(m),x[1],x[2],1,1) , 
              map(x -> (seqMap(x[2]),x[1]), 
                filter(x -> x[2] != 'N', enumerate(seq))))
  m[index] = 1
  return m
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

function weights(;w=[],winit=0.1)
    if length(w) == 8
      return map(x->convert(KnetArray{Float32},x),w)
    else
      return Any[convert(KnetArray{Float32},0.01*randn(4,10,1,1000)*winit), 
                 convert(KnetArray{Float32},zeros(1,1,1000,1)),
                 convert(KnetArray{Float32},0.01*randn(4,20,1,500)*winit), 
                 convert(KnetArray{Float32},zeros(1,1,500,1)),
                 convert(KnetArray{Float32},0.01*randn(1500,1500)),
                 convert(KnetArray{Float32},zeros(1500,1)),
                 convert(KnetArray{Float32},0.01*randn(2,1500)*winit),
                 convert(KnetArray{Float32},zeros(2,1))]
    end    
end

function initparams(weights;learningRate=0.005)
  return map(x -> Adagrad(;lr=learningRate), weights)
end

function predict(w,x)
    pool_1 = pool(dropout(relu(conv4(w[1],x) .+ w[2]), 0.25);stride=191,window=191)
    pool_2 = pool(dropout(relu(conv4(w[3],x) .+ w[4]), 0.25);stride=181,window=181)
    pool_1 = reshape(pool_1, (size(pool_1,1)*size(pool_1,2)*size(pool_1,3),size(pool_1,4)))
    pool_2 = reshape(pool_2, (size(pool_2,1)*size(pool_2,2)*size(pool_2,3),size(pool_2,4)))
    pool_out = [pool_1 ; pool_2]
    y = dropout(relu(w[5]*pool_out .+ w[6]), 0.5)
    y = w[7] * y .+ w[8]
    return y
end

function pred(w,x)
    pool_1 = pool(relu(conv4(w[1],x) .+ w[2]);stride=191,window=191)
    pool_2 = pool(relu(conv4(w[3],x) .+ w[4]);stride=181,window=181)
    pool_1 = reshape(pool_1, (size(pool_1,1)*size(pool_1,2)*size(pool_1,3),size(pool_1,4)))
    pool_2 = reshape(pool_2, (size(pool_2,1)*size(pool_2,2)*size(pool_2,3),size(pool_2,4)))
    pool_out = [pool_1 ; pool_2]
    y = relu(w[5]*pool_out .+ w[6])
    y = w[7] * y .+ w[8]
    y = y .- maximum(y,1)
    expsum = sum(exp(y),1)
    pred = exp(y)./expsum
    return pred
end

function loss(w,x,ygold)
    y = predict(w,x)
    y = y .- maximum(y,1)
    expy = exp(y)
    logphat = y .- log(sum(expy,1))
    lost = -sum(ygold .* logphat) / size(x, 4)
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

function accuracy(w,dtst)
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

function main(args=ARGS)
  opts = parse_commandline()
  println("opts=",[(k,v) for (k,v) in opts]...)
  prepData()
  chunk_size = 12800
  if opts[:weight] != ""
    println("reading initial weight from $(opts[:weight]) file.")
    w = weights(;w=load(opts[:weight])["w"])
  else
    w = weights()
  end
  params = initparams(w;learningRate=opts[:lr])
  dtrain,dtest,dvalid = read_data()
  itrain, ival, itest = getIter(length(dtrain),length(dtest),length(dvalid),chunk_size)
  patience = 0
  bests = Inf
  bestw = Any[]
  validLost = Any[]
  @time for epoch=1:opts[:epochs]
    if patience > 10
      print("Early stopping no progess for 10 epoch... \n")
      break
    end
    print("epoch $epoch... \n")
    average_loss = 0
    @time for i=1:itrain
      data = getChunk(dtrain, chunk_size, i)
      xtrn, ytrn = preprocess(data)
      dtrn = minibatch(xtrn,ytrn,opts[:batchsize])
      train(w,dtrn,params)
    end
    nloss = 0;ninstance = 0
    tloss = 0;tinstance = 0
    for i=1:ival
      data = getChunk(dvalid, chunk_size, i)
      xva, yva = preprocess(data)
      dva = minibatch(xva,yva,opts[:batchsize])
      nloss, ninstance= avgloss(w,dva)
      tloss += nloss; tinstance += ninstance
      xva = 0;yva = 0;data = 0;dva = 0;
    end
    average_loss = tloss/tinstance
    push!(validLost,average_loss)
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
    dtst = minibatch(xtst,ytst,opts[:batchsize])
    ncorr, nwrong, ninstance = accuracy(bestw,dtst)
    corr += ncorr; wrong += nwrong; instance += ninstance
    xtst = 0;ytst = 0;data = 0;dtst = 0;
  end
  print("Test accuracy is $(corr/instance) $(wrong/instance) \n")
  print("Writing the weights of the best model to weights.jld \n")
  save("weights.jld","w",map(x->convert(Array{Float32},x),bestw),"J",validLost,"s",bests)
  return bestw
end

w = main()
