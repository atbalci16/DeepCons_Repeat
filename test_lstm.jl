using Knet, JLD, ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=100; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=128; help="Number of sequences to train on in parallel.")
        ("--lr"; arg_type=Float64; default=0.005; help="Initial learning rate.")
        ("--weight"; arg_type=String; default=""; help="Initial weights instead of randomly initialized ones.")
        ("--state"; arg_type=Int; default=150; help="Length of cell and hidden vector.")
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
  writeData("data_tr.seq","data_train.seq",1024) #1290000
  writeData("data_te.seq","data_test.seq",128) #165000
  writeData("data_va.seq","data_valid.seq",280) #165000
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

function vocab(na)
  return Dict{Char,UInt8}('A' => 1, 'G' => 2, 'C' => 3, 'T' => 4, 'N' => 5)[na]
end

function generateStep(x)
  table = [eye(Bool,4,4); falses(1,4)]
  return table[x,:]
end

function preprocess(data)
  xdata = [elm[1] for elm in data]
  ydata = [elm[2] for elm in data]
  y_prep = zeros(Bool,length(ydata),2)
  y = [parse(elm)+1 for elm in ydata]
  index = map(x -> sub2ind((2,length(y)),x[1],x[2]),enumerate(y))
  y_prep[index] = 1
  return xdata, y_prep
end

function weights(lstate;w=[],winit=0.1)
  weights = Any[convert(KnetArray{Float32}, randn(lstate+4,lstate))
                convert(KnetArray{Float32}, zeros(1,lstate))
                convert(KnetArray{Float32}, randn(lstate+4,lstate))
                convert(KnetArray{Float32}, zeros(1,lstate))
                convert(KnetArray{Float32}, randn(lstate+4,lstate))
                convert(KnetArray{Float32}, zeros(1,lstate))
                convert(KnetArray{Float32}, randn(lstate+4,lstate))
                convert(KnetArray{Float32}, zeros(1,lstate))
                convert(KnetArray{Float32}, randn(lstate,2))
                convert(KnetArray{Float32}, zeros(1,2))]
  return weights
end

function initparams(weights;learningRate=0.005)
  return map(x -> Adam(;lr=learningRate), weights)
end

function initstate(lstate,sz)
    cell = convert(KnetArray{Float32}, zeros(sz,lstate))
    hidden = convert(KnetArray{Float32}, zeros(sz,lstate))
    return cell, hidden
end

function minibatch(x,y,sz)
  nbatch = div(length(x), sz)
  x = [lpad(elm,200,'N') for elm in x]
  data = Any[]
  for j=1:nbatch
    batch = Any[]
    minix = x[((j-1)*sz+1):sz*j]
    for i=1:20
      push!(batch, convert(KnetArray{Float32}, generateStep([vocab(elm[i]) for elm in minix])))
    end
      push!(data,(batch, convert(KnetArray{Float32},y[((j-1)*sz+1):sz*j,:])))
  end
  return data
end

function lstm(cell,x,hidden,W)
  input = [x hidden]
  fgate = sigm(input * W[1] .+ W[2])
  igate = sigm(input * W[3] .+ W[4])
  candidate = tanh(input * W[5] .+ W[6])
  cell = fgate .* cell + igate .* candidate
  ogate = sigm(input * W[7] .+ W[8])
  hidden = ogate .* tanh(cell)
  return (cell,hidden)
end

function predict(w,x,cell,hidden)
  for input in x
    cell, hidden = lstm(cell,input,hidden,w)
  end

  y = hidden * w[9] .+ w[10]
  return y
end

function pred(w,x,cell,hidden)
  for input in x
    cell, hidden = lstm(cell,input,hidden,w)
  end
  y = hidden * w[9] .+ w[10]
  y = y .- maximum(y,2)
  expsum = sum(exp(y),2)
  pred = exp(y)./expsum
  return pred
end

function loss(w,x,ygold,cell,hidden)
  ypred = predict(w,x,cell,hidden)
  ypred = ypred .- maximum(ypred,2)
  expy = exp(ypred)
  ynorm = logp(ypred,2)
  lost = -sum(ygold .* ynorm) / size(ygold, 1)
  println("sum")
  return lost
end

lossgradient =  grad(loss)

function train(w,dtrn,params,cell,hidden)
    for (x,y) in dtrn
        w_grad = lossgradient(w, x, y, cell, hidden)
        for i=1:length(w)
          update!(w[i],w_grad[i],params[i])
        end
    end
    return w
end

function accuracy(w,dtst,cell,hidden)
    ncorrect = 0
    ninstance = 0
    nloss = 0
    for (x,y) in dtst
        pred_y = pred(w,x,cell,hidden)
        pred_y = pred_y .== maximum(pred_y,1)
        ncorrect += sum(pred_y .* y)
        ninstance += size(pred_y,2)
    end
    nloss = ninstance - ncorrect
    return (ncorrect, nloss,ninstance)
end

function avgloss(w,data,cell,hidden)
    sum = cnt = 0
    for (x,y) in data
        sum += loss(w,x,y,cell,hidden)
        cnt += 1
    end
    return sum, cnt
end

function main()
  opts = parse_commandline()
  println("opts=",[(k,v) for (k,v) in opts]...)
  prepData()
  chunk_size = 12800
  if opts[:weight] != ""
    println("reading initial weight from $(opts[:weight]) file.")
    w = weights(opts[:state];w=load(opts[:weight])["w"])
  else
    w = weights(opts[:state])
  end
  cell,hidden = initstate(opts[:state],opts[:batchsize])
  params = initparams(w;learningRate=opts[:lr])
  dtrain,dtest,dvalid = read_data()
  itrain, ival, itest = getIter(length(dtrain),length(dtest),length(dvalid),chunk_size)
  patience = 0
  bests = Inf
  bestw =Any[]
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
      train(w,dtrn,params,copy(cell),copy(hidden))
    end
    nloss = 0;ninstance = 0
    tloss = 0;tinstance = 0
    for i=1:ival
      data = getChunk(dvalid, chunk_size, i)
      xva, yva = preprocess(data)
      dva = minibatch(xva,yva,opts[:batchsize])
      nloss, ninstance= avgloss(w,dva,copy(cell),copy(hidden))
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
    ncorr, nwrong, ninstance = accuracy(bestw,dtst,copy(cell),copy(hidden))
    corr += ncorr; wrong += nwrong; instance += ninstance
    xtst = 0;ytst = 0;data = 0;dtst = 0;
  end
  print("Test accuracy is $(corr/instance) $(wrong/instance) \n")
  print("Writing the weights of the best model to weights.jld \n")
  save("weights.jld","w",map(x->convert(Array{Float32},x),bestw),"J",validLost,"s",bests)
  return bestw
end

w = main()