using JLD,ArgParse,Knet

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=100; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=128; help="Number of sequences to train on in parallel.")
        ("--lr"; arg_type=Float64; default=0.005; help="Initial learning rate.")
        ("--weight"; arg_type=String; default=""; help="Initial weights instead of randomly initialized ones.")
        ("--state"; arg_type=Int; default=30; help="Length of cell and hidden vector.")
        ("--chunksize"; arg_type=Int; default=12800; help="Chunk size.")
        ("--ltrain"; arg_type=Int; default=1024; help="Training size.")
        ("--ltest"; arg_type=Int; default=128; help="Test size.")
        ("--lval"; arg_type=Int; default=128; help="Validation size.")
    end
    return parse_args(s;as_symbols = true)        
end

function writeData(itxt,otxt,n)
  ofile = open(otxt, "w")
  index = 1
  open(itxt, "r") do file
    for line in readlines(file)
      if length(line)-3 == 30
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

function prepData(trsz,tssz,vasz)
  isfile("data_te.seq") || download("https://cbcl.ics.uci.edu/public_data/DeepCons/data_te.seq", "data_te.seq")
  isfile("data_tr.seq") || download("https://cbcl.ics.uci.edu/public_data/DeepCons/data_tr.seq", "data_tr.seq")
  isfile("data_va.seq") || download("https://cbcl.ics.uci.edu/public_data/DeepCons/data_va.seq", "data_va.seq")
  writeData("data_tr.seq","data_train.seq",trsz) #1290000
  writeData("data_te.seq","data_test.seq",tssz) #165000
  writeData("data_va.seq","data_valid.seq",vasz) #165000
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
    return data[((i-1)*sz+1):length(data)]
  else
    return data[((i-1)*sz+1):((i-1)*sz+sz)]
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
  data = map(x->split(x), data)
  xdata = [elm[1] for elm in data]
  ydata = [elm[2] for elm in data]
  y_prep = zeros(Bool,length(ydata),2)
  y = [parse(elm)+1 for elm in ydata]
  index = map(x -> sub2ind((length(ydata),2),x[1],x[2]),enumerate(y))
  y_prep[index] = true
  return xdata, y_prep
end

function minibatch(x,y,sz;atype=KnetArray{Float32})
  nbatch = div(length(x), sz)
  padding = maximum(map(a -> length(a), x))
  x = [lpad(elm,padding,'N') for elm in x]
  data = Any[]
  for j=1:nbatch
    batch = Any[]
    minix = x[((j-1)*sz+1):sz*j]
    for i=1:padding
      push!(batch, convert(atype, generateStep([vocab(elm[i]) for elm in minix])))
    end
    push!(data,(batch, convert(atype,y[((j-1)*sz+1):sz*j,:])))
  end
  return data
end

function lstm(cell,x,hidden,W)
  input = hcat(x, hidden)
  result = input * W[1] .+ W[2]
  sz = size(hidden,2)
  fgate = sigm(result[:,1:sz])
  igate = sigm(result[:,sz+1:2*sz])
  candidate = tanh(result[:,2*sz+1:3*sz])
  cell = fgate .* cell + igate .* candidate
  ogate = sigm(result[:,3*sz+1:end])
  hidden = ogate .* tanh(cell)
  return (cell,hidden)
end

function weights(lstate,vocabsz,outsz;atype=KnetArray{Float32})
  weights = Any[convert(atype,xavier((vocabsz+lstate), 4*lstate)),
                convert(atype, zeros(1,4*lstate)),
                convert(atype,xavier(lstate,outsz)),
                convert(atype, zeros(1,outsz))]
  return weights
end

function initparams(weights;learningRate=0.005)
  return map(x -> Adam(;lr=learningRate), weights)
end

function initstate(lstate,batchsz)
    cell = convert(KnetArray{Float32}, zeros(batchsz,lstate))
    hidden = convert(KnetArray{Float32}, zeros(batchsz,lstate))
    return cell, hidden
end

function predict(w,x,cell,hidden)
  for input in x
    cell, hidden = lstm(cell,input,hidden,w[1:2])
  end
  y = hidden * w[3] .+ w[4]
  return y
end

function pred(w,x,cell,hidden)
  for input in x
    cell, hidden = lstm(cell,input,hidden,w[1:2])
  end
  y = hidden * w[3] .+ w[4]
  y = y .- maximum(y,2)
  y = exp(y)
  expsum = sum(y,2)
  pred = y./expsum
  return pred
end

function loss(w,x,ygold,cell,hidden)
  ypred = predict(w,x,cell,hidden)
  ynorm = logp(ypred,2)
  lost = -sum(ygold .* ynorm) / size(ygold, 1)
  return lost
end

lossgradient =  grad(loss)

function train(w,dtrn,params,cell,hidden)
    for (x,y) in dtrn
        w_grad = lossgradient(w, x, y, cell, hidden)
        @show gradcheck(loss,w,x,y,cell,hidden;verbose=true,atol=0.01)
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
        pred_y = pred(w,x,copy(cell),copy(hidden))
        # println(convert(Array{Float32},pred_y))
        pred_y = pred_y .== maximum(pred_y,2)
        # println(size(pred_y))
        # println(size(y))
        ncorrect += sum(pred_y .* y)
        ninstance += size(pred_y,1)
    end
    nloss = ninstance - ncorrect
    return (ncorrect, nloss,ninstance)
end

function avgloss(w,data,cell,hidden)
    sum = cnt = 0
    for (x,y) in data
        sum += loss(w,x,y,copy(cell),copy(hidden))
        cnt += 1
    end
    return sum, cnt
end

function main()
  opts = parse_commandline()
  println("opts=",[(k,v) for (k,v) in opts]...)
  prepData(opts[:ltrain],opts[:ltest],opts[:lval])
  dtrain, dtest, dval = read_data()
  cell, hidden = initstate(opts[:state],opts[:batchsize])
  w = weights(opts[:state],4,2)
  params = initparams(w)
  itrain, ival, itest = getIter(opts[:ltrain],opts[:ltest],opts[:lval],opts[:chunksize])
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
      data = getChunk(dtrain, opts[:chunksize], i)
      xtrn, ytrn = preprocess(data)
      dtrn = minibatch(xtrn,ytrn,opts[:batchsize])
      train(w,dtrn,params,copy(cell),copy(hidden))
    end
    nloss = 0;ninstance = 0
    tloss = 0;tinstance = 0
    for i=1:ival
      data = getChunk(dval, opts[:chunksize], i)
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
    data = getChunk(dtest, opts[:chunksize], i)
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

