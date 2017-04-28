using Knet, JLD, ArgParse


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=100; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=128; help="Number of sequences to train on in parallel.")
        ("--lr"; arg_type=Float64; default=0.005; help="Initial learning rate.")
        ("--weight"; arg_type=String; default=""; help="Initial weights instead of randomly initialized ones.")
        ("--chunksize"; arg_type=Int; default=12800; help="Chunk size.")
        ("--ltrain"; arg_type=Int; default=1024; help="Training size.")
        ("--ltest"; arg_type=Int; default=128; help="Test size.")
        ("--lval"; arg_type=Int; default=128; help="Validation size.")
        ("--lsequence"; arg_type=Int; default=200; help="Sequence length.")
    end
    return parse_args(s;as_symbols = true)        
end


function writeData(itxt,otxt,n,sz)
  ofile = open(otxt, "w")
  index = 1
  open(itxt, "r") do file
    for line in readlines(file)
      if length(line)-3 <= sz
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

function prepData(trsz,tssz,vasz,lsequence)
  isfile("data_te.seq") || download("https://cbcl.ics.uci.edu/public_data/DeepCons/data_te.seq", "data_te.seq")
  isfile("data_tr.seq") || download("https://cbcl.ics.uci.edu/public_data/DeepCons/data_tr.seq", "data_tr.seq")
  isfile("data_va.seq") || download("https://cbcl.ics.uci.edu/public_data/DeepCons/data_va.seq", "data_va.seq")
  writeData("data_tr.seq","data_train.seq",trsz,lsequence) #1290000
  writeData("data_te.seq","data_test.seq",tssz,lsequence) #165000
  writeData("data_va.seq","data_valid.seq",vasz,lsequence) #165000
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

function seqToVector!(matrix,seq,index)
  seqMap = Dict{Char,UInt8}('A' => 1, 'G' => 2, 'C' => 3, 'T' => 4)
  for i=1:length(seq)
    if seq[i] != 'N'
      matrix[(i-1)*4+seqMap[seq[i]],index] = 1
    end
  end
end

function oneHot(y)
  matrix = zeros(UInt8,2,length(y))
  for i=1:length(y)
    matrix[y[i]+1, i] = 1
  end
  return matrix
end

function preprocess(data,sz)
  x = zeros(UInt8,sz*4,length(data))
  ylabel =[parse(elm[2]) for elm in data]
  y = oneHot(ylabel)
  for (index,elm) in enumerate(data)
    seqToVector!(x,elm[1],index)
  end
  return x, y
end

function initparams(weights;learningRate=0.005)
  return map(x -> Adagrad(;lr=learningRate), weights)
end

function minibatch(x,y,sz)
    data = Any[]
    for i=1:sz:size(x,2)
      if i+sz-1 > size(x,2)
        push!(data,(x[:,i:end],y[:,i:end]))
      else
        push!(data,(x[:,i:i+sz-1],y[:,i:i+sz-1]))
      end
    end
    return data
end

function weights(h;winit=0.1)
  w = Any[]
    x = h[1]
    for y in h[2:end]
        push!(w, convert(Array{Float32}, randn(y, x)*winit))
        push!(w, zeros(Float32, y,1))
        x = y 
    end
    return w
end

function predict(w,x)
    for i=1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
    end
    return sigm(w[end-1]*x .+ w[end])
end

function loss(w,x,ygold)
    y = predict(w,x)
    lost = -sum(ygold .* log(y) + (1-ygold) .* log(1-y)) / length(ygold)
    return lost
end

function avgloss(w,data)
    return sum([loss(w,x,y) for (x,y) in data]), length(data)
end

function accuracy(w, data)
    ncorrect = 0
    ninstance = 0
    nloss = 0
    for (x,y) in data
        pred_y = predict(w,x)
        pred_y = [1-pred_y; pred_y]
        pred_y = pred_y .== maximum(pred_y,1)
        ncorrect += sum(pred_y .* y)
        ninstance += size(pred_y,2)
    end
    nloss = ninstance - ncorrect
    return (ncorrect, nloss, ninstance)
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

function main(args=ARGS)
  opts = parse_commandline()
  println("opts=",[(k,v) for (k,v) in opts]...)
  prepData(opts[:ltrain],opts[:ltest],opts[:lval],opts[:lsequence])
  w = weights([opts[:lsequence]*4 1])
  params = initparams(w;learningRate=opts[:lr])
  dtrain,dtest,dvalid = read_data()
  itrain, ival, itest = getIter(opts[:ltrain],opts[:ltest],opts[:lval],opts[:chunksize])
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
      data = getChunk(dtrain, opts[:chunksize], i)
      xtrn, ytrn = preprocess(data,opts[:lsequence])
      dtrn = minibatch(xtrn,ytrn,opts[:batchsize])
      train(w,dtrn,params)
    end
    nloss = 0;ninstance = 0
    tloss = 0;tinstance = 0
    for i=1:ival
      data = getChunk(dvalid, opts[:chunksize], i)
      xva, yva = preprocess(data,opts[:lsequence])
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
    data = getChunk(dtest, opts[:chunksize], i)
    xtst, ytst = preprocess(data,opts[:lsequence])
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