include("preprocess.jl")
include("lstm.jl")
include("training.jl")

using JLD,ArgParse

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
      train(w,dtrn,params,copy(cell),copy(hidden),predict)
    end
    nloss = 0;ninstance = 0
    tloss = 0;tinstance = 0
    for i=1:ival
      data = getChunk(dval, opts[:chunksize], i)
      xva, yva = preprocess(data)
      dva = minibatch(xva,yva,opts[:batchsize])
      nloss, ninstance= avgloss(w,dva,copy(cell),copy(hidden),loss,predict)
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
    ncorr, nwrong, ninstance = accuracy(bestw,dtst,copy(cell),copy(hidden),predict)
    corr += ncorr; wrong += nwrong; instance += ninstance
    xtst = 0;ytst = 0;data = 0;dtst = 0;
  end
  print("Test accuracy is $(corr/instance) $(wrong/instance) \n")
  print("Writing the weights of the best model to weights.jld \n")
  save("weights.jld","w",map(x->convert(Array{Float32},x),bestw),"J",validLost,"s",bests)
  return bestw
end

w = main()

