# for p in ("Knet")
#     Pkg.installed(p) == nothing && Pkg.add(p)
# end

using Knet, HDF5

function dataToh5(sz,data)
  ifile = open(data, "r")
  ofile = h5open(string(data[1:end-4],".h5"), "w")
  train = readlines(ifile)
  train = map(line -> split(line), train)
  x = convert(Array{String},[elm[1] for elm in train])
  y = convert(Array{String},[elm[2] for elm in train])
  index = 1
  for i=1:sz:length(x)
    g = g_create(ofile,dec(index))
    if i+sz-1 > length(x)
      g["x"] = x[i:end]
      g["y"] = y[i:end]
    else
      g["x"] = x[i:i+sz-1]
      g["y"] = y[i:i+sz-1]
    end
    index += 1
  end
  close(ifile)
  close(ofile)
end

function prepData()
  isfile("data_train.h5") || dataToh5(19200,"data_train.seq")
  isfile("data_test.h5") || dataToh5(19200,"data_test.seq")
  isfile("data_valid.h5") || dataToh5(19200,"data_valid.seq")
end

function read_data(file,index)
  f = h5open(file)
  data = read(f)[dec(index)]
  return data["x"], data["y"]
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

function preprocess(xdata,ydata)
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

function initparams(weights;learning_rate=0.001)
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
        map(x -> update!(x[1],x[2],x[3]), zip(w, w_grad, params))
    end
    #print("Free gpu memory $(Knet.gpufree())\n")
    return w
end

function avgloss(w,data)
    sum = cnt = 0
    for (x,y) in data
        sum += loss(w,x,y)
        cnt += 1
    end
    return sum/cnt
end

function main()
  print("free gpu memory $(Knet.gpufree())\n")
  prepData()
  w = weights()
  params = initparams(w)
  @time for epoch=1:100
    average_loss = 0
    for i=1:68
      print("loading data #$i...\n")
      @time x, y = read_data("data_train.h5",i)
      @time xtrn, ytrn = preprocess(x,y)
      @time dtrn = minibatch(xtrn,ytrn,128)
      @time train(w,dtrn,params)
      @time average_loss += avgloss(w,dtrn)
      xtrn = 0
      ytrn = 0
      x = 0
      y = 0
      dtrn = 0
      @time gc()
    end
    print("$epoch; average lost is $(average_loss/51) \n")
    average_loss = 0
    for i=1:9
      @time x, y = read_data("data_valid.h5",i)
      @time xtrn, ytrn = preprocess(x,y)
      @time dtrn = minibatch(xtrn,ytrn,128)
      @time average_loss += avgloss(w,dtrn)
      xtrn = 0
      ytrn = 0
      x = 0
      y = 0
      dtrn = 0
      @time gc()
    end
    print("$epoch; validation lost is $(average_loss/9) \n")
  end
  for i=1:9
    @time x, y = read_data("data_test.h5",i)
    @time xtrn, ytrn = preprocess(x,y)
    @time dtrn = minibatch(xtrn,ytrn,128)
    @time average_loss += avgloss(w,dtrn)
    xtrn = 0
    ytrn = 0
    x = 0
    y = 0
    dtrn = 0
    @time gc()
  end
  print("$epoch; test lost is $(average_loss/9) \n")
end

main()
