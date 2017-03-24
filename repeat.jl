function read_data()
  f = open("sample_tr.seq")
  train = readlines(f)
  train = map(line -> split(line), train)
  close(f)
  f = open("sample_te.seq")
  test = readlines(f)
  test = map(line -> split(line), test)
  close(f)
  return train,test 
end

function seqMap(na)
  seqMap = Dict{Char,UInt8}('A' => 1, 'G' => 2, 'C' => 3, 'T' => 4)
  return seqMap[na]
end

function seqToVector!(matrix, seq, index)
  index = map(x-> sub2ind((4,1000,1,index),x[1],x[2],1,index) , 
              map(x -> (seqMap(x[2]),x[1]), 
                filter(x -> x[2] != 'N', enumerate(seq))))
  matrix[index] = 1
end

function preprocess(data)
  preprocessed = zeros(UInt8,4,1000,1,length(data))
  map( x -> seqToVector!(preprocessed, x[2][1], x[1]),enumerate(data))
  return preprocessed, convert(Array{UInt8}, [parse(elm[2]) for elm in data])
end

function minibatch(x,y,sz)
    data = Any[]
    for i=1:sz:size(x,2)
        push!(data,(x[:,:,:,i:i+sz-1],y[i:i+sz-1]))
    end
    return data
end

function main()
  trn, tst = read_data()
  xtrn, ytrn = preprocess(trn)
  xtst, ytst = preprocess(tst)
  dtrn = minibatch(xtrn,ytrn,100)
end

main()