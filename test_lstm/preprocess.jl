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