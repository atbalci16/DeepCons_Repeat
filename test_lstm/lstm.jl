using Knet

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