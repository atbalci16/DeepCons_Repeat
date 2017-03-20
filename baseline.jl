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

function preprocess(data)
	x = zeros(UInt8,4000,length(data))
	ylabel =[parse(elm[2]) for elm in data]
	y = oneHot(ylabel)
	for (index,elm) in enumerate(data)
		seqToVector!(x,elm[1],index)
	end
	return x, y
end

function minibatch(x,y,sz)
    data = Any[]
    for i=1:sz:size(x,2)
        push!(data,(x[:,i:i+sz-1],y[:,i:i+sz-1]))
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
    y = w[end-1]*x .+ w[end]
    y = y .- maximum(y,1)
    expsum = sum(exp(y),1)
    pred = exp(y)./expsum
    return pred
end

function loss(w,x,ygold)
    pred_y = predict(w,x)
    lost = -sum(ygold .*pred_y) / size(x, 2)
    return lost
end

function avg_loss(w,data)
    return sum([loss(w,x,y) for (x,y) in data])/length(data)
end

function accuracy(w, data)
    ncorrect = 0
    ninstance = 0
    nloss = 0
    for (x,y) in data
        pred_y = predict(w,x)
        pred_y = pred_y .== maximum(pred_y,1)
        ncorrect += sum(pred_y .* y)
        ninstance += size(pred_y,2)
    end
    nloss = ninstance - ncorrect
    return (ncorrect/ninstance, nloss/ninstance)
end

function main()
	trn, tst = read_data()
	xtrn, ytrn = preprocess(trn)
	xtst, ytst = preprocess(tst)
	w = weights([4000 2])
	dtrn = minibatch(xtrn,ytrn,100)
    dtst = minibatch(xtst,ytst,100)
	print("training accuracy        : $(accuracy(w, dtrn))\ntesting accuracy         : $(accuracy(w, dtst))\n")
    print("average loss for training: $(avg_loss(w,dtrn))\naverage loss for testing : $(avg_loss(w,dtst))\n")

end

main()
