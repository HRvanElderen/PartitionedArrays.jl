function dot_tturbo(a::PVector, b::PVector)
	s = zero(eltype(a))
	s = map(own_values(a), own_values(b)) do a_own, b_own
		@tturbo for i ∈ eachindex(a_own, b_own)
			s += a_own[i] * b_own[i]
		end
		s
	end
	sum(s)
end


muladd!(b, A, x) = mul!(b, A, x, one(eltype(b)), one(eltype(b)))

function tspmv!(y, A, x)
	t = consistent!(x)
	@tturbo foreach(mul!, own_values(y), own_own_values(A), own_values(x))
	wait(t)
	@tturbo foreach(muladd!, own_values(y), own_ghost_values(A), ghost_values(x))
	y
end

function waxpby(alpha, x, beta, y)
	y_vals = map(partition(x), partition(y)) do x_vals, y_vals
		if alpha == 1
			@tturbo @. y_vals = x_vals + beta * y_vals
		elseif beta == 1
			@tturbo @. y_vals = alpha * x_vals + y_vals
		else
			@tturbo @. y_vals = alpha * x_vals + beta * y_vals
		end
	end
	PVector(y_vals, partition(axes(y, 1)))
end
