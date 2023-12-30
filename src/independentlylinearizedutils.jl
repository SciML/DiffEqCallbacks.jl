using SciMLBase

export IndependentlyLinearizedSolution

"""
    IndependentlyLinearizedSolutionChunks

When constructing an `IndependentlyLinearizedSolution` via the `IndependentlyLinearizingCallback`,
we use this indermediate structure to reduce allocations and collect the unknown number of timesteps
that the solve will generate.
"""
mutable struct IndependentlyLinearizedSolutionChunks{T, S}
    t_chunks::Vector{Vector{T}}
    u_chunks::Vector{Vector{Matrix{S}}}
    time_masks::Vector{BitMatrix}

    # Index of next write into the last chunk
    u_offsets::Vector{Int}
    t_offset::Int

    function IndependentlyLinearizedSolutionChunks{T, S}(num_us::Int, num_derivatives::Int = 0,
            chunk_size::Int = 100) where {T, S}
        return new([Vector{T}(undef, chunk_size)],
            [[Matrix{S}(undef, num_derivatives+1, chunk_size)] for _ in 1:num_us],
            [BitMatrix(undef, num_us, chunk_size)],
            [1 for _ in 1:num_us],
            1,
        )
    end
end

function chunk_size(ilsc::IndependentlyLinearizedSolutionChunks)
    # If we've been finalized, just return `0`
    if isempty(ilsc.t_chunks)
        return 0
    end
    return length(first(ilsc.t_chunks))
end

function num_us(ilsc::IndependentlyLinearizedSolutionChunks)
    # If we've been finalized, just return `0`
    if isempty(ilsc.t_chunks)
        return 0
    end
    return length(ilsc.u_chunks)
end

function num_derivatives(ilsc::IndependentlyLinearizedSolutionChunks)
    # If we've been finalized, just return `0` (which means only the primal)
    if isempty(ilsc.t_chunks)
        return 0
    end
    return size(first(first(ilsc.u_chunks)), 1) - 1
end

function Base.isempty(ilsc::IndependentlyLinearizedSolutionChunks)
    return length(ilsc.t_chunks) == 1 && ilsc.t_offset == 1
end

function get_chunks(ilsc::IndependentlyLinearizedSolutionChunks{T, S}) where {T, S}
    # Check if we need to allocate new `t` chunk
    chunksize = chunk_size(ilsc)
    if ilsc.t_offset > chunksize
        push!(ilsc.t_chunks, Vector{T}(undef, chunksize))
        push!(ilsc.time_masks, BitMatrix(undef, length(ilsc.u_offsets), chunksize))
        ilsc.t_offset = 1
    end

    # Check if we need to allocate any new `u` chunks (but only for those with `u_mask`)
    for (u_idx, u_chunks) in enumerate(ilsc.u_chunks)
        if ilsc.u_offsets[u_idx] > chunksize
            push!(u_chunks, Matrix{S}(undef, num_derivatives(ilsc)+1, chunksize))
            ilsc.u_offsets[u_idx] = 1
        end
    end

    # return the last chunk for each
    return (
        ilsc.t_chunks[end],
        ilsc.time_masks[end],
        [u_chunks[end] for u_chunks in ilsc.u_chunks],
    )
end

function get_prev_t(ilsc::IndependentlyLinearizedSolutionChunks)
    # Are we set to write into the beginning of a new chunk?
    if ilsc.t_offset == 1
        # Try to reach back to the previous chunk
        if length(ilsc.t_chunks) == 1
            # If this is the absolute first timepoint, just return -Inf
            return -Inf
        end
        # Otherwise, return the last element of the previous chunk
        return ilsc.t_chunks[end-1][end]
    end
    # Otherwise return the previous element of the current chunk
    return ilsc.t_chunks[end][ilsc.t_offset-1]
end

function get_prev_u(ilsc::IndependentlyLinearizedSolutionChunks{T,S}, u_out::Vector{S}) where {T,S}
    for u_idx in 1:length(ilsc.u_offsets)
        if ilsc.u_offsets[u_idx] == 1
            if length(ilsc.u_chunks[u_idx]) == 1
                # If we have never stored anything for this `u`, just return `0.0`
                u_out[u_idx] = S(0)
            end
            # Otherwise, get the last element of the previous chunk
            u_out[u_idx] = ilsc.u_chunks[u_idx][end-1][end]
        else
            # Otherwise, return the previous element of the current chunk
            u_out[u_idx] = ilsc.u_chunks[u_idx][end][ilsc.u_offsets[u_idx]-1]
        end
    end
    return u_out
end

"""
    store!(ilsc::IndependentlyLinearizedSolutionChunks, t, us, u_mask)

Store a new `us` matrix (one row per derivative level) into our `ilsc`,
but only the values identified by the given `u_mask`.  The `us` matrix
should be of the size `(num_us(ilsc), num_derivatives(ilsc))`.
"""
function store!(ilsc::IndependentlyLinearizedSolutionChunks{T, S},
                t::T,
                u::AbstractMatrix{S},
                u_mask::BitVector) where {T, S}
    # If `t` has been stored before, drop it.
    # We don't store duplicate timepoints, even though the solver sometimes does.
    if get_prev_t(ilsc) == t
        return
    end

    # Otherwise, let's get new chunks to deal with
    ts, time_mask, us = get_chunks(ilsc)

    # Store into the chunks, gated by `u_mask`
    for u_idx in 1:size(u, 2)
        if u_mask[u_idx]
            for deriv_idx in 1:size(u, 1)
                us[u_idx][deriv_idx, ilsc.u_offsets[u_idx]] = u[deriv_idx, u_idx]
            end
            ilsc.u_offsets[u_idx] += 1
        end
    end
    ts[ilsc.t_offset] = t
    time_mask[:, ilsc.t_offset] .= u_mask
    ilsc.t_offset += 1
end



"""
    IndependentlyLinearizedSolution

Efficient datastructure that holds a set of independently linearized solutions
(obtained via the `LinearizingSavingCallback`) with related, but slightly
different time vectors.  Stores a single time vector with a packed `BitMatrix`
denoting which `u` vectors are sampled at which timepoints.  Provides an
efficient `iterate()` method that can be used to reconstruct coherent views
of the state variables at all timepoints, as well as an efficient `sample!()`
method that can sample at arbitrary timesteps.
"""
mutable struct IndependentlyLinearizedSolution{T, S}
    # All timepoints, shared by all `us`
    ts::Vector{T}

    # Ragged matrix of `us`, where each `u` gets a matrix that grows with the
    # number of derivatives stored.
    us::Vector{Matrix{S}}

    # Bitmatrix denoting which time indices are used for which us.
    time_mask::BitMatrix
    
    # Temporary object used during construction, will be set to `nothing` at the end.
    ilsc::Union{Nothing,IndependentlyLinearizedSolutionChunks{T,S}}
end
# Helper function to create an ILS wrapped around an in-progress ILSC
function IndependentlyLinearizedSolution(ilsc::IndependentlyLinearizedSolutionChunks{T,S}) where {T,S}
    ils = IndependentlyLinearizedSolution(
        T[],
        Matrix{S}[],
        BitMatrix(undef, 0,0),
        ilsc,
    )
    return ils
end
# Automatically create an ILS wrapped around an ILSC from a `prob`
function IndependentlyLinearizedSolution(prob::SciMLBase.AbstractDEProblem, num_derivatives = 0)
    return IndependentlyLinearizedSolution(
        IndependentlyLinearizedSolutionChunks{eltype(prob.tspan),eltype(prob.u0)}(length(prob.u0), num_derivatives)
    )
end

num_derivatives(ils::IndependentlyLinearizedSolution) = size(first(ils.us), 1)
num_us(ils::IndependentlyLinearizedSolution) = length(ils.us)
Base.size(ils::IndependentlyLinearizedSolution) = size(ils.time_mask)
Base.length(ils::IndependentlyLinearizedSolution) = length(ils.ts)

function finish!(ils::IndependentlyLinearizedSolution)
    function trim_chunk(chunks::Vector, offset)
        chunks = [chunk for chunk in chunks]
        if eltype(chunks) <: AbstractVector
            chunks[end] = chunks[end][1:(offset - 1)]
        elseif eltype(chunks) <: AbstractMatrix
            chunks[end] = chunks[end][:, 1:(offset - 1)]
        else
            error(eltype(chunks))
        end
        if isempty(chunks[end])
            pop!(chunks)
        end
        return chunks
    end

    ilsc = ils.ilsc::IndependentlyLinearizedSolutionChunks
    ts = vcat(trim_chunk(ilsc.t_chunks, ilsc.t_offset)...)
    time_mask = hcat(trim_chunk(ilsc.time_masks, ilsc.t_offset)...)
    us = [hcat(trim_chunk(ilsc.u_chunks[u_idx], ilsc.u_offsets[u_idx])...)
          for u_idx in 1:length(ilsc.u_chunks)]

    # Sanity-check lengths
    if length(ts) != size(time_mask, 2)
        throw(ArgumentError("`length(ts)` must equal `size(time_mask, 2)`!"))
    end

    # All time masks must start and end with `1`:
    if !all(@view time_mask[:, 1]) || !all(@view time_mask[:, end])
        throw(ArgumentError("Time mask must start and end with 1s!"))
    end

    # Length of time mask enable vectors must equal the lengths of our `us`:
    time_mask_lens = vec(sum(time_mask; dims = 2))
    us_lens = [size(u, 2) for u in us]
    if !all(time_mask_lens .== us_lens)
        throw(ArgumentError("Time mask must indicate same length as `us` ($(time_mask_lens) != $(us_lens))"))
    end

    # Update our struct, release the `ilsc`
    ils.ilsc = nothing
    ils.ts = ts
    ils.us = us
    ils.time_mask = time_mask
    return ils
end

struct ILSStateCursor
    # The index into `us` that identifies this state
    u_idx::Int

    idx_u₀::Int
    # idx_u₁ is by definition idx_u₀ + 1, so we don't store it

    # Time index of u₀
    idx_t₀::Int
    # Time index of u₁
    idx_t₁::Int
end
# Helper to construct a state cursor off of an ILS, at a particular time index
function ILSStateCursor(ils::IndependentlyLinearizedSolution, u_idx::Int, t_idx::Int = 1)
    cursor = ILSStateCursor(u_idx,
        1,
        1,
        findfirst(@view ils.time_mask[u_idx, 2:end]) + 1)
    return seek_forward(ils, cursor, ils.ts[t_idx])
end
function interpolate(ils::IndependentlyLinearizedSolution{T},
        cursor::ILSStateCursor,
        t::T,
        deriv_idx::Int) where {T}
    u₀ = ils.us[cursor.u_idx][deriv_idx + 1, cursor.idx_u₀]
    u₁ = ils.us[cursor.u_idx][deriv_idx + 1, cursor.idx_u₀ + 1]
    t₀ = ils.ts[cursor.idx_t₀]
    t₁ = ils.ts[cursor.idx_t₁]
    return (u₁ - u₀) / (t₁ - t₀) * (t - t₀) + u₀
end

"""
    seek_forward(ils::IndependentlyLinearizedSolution, cursor::ILSStateCursor, t_target)

Seek the given `cursor` forward until it contains `t_target`.  Does not seek backward, use `seek()`
for the more general formulation, this form is optimized for the inner loop of `iterate()`.
"""
function seek_forward(ils::IndependentlyLinearizedSolution{T},
        cursor::ILSStateCursor,
        t_target::T) where {T}
    # We do not test `t_start` because we don't support seeking backward here
    while ils.ts[cursor.idx_t₁] < t_target
        next_t = findfirst(@view ils.time_mask[cursor.u_idx, (cursor.idx_t₁ + 1):end])
        cursor = ILSStateCursor(cursor.u_idx,
            cursor.idx_u₀ + 1,
            cursor.idx_t₁,
            next_t + cursor.idx_t₁)
    end
    return cursor
end

function Base.seek(ils::IndependentlyLinearizedSolution{T},
        cursor::ILSStateCursor,
        t_target::T) where {T}
    # If we need to rewind, just start from the beginning
    if t_target < ils.ts[cursor.idx_t₀]
        cursor = ILSStateCursor(ils, cursor.u_idx)
    end
    return seek_forward(ils, cursor, t_target)
end

function Base.seek(ils::IndependentlyLinearizedSolution, t_idx = 1)
    return [ILSStateCursor(ils, u_idx, t_idx) for u_idx in 1:length(ils.us)]
end
function iteration_state(ils::IndependentlyLinearizedSolution{T, S}, t_idx=1) where {T, S}
    # Nice little hack so we don't have to allocate `u` over and over again
    u = zeros(S, (num_us(ils), num_derivatives(ils)))
    cursors = seek(ils, t_idx)
    return (t_idx, u, cursors)
end
function Base.iterate(ils::IndependentlyLinearizedSolution{T, S},
        (t_idx, u, cursors) = iteration_state(ils)) where {T, S}
    if t_idx > length(ils.ts)
        return nothing
    end

    # We iteratively inch `offsets` forward, efficiently reconstructing a full set of `u`'s
    t = ils.ts[t_idx]
    for deriv_idx in 0:(size(u, 2) - 1)
        for u_idx in 1:size(u, 1)
            cursors[u_idx] = seek_forward(ils, cursors[u_idx], t)
            u[u_idx, deriv_idx+1] = interpolate(ils, cursors[u_idx], t, deriv_idx)
        end
    end

    return (t, u), (t_idx + 1, u, cursors)
end

"""
    sample!(out::Matrix{S}, ils::IndependentlyLinearizedSolution, ts::Vector{T}, deriv_idx::Int = 0)

Batch-sample `ils` at the given timepoints for the given derivative level, storing into `out`.
"""
function sample!(out::Matrix{S},
                 ils::IndependentlyLinearizedSolution{T, S},
                 ts::AbstractVector{T},
                 deriv_idx::Int = 0) where {T, S}
    sampled_size = (length(ts), length(ils.us))
    if size(out) != sampled_size
        throw(ArgumentError("Output size ($(size(out))) != sampled size ($(sampled_size))"))
    end

    # We don't make use of `iterate` here because we're sampling at arbitrary timepoints
    cursors = seek(ils)
    for (t_idx, t) in enumerate(ts)
        for u_idx in 1:length(ils.us)
            cursors[u_idx] = seek_forward(ils, cursors[u_idx], t)
            out[t_idx, u_idx] = interpolate(ils, cursors[u_idx], t, deriv_idx)
        end
    end
    return out
end
function sample(ils::IndependentlyLinearizedSolution{T, S},
                ts::AbstractVector{T},
                deriv_idx::Int = 0) where {T, S}
    out = Matrix{S}(undef, length(ts), length(ils.us))
    return sample!(out, ils, ts, deriv_idx)
end
