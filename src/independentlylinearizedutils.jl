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
    u_chunks::Vector{Vector{Vector{S}}}
    time_masks::Vector{BitMatrix}

    chunk_size::Int

    # Index of next write into the last chunk
    u_offsets::Vector{Int}
    t_offset::Int

    function IndependentlyLinearizedSolutionChunks{T, S}(num_us::Int,
            chunk_size::Int = 100) where {T, S}
        return new([Vector{T}(undef, chunk_size)],
            [[Vector{S}(undef, chunk_size)] for _ in 1:num_us],
            [BitMatrix(undef, chunk_size, num_us)],
            chunk_size,
            [1 for _ in 1:num_us],
            1,
        )
    end
end

function get_chunks(ilsc::IndependentlyLinearizedSolutionChunks{T, S}) where {T, S}
    # Check if we need to allocate new `t` chunk
    if ilsc.t_offset > ilsc.chunk_size
        push!(ilsc.t_chunks, Vector{T}(undef, ilsc.chunk_size))
        push!(ilsc.time_masks, BitMatrix(undef, ilsc.chunk_size, length(ilsc.u_offsets)))
        ilsc.t_offset = 1
    end

    # Check if we need to allocate any new `u` chunks (but only for those with `u_mask`)
    for (u_idx, u_chunks) in enumerate(ilsc.u_chunks)
        if ilsc.u_offsets[u_idx] > ilsc.chunk_size
            push!(u_chunks, Vector{S}(undef, ilsc.chunk_size))
            ilsc.u_offsets[u_idx] = 1
        end
    end

    # return the last chunk for each
    return ilsc.t_chunks[end],
    ilsc.time_masks[end],
    [u_chunks[end] for u_chunks in ilsc.u_chunks]
end

"""
    store!(ilsc::IndependentlyLinearizedSolutionChunks, t, u, u_mask)

Store a new `u` vector into our `ilsc`, but only the values identified by the
given `u_mask`.
"""
function store!(ilsc::IndependentlyLinearizedSolutionChunks{T, S},
        t::T,
        u::AbstractVector{S},
        u_mask::BitVector) where {T, S}
    ts, time_mask, us = get_chunks(ilsc)

    # Store into the chunks, gated by `u_mask`
    for u_idx in 1:length(us)
        if u_mask[u_idx]
            us[u_idx][ilsc.u_offsets[u_idx]] = u[u_idx]
            ilsc.u_offsets[u_idx] += 1
        end
    end
    ts[ilsc.t_offset] = t
    time_mask[ilsc.t_offset, :] .= u_mask
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

    # Ragged matrix of `us`
    us::Vector{Vector{S}}

    # Bitmatrix denoting which time indices are used for which us.
    time_mask::BitMatrix
    
    # Temporary object used during construction, will be set to `nothing` at the end.
    ilsc::Union{Nothing,IndependentlyLinearizedSolutionChunks{T,S}}
end
# Helper function to create an ILS wrapped around an in-progress ILSC
function IndependentlyLinearizedSolution(ilsc::IndependentlyLinearizedSolutionChunks{T,S}) where {T,S}
    ils = IndependentlyLinearizedSolution(
        T[],
        Vector{S}[],
        BitMatrix(undef, 0,0),
        ilsc,
    )
    return ils
end
# Automatically create an ILS wrapped around an ILSC from a `prob`
function IndependentlyLinearizedSolution(prob::SciMLBase.AbstractDEProblem)
    return IndependentlyLinearizedSolution(
        IndependentlyLinearizedSolutionChunks{eltype(prob.tspan),eltype(prob.u0)}(length(prob.u0))
    )
end

Base.size(ils::IndependentlyLinearizedSolution) = size(ils.time_mask)
Base.length(ils::IndependentlyLinearizedSolution) = length(ils.ts)

function finish!(ils::IndependentlyLinearizedSolution)
    function trim_chunk(chunks::Vector, offset)
        chunks = [chunk for chunk in chunks]
        if eltype(chunks) <: Vector
            chunks[end] = chunks[end][1:(offset - 1)]
        elseif eltype(chunks) <: BitMatrix
            chunks[end] = chunks[end][1:(offset - 1), :]
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
    time_mask = vcat(trim_chunk(ilsc.time_masks, ilsc.t_offset)...)
    us = [vcat(trim_chunk(ilsc.u_chunks[u_idx], ilsc.u_offsets[u_idx])...)
          for u_idx in 1:length(ilsc.u_chunks)]

    # Sanity-check lengths
    if length(ts) != size(time_mask, 1)
        throw(ArgumentError("`length(ts)` must equal `size(time_mask, 1)`!"))
    end

    # All time masks must start and end with `1`:
    if !all(@view time_mask[1, :]) || !all(@view time_mask[end, :])
        throw(ArgumentError("Time mask must start and end with 1s!"))
    end

    # Length of time mask enable vectors must equal the lengths of our `us`:
    time_mask_lens = vec(sum(time_mask; dims = 1))
    if !all(time_mask_lens .== length.(us))
        throw(ArgumentError("Time mask must indicate same length as `us` ($(time_mask_lens) != $(length.(us)))"))
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
        findfirst(@view ils.time_mask[2:end, u_idx]) + 1)
    return seek_forward(ils, cursor, ils.ts[t_idx])
end
function interpolate(ils::IndependentlyLinearizedSolution{T},
        cursor::ILSStateCursor,
        t::T) where {T}
    u₀ = ils.us[cursor.u_idx][cursor.idx_u₀]
    u₁ = ils.us[cursor.u_idx][cursor.idx_u₀ + 1]
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
        next_t = findfirst(@view ils.time_mask[(cursor.idx_t₁ + 1):end, cursor.u_idx])
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
    u = S[S(0) for _ in ils.us]
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
    for u_idx in 1:length(u)
        cursors[u_idx] = seek_forward(ils, cursors[u_idx], t)
        u[u_idx] = interpolate(ils, cursors[u_idx], t)
    end

    return (t, u), (t_idx + 1, u, cursors)
end

"""
    sample!(out::Matrix{S}, ils::IndependentlyLinearizedSolution, ts::Vector{T})

Batch-sample `ils` at the given timepoints, storing into `out`.
"""
function sample!(out::Matrix{S},
        ils::IndependentlyLinearizedSolution{T, S},
        ts::AbstractVector{T}) where {T, S}
    sampled_size = (length(ts), length(ils.us))
    if size(out) != sampled_size
        throw(ArgumentError("Output size ($(size(out))) != sampled size ($(sampled_size))"))
    end

    # We don't make use of `iterate` here because we're sampling at arbitrary timepoints
    cursors = seek(ils)
    for (t_idx, t) in enumerate(ts)
        for u_idx in 1:length(ils.us)
            cursors[u_idx] = seek_forward(ils, cursors[u_idx], t)
            out[t_idx, u_idx] = interpolate(ils, cursors[u_idx], t)
        end
    end
    return out
end
function sample(ils::IndependentlyLinearizedSolution{T, S},
        ts::AbstractVector{T}) where {T, S}
    out = Matrix{S}(undef, length(ts), length(ils.us))
    return sample!(out, ils, ts)
end
