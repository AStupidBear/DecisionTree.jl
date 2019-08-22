# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of DecisionTree.jl is released under the MIT license.

# written by Poom Chiarawongse <eight1911@gmail.com>

module treeclassifier
    include("../util.jl")
    import Random
    using Printf

    export fit, fit_zero_one

    mutable struct NodeMeta{S}
        l           :: NodeMeta{S}      # right child
        r           :: NodeMeta{S}      # left child
        label       :: Int              # most likely label
        feature     :: Int              # feature used for splitting
        threshold   :: S                # threshold value
        is_leaf     :: Bool
        depth       :: Int
        region      :: UnitRange{Int}   # a slice of the samples used to decide the split of the node
        features    :: Vector{Int}      # a list of features not known to be constant
        split_at    :: Int              # index of samples

        function NodeMeta{S}(
                features :: Vector{Int},
                region   :: UnitRange{Int},
                depth    :: Int) where S
            node = new{S}()
            node.depth = depth
            node.region = region
            node.features = features
            node.is_leaf = false
            node
        end
    end

    struct Tree{S, T}
        root   :: NodeMeta{S}
        list   :: Vector{T}
        labels :: Vector{Int}
    end

    # find an optimal split that satisfy the given constraints
    # (max_depth, min_samples_split, min_purity_increase)
    function _split!(
            X                   :: Matrix{S},   # the feature array
            Y                   :: Vector{T},   # the label array
            W                   :: Vector{U},   # the weight vector
            cinds               :: I,           # column indices
            purity_function     :: Function,
            node                :: NodeMeta{S}, # the node to split
            max_features        :: Int,         # number of features to consider
            max_depth           :: Int,         # the maximum depth of the resultant tree
            min_samples_leaf    :: Int,         # the minimum number of samples each leaf needs to have
            min_samples_split   :: Int,         # the minimum number of samples in needed for a split
            min_purity_increase :: Float64,     # minimum purity needed for a split
            indX                :: Vector{Int}, # an array of sample indices,
                                                # we split using samples in indX[node.region]
            # the six arrays below are given for optimization purposes
            nc                  :: Vector{U},   # nc maintains a dictionary of all labels in the samples
            ncl                 :: Vector{U},   # ncl maintains the counts of labels on the left
            ncr                 :: Vector{U},   # ncr maintains the counts of labels on the right
            Xf                  :: Vector{S},
            Yf                  :: Vector{T},
            Wf                  :: Vector{U},
            rng                 :: Random.AbstractRNG) where {S, T, U, I}
        treeopt = methods(purity_function).ms[1].nargs > 3
        region = node.region
        n_samples = length(region)
        n_classes = length(nc)

        nc[:] .= zero(U)
        @simd for i in region
            @inbounds nc[Y[indX[i]]] += W[indX[i]]
        end
        nt = sum(nc)
        node.label = argmax(nc)

        r_start = region.start - 1
        features = node.features
        n_features = length(features)
        best_purity = typemin(U)

        base_purity = !treeopt ? typemin(U) : -purity_function(Y, indX, region, 0, cinds)
        best_feature = -1
        threshold_lo = X[1]
        threshold_hi = X[1]

        if (min_samples_leaf * 2 >  n_samples
            || min_samples_split    >  n_samples
            || max_depth            <= node.depth
            || nc[node.label]       == nt && !treeopt)
            node.is_leaf = true
            min_samples_leaf * 2 > n_samples && println("min_samples_leaf * 2 > n_samples")
            min_samples_split > n_samples && println("min_samples_split > n_samples")
            max_depth <= node.depth && println("max_depth <= node.depth")
            return (a...) -> nothing, [[base_purity, ()]]
        end

        indf = 1
        # the number of new constants found during this split
        n_const = 0
        # true if every feature is constant
        unsplittable = true
        # the number of non constant features we will see if
        # only sample n_features used features
        # is a hypergeometric random variable
        total_features = size(X, 2)
        # this is the total number of features that we expect to not
        # be one of the known constant features. since we know exactly
        # what the non constant features are, we can sample at 'non_consts_used'
        # non constant features instead of going through every feature randomly.
        non_consts_used = util.hypergeometric(n_features, total_features-n_features, max_features, rng)
        purities = []
        @inbounds while (unsplittable || indf <= non_consts_used) && indf <= n_features
            feature = let
                indr = rand(rng, indf:n_features)
                features[indf], features[indr] = features[indr], features[indf]
                features[indf]
            end

            # in the begining, every node is
            # on right of the threshold
            ncl[:] .= zero(U)
            ncr[:] = nc
            @simd for i in 1:n_samples
                Xf[i] = X[indX[i + r_start], feature]
            end

            # sort Yf and indX by Xf
            util.q_bi_sort!(Xf, indX, 1, n_samples, r_start)

            @simd for i in 1:n_samples
                Yf[i] = Y[indX[i + r_start]]
                Wf[i] = W[indX[i + r_start]]
            end

            hi = 0
            nl, nr = zero(U), nt
            is_constant = true
            last_f = Xf[1]
            while hi < n_samples
                lo = hi + 1
                curr_f = Xf[lo]
                hi = (lo < n_samples && curr_f == Xf[lo+1]
                    ? searchsortedlast(Xf, curr_f, lo, n_samples, Base.Order.Forward)
                    : lo)

                (lo != 1) && (is_constant = false)
                # honor min_samples_leaf
                # if nl >= min_samples_leaf && nr >= min_samples_leaf
                # @assert nl == lo-1,
                # @assert nr == n_samples - (lo-1) == n_samples - lo + 1
                if lo-1 >= min_samples_leaf && n_samples - (lo-1) >= min_samples_leaf
                    unsplittable = false
                    purity = treeopt ? -purity_function(Y, indX, region, lo - 1, cinds) :
                            -nl * purity_function(ncl, nl) - nr * purity_function(ncr, nr)
                    push!(purities, [purity, (feature, last_f, curr_f)])
                    if purity > best_purity
                        # will take average at the end
                        threshold_lo = last_f
                        threshold_hi = curr_f
                        best_purity  = purity
                        best_feature = feature
                    end
                end

                # fill ncl and ncr in the direction
                # that would require the smaller number of iterations
                # i.e., hi - lo < n_samples - hi
                if (hi << 1) < n_samples + lo
                    @simd for i in lo:hi
                        ncr[Yf[i]] -= Wf[i]
                    end
                else
                    ncr[:] .= zero(U)
                    @simd for i in (hi+1):n_samples
                        ncr[Yf[i]] += Wf[i]
                    end
                end

                nr = zero(U)
                @simd for lab in 1:n_classes
                    nr += ncr[lab]
                    ncl[lab] = nc[lab] - ncr[lab]
                end

                nl = nt - nr
                last_f = curr_f
            end

            # keep track of constant features to be used later.
            if is_constant
                n_const += 1
                features[indf], features[n_const] = features[n_const], features[indf]
            end

            indf += 1
        end

        @printf("best_purity: %.4g, base_purity: %.4g\n", best_purity, base_purity)
        # no splits honor min_samples_leaf
        @inbounds if unsplittable || 
            treeopt ? best_purity - base_purity < min_purity_increase :
            best_purity / nt + util.entropy(nc, nt) < min_purity_increase
            node.is_leaf = true
            treeopt && purity_function(Y, indX, region, 0, cinds)
            unsplittable ? println("node is unsplittable") :
            println("purity increase is not significant")
            return (a...) -> nothing, [[base_purity, ()]]
        else
            function (node, Y, indX, (best_feature, threshold_lo, threshold_hi))
                bf = Int(best_feature)
                @simd for i in 1:n_samples
                    Xf[i] = X[indX[i + r_start], bf]
                end

                try
                    node.threshold = (threshold_lo + threshold_hi) / 2.0
                catch
                    node.threshold = threshold_hi
                end
                # split the samples into two parts: ones that are greater than
                # the threshold and ones that are less than or equal to the threshold
                #                                 ---------------------
                # (so we partition at threshold_lo instead of node.threshold)
                node.split_at = util.partition!(indX, Xf, threshold_lo, region)
                treeopt && purity_function(Y, indX, region, node.split_at, cinds)
                node.feature = best_feature
                node.features = features[(n_const+1):n_features]
            end, purities
        end

    end
    @inline function fork!(node::NodeMeta{S}) where S
        ind = node.split_at
        region = node.region
        features = node.features
        # no need to copy because we will copy at the end
        node.l = NodeMeta{S}(features, region[    1:ind], node.depth+1)
        node.r = NodeMeta{S}(features, region[ind+1:end], node.depth+1)
    end

    function check_input(
            X                   :: Matrix{S},
            Y                   :: Vector{T},
            W                   :: Vector{U},
            max_features        :: Int,
            max_depth           :: Int,
            min_samples_leaf    :: Int,
            min_samples_split   :: Int,
            min_purity_increase :: Float64) where {S, T, U}
        n_samples, n_features = size(X)
        if length(Y) != n_samples
            throw("dimension mismatch between X and Y ($(size(X)) vs $(size(Y))")
        elseif length(W) != n_samples
            throw("dimension mismatch between X and W ($(size(X)) vs $(size(W))")
        elseif max_depth < -1
            throw("unexpected value for max_depth: $(max_depth) (expected:"
                * " max_depth >= 0, or max_depth = -1 for infinite depth)")
        elseif n_features < max_features
            throw("number of features $(n_features) is less than the number "
                * "of max features $(max_features)")
        elseif max_features < 0
            throw("number of features $(max_features) must be >= zero ")
        elseif min_samples_leaf < 1
            throw("min_samples_leaf must be a positive integer "
                * "(given $(min_samples_leaf))")
        elseif min_samples_split < 2
            throw("min_samples_split must be at least 2 "
                * "(given $(min_samples_split))")
        end
    end

    Base.:(==)(x::NodeMeta, y::NodeMeta) = x.depth == y.depth && x.region == y.region && x.features == y.features 

    function findnode(root, node)
        root == node && return root
        root.is_leaf && return
        if isdefined(root, :l)
            nl = findnode(root.l, node)
            !isnothing(nl) && return nl
        end
        if isdefined(root, :r)
            nr = findnode(root.r, node)
            !isnothing(nr) && return nr
        end
        return nothing
    end

    function _fit(
            X                     :: Matrix{S},
            Y                     :: Vector{T},
            W                     :: Vector{U},
            cinds                 :: I,
            loss                  :: Function,
            n_classes             :: Int,
            beam_width            :: Int,
            max_features          :: Int,
            max_depth             :: Int,
            min_samples_leaf      :: Int,
            min_samples_split     :: Int,
            min_purity_increase   :: Float64,
            rng=Random.GLOBAL_RNG :: Random.AbstractRNG) where {S, T, U, I}
        n_samples, n_features = size(X)

        nc  = Array{U}(undef, n_classes)
        ncl = Array{U}(undef, n_classes)
        ncr = Array{U}(undef, n_classes)
        Wf  = Array{U}(undef, n_samples)
        Xf  = Array{S}(undef, n_samples)
        Yf  = Array{T}(undef, n_samples)

        Ys = [copy(Y) for i in 1:beam_width]
        indXs = [collect(1:n_samples) for i in 1:beam_width]
        roots = [NodeMeta{S}(collect(1:n_features), 1:n_samples, 0) for i in 1:beam_width]
        stacks = [i == 1 ? NodeMeta{S}[roots[i]] : NodeMeta{S}[] for i in 1:beam_width]
        purities = [[] for i in 1:beam_width]
        updates = Any[undef for i in 1:beam_width]
        nodes = deepcopy(roots)
        while any(!isempty, stacks)
            for i in 1:beam_width
                isempty(stacks[i]) && continue
                nodes[i] = pop!(stacks[i])
                updates[i], purities[i] = _split!(
                    X, Ys[i], W, cinds,
                    loss, nodes[i],
                    max_features,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split,
                    min_purity_increase,
                    indXs[i],
                    nc, ncl, ncr, Xf, Yf, Wf, rng)
            end
            beam_purities = vcat([push!.(p, i) for (i, p) in enumerate(purities)]...)
            beam_purities = sort(beam_purities, by = first, rev = true)[1:min(end, beam_width)]
            beam_width > 1 && println("beam_purities: ", first.(beam_purities))
            Ys, indXs, roots, stacks, nodes = map(beam_purities) do (purity, conf, i)
                root = deepcopy(roots[i])
                node = findnode(root, nodes[i])
                stack = [findnode(root, node′) for node′ in stacks[i]]
                @assert !isnothing(node) && all(!isnothing, stack)
                Y′, indX = copy(Ys[i]), copy(indXs[i])
                updates[i](node, Y′, indX, conf)
                return Y′, indX, root, stack, node
            end |> z -> collect.(zip(z...))
            for i in 1:beam_width
                if !nodes[i].is_leaf
                    fork!(nodes[i])
                    push!(stacks[i], nodes[i].r)
                    push!(stacks[i], nodes[i].l)
                end
            end
        end
        root, indX = roots[1], indXs[1]
        copyto!(Y, Ys[1])
        return (root, indX)
    end

    using Statistics
    function fit(;
            X                     :: Matrix{S},
            Y                     :: Vector{T},
            W                     :: Union{Nothing, Vector{U}},
            cinds                 :: I,
            beam_width            :: Int,
            max_features          :: Int,
            max_depth             :: Int,
            min_samples_leaf      :: Int,
            min_samples_split     :: Int,
            min_purity_increase   :: Float64,
            rng=Random.GLOBAL_RNG :: Random.AbstractRNG,
            purity_function = util.entropy) where {S, T, U, I}

        n_samples, n_features = size(X)
        list, Y_ = util.assign(Y)
        treeopt = methods(purity_function).ms[1].nargs > 3
        Y_ = treeopt ? fill!(Y, median(Y)) : Y_
        if W == nothing
            W = fill(1.0, n_samples)
        end

        check_input(
            X, Y, W,
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase)

        @eval Main yy = $Y
        root, indX = _fit(
            X, Y_, W, cinds,
            purity_function,
            length(list),
            beam_width,
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            rng)

        return Tree{S, T}(root, list, indX)
    end

    function fit_zero_one(;
            X                     :: Matrix{S},
            Y                     :: Vector{T},
            W                     :: Union{Nothing, Vector{U}},
            max_features          :: Int,
            max_depth             :: Int,
            min_samples_leaf      :: Int,
            min_samples_split     :: Int,
            min_purity_increase   :: Float64,
            rng=Random.GLOBAL_RNG :: Random.AbstractRNG) where {S, T, U}

        n_samples, n_features = size(X)
        list, Y_ = util.assign(Y)
        if W == nothing
            W = fill(1.0, n_samples)
        end

        check_input(
            X, Y, W,
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase)

        root, indX = _fit(
            X, Y_, W,
            util.zero_one,
            length(list),
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            rng)

        return Tree{S, T}(root, list, indX)
    end

end
