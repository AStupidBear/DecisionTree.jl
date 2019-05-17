using DecisionTree
using DecisionTree.util: entropy
using Statistics

x = rand(0x00:0x0f, 1000000, 600)
y = vec(Int.(mean(x, dims = 2) .> 7))

model = build_tree(y, x, purity_function = entropy)
preds = apply_tree(model, x)
sum(abs, y .- preds)

# purity_function = function (y, indX, il, ir)
#     nsl, nsr = [0, 0], [0, 0]
#     for i in il
#         nsl[y[indX[i]]] += 1
#     end
#     for i in ir
#         nsr[y[indX[i]]] += 1
#     end
#     -entropy(nsl, sum(nsl)) * length(il) +
#     -entropy(nsr, sum(nsr)) * length(ir) + 100
# end

# purity_function = function (y, indX, il, ir)
#     l = 0f0
#     for i in il
#         y[indX[i]] = 1
#     end
# end
