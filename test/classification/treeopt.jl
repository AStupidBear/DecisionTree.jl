using DecisionTree
using DecisionTree.util: entropy

x = rand(10000, 10)
y = [all(z -> z > 0.5, xi) for xi in eachrow(x)]

purity_function = function (y, indX, il, ir)
    nsl, nsr = [0, 0], [0, 0]
    for i in il
        nsl[y[indX[i]]] += 1
    end
    for i in ir
        nsr[y[indX[i]]] += 1
    end
    -entropy(nsl, sum(nsl)) * length(il) +
    -entropy(nsr, sum(nsr)) * length(ir) + 100
end
model = build_tree(y, x, purity_function = entropy)
preds = apply_tree(model, x)
sum(abs, y .- preds)

# purity_function = function (y, indX, il, ir)
#     l = 0f0
#     for i in il
#         y[indX[i]] = 1
#     end
# end