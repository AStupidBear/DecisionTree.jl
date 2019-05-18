using DecisionTree
using DecisionTree.util: entropy
using Statistics

x = rand(0x00:0x0f, 100000, 20)
y = map(Int32, (x[:, 1] .> 3) .& (x[:, 2] .> 3))

treepurity = function (y, indX, region, split)
    ncl, ncr = [0, 0], [0, 0]
    for i in region[1:split]
        ncl[y[indX[i]]] += 1
    end
    for i in region[split+1:end]
        ncr[y[indX[i]]] += 1
    end
    nl, nr = sum(ncl), sum(ncr)
    nl * entropy(ncl, nl) + nr * entropy(ncr, nr)
end

model = DecisionTreeClassifier(min_purity_increase = 1)
DecisionTree.fit!(model, x, y, purity_function = treepurity)
ŷ = DecisionTree.predict(model, x)
cm = confusion_matrix(y, ŷ)