require("src/helpers.jl")
using MachineLearning
using MAT

data_path = ARGS[1]

fea = IntSet()

for subject=1:10
    f = matopen(joinpath(data_path, subject_file(subject)))
    X = read(f, "X")
    y = read(f, "y")
    println("Subject ", subject)
    features = extract_features(X)
    scores = [abs(auc(vec(y),vec(features[:,i]))-0.5) for i=1:size(features, 2)]
    for i=sortperm(scores, rev=true)[1:25]
        push!(fea, i)
    end
end
println(fea)
println(length(fea))

fea = Int[x for x=fea]

x_train = zeros(0, length(fea))
y_train = zeros(0, 1)

for subject=1:10
    f = matopen(joinpath(data_path, subject_file(subject)))
    X = read(f, "X")
    y = read(f, "y")
    println("Subject ", subject)
    x_train = vcat(x_train, extract_features(X)[:,fea])
    y_train = vcat(y_train, y)
end

forest = fit(x_train, vec(y_train), classification_forest_options(num_trees=100))

for subject=11:16
    f = matopen(joinpath(data_path, subject_file(subject)))
    X = read(f, "X")
    y = read(f, "y")
    sfreq = read(f, "sfreq")
    
    println("Subject ", subject)

    (b,a) = low_pass_filter(sfreq, 40)
    apply_filter!(X, b, a)
    x_test = extract_features(X)[:,fea]
    res    = predict(forest, x_test)
    println(@sprintf("--Test Accuracy: %0.2f%%", accuracy(res, vec(y))*100))
end