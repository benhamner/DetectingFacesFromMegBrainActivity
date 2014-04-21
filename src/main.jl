require("src/helpers.jl")
using MachineLearning
using MAT

data_path = ARGS[1]

extract_trial(X, trial_num) = reshape(X[trial_num,:,:], (size(X,2),size(X,3)))
moving_average(x, n) = filt(ones(n)/n, [1.0], float64(x))

function extract_trial_fft_features(trial, n_moving_average)
    time_samples = size(trial, 2)
    trial_fft = abs(rfft(trial, 2))
    downsample = n_moving_average:n_moving_average:size(trial_fft, 2)
    res = zeros(size(trial, 1), length(downsample))
    for i=1:size(trial, 1)
        res[i,:] = moving_average(vec(trial_fft[i,:]), n_moving_average)[downsample]
    end
    vec(res)
end

function extract_trial_time_features(trial, n_moving_average)
    time_samples = size(trial, 2)
    downsample = n_moving_average:n_moving_average:size(trial, 2)
    res = zeros(size(trial, 1), length(downsample))
    for i=1:size(trial, 1)
        res[i,:] = moving_average(vec(abs(trial[i,:])), n_moving_average)[downsample]
    end
    vec(res)
end

function extract_features(X)
    n_moving_average = 20
    num_trials   = size(X, 1)
    num_channels = size(X, 2)
    fft_features_per_channel  = length(n_moving_average:n_moving_average:floor(size(X,3)/2))
    time_features_per_channel = length(n_moving_average:n_moving_average:floor(size(X,3)))
    fft_features  = zeros(num_trials, num_channels*fft_features_per_channel)
    time_features = zeros(num_trials, num_channels*time_features_per_channel)
    for i=1:num_trials
        trial = extract_trial(X, i)
        fft_features[i, :]  = extract_trial_fft_features(trial, n_moving_average)
        time_features[i, :] = extract_trial_time_features(trial, n_moving_average)
    end
    #time_features
    hcat(fft_features, time_features)
end

function evaluate_subject(X, y)
    features = extract_features(X)
    x_train, y_train, x_test, y_test = split_train_test(features, vec(y))
    scores = [abs(auc(vec(y_train),vec(x_train[:,i]))-0.5) for i=1:size(features, 2)]
    fea = sortperm(scores, rev=true)[1:100]
    println(@sprintf("--Score: %0.4f", scores[fea[1]]))
    println(@sprintf("--Score: %0.4f", scores[fea[end]]))
    forest = fit(x_train[:,fea], y_train, classification_forest_options(num_trees=100))
    res    = predict(forest, x_test[:,fea])
    println(@sprintf("--Test Accuracy: %0.2f%%", accuracy(res, y_test)*100))
end

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
    println("Subject ", subject)
    x_test = extract_features(X)[:,fea]
    res    = predict(forest, x_test)
    println(@sprintf("--Test Accuracy: %0.2f%%", accuracy(res, vec(y))*100))
end