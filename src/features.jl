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
    #hcat(fft_features, time_features)
    time_features
end

for subject=1:16
    println("Subject ", subject)
    f = matopen(joinpath(data_path, subject_file(subject)))
    X = read(f, "X")
    y = read(f, "y")
    features = extract_features(X)
    println(size(features))
    println(auc(vec(y), vec(features[:,1])))
    scores = [auc(vec(y),vec(features[:,i])) for i=1:size(features, 2)]
    println(@sprintf("Min Score: %0.4f", minimum(scores)))
    println(@sprintf("Max Score: %0.4f", maximum(scores)))
end