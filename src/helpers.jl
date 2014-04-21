using PyCall

function ensure_empty_directory_exists(directory)
    try
        run(`rm -r $directory`)
    end
    mkdir(directory)
    directory
end

is_train_subject(subject) = subject <= 16

train_subjects = [1:16]
test_subjects  = [17:23]

function subject_file(subject)
    if is_train_subject(subject)
        return @sprintf("train_subject%02d.mat", subject)
    else
        return @sprintf("test_subject%02d.mat",  subject)
    end
end

function low_pass_filter(sampling_rate, frequency)
    @pyimport scipy.signal as signal
    (N, Wn) = signal.buttord(wp=(frequency-10)/(sampling_rate/2), ws=frequency/(sampling_rate/2), gpass=1.0, gstop=30.0)
    (b, a)  = signal.butter(N, Wn)
    (b, a)
end

function power_filter(sampling_rate, power_frequency=50)
    @pyimport scipy.signal as signal
    ws = [power_frequency-2,power_frequency+2]/(sampling_rate/2.0)
    wp = [power_frequency-3,power_frequency+3]/(sampling_rate/2.0)
    (N, Wn) = signal.buttord(wp=wp, ws=ws, gpass=0.1, gstop=40.0)
    (b, a)  = signal.butter(N, Wn)
    (b, a)
end

function apply_filter!(X, b, a)
    for i=1:size(X, 1)
        for j=1:size(X,2)
            X[i,j,:] = filt(b, a, float64(vec(X[i,j,:])))
        end
    end
end

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
