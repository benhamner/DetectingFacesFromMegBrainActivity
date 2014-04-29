using MachineLearning
using GLM
using MAT
using PyCall
@pyimport sklearn.linear_model as linear_model

function ensure_empty_directory_exists(directory)
    try
        run(`rm -r $directory`)
    end
    mkdir(directory)
    directory
end

function read_subject(subject)
    f = matopen(joinpath(data_path, subject_file(subject)))
    X = read(f, "X")
    y = read(f, "y")
    sfreq = read(f, "sfreq")
    (X, y, sfreq)
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

function extract_channel_features(X, channel)
    time_samples = [126:5:size(X,3)]
    features = reshape(X[:,channel,time_samples], size(X,1), length(time_samples))
    float64(features)
end

function evaluate_subject(subject)
    X, y, sfreq = read_subject(subject)
    (b,a) = low_pass_filter(sfreq, 40)
    apply_filter!(X, b, a)
    features = extract_features(X)
    x_train, y_train, x_test, y_test = split_train_test(features, vec(y), seed=1)
    zmuv = fit(features, ZmuvOptions())
    x_train = transform(zmuv, x_train)
    x_test  = transform(zmuv, x_test)
    println("****50 Features****")
    run_models(x_train, y_train, x_test, y_test, 50)

    println("****100 Features****")
    run_models(x_train, y_train, x_test, y_test, 100)

    println("****250 Features****")
    run_models(x_train, y_train, x_test, y_test, 250)

    println("****500 Features****")
    run_models(x_train, y_train, x_test, y_test, 500)
end

function run_models(x_train, y_train, x_test, y_test, num_features)
    scores = [abs(auc(vec(y_train),vec(x_train[:,i]))-0.5) for i=1:size(x_train, 2)]
    fea = sortperm(scores, rev=true)[1:num_features]
    println(@sprintf("--Score: %0.4f", scores[fea[1]]))
    println(@sprintf("--Score: %0.4f", scores[fea[end]]))
    y_train_mod = Float64[yy==1?1:-1 for yy=vec(y_train)]
    #try
    #    m = fit(GlmMod, x_train[:,fea], y_train_mod, Normal())
    #    res = Float64[r>0?1:0 for r=vec(x_test[:,fea]*coef(m))]
    #    println(@sprintf("--Glm   Accuracy: %0.2f%%", accuracy(res, y_test)*100))
    #catch
    #    println("--Glm Error")
    #end
    forest = fit(x_train[:,fea], y_train, classification_forest_options(num_trees=10))
    res    = predict(forest, x_test[:,fea])
    println(@sprintf("--RF10  Accuracy: %0.2f%%", accuracy(res, y_test)*100))

    forest = fit(x_train[:,fea], y_train, classification_forest_options(num_trees=100))
    res    = predict(forest, x_test[:,fea])
    println(@sprintf("--RF100 Accuracy: %0.2f%%", accuracy(res, y_test)*100))
    model = linear_model.LogisticRegression(C=1.0, penalty="l1")
    model[:fit](x_train, y_train_mod)
    res = Float64[r>0?1:0 for r=vec(model[:predict](x_test))]
    println(@sprintf("--Logit Accuracy: %0.2f%%", accuracy(res, y_test)*100))
    #net    = fit(x_train[:,fea], y_train, neural_net_options(stop_criteria=StopAfterIteration(1000)))
    #res    = predict(net, x_test[:,fea])
    #println(@sprintf("--Net   Accuracy: %0.2f%%", accuracy(res, y_test)*100))
end