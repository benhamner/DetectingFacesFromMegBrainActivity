require("src/helpers.jl")

data_path   = ARGS[1]
output_path = ensure_empty_directory_exists(ARGS[2])

for subject=train_subjects
    println("Subject ", subject)
    X, y, sfreq = read_subject(subject)
    (b,a) = low_pass_filter(sfreq, 40)
    apply_filter!(X, b, a)
    num_channels = size(X,2)
    channel_performance = zeros(num_channels, 2)
    channel_performance[1:end, 1] = 1:num_channels
    for channel=1:num_channels
        println("--", channel)
        features = extract_channel_features(X, channel)
        x_train, y_train, x_test, y_test = split_train_test(features, vec(y), seed=1)
        zmuv = fit(features, ZmuvOptions())
        x_train = transform(zmuv, x_train)
        x_test  = transform(zmuv, x_test)

        forest = fit(x_train, y_train, classification_forest_options(num_trees=10))
        res    = predict(forest, x_test)
        channel_performance[channel, 2] = accuracy(res, y_test)*100
    end

    writecsv(joinpath(output_path, @sprintf("%d.csv", subject)), channel_performance)
end

