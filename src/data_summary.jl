using MAT

data_path = ARGS[1]

for train_subject=1:16
    f = matopen(joinpath(data_path, @sprintf("train_subject%02d.mat", train_subject)))
    X = read(f, "X")
    y = read(f, "y")
    sampling_frequency = read(f, "sfreq")
    tmin = read(f, "tmin")
    samples_before = int(-tmin*sampling_frequency)
    println("Subject ", train_subject)
    println("--Sampling Frequency: \t", sampling_frequency)
    println("--Time Before Stimuli:\t", tmin)
end
