require("src/helpers.jl")
using MAT

data_path = ARGS[1]

for subject=1:23
    f = matopen(joinpath(data_path, subject_file(subject)))
    X = read(f, "X")
    if is_train_subject(subject)
        println("Train Subject ", subject)
        y = read(f, "y")
        println("--Number of Faces:    \t", sum(y))
    else
        println("Test  Subject ", subject)
    end
    sampling_frequency = read(f, "sfreq")
    tmin = read(f, "tmin")
    X = read(f, "X")
    samples_before = int(-tmin*sampling_frequency)
    println("--Sampling Frequency: \t", sampling_frequency)
    println("--Time Before Stimuli:\t", tmin)
    println("--Size of X:          \t", size(X))
end
