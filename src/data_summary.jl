using MAT

data_path = ARGS[1]

is_train_subject(subject) = subject <= 16

function subject_file(subject)
    if is_train_subject(subject)
        return @sprintf("train_subject%02d.mat", subject)
    else
        return @sprintf("test_subject%02d.mat",  subject)
    end
end

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
    samples_before = int(-tmin*sampling_frequency)
    println("--Sampling Frequency: \t", sampling_frequency)
    println("--Time Before Stimuli:\t", tmin)
end
