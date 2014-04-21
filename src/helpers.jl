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
    println(N)
    (b, a)  = signal.butter(N, Wn)
    (b, a)
end

function power_filter(sampling_rate, power_frequency=50)
    @pyimport scipy.signal as signal
    ws = [power_frequency-2,power_frequency+2]/(sampling_rate/2.0)
    wp = [power_frequency-3,power_frequency+3]/(sampling_rate/2.0)
    (N, Wn) = signal.buttord(wp=wp, ws=ws, gpass=0.1, gstop=40.0)
    println(N)
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