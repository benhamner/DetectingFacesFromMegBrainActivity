require("src/helpers.jl")
using MachineLearning
using MAT

data_path = ARGS[1]

for subject=train_subjects
    f = matopen(joinpath(data_path, subject_file(subject)))
    X = read(f, "X")
    y = read(f, "y")
    sfreq = read(f, "sfreq")

    #(b,a) = low_pass_filter(sfreq, 40)
    #apply_filter!(X, b, a)
    println("Subject ", subject)
    evaluate_subject(X, y)
end