require("src/helpers.jl")

data_path = ARGS[1]

for subject=train_subjects
    println("Subject ", subject)
    evaluate_subject(subject)
end