
is_train_subject(subject) = subject <= 16

function subject_file(subject)
    if is_train_subject(subject)
        return @sprintf("train_subject%02d.mat", subject)
    else
        return @sprintf("test_subject%02d.mat",  subject)
    end
end
