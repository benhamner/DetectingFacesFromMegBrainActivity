require("src/helpers.jl")

data_path   = ARGS[1]
# output_path = ensure_empty_directory_exists(ARGS[2])

channels = IntSet()

for subject=train_subjects
    println("Subject ", subject)
    channel_performance = readcsv(joinpath(data_path, @sprintf("%d.csv", subject)), Float64)

    p = sortperm(vec(channel_performance[:,2]), rev=true)
    channels_sorted   = [1:length(p)][p]
    channels_selected = channels_sorted[1:20]
    print(channels_selected', "\n")
    for c=channels_selected
        push!(channels, c)
    end
end

print(@sprintf("Number of Channels Selected: %d\n", length(channels)))