using DataFrames
using DataStructures

require("src/helpers.jl")

data_path   = ARGS[1]
output_path = ensure_empty_directory_exists(ARGS[2])

channels = IntSet()
counter  = counter(Int)

for subject=train_subjects
    println("Subject ", subject)
    channel_performance = readcsv(joinpath(data_path, @sprintf("%d.csv", subject)), Float64)

    p = sortperm(vec(channel_performance[:,2]), rev=true)
    channels_sorted   = [1:length(p)][p]
    channels_selected = channels_sorted[1:20]
    print(channels_selected', "\n")
    for c=channels_selected
        push!(channels, c)
        add!(counter,   c)
    end
end

#print(@sprintf("Number of Channels Selected: %d\n", length(channels)))
#for t=sortby([x for x=counter], x->-x[2])
#    print(t, "\n")
#end

counts = sortby([x for x=counter], x->-x[2])
selected_counts = DataFrame(Channel=Int[x[1] for x=counts], Count=[x[2] for x=counts])
writetable(joinpath(output_path, "SelectedCounts.csv"), selected_counts)
writetable(joinpath(output_path, "CombinedSelected.csv"), selected_counts[selected_counts[:Count].>=3,:])
