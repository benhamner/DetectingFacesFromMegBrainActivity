require("src/helpers.jl")
using DataFrames
using Gadfly
using MAT
using PyCall

data_path   = ARGS[1]
output_path = ARGS[2]

function butterworth_filter()
    @pyimport scipy.signal as signal
    (N, Wn) = signal.buttord(wp=0.2, ws=0.3, gpass=0.1, gstop=20.0)
    (b, a)  = signal.butter(N, Wn)
    (b, a)
end

function plot_grand_average(average, title)
    channels = repmat([1:size(average, 1)], 1, size(average, 2)) 
    time     = repmat(transpose([1:size(average, 2)]), size(average, 1), 1)
    df = DataFrame(Response=vec(average), Channels=vec(channels), Time=vec(time))
    p = plot(df, x="Time", y="Channels", color="Response", Geom.rectbin)
    draw(PNG(joinpath(output_path, @sprintf("%s.png", title)), 20cm, 15cm), p)

    grand_average = reshape(mean(average, 1), size(average,2))
    df_grand_average = DataFrame(Response=grand_average, Time=[1:length(grand_average)])
    p = plot(df_grand_average, x="Time", y="Response", Geom.line)
    draw(PNG(joinpath(output_path, @sprintf("Average-%s.png", title)), 20cm, 15cm), p)
end

for subject=train_subjects
    f = matopen(joinpath(data_path, subject_file(subject)))
    X = read(f, "X")
    num_channels     = size(X,2)
    num_time_samples = size(X,3)
    y = read(f, "y")
    face    = reshape(mean(X[vec(y).==1,:,:], 1), num_channels, num_time_samples)
    no_face = reshape(mean(X[vec(y).!=1,:,:], 1), num_channels, num_time_samples)
    plot_grand_average(face, @sprintf("Subject %d-Face", subject))
    plot_grand_average(no_face, @sprintf("Subject %d-No Face", subject))
end
