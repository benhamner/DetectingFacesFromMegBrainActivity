require("src/helpers.jl")
using Color
using DataFrames
using Gadfly
using MAT
using PyCall

@pyimport scipy.cluster.hierarchy as hcluster
@pyimport scipy.spatial.distance as dist
@pyimport mne

data_path   = ARGS[1]
output_path = ensure_empty_directory_exists(ARGS[2])

grand_average_path = joinpath(output_path, "GrandAverage")
fft_path           = joinpath(output_path, "FFT")
channel_path       = joinpath(output_path, "Channels")
z_score_path       = joinpath(output_path, "ZScore")
difference_path    = joinpath(output_path, "Difference")
topomaps_path      = joinpath(output_path, "Topomaps")
mkdir(grand_average_path)
mkdir(fft_path)
mkdir(channel_path)
mkdir(z_score_path)
mkdir(difference_path)
mkdir(topomaps_path)

rainbow = Scale.ContinuousColorScale(Scale.lab_gradient(ColorValue[color(c) for c=["#3288bd","#99d594","#e6f598","#fee08b","#fc8d59","#d53e4f"]]...))

function plot_grand_average(average, title)
    channels = repmat([1:size(average, 1)], 1, size(average, 2)) 
    time     = repmat(transpose([1:size(average, 2)]), size(average, 1), 1)
    df = DataFrame(Response=vec(average), Channels=vec(channels), Time=vec(time))
    p = plot(df, x="Time", y="Channels", color="Response", Geom.rectbin, rainbow)
    draw(PNG(joinpath(grand_average_path, @sprintf("%s.png", title)), 28cm, 21.6cm), p)

    grand_average = reshape(mean(average, 1), size(average,2))
    df_grand_average = DataFrame(Response=grand_average, Time=[1:length(grand_average)])
    p = plot(df_grand_average, x="Time", y="Response", Geom.line)
    draw(PNG(joinpath(grand_average_path, @sprintf("Average-%s.png", title)), 28cm, 21.6cm), p)
end

function plot_z_score(X_face, X_no_face, title)
    num_channels     = size(X_face,2)
    num_time_samples = size(X_no_face,3)
    difference = reshape((mean(X_face, 1)-mean(X_no_face, 1)), num_channels, num_time_samples)
    variance   = reshape(var(cat(1, X_face, X_no_face), 1), num_channels, num_time_samples)
    score = difference ./ variance

    # try showing in a better order
    res = hcluster.complete(dist.pdist(score, metric="correlation"))
    I   = sortperm(hcluster.fcluster(res, 5.0))
    I   = sortperm(vec(maximum(score,2)))

    score = score[I,:]
    channels = repmat([1:num_channels], 1, num_time_samples) 
    time     = repmat(transpose([1:num_time_samples]), num_channels, 1)
    df = DataFrame(ZScore=vec(score), Channels=vec(channels), Time=vec(time), Difference=vec(difference))
    axes = Coord.cartesian(xmin=1, xmax=num_time_samples, ymin=1, ymax=num_channels)
    p = plot(df, x="Time", y="Channels", color="ZScore", Geom.rectbin, rainbow, axes)
    writecsv(joinpath(z_score_path, @sprintf("%s.csv", title)), score)
    draw(PNG(joinpath(z_score_path, @sprintf("%s.png", title)), 2*28cm, 2*21.6cm), p)
    p = plot(df, x="Time", y="Difference", color="Channels", Geom.line, rainbow)
    draw(PNG(joinpath(difference_path, @sprintf("%s.png", title)), 2*28cm, 2*21.6cm), p)
end

function plot_fft(X, sampling_rate, title)
    power = vec(mean(abs(rfft(X, 3)), (1,2)))
    frequencies = linspace(sampling_rate/2/length(power), sampling_rate/2, length(power))
    df = DataFrame(Frequencies=frequencies, Power=power)
    p = plot(df, x="Frequencies", y="Power", Geom.line)
    draw(PNG(joinpath(fft_path, @sprintf("FFT-%s.png", title)), 28cm, 21.6cm), p)
end

function plot_channel_averages(face, no_face, subject_channel_path)
    time = [1:size(face, 2)]
    for channel=1:size(face, 1)
        df = vcat(DataFrame(Time=time, Value=vec(face[channel,:]),    Stimulus="Face"),
                  DataFrame(Time=time, Value=vec(no_face[channel,:]), Stimulus="No Face"))
        p = plot(df, x="Time", y="Value", color="Stimulus", Geom.line)
        draw(PNG(joinpath(subject_channel_path, @sprintf("%d.png", channel)), 28cm, 21.6cm), p)
    end
end

function plot_topomaps(face, no_face, times, subject)
    num_channels = size(face, 1)
    layout = mne.layouts[:read_layout]("Vectorview-all")
    evoked = mne.fiff[:Evoked](pyeval("None"))
    evoked[:data]  = face-no_face
    evoked[:times] = times
    evoked[:info]  = {"ch_names" => layout[:names],
                      "chs" => [{"kind"=>1,"unit"=>112} for i=1:num_channels],
                      "nchan"=>num_channels}
    p = mne.viz[:plot_evoked_topomap](evoked,
                                      times=[0.1:0.1:1.0],
                                      layout=layout,
                                      proj=pyeval("False"),
                                      size=3,
                                      show=pyeval("False"))
    p[:savefig](joinpath(topomaps_path, @sprintf("Subject%d.png", subject)))
end

for subject=train_subjects
    f = matopen(joinpath(data_path, subject_file(subject)))
    X     = read(f, "X")
    sfreq = read(f, "sfreq")
    y     = read(f, "y")
    tmin  = read(f, "tmin")
    tmax  = read(f, "tmax")
    times = linspace(tmin, tmax, size(X,3))
    plot_fft(X, sfreq, @sprintf("Subject %d-Face", subject))
    plot_z_score(X[vec(y).==1,:,:], X[vec(y).!=1,:,:], @sprintf("Subject %d", subject))
    #(b,a) = power_filter(sfreq, 50)
    #apply_filter!(X, b, a)
    #(b,a) = power_filter(sfreq, 100)
    #apply_filter!(X, b, a)
    (b,a) = low_pass_filter(sfreq, 40)
    apply_filter!(X, b, a)
    plot_fft(X, sfreq, @sprintf("After Subject %d-Face", subject))
    num_channels     = size(X,2)
    num_time_samples = size(X,3)
    face    = reshape(mean(X[vec(y).==1,:,:], 1), num_channels, num_time_samples)
    no_face = reshape(mean(X[vec(y).!=1,:,:], 1), num_channels, num_time_samples)
    plot_topomaps(face, no_face, times, subject)
    plot_grand_average(face, @sprintf("Subject %d-Face", subject))
    plot_grand_average(no_face, @sprintf("Subject %d-No Face", subject))
    plot_grand_average(face-no_face, @sprintf("Subject %d-Face-No Face", subject))
    subject_channel_path = ensure_empty_directory_exists(joinpath(channel_path, @sprintf("Subject%02d", subject)))
    plot_channel_averages(face, no_face, subject_channel_path)
end
