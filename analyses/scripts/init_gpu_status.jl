
using CUDA


function main()

    n_gpus = length(devices())
    println(repeat("0",n_gpus))

end


main()


