
using CUDA


function main()

    n_gpus = 0
    if CUDA.functional()
        n_gpus = length(devices())
    end
    println(repeat("0",n_gpus))

end


main()


