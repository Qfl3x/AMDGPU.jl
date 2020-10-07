@testset "Wavefront Operations" begin
    function reduce_kernel(op,X)
        idx = AMDGPU.workitemIdx().x
        X[1] = AMDGPU.wfred(op,X[idx])
        nothing
    end
    function scan_kernel(op,X)
        idx = AMDGPU.workitemIdx().x
        X[1] = AMDGPU.wfscan(op,X[idx],true)
        nothing
    end

    for T in (Cint, Clong, Cuint, Culong)
        A = rand(T(1):T(100), 64)
        for op in (Base.:+, max, min, Base.:&, Base.:|, Base.:⊻)
            RA = ROCArray(A)
            wait(@roc groupsize=64 reduce_kernel(op,RA))
            @test Array(RA)[1] == reduce(op,A)

            RA = ROCArray(A)
            wait(@roc groupsize=64 scan_kernel(op,RA))
            @show Array(RA)
        end
    end
    for T in (Float32, Float64)
        A = rand(T, 64)
        for op in (Base.:+, max, min)
            RA = ROCArray(A)
            wait(@roc groupsize=64 reduce_kernel(op,RA))
            @test Array(RA)[1] ≈ reduce(op,A)

            RA = ROCArray(A)
            wait(@roc groupsize=64 scan_kernel(op,RA))
            @show Array(RA)
        end
    end
end
