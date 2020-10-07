const WAVEFRONT_INTRINSICS = GCNIntrinsic[]

"""
    wfred(op::Function, val::T) where T -> T

Performs a wavefront-wide reduction on `val` in each lane, and returns the
result. A limited subset of functions are available to be passed as `op`. When
`op` is one of `(+, max, min, &, |, ⊻)`, `T` may be
`<:Union{Cint, Clong, Cuint, Culong}`. When `op` is one of `(+, max, min)`,
`T` may also be `<:Union{Float32, Float64}`.
"""
wfred

# FIXME: Float16

for (name,op) in ((:add,typeof(+)), (:max,typeof(max)), (:min,typeof(min)))
    for jltype in (Cint, Clong, Cuint, Culong, Float32, Float64)
        push!(WAVEFRONT_INTRINSICS, GCNIntrinsic(Symbol("wfred_$name"); roclib=:ockl, inp_args=(jltype,), out_arg=jltype))
        push!(WAVEFRONT_INTRINSICS, GCNIntrinsic(Symbol("wfscan_$name"); roclib=:ockl, inp_args=(jltype,Bool), out_arg=jltype))
    end
    @eval @inline wfred(::$op, x) = $(Symbol("wfred_$name"))(x)
    @eval @inline wfscan(::$op, x, inclusive::Bool) = $(Symbol("wfscan_$name"))(x, inclusive)
end
for (name,op) in ((:and,typeof(&)), (:or,typeof(|)), (:xor,typeof(⊻)))
    for jltype in (Cint, Clong, Cuint, Culong)
        push!(WAVEFRONT_INTRINSICS, GCNIntrinsic(Symbol("wfred_$name"); roclib=:ockl, inp_args=(jltype,), out_arg=jltype))
        push!(WAVEFRONT_INTRINSICS, GCNIntrinsic(Symbol("wfscan_$name"); roclib=:ockl, inp_args=(jltype,Bool), out_arg=jltype))
    end
    @eval @inline wfred(::$op, x) = $(Symbol("wfred_$name"))(x)
    @eval @inline wfscan(::$op, x, inclusive::Bool) = $(Symbol("wfscan_$name"))(x, inclusive)
end

for intr in WAVEFRONT_INTRINSICS
    inp_vars = [gensym() for _ in 1:length(intr.inp_args)]
    inp_expr = [:($(inp_vars[idx])::$arg) for (idx,arg) in enumerate(intr.inp_args)]
    libname = Symbol("__$(intr.roclib)_$(intr.rocname)_$(intr.suffix)")
    @eval @inline function $(intr.jlname)($(inp_expr...))
        y = _intr($(Val(libname)), $(intr.out_arg), $(inp_expr...))
        return $(intr.isinverted ? :(1-y) : :y)
    end
end
