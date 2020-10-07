# HSA dispatch packet offsets
_packet_names = fieldnames(HSA.KernelDispatchPacket)
_packet_offsets = fieldoffset.(HSA.KernelDispatchPacket, 1:length(_packet_names))

@generated function _intr(::Val{fname}, out_arg, inp_args...) where {fname,}
    JuliaContext() do ctx
        inp_exprs = [:( inp_args[$i] ) for i in 1:length(inp_args)]
        inp_types = [inp_args...]
        out_type = convert(LLVMType, out_arg.parameters[1], ctx)

        # create function
        bool_types = map(x->x===Bool, inp_types)
        T_bool = LLVM.Int1Type(ctx)
        param_types = LLVMType[convert.(LLVMType, inp_types, Ref(ctx))...]
        llvm_f, _ = create_function(out_type, param_types)
        mod = LLVM.parent(llvm_f)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            # call the intrinsic
            intr_typ = LLVM.FunctionType(out_type, param_types)
            intr = LLVM.Function(mod, string(fname), intr_typ)
            params = map(x->bool_types[x[1]] ? trunc!(builder, x[2], T_bool) : x[2], enumerate(parameters(llvm_f)))
            value = call!(builder, intr, [params...])
            ret!(builder, value)
        end

        call_function(llvm_f, out_arg.parameters[1], Tuple{inp_args...}, Expr(:tuple, inp_exprs...))
    end
end

struct GCNIntrinsic
    jlname::Symbol
    rocname::Symbol
    isbroken::Bool # please don't laugh...
    isinverted::Bool
    # FIXME: Input/output types
    inp_args::Tuple
    out_arg::Type
    roclib::Symbol
    suffix::Symbol
end

GCNIntrinsic(jlname, rocname=jlname; isbroken=false, isinverted=false,
             inp_args=(), out_arg=(), roclib=:ocml, suffix=fntypes[first(inp_args)]) =
    GCNIntrinsic(jlname, rocname, isbroken, isinverted, inp_args, out_arg, roclib, suffix)

